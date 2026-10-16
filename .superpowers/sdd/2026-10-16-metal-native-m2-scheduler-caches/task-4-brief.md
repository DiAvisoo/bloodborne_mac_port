### Task 4: Scheduler on the Encoder

**Files:**
- Modify: `gpu/shadps4/video_core/renderer_vulkan/vk_scheduler.{h,cpp}`
- Delete: `vk_semaphore.{h,cpp}`, `vk_resource_pool.{h,cpp}` (`CommandPool`, `DescriptorHeap`, generic pool)
- Modify: `gpu/CMakeLists.txt` (drop deleted sources if listed explicitly)

**Interfaces:**
- Consumes: `Metal::Device`, `Metal::Encoder` (incl. Task 2's `SubmitUntracked`, `DeclareIndirect`),
  `Metal::DepthStencilCache`, `Metal::RenderPassDesc`.
- Produces (M3 callers depend on these exact shapes):

```cpp
class Scheduler {
public:
    explicit Scheduler(Metal::Device& device, bool threaded);
    void Flush();                       // submits; no SubmitInfo
    void Finish();                      // t = Submit; device.Wait(t)
    void Wait(u64 tick);                // flushes first if tick >= CurrentTick()
    void PopPendingOperations();
    void BeginRendering(const RenderState& state);
    void EndRendering(std::source_location loc = std::source_location::current(), const char* detail = nullptr);
    void SetSubmitCallback(std::function<void()> callback);
    DynamicState& GetDynamicState();
    template <typename Func> void Record(Func&& func);   // Func(Metal::Encoder&)
    // unchanged: IsRecordingDeferred, ReserveRecordData, RecordData, KickRecording, SyncRecording,
    // WaitHostCopies, RecordHostCopy, SignalAfterHostCopies, WaitDeferredSignals, IsRendering,
    // IsRenderingWith, WillBeginRendering, direct_recordings, DeferOperation, DeferPriorityOperation
    u64 CurrentTick() const noexcept;   // device.CurrentTick()
    bool IsFree(u64 tick) const;        // device.IsFree(tick)
    Metal::Device& GetDevice() noexcept;
    Metal::DepthStencilCache& GetDepthStencilCache() noexcept;
    /// Hoisted copies: recorded now on the calling thread into the upload encoder, which is
    /// committed untracked right before the next submission.
    void RecordHoistedCopy(Metal::BufferRef src, Metal::BufferRef dst,
                           std::span<const BufferCopy> copies, bool ordered);
    static std::mutex submit_mutex;     // one committer at a time (Device::Commit requirement)
};
struct BufferCopy { u64 src_offset; u64 dst_offset; u64 size; };   // VideoCore-wide, same field order as vk::BufferCopy
struct RenderAttachment {
    void* texture{};                    // id<MTLTexture> (the image's own texture or a plain view)
    u32 level{}, slice{}, depth_plane{};
    bool clear{};
    vk::ClearValue clear_value{};
};
struct RenderState {
    std::array<RenderAttachment, 8> colors{};
    u32 num_colors{};
    RenderAttachment depth{};
    bool has_depth{}, has_stencil{}, stencil_only{};
    u32 width{}, height{};
    Metal::RenderPassDesc ToDesc() const;
    bool operator==(const RenderState&) const;   // memberwise (no memcmp: padding)
};
```

`BufferCopy` lives in a new small header `gpu/shadps4/video_core/buffer_cache/buffer_copy.h`
(namespace `VideoCore`), included by the scheduler and both caches.

- [ ] **Step 1: Convert** following the inventory (roadmap-linked agent report, summarised):
  - Members: `Metal::Device& device; Metal::Encoder encoder{device}; Metal::Encoder upload_encoder{device};`
    `bool upload_open{}; std::mutex upload_mutex; Metal::DepthStencilCache depth_stencil{device};`
    Remove `master_semaphore`/`Semaphore`, `CommandPool`s, `current_cmdbuf`, `upload_cmdbuf`,
    Tracy VkCtx, `SubmitInfo` (`vk_scheduler.h:76-98`).
  - `RecordChunk` stores `void(Metal::Encoder&)` callables; `Record` calls `func(encoder)` in direct
    mode and the recorder thread calls `cmd(encoder)`. Rename the recorder thread `bb:MtlRecorder`.
  - `CommandBuffer()` is deleted; add `Metal::Encoder& DirectEncoder()` that does what
    `CommandBuffer()` did before returning (sync the recorder, switch to direct mode) for the few
    callers that need immediate encoding (presenter/upscaler in M4/M5).
  - `SubmitExecution()` (was `.cpp:619-706`): lock `submit_mutex`; call the submit callback;
    `EndRendering()`; `SyncRecording(); WaitHostCopies();` then
    `{ std::scoped_lock l{upload_mutex}; if (upload_open) { upload_encoder.SubmitUntracked(); upload_open = false; } }`
    then `const u64 tick = encoder.Submit();` (assert `tick` equals the `CurrentTick()` read at the
    start), ImGui texture submit stays (it moves to Metal in M5; keep the call compiling or guard
    it with a TODO-free comment "ImGui textures: M5"), `PopPendingOperations()`.
  - `Finish()`: `const u64 t = CurrentTick(); Flush(); device.Wait(t);`. `Wait(tick)`: `if (tick >=
    CurrentTick()) Flush(); device.Wait(tick);`. `PopPendingOperations`: drop the 1-in-32 throttle,
    run ops whose tick `device.IsFree`. The priority-ops thread waits with `device.Wait`.
  - `RecordHoistedCopy`: under `upload_mutex`, `upload_open = true;` and
    `for (c : copies) upload_encoder.CopyBuffer({src.buffer, src.offset + c.src_offset}, {dst.buffer, dst.offset + c.dst_offset}, c.size);`
    (`ordered` needs nothing: one blit encoder executes in order).
  - `BeginRendering(state)`: keep the merge-if-equal logic and `BB_PASS_STATS`; record
    `Record([desc = state.ToDesc()](Metal::Encoder& e) { e.BeginRenderPass(desc); });`.
    `RenderState::ToDesc()` maps colors (texture, level, slice, depth_plane, clear, clear color from
    `clear_value.color.float32`), depth (texture, level, slice, has_stencil, stencil_only, clears from
    `clear_value.depthStencil`), width/height. `EndRendering` records `e.EndRenderPass()`.
  - `DynamicState::CommitWith` emits closures taking `Metal::Encoder&`:
    viewport[0] → `SetViewport`, scissor[0] → `SetScissor`, cull/front → `SetCullMode`/`SetFrontFace`
    (VkCullModeFlags 1=Front, 2=Back, 3 → Front+Back is not in Metal: log once, use Back),
    depth bias → `SetDepthBias(enabled ? constant : 0, enabled ? slope : 0, enabled ? clamp : 0)`,
    stencil references → `SetStencilReference`, blend constants → `SetBlendColor`, and the
    depth/stencil fields → build a `Metal::DepthStencilKey` **on the producer thread**, take
    `void* s = depth_stencil.Get(key)` and emit `e.SetDepthStencil(s)`. Fields Metal has no dynamic
    state for (color write masks, rasterizer discard, primitive restart, line width, depth bounds,
    feedback loop) keep their dirty tracking but emit nothing; add one comment above them: "Metal:
    these belong to the pipeline key (M3) or have no equivalent."
  - `SetSubmitCallback` takes `std::function<void()>`.
  - Delete `Commit(Instance, cmdbuf)` (`.cpp:734`), `BbPassLog` command-buffer naming (or set
    `MTLCommandBuffer.label` through `encoder.CommandBuffer()`), Tracy.
- [ ] **Step 2: Compile** `CMakeFiles/bbgpu.dir/shadps4/video_core/renderer_vulkan/vk_scheduler.cpp.o`.
  Expected: no errors. Run the handle check on `vk_scheduler.{h,cpp}`: no output (except
  `vk::ClearValue`, which is vocabulary).
- [ ] **Step 3: Core command** (unit tests unaffected) — expected PASS twice.
- [ ] **Step 4: Commit**: "Metal-native M2: scheduler records into Metal::Encoder (hoisted uploads as
  an untracked command buffer, dynamic state on the encoder, depth-stencil states cached)".

---

