# Task 4 report: Scheduler on the Encoder

**Status:** DONE_WITH_CONCERNS (minor concerns, listed at the end)
**Commit:** 8eec341 "Metal-native M2: scheduler records into Metal::Encoder (hoisted uploads as an untracked command buffer, dynamic state on the encoder, depth-stencil states cached)"

## What was implemented

- `vk_scheduler.h/.cpp` converted to Metal:
  - Members: `Metal::Device& device`, `Metal::Encoder encoder{device}`, `Metal::Encoder upload_encoder{device}`,
    `std::mutex upload_mutex`, `bool upload_open`, `Metal::DepthStencilCache depth_stencil{device}`.
    The Semaphore, the two CommandPools, `current_cmdbuf`, `upload_cmdbuf`, Tracy VkCtx, `SubmitInfo`,
    `SubmitFunc`, `GetWorkSemaphore`, `AllocateWorkerCommandBuffers`, `DynamicState::Commit(Instance, cmdbuf)`
    and `pending_polls` are gone. Constructor is `Scheduler(Metal::Device&, bool threaded)`.
  - `RecordChunk` commands take `Metal::Encoder&`; `Record` calls `func(encoder)` in direct mode; the
    recorder thread runs `chunk->Execute(encoder)` and is named `bb:MtlRecorder`. `RecordHostCopy` and
    `SignalAfterHostCopies` closures take `Metal::Encoder&`.
  - `CommandBuffer()` replaced by `Metal::Encoder& DirectEncoder()` (same sync + direct-mode switch + trace).
  - `SubmitExecution()`: submit_mutex; tick = CurrentTick(); on_submit(); EndRendering(); profiler mark
    (kept); SyncRecording(); WaitHostCopies(); under upload_mutex: SubmitUntracked the upload encoder if open;
    `ImGui::Core::TextureManager::Submit(); // ImGui textures: M5`; `encoder.Submit()` with queue_submit stats;
    ASSERT_MSG tick matches; `dynamic_state.Invalidate()` (Encoder::Submit clears render state);
    wakes the priority-ops thread; PopPendingOperations().
  - `Flush()` (no args), `Finish()` (t = CurrentTick; Flush; device.Wait(t)), `Wait(tick)` (flush if
    tick >= CurrentTick; device.Wait), `PopPendingOperations()` (no 1-in-32 throttle; runs ops whose tick
    `device.IsFree`), `CurrentTick()`/`IsFree()` forward to the device, `GetDevice()`, `GetDepthStencilCache()`,
    `SetSubmitCallback(std::function<void()>)`, static `submit_mutex` kept.
  - `RecordHoistedCopy(BufferRef, BufferRef, span<const VideoCore::BufferCopy>, bool ordered)`: under
    upload_mutex, `upload_open = true` and one `upload_encoder.CopyBuffer` per region; `ordered` unused.
  - `RenderAttachment`/`RenderState` per the brief with memberwise `operator==` and `RenderState::ToDesc()`.
    `BeginRendering` keeps the merge-if-equal logic and BB_PASS_STATS, records `e.BeginRenderPass(desc)`;
    `EndRendering` records `e.EndRenderPass()`.
  - `DynamicState::CommitWith(Metal::DepthStencilCache&, Emit&&)`: viewport[0] -> SetViewport, scissor[0] ->
    SetScissor, depth/stencil fields -> DepthStencilKey -> `depth_stencil.Get(key)` on the producer thread ->
    SetDepthStencil, stencil references -> SetStencilReference, depth bias (folded enable) -> SetDepthBias,
    cull -> SetCullMode (FrontAndBack: logged once, Back), front face -> SetFrontFace, blend constants ->
    SetBlendColor. Color write masks, rasterizer discard, primitive restart, line width, depth bounds and
    feedback loop keep their setters and dirty flags, are cleared at commit and emit nothing, under the
    comment "Metal: these belong to the pipeline key (M3) or have no equivalent."
- New `gpu/shadps4/video_core/buffer_cache/buffer_copy.h` (`VideoCore::BufferCopy{src_offset, dst_offset, size}`).
- Deleted `vk_semaphore.{h,cpp}` and `vk_resource_pool.{h,cpp}`. `gpu/CMakeLists.txt` globs
  `shadps4/video_core/*.cpp` (CONFIGURE_DEPENDS), so it lists nothing to remove and was not changed.

## Decisions

1. **`RenderAttachment::clear_stencil` added** (one extra bool). The brief's single `clear` cannot carry the
   rasterizer's independent depth clear and stencil clear (`vk_rasterizer.cpp:2972/2977` set them from
   separate registers). For the depth attachment `clear` = clear depth and `clear_stencil` = clear stencil;
   for colors `clear_stencil` is unused. Additive, so callers written to the brief's shape still compile.
2. **Viewport is converted to Metal's convention in CommitWith** with `Metal::RasterUtil::ToMetalViewport`
   (y + h, -h, depth clamped to [0,1]), because `Encoder::SetViewport` passes its values straight to
   `setViewport:`. This is the convention the M1 path (mtl_frame.mm) uses and the raster test proved.
3. **Scissor negative offsets are cut in CommitWith** (Metal::Scissor is unsigned; the encoder only clamps the
   far edges to the pass).
4. **Depth-stencil key is normalised** (compare stays Always / faces stay default when the test is off), and
   the depth/stencil dirty flags follow the old deferral rules: the compare op stays dirty while the depth test
   is off, stencil faces stay dirty while the stencil test is off (matches `AnyDirty()`). Stencil masks
   truncate to u8 (stencil is 8-bit).
5. **Upload lock held through the main `encoder.Submit()`** (the brief shows a scoped block that ends before
   it). Same as the old code: no hoisted copy recorded between the two commits can end up in a later upload
   command buffer than the submission that needs it. No deadlock: Submit takes no scheduler lock.
6. **Priority-ops thread waits for the tick to be committed before `device.Wait`**: `DeferPriorityOperation`
   stores CurrentTick() (not yet committed) and `Device::Wait` asserts a committed tick (the Vulkan timeline
   wait just blocked). The thread waits on `priority_pending_ops_cv` (re-checked every 1 ms, since another
   scheduler on the same device may commit the tick) until `tick < device.CurrentTick()`; SubmitExecution
   notifies it (taking its mutex first, so no lost wakeup).
7. **BbPassLog** keeps `Enabled()` and becomes `Pass(u64 tick, label)` (logs the submission's tick instead of
   a command-buffer number; `Assign` and the cb#N object naming are gone). The rasterizer caller
   (`vk_rasterizer.cpp:3662`) is converted in M3; Metal System Trace labels on command buffers are not set.
8. **GpuProfiler mark in SubmitExecution kept** (compiles; the profiler itself is Vulkan and is converted later).
9. `Scheduler(Metal::Device&, bool threaded)` has no default for `threaded` (brief signature).
10. `BbToggle::PendingPollLimit` is now unused (left in bbport_toggles.h).

## Tests and verification

- Object compile: `ninja -C out/gpu-arm64 CMakeFiles/bbgpu.dir/shadps4/video_core/renderer_vulkan/vk_scheduler.cpp.o`
  -> `[1/2] Building CXX object ...vk_scheduler.cpp.o`, no errors, no warnings (re-run after `touch` to confirm).
- Handle check on vk_scheduler.{h,cpp}: single hit, `vk_scheduler.h:825: [[nodiscard]] Metal::Device& GetDevice() noexcept {`
  — the `GetDevice\(\)` pattern matches the accessor the brief itself requires (returns Metal::Device, not
  vk::Device). No vk handle types; only vocabulary (vk::ClearValue, vk::Viewport, vk::Rect2D, enums).
- Core command (bbmetal_core + test_mtl_encoder): `PASS: Metal-native foundation (object, device ticks, encoder)`
  twice (plain and MTL_DEBUG_LAYER=1). Only noise: the pre-existing dsymutil "skipping debug map object with
  duplicate name" warning.
- The full bbgpu library no longer links (expected from this task on); pipeline_cache.h and buffer_cache.h
  include the deleted headers (converted in later tasks).

## Files changed

- gpu/shadps4/video_core/renderer_vulkan/vk_scheduler.h
- gpu/shadps4/video_core/renderer_vulkan/vk_scheduler.cpp
- gpu/shadps4/video_core/buffer_cache/buffer_copy.h (new)
- gpu/shadps4/video_core/renderer_vulkan/vk_semaphore.{h,cpp}, vk_resource_pool.{h,cpp} (deleted)

## Concerns

- The `clear_stencil` field (decision 1) extends the brief's exact shape; M3's rasterizer must set it.
- Layered rendering (`num_layers`) is dropped with the old RenderState (RenderPassDesc has no layer count);
  if the game renders to layered targets in one pass, M3 must handle it.
- Clear colors come from `clear_value.color.float32` per the brief; integer-format clears would need the
  uint32 view (Metal's clearColor is double either way).

---

## Fix round 1: per-scheduler logical ticks (Metal::Timeline)

**Commit:** see `git log` — "Metal-native M2: per-scheduler tick timelines over the shared device (deferred ops and waits see only their own scheduler's work)"

### Changes
- New `gpu/shadps4/video_core/renderer_metal/mtl_timeline.{h,cpp}` (plain C++, in `METAL_CORE_SOURCES`
  and the bbgpu `REMOVE_ITEM` list of `gpu/CMakeLists.txt`): `Metal::Timeline` with `CurrentTick()`,
  `KnownGpuTick()`, `IsFree(tick)`, `Wait(tick)`, `Committed(device_tick)` per the ruling, plus
  `PendingCount()` (the number of unpruned mappings, used by the test to check that the map stays bounded).
  - Mapping: a deque of {logical, device} ticks for logical ticks known+1 .. current-1. Each call refreshes:
    it drops entries from the front while their device tick is free (one queue completes in order) and raises
    `known`. `IsFree` has a lock-free fast path (`tick <= known`), and an uncommitted tick (`>= current`) is
    false. `Wait` asserts the tick was committed, looks up the device tick and calls `device.Wait` outside
    the lock. `Committed` asserts a non-zero device tick that is later than this timeline's previous one.
- Scheduler: owns `Metal::Timeline timeline{device}`. `SubmitExecution()` now returns the logical tick:
  `const u64 tick = timeline.Committed(encoder.Submit());` and asserts that it equals `CurrentTick()` read at
  the start under submit_mutex. `CurrentTick`, `IsFree`, the new `KnownGpuTick`, `Wait`,
  `PopPendingOperations` and the priority-ops thread all use the timeline. `Finish()` waits on the tick its
  own `SubmitExecution()` returned. New `Metal::Timeline& GetTimeline()`.
  - Priority-ops thread: because only this scheduler commits its timeline (and notifies under the mutex), it
    now uses an untimed `priority_pending_ops_cv.wait(lk, stoken, tick < timeline.CurrentTick())` and then
    `timeline.Wait`. This replaces the 1 ms re-check from decision 6.
- The `GetDevice()` accessor is kept (callers that need the Metal device itself).

### Covering test (TDD)
`TestTimelines` in `tests/test_mtl_encoder.mm` creates two Timelines and two Encoders on one Device. Commit
order: A, B, B, then A, where A's second command buffer waits on an MTLSharedEvent that the CPU signals later.
It checks that:
- each timeline numbers its own commits 1, 2, ... and advances independently
- B's tick 2 is free while A's tick 2 is not (device tick 2, which is B's first buffer, is free)
- A's `KnownGpuTick` is 1, and tick 0 is free while an uncommitted tick is not
- `Wait(A2)` blocks while the gate is closed (checked after 50 ms) and returns once A's own buffer completes
- over 300 interleaved commits, `KnownGpuTick` never goes back, waits work out of order, every committed tick is
  free after the last one, the next tick is not free, and `PendingCount()` reaches 0 on both timelines
  (mappings are pruned)

- RED: the core test compile before the Timeline existed:
  `tests/test_mtl_encoder.mm:22:10: fatal error: 'video_core/renderer_metal/mtl_timeline.h' file not found`.
  The class did not exist yet. The decisive check, `!a.IsFree(2)` with A2 gated, fails for any scheme based
  on device ticks, because device tick 2 is B's first buffer.
- GREEN, core command (ninja bbmetal_core + clang++ test + run plain + run with MTL_DEBUG_LAYER=1):
  `PASS: Metal-native foundation (object, device ticks, encoder)` twice. The only other output is the
  pre-existing dsymutil "duplicate name" warning and "Metal API Validation Enabled".
- Object compile: `ninja -C out/gpu-arm64 CMakeFiles/bbgpu.dir/shadps4/video_core/renderer_vulkan/vk_scheduler.cpp.o`
  -> `[1/2] Building CXX object ...vk_scheduler.cpp.o`, no errors or warnings.
- Handle check on vk_scheduler.{h,cpp}: there is only the one hit seen before,
  `vk_scheduler.h:837: [[nodiscard]] Metal::Device& GetDevice() noexcept {` (the accessor the brief requires).
