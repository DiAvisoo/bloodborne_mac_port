# Metal-native M3: Pipelines and rasterizer on Metal — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The pipeline cache builds Metal pipelines directly and the rasterizer, shader HLE and
scene resolution record through `Metal::Encoder`; the interception layer (`mtl_vk_layer`,
`mtl_shadow`, `mtl_frame`) is deleted; the arm64 library and `bbgpu-host` link with
undefined-symbol errors on, with not-yet-converted presenter/upscaler/motion/overlay parts
stubbed behind `BB_METAL_NATIVE_STAGE`; `bbgpu-host --msl-check` builds every cached pipeline
through the new cache.

**Architecture:** Conversion in place (spec R1). The draw-side logic proven on `metal-backend`
(`mtl_frame.mm`: binding by shader slot map, draw preparation, tessellation runs, geometry
emulation, rect lists, indirect draws) moves into a renderer-side helper the rasterizer calls
inside `Scheduler::Record` closures; PSO building moves from `mtl_shadow.mm` into the pipeline
objects. New encoder capabilities go into `bbmetal_core` with GPU unit tests.

**Tech Stack:** C++20, Objective-C++ (ARC), Metal (macOS 15), SPIRV-Cross (via `mtl_shader`),
the project's `shim/common` log/assert.

**Spec:** `docs/superpowers/specs/2026-10-16-metal-native-renderer-design.md` (components 4, 5).
**Roadmap:** `docs/superpowers/plans/2026-10-16-metal-native-roadmap.md` (M3 row, "Open items").
**M2 decisions:** `docs/superpowers/plans/2026-10-16-metal-native-m2-decisions.md`.
**Inventories (read the relevant one per task):** `<workspace>/m3-rasterizer-notes.md`,
`<workspace>/m3-pipeline-notes.md`, `<workspace>/m3-link-notes.md` (copied into the plan's SDD
workspace at execution start).

## Global Constraints

- Worktree `~/Developer/GitHub/bb-metal-wt`, branch `metal-native`; arm64 build dir `out/gpu-arm64`.
- Everything from the M2 plan's Global Constraints still holds (Metal objects cross C++ as
  `void*`/`Metal::Object`/`Metal::BufferRef`; Vulkan headers vocabulary only; no Vulkan handle
  types, VMA or barriers in converted files; MoltenVK-parity limits; per-scheduler ticks via
  the scheduler; hazard-tracked resources; file names stay until M7).
- Bindings do not survive `Encoder::Submit`: every draw/dispatch closure binds everything it
  uses (pipeline, slots, push constants, vertex/index buffers, depth-stencil state).
- Indirect/address-reached buffers: `Device::MakeResident` + `Encoder::DeclareIndirect`.
- Cross-scheduler GPU ordering only through `Timeline::DeviceTickOf` + `Encoder::WaitForDeviceTick`
  (producer tick must be committed first) or by flushing the producer.
- Stage commits with explicit paths only; never commit `.superpowers/`; never `git add -A`.
- Commits end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

**Commands** (`cd ~/Developer/GitHub/bb-metal-wt` first):
- Core + unit test: the M2 core command (`ninja -C out/gpu-arm64 bbmetal_core` then build and run
  `out/mtl-encoder-test` plain and with `MTL_DEBUG_LAYER=1`; compile `tests/data/spec_const.comp`
  first) — must print PASS twice.
- Object compile check: `ninja -C out/gpu-arm64 CMakeFiles/bbgpu.dir/shadps4/video_core/<path>.o`.
- Handle check (converted files): `grep -nE "vk::(Device|Buffer|Image|ImageView|Sampler|CommandBuffer|Semaphore|Fence|DeviceMemory|Pipeline|PipelineLayout|DescriptorSet|ShaderModule|Unique)\b|vma[A-Z]|Vma[A-Z]|pipelineBarrier|MetalLayer::" <files>` → no output (accessor names like `GetDevice()` returning `Metal::Device&` are fine).

## Review Focus

1. A draw issued right after a Submit (new command buffer) must still have its pipeline, every
   slot, push constants and depth-stencil state bound — Task 4 `TestDrawAfterSubmitRebinds`
   (helper-level GPU test).
2. Indirect draws with a GPU count buffer must draw exactly `min(count, max_count)` draws, never
   `max_count` — Task 1 `TestIndirectCountClamp`.
3. A tessellated draw run must be flushed before any other command (another draw, a copy, a
   dispatch, EndRenderPass, Submit) — Task 4 `TestTessellationFlushOrder`.
4. Draws that must not render (rasterizer discard, FrontAndBack cull, empty scissor) are skipped,
   not drawn — Task 4 `TestSkippedDraws`.
5. The library must not link with undefined non-runtime symbols — Task 8 `nm -u` check.

---

### Task 1: Encoder capabilities for the rasterizer

**Files:** `renderer_metal/mtl_encoder.{h,mm}`, `tests/test_mtl_encoder.mm`; new core kernel
source in `mtl_encoder.mm` for the count clamp.

**Interfaces (produces — exact):**

```cpp
// mtl_encoder.h
enum class FillMode : u8 { Fill, Lines };
enum class DepthClip : u8 { Clip, Clamp };
class Encoder {
    void SetVertexBuffer(u32 slot, BufferRef buffer, u32 stride);   // setVertexBuffer:offset:attributeStride:
    void SetFillMode(FillMode mode);
    void SetDepthClip(DepthClip mode);
    void DrawIndirect(Primitive primitive, BufferRef args);                       // MTLDrawPrimitivesIndirectArguments
    void DrawIndexedIndirect(Primitive primitive, IndexType type, BufferRef indices, BufferRef args);
    /// Multi-draw indirect with a GPU count: clamps the `max_count` argument records at `args`
    /// (stride `stride` bytes) so records at index >= *count draw nothing (vertex/index count 0),
    /// with a compute pass before the render pass reopens, then issues max_count indirect draws.
    void DrawIndirectCount(Primitive primitive, BufferRef args, u32 stride, BufferRef count, u32 max_count, bool indexed,
                           IndexType type = IndexType::U16, BufferRef indices = {});
    void DispatchIndirect(BufferRef args, Size3 threads_per_group);
    void SetTessellationFactorBuffer(BufferRef factors, u32 instance_stride);
    void DrawPatches(u32 control_points, u32 patch_start, u32 patch_count, BufferRef patch_index_buffer,
                     u32 instance_count, u32 base_instance);
    /// The open compute encoder (id<MTLComputeCommandEncoder>, opened if needed) for kernels encoded
    /// outside the Encoder (tessellation and geometry emulation); its state is re-applied after.
    void* ComputeEncoder();
    void PushDebugGroup(std::string_view label);
    void PopDebugGroup();
};
// texture slots raised to 128 (Stage texture masks become std::bitset<128> or 4 u32 words)
```

Behaviour: indirect argument buffers are read on the GPU (declare nothing extra: they are
bound buffers); `DrawIndirectCount` must not change the caller's compute state (dedicated
encoder like the pattern fill); FillMode/DepthClip are render state applied on change and
reapplied on a new render encoder; `ComputeEncoder()` marks the compute stage dirty so the
caller's next `Dispatch` re-applies its pipeline and slots.

- [ ] **Step 1: Failing tests** in `tests/test_mtl_encoder.mm`: `TestVertexBufferStride` (two
  attributes read with stride 32 via a vertex function with `[[stage_in]]` — vertex descriptor
  with dynamic stride `MTLBufferLayoutStrideDynamic`), `TestIndirectDraw` (args in a shared buffer
  draw a triangle), `TestIndirectCountClamp` (4 indirect records each drawing one quad into a
  different quarter, count buffer = 2: exactly two quarters coloured), `TestDispatchIndirect`,
  `TestFillModeAndDepthClip` (lines leave the interior unfilled; depth clamp keeps a triangle
  beyond the far plane), `TestDrawPatches` (a minimal post-tessellation vertex function + factor
  buffer produce a filled quad), `TestTextureSlot100` (bind a texture at slot 100 in a fragment
  function reading `texture2d<float> t [[texture(100)]]`), `TestDebugGroups` (push/pop balanced
  under the debug layer).
- [ ] **Step 2: Run** — RED (missing members).
- [ ] **Step 3: Implement** in `mtl_encoder.mm` following the existing state-on-change design
  (dirty bits for FillMode/DepthClip; vertex-buffer stride stored per slot and applied with
  `setVertexBuffer:offset:attributeStride:`; indirect via `drawPrimitives:indirectBuffer:` /
  `drawIndexedPrimitives:indexType:indexBuffer:indexBufferOffset:indirectBuffer:`; count clamp
  kernel (MSL source string, built once like `fill_u32`): thread i < max_count zeroes the
  vertex/index-count word of record i when i >= count; `dispatchThreadgroupsWithIndirectBuffer:`;
  `setTessellationFactorBuffer:offset:instanceStride:` + `drawPatches:...`; debug groups on the
  command buffer when no encoder is open, else on the open encoder).
- [ ] **Step 4: Run** — PASS twice; run the binary 5 times.
- [ ] **Step 5: Commit**: "Metal-native M3: encoder vertex strides, fill/depth-clip modes, indirect
  draws with GPU counts, indirect dispatch, patches, compute-encoder access, 128 texture slots,
  debug groups".

---

### Task 2: Pipeline headers and PSO cache cleanup

**Files:** new `renderer_vulkan/vk_pipeline_key.h` (GraphicsPipelineKey, ComputePipelineKey,
AttachmentBlendState declaration, VertexAttribute — moved out of vk_graphics_pipeline.h /
vk_compute_pipeline.h; Vulkan enums as vocabulary only); `renderer_metal/mtl_pso_cache.{h,mm}`.

- [ ] **Step 1:** Move the key structs (no member/layout change; `static_assert(sizeof(...))` of the
  old sizes) so `mtl_pso_cache.h` includes `vk_pipeline_key.h` instead of the pipeline headers.
- [ ] **Step 2:** Remove `Metal::DepthStencilKey` and `GetDepthStencilState` from `mtl_pso_cache`
  (duplicate of `mtl_depth_stencil.h`); callers use `Metal::DepthStencilCache`.
- [ ] **Step 3:** Compile `mtl_pso_cache.mm.o` and every object that includes the key header
  (grep); core test PASS twice.
- [ ] **Step 4: Commit**: "Metal-native M3: pipeline keys in a Vulkan-free header; one DepthStencilKey".

---

### Task 3: Pipelines build Metal PSOs

**Files:** `renderer_vulkan/vk_pipeline_cache.{h,cpp}`, `vk_pipeline_common.{h,cpp}`,
`vk_graphics_pipeline.{h,cpp}`, `vk_compute_pipeline.{h,cpp}`, `vk_pipeline_serialization.{h,cpp}`,
`vk_shader_util.{h,cpp}`; uses `renderer_metal/mtl_pso_cache`, `mtl_shader`, `mtl_tessellation`,
`mtl_geometry`, `mtl_spirv_*`.

**Interfaces (produces):**

```cpp
class Pipeline {                      // vk_pipeline_common.h
public:
    bool IsReady() const; void WaitReady() const; void BuildAsync(...); // unchanged semantics
    bool Failed() const noexcept;     // build produced no PSO (draws skipped, logged once)
};
class GraphicsPipeline : public Pipeline {
public:
    const Metal::RenderPipeline* Render() const;          // plain / rect-list PSO (nullable)
    const Metal::TessellationPipeline* Tessellation() const;
    const Metal::GeometryPipeline* Geometry() const;
    bool IsRectList() const; std::span<const VertexAttribute> Attributes() const;
    const GraphicsPipelineKey& GetKey() const;
};
class ComputePipeline : public Pipeline {
public:
    const Metal::ComputePipeline* Compute() const;
};
class PipelineCache {                 // ctor (Metal::Device&, Scheduler&, ...), owns
    Metal::ShaderCompiler& Compiler(); Metal::PsoCache& Psos();   // one per device, cache dir = ShaderCompiler::DefaultCacheDir()
};
```

- [ ] **Step 1:** Per `m3-pipeline-notes.md` §1-2 and §4: modules keep SPIR-V words and compile
  lazily through the cache's `ShaderCompiler`; `GraphicsPipeline::Build` = the shadow's
  `Graphics()` body (tessellation, geometry, rect list, plain; QuadList and the clip-emulation
  discard fragment log once and fail the pipeline), `ComputePipeline::Build` =
  `PsoCache::GetComputePipeline`; `Shader::Profile` gets Metal-fixed values (as the shadow's
  device assumed); `vertex_buffer_formats` always filled; capability gates: geometry always
  emulated, tessellation supported; descriptor heap / layouts / `vk::PipelineCache` deleted;
  `shader_compile` wait/skip/balanced kept (`Drawable`/`WaitBudget` check the PSO pointer);
  builds wrapped in `@autoreleasepool` (Objective-C++ helper in renderer_metal if the file must
  stay C++).
- [ ] **Step 2:** Serialization: keys stay raw (vocabulary types), `SerializationSupport` stores
  `VertexAttribute`s; bump `PipelineKeyVersion` and `ShaderBinaryVersion`; `WarmUp` submits Metal
  builds to `PipelineBuilder`; `PsoCache::Flush` every 32 new pipelines and in `Sync()`.
- [ ] **Step 3:** `vk_shader_util`: delete `CompileSPV` and `MetalLayer` use (keep GLSL→SPIR-V
  helpers if still used by host passes; otherwise delete the file and its includes).
- [ ] **Step 4:** Compile all six objects; handle check; core test PASS twice.
- [ ] **Step 5: Commit**: "Metal-native M3: pipelines build Metal PSOs (tessellation, geometry,
  rect lists), binary archive flushed, Metal shader profile".

---

### Task 4: Draw helper (from mtl_frame) with GPU tests

**Files:** new `renderer_metal/mtl_draw.{h,mm}` (in `bbgpu`; tested through a small test
executable that links `bbmetal_core` + `mtl_draw.mm` + `mtl_pso_cache.mm` + `mtl_shader.mm`
and their SPIR-V helpers if they link standalone — else test through the core by moving the
binding/skip logic into core and keeping only tessellation/geometry in bbgpu).

**Interfaces (produces):**

```cpp
namespace Metal::Draw {
/// One bound resource element (buffer, texture or sampler), indexed by the shader's unified binding.
struct Bound { BufferRef buffer{}; u64 size{}; void* texture{}; void* sampler{}; };
/// Binds every resource a stage's compiled shader uses (null fallbacks for missing ones), push
/// constants at the shader's push-constant slot, and the buffer-size table when the shader has one.
void BindStage(Encoder& e, Stage stage, const CompiledShader& shader, std::span<const Bound> bound,
               std::span<const u8> push, const Nulls& nulls);
struct DrawState { /* from PrepareDraw: viewport, scissor, cull, winding, depth bias, stencil ref,
                      blend color, depth-stencil state pointer, rasterizer_discard, cull_all, fill,
                      depth clip, target size */ };
/// False when the draw must be skipped (discard, cull both faces, empty scissor after clamping).
bool Prepare(Encoder& e, const RenderPipeline& p, const DrawState& s);
class Runs;   // tessellation runs (DrawSnap/PendingPatch/scratch pool, FlushPending) and geometry draws
}
```

- [ ] **Step 1: Failing tests** (`TestDrawAfterSubmitRebinds`, `TestSkippedDraws`,
  `TestTessellationFlushOrder`, `TestBindStageNullFallbacks`) in a new
  `tests/test_mtl_draw.mm`, built and run by `build_macos.sh --test` like the encoder test.
- [ ] **Step 2:** Move `mtl_frame.mm` 317-368 (BindStage), 375-542 (PrepareDraw, PrimitiveOf),
  1053-1123 (Snap/Apply, Scratch), 1127-1313 (TessellatedDraw, GeometryDraw, FlushPending),
  1315-1513 (Draw/DrawIndexed incl. rect lists, indirect + geometry indirect), 1515-1546
  (Dispatch/DispatchIndirect) per `m3-pipeline-notes.md` §3, rewritten against `Metal::Encoder`
  (Task 1 APIs; `ComputeEncoder()` for kernels); stats become counters in the helper. Snapshots
  must capture the bound slots, not just dynamic state.
- [ ] **Step 3:** Tests PASS (plain and debug layer).
- [ ] **Step 4: Commit**: "Metal-native M3: draw helper (slot binding, draw preparation,
  tessellation runs, geometry emulation, rect lists, indirect draws) with GPU tests".

---

### Task 5: Rasterizer — resources, render state, consumers

**Files:** `renderer_vulkan/vk_rasterizer.{h,cpp}`, `vk_draw_prep.{h,cpp}`, `vk_draw_pipe.h`,
`vk_bind_helper.h`, `vk_constant_ring.h` (callers).

- [ ] **Step 1:** Per `m3-rasterizer-notes.md` §1 and §3: per-draw `Metal::Draw::Bound` table
  indexed by `binding.unified` (filled where `set_writes` were), copied into `RecordData`; vertex
  binds as `BufferRef` + stride; index bind `{BufferRef, IndexType}`; `Handle()` → `Ref(offset)`
  at the listed lines; `image_view` → `Texture()`; `GetSampler` `void*`; every `Transit` →
  `runtime.AccessImage(&image, write)` (every bound image, incl. the helper-thread path — keep the
  runtime call on the producer thread); delete `IsBufferAccessed` calls, `FlushBarriers` calls,
  barrier plumbing, `SubmitPendingArenaBinds`; `CopyBuffer`/`FillBuffer` callers by reference with
  `VideoCore::BufferCopy`; `ConstantRing(device, scheduler)`; null buffer with the new `Buffer`
  ctor; Instance capability queries → `Metal::Limits` / constants.
- [ ] **Step 2:** RenderState producers (§2 list): `colors[]`, `num_colors`, depth with
  `clear_stencil`, `has_depth/has_stencil/stencil_only`, value-initialised `clear_value`; redirect
  structs (`SceneTargets::Target`, upscaler `Target`, `original_color`) carry `void*` textures.
- [ ] **Step 3:** Compile `vk_rasterizer.cpp.o` (errors may remain only in the draw/dispatch
  bodies that Task 6 converts and in stubbed M5 headers — list them); handle check on the
  touched lines.
- [ ] **Step 4: Commit**: "Metal-native M3: rasterizer resources, render state and cache consumers on Metal".

---

### Task 6: Rasterizer — draws, dispatches, dynamic state

**Files:** `renderer_vulkan/vk_rasterizer.cpp`, `vk_scheduler.{h,cpp}` (CommitWith signature).

- [ ] **Step 1:** DrawRecord / indirect / dispatch / indirect dispatch (§2) become `Record`
  closures that call `Metal::Draw` (Task 4): pipeline null or failed → skip (count once);
  `Prepare` false → skip; tessellated / geometry / rect-list routing by the pipeline;
  `DrawIndirectCount` for the count variants; FrontAndBack cull and rasterizer discard skip.
  `CommitWith(scheduler.GetDepthStencilCache(), emit)`; the PSO-key fields (color write masks,
  primitive restart, line width, depth bounds) come from the pipeline, not dynamic state.
  Debug labels via `Encoder::PushDebugGroup/PopDebugGroup` in `Record`; `BbPassLog::Pass(tick,
  label)`.
- [ ] **Step 2:** Compile `vk_rasterizer.cpp.o` — expected clean except references into stubbed
  M5 classes resolved in Task 8; handle check over the whole file.
- [ ] **Step 3: Commit**: "Metal-native M3: rasterizer draws and dispatches through the draw helper".

---

### Task 7: Shader HLE and scene resolution

**Files:** `renderer_vulkan/vk_shader_hle.cpp`, `vk_scene_resolution.{h,cpp}`, `vk_runtime.cpp`
(NativeAccess caller).

- [ ] **Step 1:** Shader HLE per `m3-rasterizer-notes.md` §4: `BuildHostKernel(BUFFER_MULTI_COPY_COMP)`,
  bound via slot map; `VideoCore::BufferCopy`; `RecordHoistedCopy(src->Ref(), dst->Ref(), ...)`;
  storage alignment from `Metal::Limits`.
- [ ] **Step 2:** Scene resolution per §4: `Target{void* texture}`, proxies `Metal::Object` via
  `CreateTexture`, views via `CreateTextureView`; delete layouts/Transition; `NativeAccess(bool
  write)` (update `vk_runtime.cpp`); same-size copies `CopyTexture`, scaled colour copies a
  fullscreen sampled draw (`ms_image_blit` non-MSAA function with a linear/nearest sampler —
  nearest as before), depth resample with `Metal::BlitShaders::DepthResample*` and
  `DepthStencilCache`; delete the Vulkan resample resources.
- [ ] **Step 3:** Compile `vk_shader_hle.cpp.o`, `vk_scene_resolution.cpp.o`, `vk_runtime.cpp.o`
  (now clean); handle check.
- [ ] **Step 4: Commit**: "Metal-native M3: shader HLE copies and scene resolution on Metal".

---

### Task 8: Remove the interception layer; stage gating; link; msl-check gate

**Files:** delete `renderer_metal/mtl_vk_layer.{h,mm}`, `mtl_shadow.{h,mm}`, `mtl_frame.{h,mm}`,
`renderer_vulkan/vk_metal_shadow.h`; `gpu/CMakeLists.txt`; new `gpu/shim/stubs_m3.{h,cpp}`;
gated headers `vk_presenter.h`, `vk_temporal_upscaler.h`, `vk_camera_motion.h`,
`vk_object_motion.h`; `vk_gpu_profiler.{h,cpp}`; `gpu/shim/bbgpu.cpp`; `renderer_metal/mtl_shader_check.mm`.

- [ ] **Step 1:** `BB_METAL_NATIVE_STAGE` (default 3) per `m3-link-notes.md`: exclude M4-M7 TUs,
  add `stubs_m3`, gate the headers, compile definition, gate the EXCLUDE_FROM_ALL tests.
  Presenter stub owns `Metal::Device`, a draw `Scheduler`, `Runtime` and the `Rasterizer`
  (headless; `Present` flushes). Stubs for upscaler, motion, profiler, breadcrumbs, overlay
  exactly as the notes list (constructors take `Metal::Device&`).
- [ ] **Step 2:** Replace `-undefined,dynamic_lookup` with `-undefined,error` plus `-U,_runtime_*`
  for each host-runtime symbol (derive the list from `nm -u` of the library filtered to
  `_runtime_`); remove remaining `MetalLayer::` uses; delete the three interception files.
- [ ] **Step 3:** `--msl-check` iterates the pipeline cache's PipelineKey blobs and builds every
  pipeline through the new `PipelineCache` (graphics and compute), reporting counts and failures
  like today.
- [ ] **Step 4: Gate:** `bash scripts/macos/build_gpu_arm64.sh` links; the `bbgpu-host` link line
  from `build_macos.sh` links; `nm -u out/gpu-arm64/libbbgpu.dylib | grep -v "^_runtime_\|^_objc\|^_OBJC\|^__\|^_CF\|^_dispatch\|^_MTL\|^_NS\|^_pthread\|^_os_\|^_SDL\|^_spirv\|^_fmt"` shows no project symbols (review the rest by hand); `out/bbgpu-host --msl-check out/macdata/user/cache/CUSA03173`
  (or `BB_MSL_CHECK_DIR`) reports every cached pipeline built, 0 failures except the logged
  unsupported kinds (QuadList, clip-emulation fragment); core and draw tests PASS; handle check
  over `renderer_vulkan/` excluding the M4-M7 files.
- [ ] **Step 5:** Roadmap: mark M3 done with the date; move resolved open items; list M4 inputs
  (presenter stub to replace, `Present` contract, `DirectEncoder` for the presenter).
- [ ] **Step 6: Commit and push**: "Metal-native M3 done: pipelines and rasterizer on Metal;
  interception layer removed; library links with undefined-symbol errors".
