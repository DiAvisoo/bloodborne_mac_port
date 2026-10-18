# Metal-native M6: FSR 3.1 on Metal — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** FSR 3.1 upscaling (and the presenter's FSR 1 option) runs on the Metal renderer, with a picture matching 0.4.0's and ≥ 107 FPS at 1080p FSR Quality.

**Architecture:** AMD's FidelityFX SDK 1.1.4 host code (`ffx_fsr3upscaler.cpp`) is kept as is and talks
to a new Metal back end (`gpu/shim/ffx_metal/`) through AMD's `FfxInterface` callbacks, the role
`ffx_vk.cpp` plays on Vulkan. AMD's shipped SPIR-V blobs are compiled to MSL by our
`Metal::ShaderCompiler`; resources are `MTLTexture`s; jobs are encoded on our `Metal::Encoder`
inside a `scheduler.Record` closure. A small wrapper (`Fsr3Metal`) owns the context and the three
shared resources and is called from `ResolveFsr3` in `vk_temporal_upscaler.cpp`.

**Tech Stack:** C++20 / Objective-C++ (ARC), Metal, SPIRV-Cross (MSL 3.1), FidelityFX SDK 1.1.4 (MIT).

**Spec:** `docs/superpowers/specs/2026-10-16-metal-native-renderer-design.md` (section 7, FSR 3.1;
risks), roadmap `docs/superpowers/plans/2026-10-16-metal-native-roadmap.md` (row M6).

## Decisions (made here, don't re-litigate)

1. **Reuse AMD's host code and blobs from the submodule, not its Vulkan targets.** "The FSR-Vulkan
   submodule leaves the build" means its CMake targets (`ffx-vulkan::*`, `ffx_vk.cpp`, the portable
   wrapper) are no longer built. Our `gpu/CMakeLists.txt` compiles the AMD files it needs straight
   from `gpu/third_party/fsr-vulkan/upstream/ffx-1.1.4/sdk/` (patched per its `UPSTREAM.md`) and
   `generated/ffx-1.1.4/vk/fsr3upscaler/`. Whether to vendor these files and drop the submodule is
   M7's call.
2. **Capabilities reported to FFX:** shader model 6.2, `waveLaneCountMin = waveLaneCountMax = 32`
   (no wave64 blobs), `fp16Supported = true` (`BB_FFX_FP16=0` reports false, for comparison),
   everything else false. FSR 3 is supported on `MTLGPUFamilyApple7` and later (texture atomics on
   `r32uint`, quad operations); below that `fsr3_supported` stays false and the existing fallback applies.
3. **Barriers:** FSR's internal textures and the inputs are hazard-tracked `MTLTexture`s (the default
   for `Metal::CreateTexture`) and FFX jobs run on one serial compute encoder, so `BARRIER` and
   `DISCARD` jobs are no-ops. If the validation layer or the golden test disagrees, encode
   `memoryBarrierWithScope:MTLBarrierScopeTextures` for `BARRIER`.
4. **Clears** (`CLEAR_FLOAT`) are clear-only render passes (`ColorAttachment::clear`), so every FFX
   texture is created with `render_target = true` and `shader_write = true`. **Copies** use
   `Encoder::CopyTexture`. **Constant buffers** go through `Encoder::SetBytes` (FSR's are < 4 KiB).
5. **Bindings:** each blob carries reflection tables (names, binding, count, space). The back end
   maps (space, binding) to an MSL slot with `CompiledShader::Find(set, binding, kind)`. It does not
   use `HostKernel`, whose slot arrays only cover bindings 0–7. Samplers are bindings 1000/1001
   (`s_PointClamp`, `s_LinearClamp`), created once per back end.
6. **Pipeline cache:** compiled compute PSOs are kept per device, keyed by blob data pointer, and
   survive context re-creation. A preset or size change must not recompile (the MSL disk cache covers
   cold starts).
7. **Golden reference:** a one-shot tool runs the existing Vulkan path (`ffx-vulkan::portable` on
   MoltenVK, arm64) on synthetic inputs and writes reference outputs into `tests/data/fsr3/`. These are
   checked in so that the comparison test outlives MoltenVK (M7).

## Global Constraints

- Host code is arm64 (`scripts/macos/build_gpu_arm64.sh` → `out/gpu-arm64/`); the game stays x86-64.
- Plain C++ headers never import Objective-C: Metal objects cross as `void*`. `.mm` files are ARC.
- No `vk*` function calls in new code (Vulkan-Headers as vocabulary only); the golden tool is the
  one exception and is not part of `bbgpu`.
- Never test against `~/Library/Application Support/Bloodborne`; test data is `BB_DATA_DIR=$MAIN/out/macdata`.
- Never attach lldb to the x86-64 game process. One game at a time; every game run has a watchdog
  and a 1 GB log guard (scratchpad `t11-pace.sh` pattern). No `MTL_DEBUG_LAYER_WARNING_MODE=nslog`.
- New Metal sources go in the `METAL_CORE_SOURCES` / `REMOVE_ITEM` lists in `gpu/CMakeLists.txt`.
- Commits are titled `Metal-native M6: ...` and end with the session's `Co-Authored-By` lines.
- `gpu/third_party/fsr-vulkan` shows as modified (applied patch): never commit it.

## Review Focus

1. **Preset / output size / HDR change mid-game** re-creates the context: no multi-second hitch, no
   leaked textures (Task 4 test: re-create 3 times, PSO compile count unchanged, memory back to start).
2. **Reactive mask absent, exposure and composition mask absent:** FFX substitutes its defaults; a null
   registered resource must not crash or bind garbage (Task 3 test dispatches with all three null).
3. **The luma pyramid's single-pass downsampler** syncs threadgroups through an atomic counter and
   coherent mip 5: a wrong fence would give flickering exposure, not a crash (Task 3 golden test
   compares the auto-exposure frame; run under `MTL_SHADER_VALIDATION=1` without the archive).
4. **Reset frames** (save load, camera cut) must not smear history (Task 3 golden includes a reset at
   frame 5).
5. **Odd render sizes** (Ultra Performance at 1440p: 853×480): dispatch rounding and the half-size SPD
   texture (Task 3 golden uses an odd size).

---

### Task 1: AMD host code and blobs build; every FSR 3 blob compiles to a Metal PSO

**Files:**
- Modify: `gpu/CMakeLists.txt` (new static target `ffx_fsr3_host`; drop `add_subdirectory(third_party/fsr-vulkan)` and the stage-6 `ffx-vulkan::portable` link at 289-292; `BB_METAL_NATIVE_STAGE` default 6; remove `host_passes/fsr_pass` from `BB_UNCONVERTED_6` only in Task 5)
- Modify: `scripts/macos/build_gpu_arm64.sh`, `build_macos.sh` (`-DBB_METAL_NATIVE_STAGE=6`)
- Create: `gpu/shim/ffx_metal/ffx_metal.h`, `gpu/shim/ffx_metal/ffx_metal_private.h`, `gpu/shim/ffx_metal/ffx_metal.mm`
- Test: `tests/test_ffx_metal.mm` (CMake target `ffx-metal-test`, APPLE only, run by `build_macos.sh --test` plain and with `MTL_DEBUG_LAYER=1`)

**Interfaces:**
- Produces:
  - `size_t FfxMetal::ScratchSize(uint32_t max_contexts);`
  - `FfxErrorCode FfxMetal::GetInterface(FfxInterface* out, Metal::Device& device, void* scratch, size_t scratch_size, uint32_t max_contexts);`
  - `FfxCommandList FfxMetal::CommandList(Metal::Encoder& encoder);`
  - `FfxDeviceCapabilities FfxMetal::Capabilities(Metal::Device& device);` (decision 2)
  - `bool FfxMetal::Supported(Metal::Device& device);` (Apple7+)
  - `void* FfxMetal::PipelineFor(Metal::Device&, const FfxShaderBlob&, std::string* error);` (cached `id<MTLComputePipelineState>`, decision 6)
  - `uint32_t FfxMetal::PipelineCompileCount();` (for tests)
- `ffx_fsr3_host` sources: `src/shared/ffx_assert.cpp`, `ffx_message.cpp`, `ffx_object_management.cpp`,
  `src/components/fsr3upscaler/ffx_fsr3upscaler.cpp`, `src/backends/shared/ffx_shader_blobs.cpp`,
  `src/backends/shared/blob_accessors/ffx_fsr3upscaler_shaderblobs.cpp`. Definition `FFX_FSR3UPSCALER=1` only,
  `-include src/ffx_1_1_4_portability.hpp`, include dirs as in the submodule's `ffx_vulkan_fsr3_vk_backend_1_1_4`
  (without frame interpolation / optical flow).

- [ ] **Step 1: Write the failing test** `AllFsr3BlobsCompile`: for every permutation of the six FSR 3 upscaler key bits
  (REPROJECT_USE_LANCZOS_TYPE, HDR_COLOR_INPUT, LOW_RESOLUTION_MOTION_VECTORS, JITTERED_MOTION_VECTORS,
  INVERTED_DEPTH, APPLY_SHARPENING) and every pass (`FFX_FSR3UPSCALER_PASS_*`, excluding `DEBUG_VIEW`),
  with fp16 on and off and wave64 off, get the blob through the interface's `fpGetPermutationBlobByIndex`, assert
  `blob.data` starts with 0x07230203, and `PipelineFor` returns non-null with an empty error. Then
  `PipelineFor` again for the same blob: assert `PipelineCompileCount()` is unchanged. Also assert
  `Capabilities(device).waveLaneCountMax == 32` and `Supported(device)` on this M4 Max.
- [ ] **Step 2: Run** `ninja -C out/gpu-arm64 ffx-metal-test && out/gpu-arm64/ffx-metal-test`. Expected: link failure (FfxMetal undefined).
- [ ] **Step 3: Implement** the CMake target and `ffx_metal.mm`: `GetInterface` fills `FfxInterface` with
  `fpGetSDKVersion`, `fpGetDeviceCapabilities`, `fpCreateBackendContext`/`fpDestroyBackendContext` and
  `fpGetPermutationBlobByIndex = ffxGetPermutationBlobByIndex`. The other callbacks return `FFX_ERROR_BACKEND_API_ERROR`
  until Tasks 2–3. `PipelineFor` uses `ShaderCompiler::Compile(span<const u32>(blob), hash of the blob bytes, "ffx <pass>")`
  then `newComputePipelineStateWithFunction`. Any SPIRV-Cross failure goes into `error` with the pass name.
  Compilation fixes (quad ops, texture atomics, `coherent`) go in `mtl_shader.mm` options only if the test shows them.
- [ ] **Step 4: Run** the test plain and with `MTL_DEBUG_LAYER=1`. Expected: `PASS AllFsr3BlobsCompile (N pipelines)` both times.
- [ ] **Step 5: Commit** `Metal-native M6: FSR 3 host code and blobs build; every blob compiles to a Metal PSO`.

### Task 2: Resources, clears and copies

**Files:**
- Create: `gpu/shim/ffx_metal/ffx_metal_resources.mm`
- Modify: `gpu/shim/ffx_metal/ffx_metal.mm`, `ffx_metal_private.h`
- Test: `tests/test_ffx_metal.mm`

**Interfaces:**
- Produces:
  - Callbacks `fpCreateResource`, `fpRegisterResource`, `fpUnregisterResources`, `fpGetResourceDescription`, `fpGetResource`, `fpDestroyResource`, `fpRegisterStaticResource` (returns `FFX_ERROR_BACKEND_API_ERROR`; the upscaler does not use it), `fpMapResource`/`fpUnmapResource` (same).
  - Job types `FFX_GPU_JOB_CLEAR_FLOAT`, `FFX_GPU_JOB_COPY`, `FFX_GPU_JOB_BARRIER`, `FFX_GPU_JOB_DISCARD` in `fpScheduleGpuJob` / `fpExecuteGpuJobs` (compute in Task 3).
  - `MTLPixelFormat FfxMetal::PixelFormat(FfxSurfaceFormat);` (Invalid for formats FSR does not use).
  - `FfxResource FfxMetal::Resource(void* texture, FfxResourceUsage usage, const wchar_t* name);` (describes an `id<MTLTexture>` for registration; null texture → a null FfxResource).
  - Per-mip storage views for UAV bindings with `arrayIndex` = mip (`Metal::CreateTextureView`).
  - Init data (`FfxCreateResourceDescription::initData`) uploaded through a shared staging buffer and `CopyBufferToTexture` on the first `fpExecuteGpuJobs`.
  - `uint64_t FfxMetal::LiveTextureBytes();` (for tests).

- [ ] **Step 1: Write the failing tests:**
  - `PixelFormatTable`: every format in ffx_fsr3upscaler.cpp's resource list (RGBA16F, RG16F, R16F, R32F, R32_UINT, R8_UNORM, R16_UNORM, RG32F, R8G8_UNORM) maps to a valid `MTLPixelFormat`.
  - `CreateClearCopy`: create a 64×32 RG16F texture with 3 mips and init data (a ramp). Schedule `CLEAR_FLOAT` {0.25, 0.5} on mip 1 and `COPY` of mip 0 into a second resource. Execute in an Encoder, submit, wait. Read back: mip 0 of the copy equals the ramp, and mip 1 equals 0.25/0.5. Destroy both: `LiveTextureBytes()` returns to its starting value.
  - `RegisterNull`: registering `Resource(nullptr, ...)` returns `FFX_OK`, and `fpGetResourceDescription` of it reports width 0.
- [ ] **Step 2: Run** the test. Expected: FAIL (callbacks return `FFX_ERROR_BACKEND_API_ERROR`).
- [ ] **Step 3: Implement** in `ffx_metal_resources.mm`. Keep `ffx_vk.cpp`'s split between effect-owned resources (created once) and dynamic ones (registered per dispatch, cleared by `fpUnregisterResources`). Resource states are ignored (decision 3).
- [ ] **Step 4: Run** plain and with `MTL_DEBUG_LAYER=1`. Expected: all PASS, no validation errors.
- [ ] **Step 5: Commit** `Metal-native M6: FFX Metal back end resources, clears and copies`.

### Task 3: Compute jobs and a golden comparison against the Vulkan path

**Files:**
- Create: `gpu/shim/ffx_metal/ffx_metal_compute.mm`
- Create: `tests/fsr3_golden_vk.cpp` (one-shot tool, CMake target `fsr3-golden-vk`, links `ffx-vulkan::portable` and the arm64 MoltenVK; built only when `-DBB_FSR3_GOLDEN=ON`, which re-enables the submodule's `add_subdirectory` for this target alone)
- Create: `tests/fsr3_synthetic.h` (shared input generator), `tests/data/fsr3/*.bin` (golden outputs)
- Test: `tests/test_ffx_metal.mm`

**Interfaces:**
- Consumes: Task 1 `PipelineFor`; Task 2 resources.
- Produces: callbacks `fpCreatePipeline` (fills `FfxPipelineState` from the blob's reflection tables with `CompiledShader::Find` slots; samplers per decision 5), `fpDestroyPipeline`, `fpStageConstantBufferDataFunc`, `FFX_GPU_JOB_COMPUTE` (SetComputePipeline, `Access::None`; the textures; `SetBytes` for CBs; `Dispatch(dimensions, local_size)`; indirect args if set).
- `tests/fsr3_synthetic.h`: `struct Fsr3Frame { std::vector<uint16_t> color /*RGBA16F*/; std::vector<float> depth; std::vector<uint16_t> motion /*RG16F*/; float jitter_x, jitter_y; bool reset; };` and `Fsr3Frame MakeFrame(uint32_t index, uint32_t w, uint32_t h);`. The scene is a checkerboard plus colour ramp moving 1.25 px/frame right with matching motion vectors, depth a horizontal gradient, Halton(2,3) jitter as in `vk_temporal_upscaler.cpp:297`, and reset at frames 0 and 5.
- Golden cases (render → output): `quality_hdr` 256×144 → 384×216 (HDR + auto exposure, sharpening 0.3), `odd_up` 171×96 → 512×288 (no HDR, no sharpening). 8 frames each. Outputs of frames 4 and 7 stored as RGBA16F raw (`<case>_f<N>.bin`, about 1.5 MB in total).

- [ ] **Step 1: Write the golden tool**, run it once (`-DBB_FSR3_GOLDEN=ON`, `ninja fsr3-golden-vk`, `out/gpu-arm64/fsr3-golden-vk tests/data/fsr3`). Log which blob set ffx_vk picked (fp16 or not) and record it in `tests/data/fsr3/README.md`, with the tool's command line and the SDK version. Commit the data.
- [ ] **Step 2: Write the failing test** `Fsr3MatchesVulkan`: for each case, create the context through `ffxFsr3UpscalerContextCreate` with `FfxMetal::GetInterface`, using the same flags as the tool. Dispatch 8 frames with the same inputs; the reactive, exposure and composition masks are null. Compare frames 4 and 7 to the golden: mean absolute error ≤ 0.002 and max ≤ 0.03 per channel, over finite values only. Assert no NaN/Inf. Run with the same fp16 setting as the golden (`BB_FFX_FP16` per the README).
- [ ] **Step 3: Run.** Expected: FAIL (`FFX_GPU_JOB_COMPUTE` unimplemented).
- [ ] **Step 4: Implement** `ffx_metal_compute.mm`. When a binding name in the blob is not found in the compiled shader, fail `fpCreatePipeline` with the pass and the name.
- [ ] **Step 5: Run** plain, with `MTL_DEBUG_LAYER=1`, and with `MTL_SHADER_VALIDATION=1` (Review Focus 3). Expected: PASS with errors printed (e.g. `quality_hdr f7 mean 0.0004 max 0.011`) in all three. If one case fails only at frame 7, suspect decision 3 (barriers) first.
- [ ] **Step 6: Commit** `Metal-native M6: FFX compute jobs; FSR 3 matches the Vulkan path on synthetic frames`.

### Task 4: `Fsr3Metal` wrapper and the renderer hook

**Files:**
- Create: `gpu/shim/ffx_metal/fsr3_metal.h`, `gpu/shim/ffx_metal/fsr3_metal.mm`
- Modify: `gpu/shadps4/video_core/renderer_vulkan/vk_temporal_upscaler.{h,cpp}` (`ResolveInputs` h:147-158, `ResolveFsr3` cpp:711-717, `EnsureResources` cpp:333)
- Modify: `gpu/shadps4/video_core/renderer_vulkan/vk_rasterizer.cpp:68` (`ConfigureUpscalerSupport(BbMetalFx::Supported(device), FfxMetal::Supported(device))`)
- Modify: `gpu/shim/bbport_overlay.cpp:631-698` (drop `kFsr3Later`), `gpu/shim/bbport_settings.h:92` comment
- Test: `tests/test_ffx_metal.mm`, `tests/test_upscaler_support.cpp`

**Interfaces:**
- Produces:
  ```cpp
  class Fsr3Metal {
  public:
      struct CreateInfo { u32 max_render_w, max_render_h, max_output_w, max_output_h; bool hdr; };
      struct DispatchInfo {
          void* color; void* depth; void* motion; void* reactive /*nullable*/; void* output; // id<MTLTexture>
          u32 render_w, render_h, output_w, output_h;
          float jitter_x, jitter_y, frame_time_ms, camera_near, camera_far, vertical_fov, sharpness;
          bool sharpen, reset;
      };
      bool Create(Metal::Device& device, const CreateInfo& info, std::string* error);
      bool Dispatch(Metal::Encoder& encoder, const DispatchInfo& info);
      void Destroy();               // caller has run scheduler.Finish()
      bool Matches(const CreateInfo& info) const;
      u64 MemoryBytes() const;
  };
  ```
  Context flags: `FFX_FSR3UPSCALER_ENABLE_HDR | FFX_FSR3UPSCALER_ENABLE_AUTO_EXPOSURE` when `hdr`, else 0. Depth is not inverted; motion vectors are render-resolution and unjittered (as master 397-430). It owns the three shared resources from `ffxFsr3UpscalerGetSharedResourceDescriptions` and the scratch memory.
- `ResolveInputs` gains `float frame_time_ms, camera_near, camera_far, vertical_fov;` filled as on master. Frame time comes from the steady clock, clamped to 16.6 when ≤ 0 or > 200 (master 1082). `camera_near = camera_motion.Near()`, `camera_far = 3000`, `vertical_fov = camera_motion.VerticalFov()`. Jitter sign as master: −1 unless toggle 1<<26. `sharpness = min(settings.sharpness, 1)`, and ExtraSharpen covers sharpness above 1.

- [ ] **Step 1: Write the failing tests:**
  - `WrapperRecreate` in `test_ffx_metal.mm`: `Create` at 1280×720 → 1920×1080, dispatch 2 frames, then `Destroy`/`Create` at 960×540 → 1920×1080, then 2560×1440 output, then back. Assert `PipelineCompileCount()` is unchanged after the first create (Review Focus 1), and `LiveTextureBytes()` after the final `Destroy` equals the value before the first `Create`.
  - In `test_upscaler_support.cpp`: with `ConfigureUpscalerSupport(true, true)`, `ChooseUpscaler(Fsr3)` returns `Fsr3`. With `(true, false)` it still falls back to MetalFX (existing assertion kept).
- [ ] **Step 2: Run both.** Expected: FAIL (no `Fsr3Metal`; Fsr3 still falls back).
- [ ] **Step 3: Implement** the wrapper and `ResolveFsr3`. Record inside `scheduler.Record`. Re-create in `EnsureResources` when `!Matches`, after `scheduler.Finish()`. On `Dispatch` failure, `Run()` keeps its existing `Fail(...)` path, which logs once and falls back.
- [ ] **Step 4: Run** both tests plain and with `MTL_DEBUG_LAYER=1`, then the whole `bash build_macos.sh --test`. Expected: all PASS.
- [ ] **Step 5: Commit** `Metal-native M6: FSR 3.1 upscaler on the Metal renderer`.

### Task 5: FSR 1 (EASU + RCAS) in the presenter

**Files:**
- Modify: `gpu/shadps4/video_core/renderer_vulkan/host_passes/fsr_pass.{h,cpp}` (rewrite on Metal: two `BuildHostKernel` kernels from `FSR_EASU_COMP`/`FSR_RCAS_COMP`, a linear clamp sampler, per-size intermediate and output RGBA16F textures; `FSRConstants` 5×uvec4 via `SetBytes` at the push-constant slot)
- Modify: `gpu/CMakeLists.txt` (empty `BB_UNCONVERTED_6`), `vk_presenter.cpp:126-129` (call `fsr_pass.Render` as on master 156/422 instead of the warning)
- Test: `tests/test_mtl_host_kernel.mm` (new case)

**Interfaces:**
- Produces: `FsrPass::Create(Metal::Device&)`, `void* FsrPass::Render(Metal::Encoder&, void* input_texture, Size2 input, Size2 output, const Settings&, bool hdr)` returning the output `id<MTLTexture>`. `Settings{enable, use_rcas, rcas_attenuation = 0.25}` are unchanged, and `fsr.comp` bindings 0–2 fit `HostKernel`.

- [ ] **Step 1: Write the failing test** `FsrPassUpscalesEdge`: a 64×64 RGBA16F image with a vertical black/white edge at x = 32, upscaled to 128×128 with RCAS on and off. Assert output pixels at x ≤ 56 are < 0.05 and at x ≥ 72 are > 0.95. Assert the RCAS output differs from the EASU-only output near the edge, and both sizes are 128×128.
- [ ] **Step 2: Run.** Expected: FAIL to compile (no Metal `FsrPass`).
- [ ] **Step 3: Implement** the rewrite.
- [ ] **Step 4: Run** plain and with `MTL_DEBUG_LAYER=1`. Expected: PASS.
- [ ] **Step 5: Commit** `Metal-native M6: FSR 1 (EASU + RCAS) in the presenter`.

### Task 6: M6 gate

**Files:**
- Modify: `docs/FINDINGS.md`, `docs/MACOS_BACKLOG.md`, `docs/superpowers/plans/2026-10-16-metal-native-roadmap.md` (M6 row)

Scratchpad scripts `t11-pace.sh` / `t11-walk.sh` (copy as `t12-*`) with `T_UPSCALER=fsr3`; 0.4.0 baseline from the main checkout on `master`.

- [ ] **Step 1: Picture.** At `cages-20261016` and `livecopy-20261016`, capture screenshots at +40/+50 s for 1080p FSR Quality and 1440p FSR Quality, on both builds. Compare side by side: no missing geometry, no ghosting trails behind the moving character, and no flicker in exposure between consecutive shots. Note any difference in FINDINGS with the image paths.
- [ ] **Step 2: Speed and pacing.** Uncapped (`BB_PRESENT_MODE=Immediate BB_VBLANK_HZ=480`), 30 s stand, at the four test scenes, 1080p FSR Quality and 1440p FSR Quality, both builds. Gate: ≥ 107 FPS at 1080p Quality, and ahead of or level with 0.4.0 elsewhere. Report worst frames and spikes before FPS. Also run once with `BB_FFX_FP16=0` and record the difference.
- [ ] **Step 3: Validation.** `MTL_DEBUG_LAYER=1 MTL_DEBUG_LAYER_ERROR_MODE=nslog` at the three scenes with FSR Quality, 60 cap: no errors.
- [ ] **Step 4: Preset switch.** In-game, switch the FSR preset Quality → Performance → Quality through the overlay (or `BB_PAD_FILE` + settings reload). Log the frame time of the switch frame; it must be < 100 ms with warm caches (Review Focus 1).
- [ ] **Step 5: Walk.** 15 minutes at a 60 cap with 1080p FSR Quality at `cages-20261016`: 60.0 FPS mean, no errors, GPU and game memory flat.
- [ ] **Step 6: Docs and commit** `Metal-native M6: FSR gate`. Update FINDINGS (matrix entry), the roadmap M6 row (**done** + numbers), and the backlog (any shortcuts, e.g. barriers if decision 3 had to change).
