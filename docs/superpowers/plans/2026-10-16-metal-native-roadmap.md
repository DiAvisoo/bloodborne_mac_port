# Metal-native renderer — roadmap

> **For agentic workers:** this is the milestone map. Each milestone gets its own task-by-task
> plan (`2026-10-16-metal-native-mN-*.md`), written when the previous milestone is done, against
> the code that exists then. Only M1's plan exists now.

**Goal:** 0.5.0 renders through Metal only; MoltenVK and the Vulkan loader are removed.

**Spec:** `docs/superpowers/specs/2026-10-16-metal-native-renderer-design.md` (approved 2026-10-16).

**Branch / worktree:** `metal-native` in `~/Developer/GitHub/bb-metal-wt` (the main checkout stays
on `master`, 0.4.0, which the user plays). The branch is allowed to be broken from M2 to M4.

## Global constraints (every milestone)

- Host code is arm64 (`scripts/macos/build_gpu_arm64.sh` → `out/gpu-arm64/libbbgpu.dylib`,
  `out/bbgpu-host`); the game stays x86-64 under Rosetta.
- Plain C++ headers never import Objective-C: Metal objects cross into C++ as `void*` or
  `Metal::Object` (M1). `.mm` files are ARC (`-fobjc-arc`, set by `gpu/CMakeLists.txt`).
- Vulkan-Headers stay as vocabulary only (enums, plain structs); no `vk*` function calls in new code.
- Limits keep MoltenVK's values for parity: uniform/storage/texel alignment 16, non-coherent atom
  16, anisotropy 16, LOD bias 16, sampler count 500000, max buffer = `maxBufferLength`.
- Never test against `~/Library/Application Support/Bloodborne` (the user's save); test data is
  `BB_DATA_DIR=$MAIN/out/macdata` with saves restored from `out/macdata/save-backups/`.
- Never attach lldb to the x86-64 game process (wedges Rosetta).
- Commits end with the `Co-Authored-By` line from the session.

## Milestones

| # | Plan | What | Main files | Gate |
|---|---|---|---|---|
| M1 | `…-m1-device-encoder.md` | `Metal::Object`, Device with ticks (`MTLSharedEvent`) and limits, `Metal::Encoder` (native-typed, from `mtl_frame`'s state machine) | `renderer_metal/mtl_object.*`, `mtl_device.*`, `mtl_encoder.*`, `tests/test_mtl_encoder.mm` | Unit test passes; old path still builds and runs — **done** (2026-10-16) |
| M2 | `…-m2-scheduler-caches.md` | Scheduler on the Encoder (`Record(lambda(Metal::Encoder&))`, recorder thread, host copies, hoisting); caches on Metal: `Buffer`/`UniqueBuffer`, guest arena (private `MTLBuffer`), staging, `Image`/`ImageView`/`Sampler`, `mtl_runtime` transfers, detile kernels, fault parser, blit helper; layouts/barriers deleted; sparse code deleted | `vk_scheduler.*`→`mtl_scheduler.*`, `buffer_cache/`, `texture_cache/`, `vk_runtime`→`mtl_runtime`, `vk_staging_buffer_pool` | Builds (library may not link the old rasterizer); staging/copy unit test; detile kernels equal the CPU detiler on sample tiles — **done** (2026-10-16; vk_runtime.cpp compiles except the M3 header vk_scene_resolution.h) |
| M3 | written after M2 | Pipelines (`mtl_pso_cache`, `mtl_shader`, binary archive, background compiles, `shader_compile` modes); rasterizer, draw prep, shader HLE, scene resolution on the Encoder; tessellation runs and geometry emulation moved from `mtl_frame`; `mtl_vk_layer`, `mtl_shadow`, `mtl_frame` deleted | `vk_pipeline_cache`, `vk_graphics_pipeline`, `vk_compute_pipeline`, `vk_pipeline_serialization`, `vk_rasterizer`, `vk_draw_prep`, `vk_shader_hle`, `vk_scene_resolution` | `bbgpu-host --msl-check` builds every cached pipeline through the new cache |
| M4 | written after M3 | Presenter on `CAMetalLayer` (2 drawables, `displaySyncEnabled`, fps cap, Retina), post-process pass; first light | `vk_presenter`, `vk_swapchain`, `host_passes/pp_pass`, `gpu/shim/bbgpu.cpp` | Title screen correct at 60 FPS; a save loads |
| M5 | written after M4 | ImGui on Metal, MetalFX direct, TAA, camera/object motion, frame capture, GPU time stats; gameplay parity | `gpu/shim/bbport_overlay.cpp`, `bbport_metalfx.mm`, `vk_temporal_upscaler`, `vk_camera_motion`, `vk_object_motion`, `vk_frame_capture` | Three test scenes match 0.4.0; 15-min walks; `MTL_DEBUG_LAYER=1` clean |
| M6 | written after M5 | FSR 3.1 through a Metal `FfxInterface` back end (AMD's SPIR-V → MSL) | `gpu/shim/ffx_metal/`, `host_passes/fsr_pass` | FSR picture matches 0.4.0; ≥ 107 FPS at 1080p Quality |
| M7 | written after M6 | Remove MoltenVK, Vulkan loader, FSR 4 / 4.1.1, single-process mode, `renderer=` and the launcher picker; x86-64 library trimmed to the remote shim; packaging and CI | `bootstrap_deps.sh`, `gpu/CMakeLists.txt`, `build_macos.sh`, `run.sh`, `package_app.sh`, `.github/workflows/macos-release.yml`, `launcher/` | Spec success criteria 1–5 |

## Interfaces between milestones

- **M1 → M2/M3:** `Metal::Device` (`CurrentTick`, `KnownGpuTick`, `IsFree`, `Wait`, `Finish`,
  `Commit`, `GetLimits`, `ErrorCount`, `Native`), `Metal::Encoder` (passes, state, draws,
  dispatches, blits, `KeepAlive`, `Submit`, `CommandBuffer`), `Metal::Object`, `Metal::BufferRef`.
  Exact signatures in the M1 plan.
- **M2 → M3:** the scheduler's `Record(Func)` with `Func(Metal::Encoder&)`; cache objects expose
  `Metal::BufferRef` / texture pointers instead of Vulkan handles.
- **M3 → M4:** the rasterizer hands the presenter a finished frame texture.
- **M4 → M5/M6:** the presenter's per-frame command buffer and drawable texture.

## Test scenes and protocols (from M4)

Saves in `out/macdata/save-backups/`: `loadcrash-20261014-134937` (Hunter's Dream headstone),
`cages-20261016` (Central Yharnam lamp), `livecopy-20261016` (Cathedral Ward). Scripts in the
session scratchpad drive them with `BB_PAD_FILE` (title +30 s, Cross twice, walk/turn,
screenshots); speed by the head-to-head protocol against 0.4.0.

## Open items carried from M1 (final review, 2026-10-16)

For the M2 plan:
- **Binding lifetime:** Encoder bindings are unretained and persist across Submit (also >4 KiB
  `SetBytes` temporaries). The scheduler must rebind per command buffer or reset bindings at
  Submit before caches start freeing objects at `IsFree(tick)`.
- `FillBuffer` writes only the low byte of a 32-bit pattern: add a compute fill if the caches
  use non-uniform patterns.
- `SetBytes` over 4 KiB allocates a buffer each time: use a tick-recycled ring.
- Attachments use `slice` only (3D `depthPlane`, stencil-only formats not handled yet).
- `~Encoder` should end an open clear-only pass before checking for a command buffer; `~Device`
  `Finish` has no timeout; `Wait` must not be called from a completion handler; `IsFree` assumes
  same-queue completion order; logging from handlers may outlive logging at exit.

## Open items carried from M2 (2026-10-16)

For M5 (fidelity/performance):
- RGB32 formats need an expand step (no Metal RGB32 texture); view min LOD, sampler LOD bias,
  custom border colours are not carried over; `PixelFormatView` disables lossless compression
  (bandwidth cost). 8 GB Macs: recommended working set (~5.3 GB) is above the texture cache's
  4 GB low-VRAM cutoff, so tune `BB_GC_*` there.
- `BlitHelper` is owned by `Runtime` only; `TextureCache` no longer has one.
- `Image` no longer has `aspect_mask`; buffer_cache/texture_cache derive it from `props.is_depth`.

For the M3 plan, external consumers still on the old API:
- Rasterizer: `Handle()` x13, `image_view` x5 (2483, 2491, 2901, 2964, 3035), `GetSampler` (2582,
  now returns `void*`), `Transit` x8 -> `AccessImage(image, write)`, `AccessBuffer` x8;
  `IsBufferAccessed` callers 1249, 1255, 1411, 1844, 1907, 2097, 2192; `backing` (2491).
- Shader HLE: `IsBufferAccessed` 107-108.
- `Runtime::CopyBuffer`/`FillBuffer` take `const Buffer&` (callers pass pointers today).
- Scene resolution: `vk_scene_resolution.h` uses `VideoCore::UniqueImage` (does not exist);
  `GetImage` 161-162, `Transit` x2, `aspect_mask` x10, `usage_flags`/`supported_samples` x3,
  `backing` x6. This is the only error left in `vk_runtime.cpp`.
- Upscaler: `GetImage` x5, `Transit` x9, `usage_flags`/`supported_samples` x7, `backing` x2.
- Camera motion: `Transit` 290-292, usage/samples, `backing` x2. Object motion `image_view` 204;
  post-process pass `image_view` 199; presenter `image_view` 416, `Transit` 418.
- `Runtime` uses `GetImage` x10 and `aspect_mask` x10 internally (done) and buffer_cache 726.
- `RenderAttachment` gained `clear_stencil` and dropped `num_layers`; viewports go through
  `RasterUtil::ToMetalViewport`.
- `GetWorkSemaphore` residue in pipeline_cache, constant_ring, presenter. The presenter's three
  schedulers share one device with per-scheduler timelines.

For the M3 plan, from the M2 final review:
- The presenter and other consumers of the draw scheduler's frame must order after it on the
  GPU: `Timeline::DeviceTickOf(logical)` + `Encoder::WaitForDeviceTick(tick)`, or `Flush` the
  draw scheduler first. The tick must already be committed (one queue starts command buffers in
  commit order: a wait for a later commit never ends; `WaitForDeviceTick` asserts it).
- `FrontAndBack` cull must discard the draw (Metal has no such cull mode).
- Value-initialise `RenderAttachment::clear_value`.
- Every rasterizer image path must call `Runtime::AccessImage`: `FlushBarriers` no longer runs
  `BeforeImageAccess`.
