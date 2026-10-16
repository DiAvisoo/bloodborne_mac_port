# Metal-native M5 — Overlay, MetalFX, TAA, motion, profiling — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Gameplay parity with 0.4.0 on the native Metal renderer: ImGui overlay, MetalFX on native
textures, TAA, camera/object motion, GPU time stats and capture; the `stubs_m3` stand-ins are gone
and the build is at stage 5.

**Architecture:** `gpu/shim/stubs_m3.h` already defines the M5 API the rasterizer and presenter
call (`Metal::Device&` constructors, `void*` textures in `Target`/`Display`/`RedirectSampled`).
M5 rewrites the implementations behind that API. Compute host shaders (TAA, camera motion, merge,
sharpen, reactive) keep their GLSL→SPIR-V build step and are compiled at run time through
`Metal::BuildHostKernel` (SPIRV-Cross + MSL disk cache), as the detilers already are; render-pass
work (scaled colour copies, depth resample/clear) uses the hand-written MSL in
`mtl_blit_shaders.h`. Each GPU step is a `scheduler.Record` closure on the frame's command buffer;
MetalFX encodes directly into that command buffer. The overlay renders with `imgui_impl_metal`
into the drawable inside the present command buffer.

**Tech Stack:** C++20, Objective-C++ (ARC), Metal, MetalFX, QuartzCore, SPIRV-Cross, ImGui 1.92.9b, CMake.

**Spec:** `docs/superpowers/specs/2026-10-16-metal-native-renderer-design.md`. Roadmap:
`docs/superpowers/plans/2026-10-16-metal-native-roadmap.md` (M5 row; open items from M2-M4).
Survey notes behind this plan (read the section each task names; file:line references at f530ede):
`$SP/m5-notes.md`, `$SP` = `/private/tmp/claude-501/-Users-markhall-Developer-GitHub-bloodborne-mac-port/b807637a-38fd-4db6-a12b-0b68d93bb6cb/scratchpad`.
Reference (read only, GPL-2.0-or-later): `~/Developer/others/bloodborne-macos/gpu/shadps4/video_core/renderer_metal/`
(`mtl_overlay.mm`, MetalFX and TAA code).

## Global Constraints

- Worktree `/Users/markhall/Developer/GitHub/bb-metal-wt`, branch `metal-native`; main checkout
  `$MAIN=/Users/markhall/Developer/GitHub/bloodborne_mac_port` (master, 0.4.0) is never edited
  except its `out/macdata` test data.
- Host code arm64 (`bash scripts/macos/build_gpu_arm64.sh`); `bash build_macos.sh --test` must end
  with no FAIL (msl-check with `BB_MSL_CHECK_DIR=$SP/archive-cache/CUSA03173`).
- Plain C++ headers never import Objective-C; Metal objects cross as `void*`/`Metal::Object`;
  `.mm` files ARC (imgui_impl_metal.mm too). Vulkan-Headers vocabulary only: no `vk*` calls, no
  VkDevice/VkImage/VMA in converted code.
- `BB_METAL_NATIVE_STAGE` is a CMake cache variable: build scripts pass `-D`. New core files go in
  `METAL_CORE_SOURCES` and the `REMOVE_ITEM` list.
- Existing test assertions are ported unchanged (TAA, camera motion); a test whose assertion has
  to change needs a ledgered ruling with the reason.
- Game runs: `BB_DATA_DIR=$MAIN/out/macdata`, `BB_GPU_USER_DIR=<worktree>/out/macdata-gpu`, scratch
  ini from `$SP/shot.sh` (skip_intro=1, windowed, fps_cap 60), test saves restored from
  `$MAIN/out/macdata/save-backups/` before and after; never `~/Library/Application Support/Bloodborne`;
  never lldb/debugserver on game processes; kill bb-probe/bbgpu-host after each run;
  `BB_TIMEOUT` dumps are not crash evidence. Pictures: `screencapture -l` or `BB_SNAPSHOT_TRIGGER`.
- Upscaler settings that must work at the end of M5: off, MetalFX (Native AA + all presets, live
  output sizes, scaled UI path), TAA (native), jitter, sharpness 0..2 (ExtraSharpen), object
  motion on/off, `debug_view=motion`, overlay menu and FPS counter. `upscaler=fsr3` (M6) and
  `fsr4`/`fsr411` (removed) fall back without crashing and without writing the fallback into the
  user's ini.
- Never commit `.superpowers/` or `out/`; stage explicit paths; commit messages end with
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. **Resolution / preset / upscaler changes at run time** (menu, live output sizes): every texture
   the upscaler, motion and MetalFX own is recreated without use-after-free (KeepAlive or Finish
   before release) and without a stale size reaching a kernel or the MetalFX scaler.
2. **Untracked pointer-accessed buffers** (object motion history, fault/page buffers): a frame's
   writes are ordered before the next frame's reads (same queue, declared resident/indirect), and
   history is invalidated on load screens/teleports as in 0.4.0.
3. **Upscaler fallbacks**: `upscaler=fsr3/fsr4/fsr411` in an existing ini, MetalFX unsupported,
   TAA with a scaled preset → a working picture, a log line, and the user's ini untouched.
4. **Overlay input and text entry** while the game runs (Cmd+, , gamepad L3+R3, text fields, save
   debounce) and its effect on Present (the overlay renders every present; FPS average and
   settings save keep working when the overlay is hidden).
5. **Validation clean**: `MTL_DEBUG_LAYER=1` and `MTL_SHADER_VALIDATION=1` runs at the three test
   scenes with each upscaler, including the depth-format mismatch fixed in Task 3.

---

### Task 1: Stage 5 build skeleton; remove FSR 4 and breadcrumbs

**Files:** `gpu/CMakeLists.txt`, `build_macos.sh`, `scripts/macos/build_gpu_arm64.sh`; delete
`gpu/shadps4/video_core/renderer_vulkan/vk_fsr4.{h,cpp}`, `.../fsr411/`, `.../vk_breadcrumbs.{h,cpp}`
and the `fsr4-bench` target; `gpu/shim/bbgpu.cpp` (~560-567, ReportStuck caller → plain log line
with the Metal device status); `gpu/shim/stubs_m3.{h,cpp}` (keep compiling the parts later tasks
replace); `tests/test_upscaler_support.cpp` if FSR 4 symbols disappear.

**Interfaces:** Produces a build at `BB_METAL_NATIVE_STAGE=5` in both scripts where the stage-5
files that later tasks convert are still excluded individually (a `BB_M5_PENDING` list in CMake that
each task shrinks), no `ffx-vulkan` / FSR 4 link at any stage below 6, and `taa-shader-test` /
`camera-motion-test` gated on stage ≥ 5 but excluded until Tasks 5/7 port them.

- [ ] Read `$SP/m5-notes.md` §0, §1e, §1i.
- [ ] Change the gating: stage 5 in both scripts; replace the stage-5 exclusion with the
  `BB_M5_PENDING` list (vk_temporal_upscaler, vk_camera_motion, vk_object_motion,
  vk_gpu_profiler, shim/bbport_overlay, shim/bbport_metalfx, imgui_impl_vulkan) so that removing a
  name from the list compiles the real file and the matching stub must go in the same task;
  `stubs_m3.cpp` compiled while any stubbed class remains. Delete FSR 4 / fsr411 / breadcrumbs and
  their callers; `upscaler=fsr4|fsr411` maps to the fallback rule of Task 9 (until then: off, log once).
- [ ] Gates: both builds succeed; `nm -u out/gpu-arm64/libbbgpu.dylib | grep -iE 'ffx|fsr4|_vk[A-Z]'`
  prints nothing; `bash build_macos.sh --test` no FAIL; title-screen smoke run (`$SP/shot.sh`
  variant) still 60 FPS.
- [ ] Commit `Metal-native M5: stage 5 build skeleton; FSR 4 and breadcrumbs removed`.

### Task 2: Host kernels with samplers; shared scale helper

**Files:** `renderer_metal/mtl_host_kernel.{h,mm}`, `renderer_metal/mtl_blit_shaders.h`,
`vk_scene_resolution.{h,cpp}` (or a new core `renderer_metal/mtl_scale.{h,mm}`), tests
`tests/test_mtl_host_kernel.mm` (new or extend an existing core test), `build_macos.sh`.

**Interfaces:** Produces `HostKernel::sampler_slot` (`std::array<u32, 8>`, ~0u unbound, filled from
`ShaderBinding::Kind::Sampler` per SPIR-V binding) and a scale helper usable on an Encoder:
`ScaleColor(Encoder&, void* src, void* dst, rect, linear)`, `ResampleDepth(Encoder&, void* src_depth, void* dst_depth)`,
`ClearDepth(Encoder&, void* depth, float value)` — exact names are the implementer's, documented in
the header and reused by Tasks 6-8.

- [ ] Read `$SP/m5-notes.md` §2, §1a (RunUiOnly / PrepareUiDepth / RunScaled copies).
- [ ] RED: a core test compiling a small GLSL compute shader with a combined image sampler (add it
  as a test host shader via `add_host_shader`, or embed SPIR-V) through `BuildHostKernel`, asserting
  `sampler_slot[b]` is set and a dispatch that samples a 4×4 texture with linear filtering returns
  the bilinear value; scale-helper tests (RGBA16F→RGBA8 scaled copy, D32 resample, depth clear)
  read back expected texels. Watch them fail.
- [ ] GREEN: implement; run plain and `MTL_DEBUG_LAYER=1`; suite no FAIL.
- [ ] Commit `Metal-native M5: host kernels bind samplers; scale helper`.

### Task 3: Pipeline depth/stencil format matches the bound attachment

**Files:** `gpu/shadps4/video_core/renderer_vulkan/vk_pipeline_cache.cpp` (~507-513, ~540
`motion_possible`), pipeline key version (`vk_pipeline_serialization.cpp` `PipelineKeyVersion`),
a test where the key is computed (CPU test if one exists for keys; else a small new one).

- [ ] Read `$SP/m5-notes.md` §6.1.
- [ ] RED: test that a key for registers with a valid depth buffer but depth and stencil tests
  off has no depth/stencil format (and the converse cases), matching the rasterizer's bind rule
  (`vk_rasterizer.cpp` ~277-294). GREEN: compute `db_bound` once and use it for `z_format`,
  `stencil_format` and `motion_possible`; bump the key version (one cache rebuild; note it).
- [ ] Game check: `MTL_DEBUG_LAYER=1 MTL_DEBUG_LAYER_ERROR_MODE=nslog` title + cages run → zero
  "depth attachment pixel format" messages; list any remaining validation messages in the report.
  Re-check the caged figure at the cages save (snapshot +40/+50 s) and report whether it is back.
- [ ] Commit `Metal-native M5: pipeline depth formats follow the bound depth buffer`.

### Task 4: Object motion on Metal

**Files:** `vk_object_motion.{h,cpp}`, remove its part of `stubs_m3.*` and of `BB_M5_PENDING`,
`tests/test_motion_shaders.cpp` (compile the motion variants to MSL through `Metal::ShaderCompiler`).

**Interfaces:** Consumes `ObjectMotion(Metal::Device&, Scheduler&)` as declared in `stubs_m3.h`;
history buffers via `CreateBuffer` + `gpuAddress` + `Device::MakeResident`; motion target RGBA32F
as colour attachment 7 through `RenderState.colors`.

- [ ] Read `$SP/m5-notes.md` §1c, §5 and Review Focus 2.
- [ ] RED: motion-shader-test compiles every motion variant to MSL (device pointers, MSL 3.1)
  and fails before the port. Implement; decide and document the frame-to-frame ordering of the
  pointer-accessed history buffers (same queue + declared indirect/resident; ledger it).
- [ ] Gate: suite no FAIL; game run with `object_motion=1`, `debug_view=motion` at the cages save
  shows moving-object vectors (snapshot), no validation errors with `MTL_DEBUG_LAYER=1`.
- [ ] Commit `Metal-native M5: object motion on Metal`.

### Task 5: Camera motion on Metal; camera-motion-test ported

**Files:** `vk_camera_motion.{h,cpp}`, its stub/pending entry, `tests/test_camera_motion.cpp`,
`gpu/CMakeLists.txt` test target, `build_macos.sh --test`.

- [ ] Read `$SP/m5-notes.md` §1b, §7.
- [ ] Port the test first (Metal device, `Scheduler(device,false)`, `CreateTexture`, uploads,
  readback, assertions unchanged, argv[1] SPIR-V override kept) against a free function
  `CameraMotion::Record(Encoder&, …)` that the production class also uses; it fails to build/RED.
- [ ] Implement with `BuildHostKernel(camera_motion…)`, debug overlay with `camera_motion_debug`.
  GREEN plain and `MTL_DEBUG_LAYER=1`.
- [ ] Commit `Metal-native M5: camera motion on Metal; test ported`.

### Task 6: Temporal upscaler skeleton on Metal (textures, redirects, UI path, merge, reactive)

**Files:** `vk_temporal_upscaler.{h,cpp}`, its stub/pending entry; `vk_presenter.cpp` only if the
`DisplayOverride` contract needs it.

**Interfaces:** Consumes the `stubs_m3.h` TemporalUpscaler API (callers unchanged), Task 2 sampler
slots and scale helper, Tasks 4/5 motion. Produces `Run`/`RunScaled` with a pluggable resolve step
(TAA in Task 7, MetalFX in Task 8; until then the resolve is a plain copy/scale so the game renders
with `upscaler=taa|metalfx` set) and the FSR 3 hook returning false.

- [ ] Read `$SP/m5-notes.md` §1a, §3, §5 (all of it: this is the largest file, 2049 lines).
- [ ] Convert textures to `Metal::Object` (`CreateTexture`), `Transit` → `Runtime::AccessImage`,
  `GetImage()`/`backing`/`usage_flags` per §3, `UniqueImage` → `Metal::Object`, every
  `scheduler.CommandBuffer()` use → `Record` closures; RunUiOnly / EnsureUiResources /
  PrepareUiDepth / RunScaled copies via the Task 2 helper; merge, reactive/opaque snapshot,
  Redirect*/Mirror/DisplayOverride. `DumpImages`/`BB_PRESENT_DUMP_*`: port if cheap, else remove
  and backlog.
- [ ] Tests: `tests/test_ui_composition.cpp` and `tests/test_upscaler_support.cpp` keep passing;
  add a unit test for any new pure function. Game gate: cages save with `upscaler=metalfx`
  preset Quality and with `upscaler=taa` renders a correct (unresolved) picture, UI correct,
  live output-size change in the menu or via ini restart does not crash; `MTL_DEBUG_LAYER=1` clean.
- [ ] Commit `Metal-native M5: temporal upscaler on Metal (skeleton, UI path, merge)`.

### Task 7: TAA on Metal; taa-shader-test ported

**Files:** `vk_temporal_upscaler.cpp` (TAA resolve), a shared `Taa::Record` helper (header next
to the upscaler), `tests/test_taa_shader.cpp`, CMake target, `build_macos.sh --test`.

- [ ] Read `$SP/m5-notes.md` §1a (TAA), §7.
- [ ] Port the test first (assertions unchanged, argv[1] override kept) against `Taa::Record`; RED.
- [ ] Implement TAA history ping-pong, sharpen, LDR variant (`TAA_SHARPEN_LDR_COMP`), extra sharpen.
  GREEN plain and `MTL_DEBUG_LAYER=1`.
- [ ] Game gate: `upscaler=taa` at the cages and Cathedral Ward (`livecopy-20261016`) saves vs 0.4.0
  snapshots (no ghost trails standing still, no shimmer regression), 60 FPS.
- [ ] Commit `Metal-native M5: TAA on Metal; test ported`.

### Task 8: MetalFX on native textures

**Files:** `gpu/shim/bbport_metalfx.{h,mm}` (new API on `id<MTLTexture>` as `void*`), its pending
entry, `vk_temporal_upscaler.cpp` (resolve hook), a core/unit smoke test, `CLAUDE.md` (remove the
synchronous-submit note for MetalFX on this branch's docs only if it applies — see notes §1h).

- [ ] Read `$SP/m5-notes.md` §1h, §5.
- [ ] RED: smoke test creating the temporal scaler with the upscaler's formats (colour RGBA16F,
  depth, motion RG16F/RGBA16F, output RGBA16F/RGBA8) and encoding one frame into an `Encoder`
  command buffer under `MTL_DEBUG_LAYER=1` (skip with a PASS-note if `!Supported`).
- [ ] Implement: encode inside a `Record` closure into the frame's command buffer (end the open
  encoder, encode, KeepAlive the scaler); jitter sign handling identical in `Run` and `RunScaled`;
  per-axis `BB_METALFX_JITTER="sx,sy"` (keep `BB_METALFX_JITTER_SIGN`); ExtraSharpen (sharpness > 1).
- [ ] Game gate: MetalFX Native AA and Quality at the three saves vs 0.4.0 MetalFX snapshots;
  FPS vs 0.4.0 at the city (`FINDINGS.md` numbers; same ini); output-size change at run time.
- [ ] Commit `Metal-native M5: MetalFX on native textures`.

### Task 9: ImGui overlay on Metal; upscaler fallbacks

**Files:** `gpu/shim/bbport_overlay.{h,cpp}`, `gpu/CMakeLists.txt` (imgui_impl_metal.mm with ARC;
drop imgui_impl_vulkan.cpp), `vk_presenter.cpp` (Init in the ctor; Render in Present after the
blit, before snapshot/commit), `gpu/shim/bbport_settings.cpp` (fallback without persisting),
`tests/test_upscaler_support.cpp`, overlay test (`tests/test_mtl_overlay.mm`: ImGui frame into an
offscreen BGRA8 texture under `MTL_DEBUG_LAYER=1`), launcher note if fsr3 is shown.

- [ ] Read `$SP/m5-notes.md` §1g, §4, §8 and Review Focus 3-4.
- [ ] RED: overlay test; settings test that `upscaler=fsr3|fsr4|fsr411` resolves to the fallback
  (MetalFX if supported, else off) while the saved ini keeps the user's value. GREEN.
- [ ] Implement backend-neutral `BbOverlay::Init(Metal::Device&, u32 pixel_format)` /
  `Render(void* command_buffer, void* target, u32 w, u32 h)` (Huax `mtl_overlay.mm` shape);
  Render runs every present (FPS average, debounced save) even when hidden; menu shows FSR 3.1 as
  returning in M6 (disabled). Look at the known menu bugs while here ("Subpixel jitter" unticks
  itself; choices not persisted — `menu_trace`); fix if the cause is found, else backlog.
- [ ] Game gate: Cmd+, opens the menu at the title and in game, tabs work, a text field accepts
  typing, FPS counter shows, a preset change applies, the ini keeps `upscaler=fsr3` after a session
  that started with it.
- [ ] Commit `Metal-native M5: ImGui overlay on Metal; upscaler fallbacks`.

### Task 10: GPU time stats, profiler, capture; stubs gone

**Files:** `renderer_metal/mtl_device.{h,mm}` (GPU start/end times per command buffer →
per-frame busy time), `BbStats`/`driver.cpp` frame-stats line, `vk_gpu_profiler.{h,cpp}`,
`gpu/shim/bbport_gpu_capture.mm` (capture our device), delete `gpu/shim/stubs_m3.{h,cpp}`, the
`< 5` header blocks and `BB_M5_PENDING`; docs.

- [ ] Read `$SP/m5-notes.md` §1d, §1f, §0 (end state).
- [ ] RED: device test that a committed command buffer reports a GPU busy time > 0 through the
  new accessor. GREEN. Frame-stats line prints GPU ms/frame. GpuProfiler: per-label counter
  samples if cheap on Apple GPUs (encoder boundaries), else a documented no-op + backlog.
- [ ] `BB_GPU_CAPTURE` captures the renderer's own MTLDevice (smoke: a capture file is written).
- [ ] Remove stubs; `grep -rn "stubs_m3\|BB_METAL_NATIVE_STAGE < 5\|BB_M5_PENDING" gpu build_macos.sh scripts`
  prints nothing; suite no FAIL.
- [ ] Commit `Metal-native M5: GPU time stats, profiler, capture; M5 stubs removed`.

### Task 11: M5 gate — parity runs

**Files:** fixes as found; `docs/FINDINGS.md`, `docs/MACOS_BACKLOG.md`, roadmap M5 row and open items.

- [ ] Read `$SP/m5-notes.md` §6.
- [ ] Three scenes (loadcrash Hunter's Dream, cages Central Yharnam, livecopy Cathedral Ward) ×
  {off, MetalFX Quality, TAA} on metal-native and 0.4.0 with the same ini: snapshots side by side,
  FPS, worst frames (display on). Differences are fixed or recorded precisely.
- [ ] Two 15-minute walks (scripted pad input, e.g. walk/turn loops in the city) with
  `BB_FRAME_STATS=1`: no crash, no command-buffer errors (`Device::ErrorCount`), memory stable.
- [ ] `MTL_DEBUG_LAYER=1` and `MTL_SHADER_VALIDATION=1` runs at each scene: list every distinct
  message; fix renderer-side ones.
- [ ] Full-screen switch (`BB_FULLSCREEN=1` and green button/Ctrl-Cmd-F if the screen is
  unlocked); second-frame stall measured (fix with a budgeted WarmUp wait if simple).
- [ ] Caged figure: if still missing after Task 3, follow `$SP/m5-notes.md` §6.2 b-f (draw-list
  diff with `BB_CAPTURE_TRIGGER`, `BB_SKIP_GS=1`, `BB_DROP_TESS`, `BB_MSL_SAFE_MATH=1`) and fix or
  record the root cause.
- [ ] Docs: FINDINGS entry, backlog, roadmap M5 row done only if the gate is met. Commit
  `Metal-native M5: parity gate`.
