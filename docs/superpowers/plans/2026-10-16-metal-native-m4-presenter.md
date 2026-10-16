# Metal-native M4 — Presenter on CAMetalLayer, first light — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The game presents through a `CAMetalLayer` with no Vulkan swapchain: the title screen renders correctly at 60 FPS and a save loads.

**Architecture:** A core helper (`renderer_metal/mtl_present.{h,mm}`, in `bbmetal_core`, unit-tested
without the game) owns everything Objective-C: layer configuration, the post-process and present
shaders, the aspect fit, and a present queue (a second `MTLCommandQueue`) whose command buffers wait
on the device's `MTLSharedEvent` for the frame's device tick, present the drawable, and call back
on completion. `vk_presenter.{h,cpp}` (plain C++) is rewritten on that helper, `pp_pass` becomes a
thin Metal wrapper, `vk_swapchain.*` is deleted, the `BB_METAL_NATIVE_STAGE < 4` stand-in goes away
and the build moves to stage 4. The binary-archive serialization failure found in M3 is fixed
(test first). First light is run against test saves.

**Tech Stack:** C++20, Objective-C++ (ARC), Metal, QuartzCore (`CAMetalLayer`), MSL, CMake, SDL3.

**Spec:** `docs/superpowers/specs/2026-10-16-metal-native-renderer-design.md`. Roadmap:
`docs/superpowers/plans/2026-10-16-metal-native-roadmap.md` (M4 row, "Open items carried from M3").
Survey notes behind this plan: `$SP/m4-presenter-notes.md` where `$SP` is the session scratchpad
`/private/tmp/claude-501/-Users-markhall-Developer-GitHub-bloodborne-mac-port/b807637a-38fd-4db6-a12b-0b68d93bb6cb/scratchpad`
(read the section a task names; file:line references are at commit 82d0c4a).
Reference implementation (GPL-2.0-or-later, read only, port ideas not files):
`~/Developer/others/bloodborne-macos/gpu/shadps4/video_core/renderer_metal/mtl_presenter.{h,mm}`.

## Global Constraints

- Worktree `/Users/markhall/Developer/GitHub/bb-metal-wt`, branch `metal-native`. Main checkout
  `/Users/markhall/Developer/GitHub/bloodborne_mac_port` (`$MAIN`) stays on master: never edit it.
- Host code is arm64 (`bash scripts/macos/build_gpu_arm64.sh` → `out/gpu-arm64/libbbgpu.dylib`,
  `out/bbgpu-host`); `bash build_macos.sh` builds the x86-64 side and the host; `bash build_macos.sh --test`
  runs every test suite (must end with no FAIL).
- Plain C++ headers never import Objective-C; Metal objects cross as `void*` / `Metal::Object`.
  `.mm` files are ARC.
- Vulkan-Headers are vocabulary only; no `vk*` calls in new code.
- `BB_METAL_NATIVE_STAGE` is a CMake **cache** variable: build scripts pass it with `-D`.
- New core files go in both `METAL_CORE_SOURCES` and the `REMOVE_ITEM` list in `gpu/CMakeLists.txt`.
- Never test against `~/Library/Application Support/Bloodborne`. Game runs use
  `BB_DATA_DIR=$MAIN/out/macdata`, a scratch ini and `BB_GPU_USER_DIR=<worktree>/out/macdata-gpu`;
  restore saves from `$MAIN/out/macdata/save-backups/` and put the original `savedata` back after.
- Never attach lldb/debugserver to the x86-64 game process. `BB_TIMEOUT` stack dumps are not crash
  evidence (the watchdog derails a worker).
- Never commit `.superpowers/`, game files or `out/`. Stage explicit paths. Commit messages end with
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Present behaviour matches 0.4.0: BGRA8Unorm drawable, sRGB colorspace, 2 drawables, vsync on
  unless `BB_PRESENT_MODE=Immediate`, Retina from SDL (drawable size = backing pixels unless
  `BB_HIGH_DPI=0`), letterbox in black, linear scaling, fps cap stays in the driver
  (`driver.cpp` PresentThread), `BB_FRAMES_AHEAD` kept.

## Review Focus

1. **Window resize / fullscreen toggle / minimise** while presenting: the drawable size changes on
   the main thread; Present must fit into each drawable's own size, skip when the window is 0×0,
   and never stall (nextDrawable nil → frame skipped and returned to the pool).
2. **Frame pool exhaustion / ordering**: a frame returns to the pool only after its present command
   buffer completed (the next post-process write to it must not race the present read);
   `PrepareFrame` must never return null; the dtor waits for presents in flight.
3. **Cross-queue synchronisation**: the present command buffer waits for the frame's device tick
   (`Timeline::DeviceTickOf` after the draw scheduler flushed); a blank frame (tick 0) waits for
   nothing; a nullopt tick must never reach the wait.
4. **Autorelease**: every Present runs inside `@autoreleasepool` (the `bb:Present` thread has none);
   a long run must not leak drawables (nextDrawable keeps working after thousands of frames).
5. **Colour**: gamma slider 1.0 round-trips sRGB bytes; `A2R10G10B10` sources take the degamma
   path; alpha is forced to 1 (swizzle One) so a transparent game buffer is not shown see-through.

---

### Task 1: Core present helper (`mtl_present`) with unit tests

**Files:**
- Create: `gpu/shadps4/video_core/renderer_metal/mtl_present.h` (plain C++)
- Create: `gpu/shadps4/video_core/renderer_metal/mtl_present.mm`
- Modify: `gpu/shadps4/video_core/renderer_metal/mtl_blit_shaders.h` (append present shaders)
- Modify: `gpu/CMakeLists.txt` (`METAL_CORE_SOURCES` + `REMOVE_ITEM`)
- Modify: `build_macos.sh` (`--test`: build and run `out/mtl-present-test`, plain and with
  `MTL_DEBUG_LAYER=1`, linking `-framework QuartzCore` in addition to the other Metal tests' flags)
- Test: `tests/test_mtl_present.mm`

**Interfaces:**
- Consumes: `Metal::Device` (`NativeDevice()`, `NativeQueue()`; the shared event is
  `Device::Impl::event` in `mtl_device_impl.h`, includable from core `.mm` files),
  `Metal::Encoder` (BeginRenderPass/SetRenderPipeline/SetViewport/SetScissor/SetTexture/SetSampler/
  SetBytes/Draw/EndRenderPass, KeepAlive), `Metal::CompileLibrary`, `Metal::CreateRenderPipeline`,
  `Metal::CreateSampler`, `Metal::CreateTexture`, `Metal::CreateTextureView`,
  `BlitShaders::Source` / `fs_tri_vs` / `FsTriOut` (mtl_blit_shaders.h).
- Produces (exact; Task 2 uses these names):

```cpp
// mtl_present.h
namespace Metal {
struct FitRect { s32 x, y; u32 width, height; };
/// Largest rect of the source's aspect inside dst, centred (letterbox / pillarbox); same rounding
/// as vk_presenter.cpp FitImage at 82d0c4a. A zero-sized source or destination gives {0,0,0,0}.
FitRect FitImage(u32 src_width, u32 src_height, u32 dst_width, u32 dst_height);

struct PostProcessSettings { float gamma{1.0f}; u32 hdr{}; u32 srgb_input{}; }; // 12 bytes, shader layout

/// Pipelines and sampler for the post-process pass and the present blit (compiled once).
class PresentPasses {
public:
    PresentPasses(Device& device, u32 frame_format /*MTLPixelFormat of frames*/,
                  u32 drawable_format /*MTLPixelFormat of the layer*/);
    ~PresentPasses();
    /// Encodes the post-process pass: samples `input` (id<MTLTexture>, any float-sampleable
    /// format; linear filter) over the whole `output` (render target of frame_format).
    void PostProcess(Encoder& encoder, void* input, void* output, u32 out_width, u32 out_height,
                     const PostProcessSettings& settings);
    /// Encodes into a raw id<MTLCommandBuffer> (present queue): clears `target` black, then if
    /// `frame` is non-null draws it into FitImage(frame size, target size), linear.
    void Present(void* command_buffer, void* frame, u32 frame_width, u32 frame_height,
                 void* target, u32 target_width, u32 target_height);
private:
    struct Impl; std::unique_ptr<Impl> impl;
};

struct LayerConfig {
    u32 pixel_format{80};       // MTLPixelFormatBGRA8Unorm
    u32 max_drawables{2};
    bool display_sync{true};
    bool framebuffer_only{true};
};
/// Configures a CAMetalLayer (void* = CAMetalLayer*, e.g. SDL_Metal_GetLayer): device, pixel
/// format, sRGB colorspace, maximumDrawableCount, displaySyncEnabled, framebufferOnly.
void ConfigureLayer(Device& device, void* layer, const LayerConfig& config);

struct Drawable {
    Object drawable;       // id<CAMetalDrawable> (retained)
    void* texture{};       // its id<MTLTexture>, unretained (lives as long as `drawable`)
    u32 width{}, height{};
};
/// [layer nextDrawable]; empty Drawable (drawable null) on timeout/nil. Caller holds an autorelease pool.
Drawable NextDrawable(void* layer);

/// Runs `body` inside @autoreleasepool (for plain C++ callers on threads without a pool).
void WithAutoreleasePool(const std::function<void()>& body);

/// A second command queue for presents. Its command buffers can wait for a device tick of the
/// main queue (the shared event) and report completion through a callback.
class PresentQueue {
public:
    explicit PresentQueue(Device& device);
    ~PresentQueue();                       // waits for every command buffer in flight
    void* Begin();                         // new id<MTLCommandBuffer> (unretained, valid until Commit)
    /// Encodes a wait for the device's shared event to reach `device_tick` (0: no wait).
    void WaitForDeviceTick(void* command_buffer, u64 device_tick);
    /// presentDrawable (if `drawable` non-null), then commit; `on_complete` runs on a Metal
    /// completion thread once the command buffer completed (also on error).
    void Commit(void* command_buffer, void* drawable, std::function<void()> on_complete);
    void WaitIdle();                       // blocks until nothing is in flight
    [[nodiscard]] u32 InFlight() const;
private:
    struct Impl; std::unique_ptr<Impl> impl;
};
}
```

- [ ] **Step 1: Write the failing test** `tests/test_mtl_present.mm`, in the style of
  `tests/test_mtl_draw.mm` (`Expect(bool, what)`, `failures`, exit code = failures != 0). Cases:
  1. `FitImage`: (1920,1080)→(1920,1200) = {0,60,1920,1080}; (1920,1080)→(2560,1080) =
     {320,0,1920,1080}; (1280,720)→(3456,2234) = width 3456, height 1944, y 145; equal aspect →
     full rect; any zero → {0,0,0,0}. Before writing these numbers, read `FitImage`/`MakeImageBlitFit`
     in `vk_presenter.cpp` (lines ~63-128) and make the test match its rounding exactly.
  2. Post-process, gamma 1.0: a 4×4 `BGRA8Unorm_sRGB` source with bytes {0,1,64,128,200,255}
     spread over channels, output `BGRA8Unorm` 4×4 → every output byte equals the source byte ±1.
  3. Post-process, gamma 1.2: output equals the CPU formula from
     `gpu/shadps4/video_core/host_shaders/post_process.frag` (port `gamma()` to C++ in the test) ±1.
  4. Post-process, `srgb_input=1` with an `RGB10A2Unorm` source: equals the CPU
     `gamma(degamma(x))` ±1 (10-bit inputs chosen so the result is not the identity).
  5. Alpha: source alpha 0 viewed through `CreateTextureView(..., swizzle {R,G,B,One})` → output alpha 255.
  6. Present with letterbox: frame 16×9 solid red, offscreen `CAMetalLayer` (no window;
     `ConfigureLayer` with `framebuffer_only=false`, `layer.drawableSize = 32×32`), `NextDrawable`,
     `PresentQueue::Begin`, `PresentPasses::Present`, then before Commit blit the drawable texture
     into a shared readback buffer on the same command buffer; `Commit(cb, nullptr, ...)` (no
     presentDrawable without a window); wait; rows inside the fit rect are red, rows outside are black.
  7. Event wait: commit a main-queue command buffer through `Device::Commit` that writes a value
     (blit fill) and get its tick; a present-queue command buffer that `WaitForDeviceTick(tick)` then
     copies the value → reads the written value. Tick 0 → no wait (completes).
  8. Autorelease: 200 iterations of `WithAutoreleasePool([&]{ NextDrawable; Begin; Present; Commit })`
     with `max_drawables=2` all get a drawable (none nil) and `InFlight()` returns to 0 after `WaitIdle()`.
  9. `~PresentQueue` with command buffers in flight returns only after their callbacks ran (counter).

- [ ] **Step 2: Run it to verify it fails.** Add the build/run lines to `build_macos.sh --test`
  next to the other Metal tests (`out/mtl-present-test`). Run `bash build_macos.sh --test`.
  Expected: compile FAIL (`mtl_present.h` not found).

- [ ] **Step 3: Implement.** Shaders appended to `BlitShaders::Source` in `mtl_blit_shaders.h`
  (use `fs_tri_vs`; `select(a, b, cond)` = GLSL `mix(a, b, bvec)`):

```metal
struct PpSettings { float gamma; uint hdr; uint srgb_input; };
constant float pp_cutoff = 0.0031308, pp_a = 1.055, pp_b = 0.055, pp_d = 12.92;
static float3 pp_gamma(float3 c, float g) {
    return select(pp_a * pow(c, float3(1.0 / (2.4 + 1.0 - g))) - pp_b, pp_d * c / g, c < pp_cutoff);
}
static float3 pp_degamma(float3 c) {
    return select(pow(max(c + pp_b, 0.0) / pp_a, float3(2.4)), c / pp_d, c < pp_d * pp_cutoff);
}
fragment float4 post_process_fs(FsTriOut in [[stage_in]], texture2d<float> tex [[texture(0)]],
                                sampler s [[sampler(0)]], constant PpSettings& pp [[buffer(0)]]) {
    float4 c = tex.sample(s, in.uv);
    if (pp.hdr != 0u) return c;
    if (pp.srgb_input != 0u) c.rgb = pp_degamma(c.rgb);
    return float4(pp_gamma(c.rgb, pp.gamma), c.a);
}
fragment float4 present_fs(FsTriOut in [[stage_in]], texture2d<float> tex [[texture(0)]],
                           sampler s [[sampler(0)]]) { return tex.sample(s, in.uv); }
```

  Check the GLSL in `host_shaders/post_process.frag` line by line against this port (cutoff
  compare direction, the `d*c/g` branch) and fix the MSL if they differ; the test's CPU formula is
  the GLSL. If fast-math `pow` breaks the ±1 check, compile this library with
  `MTLMathModeSafe` (add an option to `CompileLibrary` only if needed).
  `mtl_present.mm`: PresentPasses compiles `BlitShaders::Source` once, builds `post_process_fs`
  (colour `frame_format`) and `present_fs` (colour `drawable_format`) pipelines, a linear
  clamp-to-edge sampler (`SamplerDesc{.min_filter=1,.mag_filter=1,.mip_filter=0}`); PostProcess
  follows the draw idiom of `vk_scene_resolution.cpp` (~83-91 `SetFullscreenState`, ~259-270:
  BeginRenderPass with a non-clearing colour attachment, viewport/scissor = output size,
  SetTexture/SetSampler/SetBytes(Fragment, 0, &settings, 12), Draw(Triangle, 3), EndRenderPass).
  Present encodes directly with a `MTLRenderPassDescriptor` (loadAction Clear black, store Store),
  viewport = FitImage rect. PresentQueue: `[device newCommandQueue]`, `commandBuffer`,
  `encodeWaitForEvent:impl->event value:tick`, `presentDrawable:`, `addCompletedHandler:` that
  calls the callback then decrements an atomic in-flight counter under a mutex + condition variable
  (WaitIdle and the dtor wait on it). Never call `Device::Commit` from the present queue.

- [ ] **Step 4: Run the tests to verify they pass.** `bash build_macos.sh --test`. Expected: all
  suites pass including `mtl-present-test` plain and under `MTL_DEBUG_LAYER=1` (0 errors).

- [ ] **Step 5: Commit**

```bash
git add gpu/shadps4/video_core/renderer_metal/mtl_present.h gpu/shadps4/video_core/renderer_metal/mtl_present.mm \
  gpu/shadps4/video_core/renderer_metal/mtl_blit_shaders.h gpu/CMakeLists.txt build_macos.sh tests/test_mtl_present.mm
git commit -m "Metal-native M4: present helper (layer, post-process and present passes, present queue)"
```

---

### Task 2: Presenter and post-process pass on Metal; stage 4

**Files:**
- Modify (rewrite): `gpu/shadps4/video_core/renderer_vulkan/vk_presenter.h`, `vk_presenter.cpp`
- Modify: `gpu/shadps4/video_core/renderer_vulkan/host_passes/pp_pass.{h,cpp}` (thin wrapper over
  `Metal::PresentPasses::PostProcess`; keep `PostProcessingPass::Settings` = `{float gamma; u32 hdr; u32 srgb_input;}`
  because `video_out.cpp:470` uses it — or alias it to `Metal::PostProcessSettings`)
- Delete: `gpu/shadps4/video_core/renderer_vulkan/vk_swapchain.{h,cpp}`
- Modify: `gpu/shim/stubs_m3.{h,cpp}` (remove the stage<4 Presenter stand-in and its includes; the
  stage<5 stubs stay), `gpu/CMakeLists.txt` (drop `BB_UNCONVERTED_4`; the stage-4 files compile;
  remove `vk_swapchain`; remove `add_host_shader` lines for `fs_tri.vert` and `post_process.frag` if
  nothing includes their generated headers any more), `build_macos.sh` and
  `scripts/macos/build_gpu_arm64.sh` (`-DBB_METAL_NATIVE_STAGE=4`), the stage default in
  `gpu/CMakeLists.txt` → 4.
- Test: build gates below + `tests/test_mtl_present.mm` stays green.

**Interfaces:**
- Consumes (Task 1): `Metal::FitImage`, `Metal::PresentPasses`, `Metal::ConfigureLayer`,
  `Metal::LayerConfig`, `Metal::NextDrawable`, `Metal::WithAutoreleasePool`, `Metal::PresentQueue`.
  Existing: `Vulkan::Scheduler` (draw scheduler: `Record`, `Flush`, `CurrentTick`, `Wait`,
  `GetTimeline().DeviceTickOf(tick)` / `.Wait(tick)`), `Runtime::AccessImage(&image, false)`,
  texture cache `FindImage`/`UpdateImage`/`GetImage`, `Image::Handle()` (id<MTLTexture>),
  `GetUpscaler().DisplayOverride(VAddr, Display&)` (stub: false; `Display::texture` is a Metal
  texture), `BbOverlay::Visible()` (stub false), `BbGpuCapture::OnPresent()`, `BbStats`,
  `DebugState.IncFlipFrameNum()`, `window.GetWindowInfo().render_surface` (CAMetalLayer*).
- Produces: the public API of `vk_presenter.h` unchanged for its callers (gnmdriver.cpp:3659,
  driver.cpp:238/600-608/908-917/966-971, video_out.cpp:427/470, bbgpu.cpp:275):
  `Presenter(Frontend::WindowSDL&, AmdGpu::Liverpool*)`, `GetPPSettingsRef`, `GetWindow`,
  `GetRasterizer`, `IsHDRSupported()` (false), `SetHDR(bool)` (no-op), `RegisterVideoOutSurface`,
  `IsVideoOutSurface`, `PrepareFrame`, `PrepareBlankFrame(bool)`, `PrepareLastFrame`,
  `Present(Frame*, bool reusing, bool game_frame, u32 present_id)`, `PresentTiming` +
  `LastPresentTiming()`, `IsDisplayTimingSupported()` (false), `TakePastTimings()` (empty).
  `GetFsrSettingsRef` is dropped (no callers) until M6.

- [ ] **Step 1: Read first.** `$SP/m4-presenter-notes.md` sections 1, 2, 4, 5, 6 (Huax excerpts);
  the current `vk_presenter.{h,cpp}` and the stand-in `gpu/shim/stubs_m3.{h,cpp}` (stage<4 part);
  Huax `mtl_presenter.mm` (ignore frame interpolation, MetalFX and overlay).

- [ ] **Step 2: Move the build to stage 4 and watch it fail.** Remove the stand-in and the
  `vk_presenter.h` gate, drop `BB_UNCONVERTED_4`, pass `-DBB_METAL_NATIVE_STAGE=4` in both build
  scripts. Run `bash scripts/macos/build_gpu_arm64.sh`. Expected: compile errors in
  `vk_presenter.cpp` / `pp_pass.cpp` (Vulkan swapchain, instance, `mtl_vk_layer.h`).

- [ ] **Step 3: Rewrite.** Frame = `{u32 width, height; Metal::Object texture; u64 ready_tick; bool is_hdr; u32 id;}`
  (BGRA8Unorm render-target texture of the fitted size; recreated when the fitted size changes,
  the old one released only after its last use — via `draw_scheduler` KeepAlive/DeferOperation).
  Pool of 2 frames (`free_queue` + mutex/cv as today; a frame returns in the present command
  buffer's completion callback, and when Present skips it).
  Constructor: `Metal::Device`, draw `Scheduler` (threaded, as the stand-in), `Runtime`,
  `Rasterizer` exactly as the stand-in builds them; `ConfigureLayer` with
  `display_sync = (BB_PRESENT_MODE != "Immediate")`, `max_drawables` from `BB_SWAPCHAIN_IMAGES`
  clamped 2..3 (default 2); `PresentPasses(device, BGRA8Unorm, BGRA8Unorm)`; `PresentQueue`.
  `PrepareFrame` (draw-recording thread): as today (DisplayOverride else FindImage+UpdateImage;
  view with `GetFrameViewFormat` and alpha swizzle One via `CreateTextureView`, kept alive with
  the command buffer; `AccessImage`; `srgb_input` for `A2R10G10B10Srgb`), frame size = the game
  image fitted into the current window pixel size (so Present is a 1:1 copy; Retina-aware),
  `Record` the post-process pass into the frame, `ready_tick = CurrentTick()`, `Flush`, timing
  into `BbStats::flip_submit_ns`, `WatchFrameCompletion` via `GetTimeline().Wait`, BB_FRAMES_AHEAD
  bound unchanged. `PrepareBlankFrame`: a frame with `ready_tick = 0` and no content (Present
  just clears). `PrepareLastFrame`: last submitted frame if it is back in the pool, else a blank
  frame. `Present` (swap thread; never touches the draw scheduler): everything inside
  `WithAutoreleasePool`; window 0×0 → free the frame, return; `NextDrawable` (empty → log once,
  free the frame, return); `Begin`; `WaitForDeviceTick(cb, *DeviceTickOf(ready_tick))` (assert
  engaged: PrepareFrame flushed; 0 = no wait); `PresentPasses::Present` (frame texture or null
  for blank); `BbOverlay::Visible()` stays false in M4 (do not call Init/Render: no stubs for
  them); `Commit(cb, drawable, [frame]{ FreeFrame(frame); })`; timings; `BbGpuCapture::OnPresent()`;
  `DebugState.IncFlipFrameNum()` for game frames. Destructor: `PresentQueue::WaitIdle`, draw
  scheduler `Finish`, pipeline cache `Sync()`. Remove every `MetalLayer::`/`mtl_vk_layer.h`/
  `BB_METAL_BACKEND` block, `vk_platform.h`, `vk_mem_alloc.h`, `vk_instance.h`, `vk_swapchain.h`,
  `fsr_pass.h` includes. If `BB_FSR1`/FSR settings ask for FSR 1, log once that it returns in M6.

- [ ] **Step 4: Build gates.** All must pass; paste the output tails into the report:
  - `bash scripts/macos/build_gpu_arm64.sh` and `bash build_macos.sh` succeed.
  - `nm -u out/gpu-arm64/libbbgpu.dylib | grep -E '_ZN6Vulkan|Swapchain|vkAcquire|vkQueuePresent'` prints nothing.
  - `bash build_macos.sh --test` → no FAIL (msl-check: run it with
    `BB_MSL_CHECK_DIR=$MAIN/out/macdata/user/cache/CUSA03173`, read-only).
  - `grep -rn "BB_METAL_NATIVE_STAGE < 4\|BB_UNCONVERTED_4\|vk_swapchain" gpu build_macos.sh scripts` prints nothing.

- [ ] **Step 5: Smoke run (title screen only, ~60 s).** Follow `$SP/m4-presenter-notes.md` §10:
  scratch ini = copy of `$MAIN/out/macdata/bench-native.ini` with `fullscreen=0`, `fps_cap=60`,
  `upscaler=off`; `mkdir -p out/macdata-gpu && cp -R $MAIN/out/macdata/user/cache out/macdata-gpu/`
  first (so the game's WarmUp clears the copy, not master's cache). Run
  `BB_DATA_DIR=$MAIN/out/macdata BB_GAME_DIR=/Users/markhall/Bloodborne/CUSA03173 BB_CONFIG=<ini> BB_GPU_USER_DIR=$PWD/out/macdata-gpu BB_FRAME_STATS=1 BB_TIMEOUT=90 bash run.sh > <log> 2>&1`,
  screenshot the window after ~45 s (`$SP/winid bbgpu-host`, `screencapture -x -o -l <id> <png>`),
  look at it. Report what the screenshot shows and the `^Frame stats` lines. A failure here is
  reported (DONE_WITH_CONCERNS) with the log tail, not hidden: Task 4 owns first-light debugging.

- [ ] **Step 6: Commit**

```bash
git add -A gpu/shadps4/video_core/renderer_vulkan/vk_presenter.h gpu/shadps4/video_core/renderer_vulkan/vk_presenter.cpp \
  gpu/shadps4/video_core/renderer_vulkan/host_passes/pp_pass.h gpu/shadps4/video_core/renderer_vulkan/host_passes/pp_pass.cpp \
  gpu/shadps4/video_core/renderer_vulkan/vk_swapchain.h gpu/shadps4/video_core/renderer_vulkan/vk_swapchain.cpp \
  gpu/shim/stubs_m3.h gpu/shim/stubs_m3.cpp gpu/CMakeLists.txt build_macos.sh scripts/macos/build_gpu_arm64.sh
git commit -m "Metal-native M4: presenter on CAMetalLayer, post-process pass on Metal, stage 4"
```

---

### Task 3: Binary archive serializes the game's pipelines

**Files:**
- Test: `tests/test_mtl_archive.mm` (pure Metal; build/run in `build_macos.sh --test`, plain and debug layer)
- Modify: `gpu/shadps4/video_core/renderer_metal/mtl_shader.mm` (unique entry names, MSL disk-cache
  version bump) and/or `mtl_pso_cache.mm` — whichever the test proves at fault.

**Interfaces:**
- Consumes: `PsoCache::Flush` / `CreateWithArchive` (mtl_pso_cache.mm ~187-302, ~539, ~625),
  `get_cleansed_entry_point_name` (mtl_shader.mm ~488), the MSL disk cache entry line (~538, parsed ~619).
- Produces: `pipelines.binaryarchive` written on Flush; a second run loads pipelines from it.

- [ ] **Step 1: Write the failing test.** Read `$SP/m4-presenter-notes.md` §7. One case per
  hypothesis, each on a fresh archive (`newBinaryArchiveWithDescriptor:` without url), adding
  descriptors with `addRenderPipelineFunctionsWithDescriptor:` then `serializeToURL:` to a temp
  file, printing PASS/FAIL per case: (a) one VS+FS pair, unique names; (b) two pipelines from four
  libraries whose functions are all named `main0`; (c) as (b) with unique names; (d) vertex-only
  pipeline (`rasterizationEnabled = NO` and a depth-only pipeline); (e) post-tessellation vertex
  function; (f) vertex descriptor with `MTLBufferLayoutStrideDynamic`; (g) descriptor whose
  `binaryArchives` already holds the archive. Then reload a serialized archive and create the
  pipeline with `MTLPipelineOptionFailOnBinaryArchiveMiss` to prove a hit. The test asserts that
  the configuration the renderer will use after the fix serializes and reloads; keep the failing
  cases as documented expected failures only if Metal genuinely refuses them (print them as INFO,
  and make the renderer avoid them).
- [ ] **Step 2: Run it.** Expected: the case matching the renderer's current configuration FAILs
  with the same message as the game ("expecting 'fragment' stage" / "missing 'vertex' stage").
  If no case reproduces it, reproduce with `bbgpu-host --msl-check` on a copy of the cache
  (`cp -R $MAIN/out/macdata/user/cache $SP/archive-cache`), bisect by which pipelines are added
  (temporary local instrumentation, not committed), then add the minimal reproducing case to the
  test; record the finding in the report.
- [ ] **Step 3: Fix** the renderer to the configuration that passes (expected: unique entry-point
  names per shader, e.g. `rename_entry_point` to `<stage>_<hash16>` before `compile()`, and bump the
  MSL disk-cache version so cached `main0` sources are rebuilt).
- [ ] **Step 4: Verify.** Test passes plain and under `MTL_DEBUG_LAYER=1`; `bash build_macos.sh --test`
  no FAIL; `out/bbgpu-host --msl-check <copy of the cache>` builds as before (≥ 644 of 645) and the
  archive file exists afterwards with no "Flush" error in the output; a second `--msl-check` run
  reports archive hits (add a counter to the existing msl-check summary if none exists).
- [ ] **Step 5: Commit**

```bash
git add tests/test_mtl_archive.mm build_macos.sh gpu/shadps4/video_core/renderer_metal/mtl_shader.mm gpu/shadps4/video_core/renderer_metal/mtl_pso_cache.mm
git commit -m "Metal-native M4: binary archive serializes (unique entry points)"
```

---

### Task 4: First light — title screen at 60 FPS, a save loads

**Files:**
- Modify: whatever the runs prove broken (renderer, presenter, shims), each fix with a test where
  a unit test can express it.
- Modify: `docs/FINDINGS.md` (first-light measurements), `docs/MACOS_BACKLOG.md` (HDR, display
  timing/`TakePastTimings`, overlay and FSR 1 off until M5/M6, anything deferred), roadmap M4 row.

**Interfaces:**
- Consumes: Tasks 1-3.
- Produces: the roadmap gate; a findings list for M5.

- [ ] **Step 1: Title screen.** Run as Task 2 Step 5 with `BB_TIMEOUT=120`. Expected: the title
  screen (logo, "PRESS ANY BUTTON"-style prompt) with correct colours, no letterbox at 16:9,
  `Frame stats` ≈ 60 FPS (vblank 60) once the pipeline cache is warm (second run; first run
  compiles every pipeline). Compare against a 0.4.0 screenshot of the same screen taken from
  `$MAIN` (`bash run.sh` there with the same ini and data dir but its own default GPU dir — the
  master cache must not be cleared: use `BB_GPU_USER_DIR=$MAIN/out/macdata/user` only for master).
- [ ] **Step 2: Save loads.** Restore `cages-20261016` into `$MAIN/out/macdata` exactly as
  `$SP/shot.sh` does (it backs up and restores `savedata`), drive the title with `BB_PAD_FILE`
  (Cross twice), screenshot at +40/+50 s. Expected: Central Yharnam at the lamp; compare with 0.4.0.
  Repeat with `loadcrash-20261014-134937` (Hunter's Dream) once.
- [ ] **Step 3: Validation.** One title-screen run with `MTL_DEBUG_LAYER=1`
  (`MTL_DEBUG_LAYER_ERROR_MODE=nslog`): list every distinct validation message; fix those in
  presenter/encoder code; ledger the rest as M5 items. Check the render-pipeline depth format vs
  image creation format (roadmap M3 open item).
- [ ] **Step 4: Resize and Retina.** Resize the window (AppleScript `set bounds of front window`
  on bbgpu-host, or `BB_FULLSCREEN=1` run) and confirm by screenshot that the picture refits with
  black bars and no stall; `BB_HIGH_DPI=0` run shows a point-sized drawable.
- [ ] **Step 5: GC on 16 GB** (roadmap M2 item): log `Device::AllocatedSize()` and `ResidentBytes()`
  every 5 s in the frame-stats line during the city run; confirm the texture-cache GC budget does
  not count the 16 GiB arena; ledger the numbers.
- [ ] **Step 6: Record.** FINDINGS entry (FPS title/city, startup time cold/warm, archive hit or
  not, validation summary), backlog entries, roadmap M4 row "done" with the date. Commit:

```bash
git add docs/FINDINGS.md docs/MACOS_BACKLOG.md docs/superpowers/plans/2026-10-16-metal-native-roadmap.md <fixed files>
git commit -m "Metal-native M4: first light"
```
