// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: stand-ins for the renderer parts not converted to Metal yet (Metal-native renderer,
// docs/superpowers/plans/2026-10-16-metal-native-roadmap.md). gpu/CMakeLists.txt leaves the
// unconverted files out of the build below BB_METAL_NATIVE_STAGE; their headers include this one
// instead and get classes with the same public API that do nothing:
//   - below stage 4 (M4): the presenter (vk_presenter.h). Headless: it owns the device, the draw
//     scheduler, the runtime and the rasterizer, and only submits the draw scheduler's work.
//   - below stage 5 (M5): the temporal upscaler, camera and object motion and the GPU profiler
//     (vk_temporal_upscaler.h, vk_camera_motion.h, vk_object_motion.h, vk_gpu_profiler.h): all off.
//     Breadcrumbs and the settings overlay keep their headers; stubs_m3.cpp defines what the
//     shim calls.
// Each milestone deletes its part; M5 deletes this file.

#pragma once

#ifndef BB_METAL_NATIVE_STAGE
#error "BB_METAL_NATIVE_STAGE must be defined (gpu/CMakeLists.txt: target_compile_definitions)"
#endif

#include <array>
#include <memory>
#include <vector>

#include "common/types.h"
#include "video_core/renderer_vulkan/motion_history.h"
#include "video_core/renderer_vulkan/vk_common.h"
#include "video_core/texture_cache/image_view.h"
#include "video_core/texture_cache/types.h"

#if BB_METAL_NATIVE_STAGE < 4
#include "core/libraries/videoout/buffer.h"
#include "video_core/renderer_metal/mtl_device.h"
#include "video_core/renderer_vulkan/vk_runtime.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/texture_cache/texture_cache.h"
#endif

namespace Metal {
class Device;
}

namespace VideoCore {
class TextureCache;
}

namespace Frontend {
class WindowSDL;
}

namespace AmdGpu {
struct Liverpool;
struct ColorBuffer;
}

namespace Vulkan {

class Scheduler;
class Runtime;
class SceneTargets;
class ObjectMotion;
struct RenderState;

// --- M5: temporal upscaling (vk_temporal_upscaler.h) -----------------------------------------

class CameraMotion;

class TemporalUpscaler {
public:
    TemporalUpscaler(Metal::Device&, Scheduler&, VideoCore::TextureCache&, Runtime&,
                     CameraMotion&, SceneTargets&) {}

    [[nodiscard]] bool Enabled() const noexcept {
        return false;
    }
    void OnSceneColor(VideoCore::ImageId) {}
    void OnBlendedSceneDraw() {}
    void OnSceneComposite() {}
    void OnDispatch(u64) {}
    bool OnFrameStart() {
        return false;
    }
    bool RasterScaling() const {
        return false;
    }
    [[nodiscard]] float SceneMipBias() const {
        return 0.0f;
    }
    [[nodiscard]] u64 RedirectState() const noexcept {
        return 0;
    }
    [[nodiscard]] std::array<float, 2> Jitter() const noexcept {
        return {};
    }
    void OnDraw(u64, VideoCore::ImageId, VideoCore::ImageId, bool) {}
    void OnColorTarget(VideoCore::ImageId) {}

    struct Target {
        void* texture{}; ///< id<MTLTexture>
        u32 width{}, height{};
        bool native_ui = false;
    };
    bool RedirectColor(VideoCore::ImageId, const VideoCore::ImageViewInfo&, Target&) {
        return false;
    }
    bool RedirectDepth(VideoCore::ImageId, Target&) {
        return false;
    }
    bool RedirectSampled(VideoCore::ImageId, const VideoCore::ImageViewInfo&, void*& /*texture*/) {
        return false;
    }
    [[nodiscard]] bool RedirectsSampled(VideoCore::ImageId) const {
        return false;
    }

    struct Display {
        void* texture{}; ///< id<MTLTexture>
        vk::Format format{};
        u32 width{}, height{};
    };
    bool DisplayOverride(VAddr, Display&) {
        return false;
    }
};

// --- M5: camera motion vectors (vk_camera_motion.h) ------------------------------------------

class CameraMotion {
public:
    CameraMotion(Metal::Device&, Scheduler&, VideoCore::TextureCache&, Runtime&) {}

    void InvalidateHistory() {}
    void SetObjectMotion(ObjectMotion*) noexcept {}
    void SetJitter(std::array<float, 2>) noexcept {}
    [[nodiscard]] bool Enabled() const noexcept {
        return false;
    }
    [[nodiscard]] bool Ready() const noexcept {
        return false;
    }
    [[nodiscard]] VideoCore::ImageId Depth() const noexcept {
        return {};
    }
    [[nodiscard]] std::array<u32, 2> RenderSize() const noexcept {
        return {};
    }
    [[nodiscard]] float VerticalFov() const noexcept {
        return 0.0f;
    }
    [[nodiscard]] float Near() const noexcept {
        return 0.0f;
    }
    [[nodiscard]] std::array<std::array<float, 4>, 3> TaaDepthParameters() const noexcept {
        return {};
    }
    void OnConstants(const float*) {}
    void OnGBufferPass(VideoCore::ImageId) {}
    void OnDisplayPass(VideoCore::ImageId) {}
};

// --- M5: object motion vectors (vk_object_motion.h) ------------------------------------------

class ObjectMotion {
public:
    ObjectMotion(Metal::Device&, Scheduler&) {}

    [[nodiscard]] bool Enabled() const noexcept {
        return false;
    }
    using DrawInfo = Motion::Draw;
    u32 PrepareDraw(const DrawInfo&) {
        return 0; // parameter 0: motion off
    }
    bool Moving(const Motion::History::GateKey&, u64) {
        return false;
    }
    /// Uncached: scans every time.
    template <class Scan>
    Motion::IndexRangeCache::Result IndexRange(const Motion::IndexRangeCache::Key&, Scan&& scan) {
        return scan();
    }
    void Attach(RenderState&, u32, u32) {}
    void OnFrameStart() {}
    void InvalidateHistory() {}
};

// --- M5: GPU time per pass (vk_gpu_profiler.h): never on -------------------------------------

class GpuProfiler {
public:
    static GpuProfiler* Get() noexcept {
        return nullptr;
    }
    static void Init(Metal::Device&, Scheduler&) {}

    template <typename Describe>
    void Mark(u64, Describe&&) {}
    [[nodiscard]] u64 Current() const noexcept {
        return 0;
    }
    void Resume(u64) {}
    [[nodiscard]] bool Records(const Scheduler*) const noexcept {
        return false;
    }
    void BeginFrame() {}
};

// --- M4: the presenter (vk_presenter.h) ------------------------------------------------------

#if BB_METAL_NATIVE_STAGE < 4

struct Frame {
    u32 width{};
    u32 height{};
    u64 ready_tick{};
    bool is_hdr{false};
    u8 id{};
};

enum SchedulerType {
    Draw,
    Present,
    CpuFlip,
};

class Rasterizer;

/// Headless: nothing reaches the screen. A frame's draws are submitted when it is prepared (where
/// the real presenter submits them, on the thread recording into the draw scheduler).
class Presenter {
public:
    /// What video_out.cpp sets (the real one is HostPasses::PostProcessingPass::Settings).
    struct PostProcessSettings {
        float gamma = 1.0f;
        u32 hdr = 0;
        u32 srgb_input = 0;
    };

    Presenter(Frontend::WindowSDL& window, AmdGpu::Liverpool* liverpool);
    ~Presenter();

    PostProcessSettings& GetPPSettingsRef() {
        return pp_settings;
    }
    Frontend::WindowSDL& GetWindow() const {
        return window;
    }
    Rasterizer& GetRasterizer() const {
        return *rasterizer;
    }
    bool IsHDRSupported() const {
        return false;
    }
    void SetHDR(bool) {}

    VideoCore::Image& RegisterVideoOutSurface(
        const Libraries::VideoOut::BufferAttributeGroup& attribute, VAddr cpu_address) {
        vo_buffers_addr.emplace_back(cpu_address);
        auto desc = VideoCore::TextureCache::ImageDesc{attribute, cpu_address};
        const auto image_id = texture_cache.FindImage(desc);
        auto& image = texture_cache.GetImage(image_id);
        image.usage.vo_surface = 1u;
        return image;
    }

    bool IsVideoOutSurface(const AmdGpu::ColorBuffer& color_buffer) const;

    Frame* PrepareFrame(const Libraries::VideoOut::BufferAttributeGroup& attribute,
                        VAddr cpu_address);
    Frame* PrepareBlankFrame(bool present_thread);
    Frame* PrepareLastFrame();
    void Present(Frame* frame, bool is_reusing_frame = false, bool is_game_frame = true,
                 u32 present_id = 0);

    struct PresentTiming {
        u64 start, acquired, submitted, presented;
    };
    const PresentTiming& LastPresentTiming() const {
        return last_present;
    }
    bool IsDisplayTimingSupported() const {
        return false;
    }
    std::vector<vk::PastPresentationTimingGOOGLE> TakePastTimings() {
        return {};
    }

private:
    Frontend::WindowSDL& window;
    AmdGpu::Liverpool* liverpool;
    Metal::Device device;
    Scheduler draw_scheduler;
    Runtime runtime;
    std::unique_ptr<Rasterizer> rasterizer;
    VideoCore::TextureCache& texture_cache;
    PostProcessSettings pp_settings{};
    Frame frame{};
    PresentTiming last_present{};
    std::vector<VAddr> vo_buffers_addr;
};

#endif // BB_METAL_NATIVE_STAGE < 4

} // namespace Vulkan
