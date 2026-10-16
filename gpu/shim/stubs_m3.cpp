// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: the out-of-line parts of the stand-ins in stubs_m3.h (renderer parts not converted to
// Metal yet), and what the shim calls of the breadcrumbs and the settings overlay, whose sources
// leave the build until M5.

#include <algorithm>
#include <cstdio>

#include "bbport_overlay.h"
#include "stubs_m3.h"
#include "video_core/renderer_vulkan/vk_breadcrumbs.h"

#if BB_METAL_NATIVE_STAGE < 4
#include "video_core/renderer_vulkan/vk_presenter.h"
#include "video_core/renderer_vulkan/vk_rasterizer.h"
#include "video_core/renderer_vulkan/vk_scene_resolution.h"
#include "video_core/texture_cache/blit_helper.h"
#endif

#if BB_METAL_NATIVE_STAGE < 4
namespace Vulkan {

Presenter::Presenter(Frontend::WindowSDL& window_, AmdGpu::Liverpool* liverpool_)
    : window{window_}, liverpool{liverpool_}, draw_scheduler{device, true},
      runtime{device, draw_scheduler},
      rasterizer{std::make_unique<Rasterizer>(device, draw_scheduler, runtime, liverpool)},
      texture_cache{rasterizer->GetTextureCache()} {
    std::printf("Presenter: headless stand-in (Metal-native M3; nothing is shown until M4)\n");
}

Presenter::~Presenter() {
    draw_scheduler.Finish();
    rasterizer->GetPipelineCache().Sync();
}

bool Presenter::IsVideoOutSurface(const AmdGpu::ColorBuffer& color_buffer) const {
    return std::ranges::find(vo_buffers_addr, color_buffer.Address()) != vo_buffers_addr.cend();
}

Frame* Presenter::PrepareFrame(const Libraries::VideoOut::BufferAttributeGroup&, VAddr) {
    // The real presenter submits the frame's draws here (vk_presenter.cpp, before the copy to
    // the frame image), on the thread that records into the draw scheduler. Present may run on
    // the swap thread, which must not flush the draw scheduler.
    draw_scheduler.Flush();
    return &frame;
}

Frame* Presenter::PrepareBlankFrame(bool) {
    return &frame;
}

Frame* Presenter::PrepareLastFrame() {
    return &frame;
}

void Presenter::Present(Frame*, bool, bool, u32) {}

} // namespace Vulkan
#endif // BB_METAL_NATIVE_STAGE < 4

namespace Vulkan::Breadcrumbs {

void ReportStuck(const char*) {}

} // namespace Vulkan::Breadcrumbs

namespace BbOverlay {

bool HandleEvent(const SDL_Event&) {
    return false;
}

void UpdateTextInput(SDL_Window*) {}

bool Visible() {
    return false;
}

bool CapturesInput() {
    return false;
}

} // namespace BbOverlay
