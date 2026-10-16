// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: what the shim calls of the breadcrumbs and the settings overlay, whose sources
// leave the build until M5.

#include "bbport_overlay.h"
#include "stubs_m3.h"
#include "video_core/renderer_vulkan/vk_breadcrumbs.h"

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
