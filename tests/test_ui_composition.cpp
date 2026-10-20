// SPDX-License-Identifier: GPL-2.0-or-later
#include <cassert>
#include <cstdio>
#include "gpu/shadps4/video_core/renderer_vulkan/ui_composition.h"

int main() {
    using namespace Vulkan::UiComposition;
    // Title/loading menus have no depth/camera. UI must still go to the native target.
    assert(Choose(true, true, false, true) == Background::Copy);
    assert(Choose(true, true, false, false) == Background::Copy);
    // In-game HUD: first reconstruct the scene, then draw the native UI over it.
    assert(Choose(true, true, true, true) == Background::Temporal);
    // Runtime FSR disable/failure must not change UI resolution.
    assert(Choose(true, true, true, false) == Background::Copy);
    assert(Choose(true, false, true, true) == Background::None);
    assert(Choose(false, true, true, true) == Background::None);
    assert(NativeViewport(1920, -1080));
    assert(!NativeViewport(960, -540));
    assert(MovieShader(0x34e8a281) && MovieShader(0x24042a9b));
    assert(!MovieShader(0x0b0acf50)); // fullscreen post/tonemap also has a native viewport
    assert(!MovieShader(0x6c62a79f)); // shadow geometry
    const auto display = Scale(960, 540, 1920, 1080, false);
    assert(display[0] == 2 && display[1] == 2);
    const auto ui = Scale(960, 540, 1920, 1080, true);
    assert(ui[0] == 1 && ui[1] == 1); // don't double an already native UI viewport
    const auto larger = Scale(960, 540, 3840, 2160, true);
    assert(larger[0] == 2 && larger[1] == 2);
    // Wide Scaleform movies already use the full width, but the scene/display-copy
    // coordinates still use the retail guest buffer. Scaling both identically doubled
    // menu text while frames without movie draws reverted to a narrow display.
    for (const auto size : {std::array<uint32_t, 2>{3840, 1080}, {2560, 1080},
                           {3440, 1440}, {5120, 1440}, {2560, 720}}) {
        assert(NativeViewport(float(size[0]), -float(size[1]), size));
        const auto wide_ui = Scale(1920, 1080, size[0], size[1], true, size);
        assert(wide_ui[0] == 1 && wide_ui[1] == 1);
        const auto wide_display = Scale(1920, 1080, size[0], size[1], false, size);
        assert(wide_display[0] == float(size[0]) / 1920);
        assert(wide_display[1] == float(size[1]) / 1080);
    }
    std::puts("UI composition: PASS (menu without camera, HUD, FSR off, viewport scaling)");
}
