// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: stand-ins for the renderer parts not converted to Metal yet (Metal-native renderer,
// docs/superpowers/plans/2026-10-16-metal-native-roadmap.md). gpu/CMakeLists.txt leaves the
// files named in BB_M5_PENDING out of the build; their headers include this one instead and get
// classes with the same public API that do nothing (each guarded by BB_M5_PENDING_<FILE NAME>,
// which CMake defines while the file is pending):
//   - M5: the GPU profiler (vk_gpu_profiler.h): off.
//     The settings overlay keeps its header; stubs_m3.cpp defines what the shim calls.
// Converting a file removes its name from BB_M5_PENDING and its stand-in from here; the last
// conversion deletes this file.

#pragma once

#ifndef BB_METAL_NATIVE_STAGE
#error "BB_METAL_NATIVE_STAGE must be defined (gpu/CMakeLists.txt: target_compile_definitions)"
#endif

#include <array>

#include "common/types.h"
#include "video_core/renderer_vulkan/vk_common.h"
#include "video_core/texture_cache/image_view.h"
#include "video_core/texture_cache/types.h"

namespace Metal {
class Device;
}

namespace VideoCore {
class TextureCache;
}

namespace Vulkan {

class Scheduler;
class Runtime;
class SceneTargets;
struct RenderState;

// --- M5: GPU time per pass (vk_gpu_profiler.h): never on -------------------------------------

#ifdef BB_M5_PENDING_VK_GPU_PROFILER
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
#endif // BB_M5_PENDING_VK_GPU_PROFILER

} // namespace Vulkan
