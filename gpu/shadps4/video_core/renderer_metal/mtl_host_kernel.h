// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: the renderer's own compute kernels (host_shaders GLSL -> SPIR-V) as Metal compute
// pipelines (Metal-native renderer).

#pragma once

#include <array>
#include <span>
#include <string_view>
#include "common/types.h"
#include "video_core/renderer_metal/mtl_object.h"
#include "video_core/renderer_metal/mtl_specialize.h"

namespace Metal {

class Device;

struct HostKernel {
    Object pipeline;                  ///< id<MTLComputePipelineState>
    std::array<u32, 8> buffer_slot{}; ///< SPIR-V binding (set 0) -> MSL buffer index (~0u: none)
    std::array<u32, 8> texture_slot{}; ///< SPIR-V binding (set 0) -> MSL texture index (~0u: none)
    std::array<u32, 3> local_size{};  ///< threads per threadgroup (the module's LocalSize)
};

/// Compiles a host GLSL kernel (SPIR-V from host_shaders) with specialisation constants baked in.
/// Asserts when the kernel does not compile (a host kernel is part of the renderer).
HostKernel BuildHostKernel(Device& device, std::span<const u32> spirv,
                           std::span<const SpecConstant> constants, std::string_view name);

} // namespace Metal
