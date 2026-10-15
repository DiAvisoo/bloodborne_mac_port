// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: Metal back end "shadow" builder (BB_METAL_SHADOW=1, and always with BB_METAL_FRAME=1) (docs/DESIGN_METAL_BACKEND.md, M1). With BB_METAL_SHADOW=1
// in the arm64 GPU host, every pipeline the Vulkan path creates is also built as a Metal pipeline
// state (renderer_metal/mtl_shadow.mm) and the results are counted; nothing draws with them.
// Everywhere else these are no-ops.

#pragma once

#include <span>
#include "common/types.h"
#include "video_core/renderer_vulkan/vk_common.h"

namespace Vulkan {
struct GraphicsPipelineKey;
struct ComputePipelineKey;
} // namespace Vulkan

namespace MetalShadow {

/// One vertex attribute as the Vulkan pipeline declares it (GraphicsPipeline::GetVertexInputs).
struct VertexInput {
    u32 location{};
    bool per_instance{};
    u32 divisor{1};
};

#ifdef BB_METAL_BACKEND
bool Enabled();
/// The SPIR-V a Vulkan shader module was made from (CompileSPV).
void RegisterModule(vk::ShaderModule module, std::span<const u32> spirv);
/// A graphics pipeline the Vulkan path built; `modules`/`has_stage` are indexed by SwStage.
/// `pipeline`: the Vulkan pipeline, so the Metal frame path finds the Metal one for it.
void Graphics(const Vulkan::GraphicsPipelineKey& key, std::span<const vk::ShaderModule> modules,
              std::span<const bool> has_stage, std::span<const VertexInput> inputs,
              bool aux_fragment, vk::Pipeline pipeline);
void Compute(const Vulkan::ComputePipelineKey& key, vk::ShaderModule module, vk::Pipeline pipeline);
/// Around the game pipelines' vkCreate*Pipelines: the layer's generic translation skips them.
struct GameBuildScope {
    GameBuildScope();
    ~GameBuildScope();
};
#else
struct GameBuildScope {};
inline bool Enabled() {
    return false;
}
inline void RegisterModule(vk::ShaderModule, std::span<const u32>) {}
inline void Graphics(const Vulkan::GraphicsPipelineKey&, std::span<const vk::ShaderModule>,
                     std::span<const bool>, std::span<const VertexInput>, bool, vk::Pipeline) {}
inline void Compute(const Vulkan::ComputePipelineKey&, vk::ShaderModule, vk::Pipeline) {}
#endif

} // namespace MetalShadow
