// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: the Metal pipelines the shadow builder made (mtl_shadow.mm), by Vulkan pipeline handle,
// for the Metal frame path (mtl_vk_layer.mm).

#pragma once

#include <array>
#include "video_core/renderer_vulkan/vk_common.h"

namespace Metal {
struct RenderPipeline;
struct ComputePipeline;
struct TessellationPipeline;
struct DepthStencilKey;
} // namespace Metal

namespace MetalShadow {

/// State a generic (host-side) pipeline fixes at creation; the game's pipelines set it all
/// dynamically.
struct FixedState {
    bool cull{}, front_face{}, depth{}, stencil{}, depth_bias{}; ///< these are not dynamic
    VkCullModeFlags cull_mode{};
    VkFrontFace front{};
    bool depth_test{}, depth_write{};
    VkCompareOp depth_compare{VK_COMPARE_OP_ALWAYS};
    bool stencil_test{};
    bool depth_bias_enable{};
};

struct GraphicsEntry {
    const Metal::RenderPipeline* render{};     ///< null: no Metal pipeline (the draw is skipped)
    const Metal::TessellationPipeline* tess{}; ///< a tessellated draw's pipelines
    bool rect_list{};                          ///< the vertex shader is the rect list rewrite
    bool generic{};                            ///< built from a Vulkan create info (host passes)
    FixedState fixed;                          ///< generic pipelines: their static state
    std::array<u32, 32> strides{};             ///< generic pipelines: vertex binding strides
};

/// Null when the Vulkan pipeline has no Metal counterpart (yet).
const GraphicsEntry* FindGraphics(VkPipeline pipeline);
const Metal::ComputePipeline* FindCompute(VkPipeline pipeline);
void Forget(VkPipeline pipeline);
/// Pipelines made outside the game's pipeline cache (host passes, detiling, copies): built from
/// their Vulkan create infos. Not while GraphicsPipeline/ComputePipeline create theirs (GameBuildScope,
/// vk_metal_shadow.h): those come through Graphics/Compute with their keys.
void ModuleCreated(VkShaderModule module, const VkShaderModuleCreateInfo* info);
void GraphicsCreated(const VkGraphicsPipelineCreateInfo* info, VkPipeline pipeline);
void ComputeCreated(const VkComputePipelineCreateInfo* info, VkPipeline pipeline);

/// id<MTLDepthStencilState> for the key (the pipeline cache's), null on failure.
void* DepthStencil(const Metal::DepthStencilKey& key);

} // namespace MetalShadow
