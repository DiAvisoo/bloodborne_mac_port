// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: the Metal frame encoder (docs/DESIGN_METAL_BACKEND.md, M2 step C). The Vulkan layer
// (mtl_vk_layer.mm) passes every command of a command buffer the Metal path owns to these
// functions, which encode it into one MTLCommandBuffer on MoltenVK's queue. Objective-C++ only.

#pragma once

#ifndef __OBJC__
#error "mtl_frame.h is for Objective-C++ (.mm) files only"
#endif

#import <Metal/Metal.h>

#include <array>
#include <atomic>
#include <unordered_map>
#include <vector>
#include "video_core/renderer_vulkan/vk_common.h"
// Metal export types (this directory builds without VK_USE_PLATFORM_METAL_EXT).
#include <vulkan/vulkan_metal.h>
#include "common/types.h"

namespace MetalShadow {
struct GraphicsEntry;
}
namespace Metal {
struct ComputePipeline;
}

namespace MetalFrame {

/// One descriptor of a push descriptor set (by binding, then array element).
struct Bound {
    id<MTLBuffer> buffer;
    u64 offset{};
    u64 size{};
    id<MTLTexture> texture;
    id<MTLSamplerState> sampler;
};

struct Binding {
    std::vector<Bound> elements; ///< array elements
};

/// What the frame encoder keeps for one owned command buffer.
struct Owned {
    id<MTLCommandBuffer> cmd;
    id<MTLRenderCommandEncoder> render;
    id<MTLComputeCommandEncoder> compute;
    id<MTLBlitCommandEncoder> blit;
    unsigned commands{};

    // Render pass (vkCmdBeginRendering .. vkCmdEndRendering).
    bool rendering{};
    u32 target_width{}, target_height{};

    // Bound state.
    const MetalShadow::GraphicsEntry* graphics{};
    const Metal::ComputePipeline* compute_pipeline{};
    bool graphics_bound{}; ///< a graphics pipeline was bound (even one without a Metal pipeline)
    std::array<u8, 256> push{};
    /// Push descriptors by (set << 16 | binding), per bind point.
    std::unordered_map<u32, Binding> graphics_bindings, compute_bindings;

    struct VertexBuffer {
        id<MTLBuffer> buffer;
        u64 offset{};
        u64 stride{};
    };
    std::array<VertexBuffer, 32> vertex_buffers{};
    id<MTLBuffer> index_buffer;
    u64 index_offset{};
    VkIndexType index_type{VK_INDEX_TYPE_UINT16};

    // Dynamic state (as the Vulkan side set it).
    std::vector<VkViewport> viewports;
    std::vector<VkRect2D> scissors;
    VkCullModeFlags cull_mode{};
    VkFrontFace front_face{VK_FRONT_FACE_COUNTER_CLOCKWISE};
    bool depth_test{}, depth_write{};
    VkCompareOp depth_compare{VK_COMPARE_OP_ALWAYS};
    bool stencil_test{};
    struct Face {
        VkStencilOp fail{}, pass{}, depth_fail{};
        VkCompareOp compare{VK_COMPARE_OP_ALWAYS};
        u32 compare_mask{0xff}, write_mask{0xff}, reference{};
    } front, back;
    bool depth_bias_enable{};
    float depth_bias[3]{}; ///< constant, clamp, slope
    float blend_constants[4]{};
    bool rasterizer_discard{};
};

/// The device, queue and export entry point (from the layer's SetDevice).
void Init(VkDevice device, id<MTLCommandQueue> queue, PFN_vkExportMetalObjectsEXT export_objects,
          PFN_vkGetBufferDeviceAddress get_address);

/// Ends whatever encoder is open (before the command buffer is committed, at barriers, ...).
void EndEncoders(Owned& o);

// Resource bookkeeping from the device functions the layer sees.
void SamplerCreated(VkSampler sampler, const VkSamplerCreateInfo* info);
void SamplerDestroyed(VkSampler sampler);
void ImageViewCreated(VkImageView view, const VkImageViewCreateInfo* info);
void ImageViewDestroyed(VkImageView view);
void DeviceAddressTaken(VkBuffer buffer);

// Commands.
void Barrier(Owned& o);
void BeginRendering(Owned& o, const VkRenderingInfo* info);
void EndRendering(Owned& o);
void BindPipeline(Owned& o, VkPipelineBindPoint point, VkPipeline pipeline);
void PushDescriptorSet(Owned& o, VkPipelineBindPoint point, u32 set, u32 count,
                       const VkWriteDescriptorSet* writes);
void PushConstants(Owned& o, u32 offset, u32 size, const void* values);
void BindVertexBuffers(Owned& o, u32 first, u32 count, const VkBuffer* buffers,
                       const VkDeviceSize* offsets, const VkDeviceSize* strides);
void BindIndexBuffer(Owned& o, VkBuffer buffer, VkDeviceSize offset, VkIndexType type);
void SetViewports(Owned& o, u32 count, const VkViewport* viewports);
void SetScissors(Owned& o, u32 count, const VkRect2D* scissors);
void Draw(Owned& o, u32 vertex_count, u32 instance_count, u32 first_vertex, u32 first_instance);
void DrawIndexed(Owned& o, u32 index_count, u32 instance_count, u32 first_index,
                 s32 vertex_offset, u32 first_instance);
void DrawIndirect(Owned& o, VkBuffer buffer, VkDeviceSize offset, u32 count, u32 stride,
                  bool indexed);
void Dispatch(Owned& o, u32 x, u32 y, u32 z);
void DispatchIndirect(Owned& o, VkBuffer buffer, VkDeviceSize offset);
void CopyBuffer(Owned& o, VkBuffer src, VkBuffer dst, u32 count, const VkBufferCopy* regions);
void FillBuffer(Owned& o, VkBuffer dst, VkDeviceSize offset, VkDeviceSize size, u32 data);
void UpdateBuffer(Owned& o, VkBuffer dst, VkDeviceSize offset, VkDeviceSize size,
                  const void* data);
void CopyImage(Owned& o, VkImage src, VkImage dst, u32 count, const VkImageCopy* regions);
void CopyBufferToImage(Owned& o, VkBuffer src, VkImage dst, u32 count,
                       const VkBufferImageCopy* regions);
void CopyImageToBuffer(Owned& o, VkImage src, VkBuffer dst, u32 count,
                       const VkBufferImageCopy* regions);
void ClearColorImage(Owned& o, VkImage image, const VkClearColorValue* color, u32 count,
                     const VkImageSubresourceRange* ranges);
void ClearDepthStencilImage(Owned& o, VkImage image, const VkClearDepthStencilValue* value,
                            u32 count, const VkImageSubresourceRange* ranges);
void BlitImage(Owned& o, VkImage src, VkImage dst, u32 count, const VkImageBlit* regions,
               VkFilter filter);

/// Counters for BB_FRAME_STATS-style reports (skipped work by reason).
struct Stats {
    std::atomic<u64> draws{}, dispatches{}, skipped_no_pipeline{}, skipped_tess{},
        skipped_emulated{}, skipped_discard{}, unsupported{};
};
Stats& GetStats();

} // namespace MetalFrame
