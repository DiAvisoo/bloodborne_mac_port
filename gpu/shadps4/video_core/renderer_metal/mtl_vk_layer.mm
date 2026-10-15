// SPDX-License-Identifier: GPL-2.0-or-later

#import <Metal/Metal.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include "video_core/renderer_metal/mtl_vk_layer.h"
#include "video_core/renderer_vulkan/vk_common.h"
// The Metal export types (this directory builds without VK_USE_PLATFORM_METAL_EXT: under ARC its
// structs break vulkan.hpp; the C header alone is fine).
#include <vulkan/vulkan_metal.h>

namespace MetalLayer {
namespace {

using Dispatcher = std::remove_reference_t<decltype(VULKAN_HPP_DEFAULT_DISPATCHER)>;

// The dispatcher members the renderer's command recording and submission use (core names and the
// extension aliases vulkan.hpp may have filled instead).
#define BB_LAYER_FUNCTIONS(X)                                                                      \
    X(vkBeginCommandBuffer) X(vkEndCommandBuffer) X(vkQueueSubmit) X(vkQueueSubmit2)               \
    X(vkQueueSubmit2KHR) X(vkCmdPipelineBarrier) X(vkCmdPipelineBarrier2)                          \
    X(vkCmdPipelineBarrier2KHR) X(vkCmdBindPipeline) X(vkCmdPushDescriptorSetKHR)                  \
    X(vkCmdPushConstants) X(vkCmdBindDescriptorSets) X(vkCmdDispatch) X(vkCmdDispatchIndirect)     \
    X(vkCmdBeginRendering) X(vkCmdBeginRenderingKHR) X(vkCmdEndRendering)                          \
    X(vkCmdEndRenderingKHR) X(vkCmdDraw) X(vkCmdDrawIndexed) X(vkCmdDrawIndirect)                  \
    X(vkCmdDrawIndexedIndirect) X(vkCmdDrawIndirectCount) X(vkCmdDrawIndexedIndirectCount)         \
    X(vkCmdBindVertexBuffers) X(vkCmdBindVertexBuffers2) X(vkCmdBindVertexBuffers2EXT)             \
    X(vkCmdBindIndexBuffer) X(vkCmdCopyImage) X(vkCmdCopyBuffer) X(vkCmdCopyImageToBuffer)         \
    X(vkCmdCopyBufferToImage) X(vkCmdBlitImage) X(vkCmdResolveImage) X(vkCmdClearColorImage)       \
    X(vkCmdClearDepthStencilImage) X(vkCmdFillBuffer) X(vkCmdUpdateBuffer)                         \
    X(vkCmdSetViewport) X(vkCmdSetScissor) X(vkCmdSetViewportWithCount)                            \
    X(vkCmdSetViewportWithCountEXT) X(vkCmdSetScissorWithCount) X(vkCmdSetScissorWithCountEXT)     \
    X(vkCmdSetDepthTestEnable) X(vkCmdSetDepthTestEnableEXT) X(vkCmdSetDepthWriteEnable)           \
    X(vkCmdSetDepthWriteEnableEXT) X(vkCmdSetDepthCompareOp) X(vkCmdSetDepthCompareOpEXT)          \
    X(vkCmdSetDepthBoundsTestEnable) X(vkCmdSetDepthBounds) X(vkCmdSetDepthBiasEnable)             \
    X(vkCmdSetDepthBias) X(vkCmdSetStencilTestEnable) X(vkCmdSetStencilTestEnableEXT)              \
    X(vkCmdSetStencilOp) X(vkCmdSetStencilOpEXT) X(vkCmdSetStencilReference)                       \
    X(vkCmdSetStencilWriteMask) X(vkCmdSetStencilCompareMask) X(vkCmdSetPrimitiveRestartEnable)    \
    X(vkCmdSetRasterizerDiscardEnable) X(vkCmdSetCullMode) X(vkCmdSetCullModeEXT)                  \
    X(vkCmdSetFrontFace) X(vkCmdSetFrontFaceEXT) X(vkCmdSetBlendConstants)                         \
    X(vkCmdSetColorWriteMaskEXT) X(vkCmdSetLineWidth) X(vkCmdSetVertexInputEXT)                    \
    X(vkCmdBeginDebugUtilsLabelEXT) X(vkCmdEndDebugUtilsLabelEXT) X(vkCmdInsertDebugUtilsLabelEXT) \
    X(vkCmdWriteTimestamp) X(vkCmdWriteTimestamp2) X(vkCmdWriteBufferMarkerAMD)

enum Id : unsigned {
#define BB_ID(name) Id_##name,
    BB_LAYER_FUNCTIONS(BB_ID)
#undef BB_ID
        Count
};
constexpr const char* Names[] = {
#define BB_NAME(name) #name,
    BB_LAYER_FUNCTIONS(BB_NAME)
#undef BB_NAME
};

bool trace;
bool present_enabled;
std::array<std::atomic<unsigned long long>, Count> calls{};
std::chrono::steady_clock::time_point since = std::chrono::steady_clock::now();
std::mutex report_mutex;

void Report() {
    const auto now = std::chrono::steady_clock::now();
    if (now - since < std::chrono::seconds(5)) {
        return;
    }
    std::unique_lock lock{report_mutex, std::try_to_lock};
    if (!lock || now - since < std::chrono::seconds(5)) {
        return;
    }
    const double seconds = std::chrono::duration<double>(now - since).count();
    since = now;
    std::vector<std::pair<unsigned long long, unsigned>> top;
    for (unsigned i = 0; i < Count; ++i) {
        if (const auto n = calls[i].exchange(0)) {
            top.emplace_back(n, i);
        }
    }
    std::sort(top.rbegin(), top.rend());
    std::string line;
    for (const auto& [n, i] : top) {
        char item[96];
        std::snprintf(item, sizeof item, "%s%s %.0f", line.empty() ? "" : ", ", Names[i] + 2,
                      double(n) / seconds);
        line += item;
    }
    std::printf("Metal layer: Vulkan calls per second: %s\n", line.c_str());
}

// ---- The Metal side ----

PFN_vkExportMetalObjectsEXT export_objects;
VkDevice vk_device;
id<MTLCommandQueue> queue;
id<MTLRenderPipelineState> blit_pipelines[4]; ///< by MTLPixelFormat slot (BlitPipeline)
std::vector<std::pair<MTLPixelFormat, id<MTLRenderPipelineState>>> blit_by_format;
id<MTLSamplerState> linear_sampler, nearest_sampler;
id<MTLLibrary> blit_library;
std::mutex metal_mutex; ///< blit pipelines (created on first use)

/// A command buffer the Metal path encodes (Own .. submit).
struct Owned {
    id<MTLCommandBuffer> cmd;
    unsigned commands{};
};
std::mutex owned_mutex;
std::unordered_map<VkCommandBuffer, Owned> owned;
std::atomic<unsigned> owned_count{0}; ///< owned.size(): the lock is skipped while it is zero

Owned* FindOwned(VkCommandBuffer cb) {
    if (owned_count.load(std::memory_order_acquire) == 0) {
        return nullptr;
    }
    std::scoped_lock lock{owned_mutex};
    auto it = owned.find(cb);
    return it == owned.end() ? nullptr : &it->second;
}

id<MTLTexture> TextureOf(VkImage image) {
    VkExportMetalTextureInfoEXT texture{VK_STRUCTURE_TYPE_EXPORT_METAL_TEXTURE_INFO_EXT};
    texture.image = image;
    texture.plane = VK_IMAGE_ASPECT_PLANE_0_BIT;
    VkExportMetalObjectsInfoEXT info{VK_STRUCTURE_TYPE_EXPORT_METAL_OBJECTS_INFO_EXT, &texture};
    export_objects(vk_device, &info);
    return texture.mtlTexture;
}

/// Fullscreen-triangle copy of a source rectangle (normalized) into the viewport, for
/// vkCmdBlitImage: Metal's blit encoder neither scales nor filters.
id<MTLRenderPipelineState> BlitPipeline(MTLPixelFormat format) {
    std::scoped_lock lock{metal_mutex};
    for (const auto& [known, state] : blit_by_format) {
        if (known == format) {
            return state;
        }
    }
    NSError* error = nil;
    if (!blit_library) {
        NSString* source =
            @"#include <metal_stdlib>\n"
             "using namespace metal;\n"
             "struct V { float4 pos [[position]]; float2 uv; };\n"
             "vertex V blit_vs(uint id [[vertex_id]], constant float4& src [[buffer(0)]]) {\n"
             "    float2 p = float2((id << 1) & 2, id & 2);\n"
             "    V v; v.pos = float4(p * 2.0 - 1.0, 0.0, 1.0);\n"
             "    float2 t = float2(p.x, 1.0 - p.y);\n" // Metal NDC y up, texture y down
             "    v.uv = mix(src.xy, src.zw, t); return v; }\n"
             "fragment float4 blit_fs(V in [[stage_in]], texture2d<float> tex [[texture(0)]],\n"
             "                        sampler s [[sampler(0)]]) { return tex.sample(s, in.uv); }\n";
        blit_library = [queue.device newLibraryWithSource:source options:nil error:&error];
        if (!blit_library) {
            std::printf("Metal layer: blit shader: %s\n", error.localizedDescription.UTF8String);
            return nil;
        }
        MTLSamplerDescriptor* sd = [MTLSamplerDescriptor new];
        sd.minFilter = sd.magFilter = MTLSamplerMinMagFilterLinear;
        sd.sAddressMode = sd.tAddressMode = MTLSamplerAddressModeClampToEdge;
        linear_sampler = [queue.device newSamplerStateWithDescriptor:sd];
        sd.minFilter = sd.magFilter = MTLSamplerMinMagFilterNearest;
        nearest_sampler = [queue.device newSamplerStateWithDescriptor:sd];
    }
    MTLRenderPipelineDescriptor* d = [MTLRenderPipelineDescriptor new];
    d.vertexFunction = [blit_library newFunctionWithName:@"blit_vs"];
    d.fragmentFunction = [blit_library newFunctionWithName:@"blit_fs"];
    d.colorAttachments[0].pixelFormat = format;
    id<MTLRenderPipelineState> state = [queue.device newRenderPipelineStateWithDescriptor:d
                                                                                    error:&error];
    if (!state) {
        std::printf("Metal layer: blit pipeline for format %lu: %s\n", (unsigned long)format,
                    error.localizedDescription.UTF8String);
    }
    blit_by_format.emplace_back(format, state);
    return state;
}

void Unsupported(unsigned id) {
    static std::array<std::atomic<bool>, Count> logged{};
    if (!logged[id].exchange(true)) {
        std::printf("Metal layer: %s is not implemented in the Metal path yet (skipped)\n",
                    Names[id]);
    }
}

void ClearColor(Owned& o, VkImage image, const VkClearColorValue* color, uint32_t count,
                const VkImageSubresourceRange* ranges) {
    id<MTLTexture> texture = TextureOf(image);
    if (!texture) {
        return;
    }
    for (uint32_t r = 0; r < count; ++r) {
        const auto& range = ranges[r];
        const uint32_t levels = range.levelCount == VK_REMAINING_MIP_LEVELS
                                    ? uint32_t(texture.mipmapLevelCount) - range.baseMipLevel
                                    : range.levelCount;
        const uint32_t layers = range.layerCount == VK_REMAINING_ARRAY_LAYERS
                                    ? uint32_t(texture.arrayLength) - range.baseArrayLayer
                                    : range.layerCount;
        for (uint32_t level = 0; level < levels; ++level) {
            for (uint32_t layer = 0; layer < layers; ++layer) {
                MTLRenderPassDescriptor* rp = [MTLRenderPassDescriptor renderPassDescriptor];
                rp.colorAttachments[0].texture = texture;
                rp.colorAttachments[0].level = range.baseMipLevel + level;
                rp.colorAttachments[0].slice = range.baseArrayLayer + layer;
                rp.colorAttachments[0].loadAction = MTLLoadActionClear;
                rp.colorAttachments[0].storeAction = MTLStoreActionStore;
                rp.colorAttachments[0].clearColor =
                    MTLClearColorMake(color->float32[0], color->float32[1], color->float32[2],
                                      color->float32[3]);
                [[o.cmd renderCommandEncoderWithDescriptor:rp] endEncoding];
            }
        }
    }
}

void Blit(Owned& o, VkImage src_image, VkImage dst_image, uint32_t count,
          const VkImageBlit* regions, VkFilter filter) {
    id<MTLTexture> src = TextureOf(src_image);
    id<MTLTexture> dst = TextureOf(dst_image);
    id<MTLRenderPipelineState> pipeline = dst ? BlitPipeline(dst.pixelFormat) : nil;
    if (!src || !dst || !pipeline) {
        return;
    }
    for (uint32_t r = 0; r < count; ++r) {
        const VkImageBlit& b = regions[r];
        const float sw = float(std::max<NSUInteger>(1, src.width >> b.srcSubresource.mipLevel));
        const float sh = float(std::max<NSUInteger>(1, src.height >> b.srcSubresource.mipLevel));
        float uv[4] = {b.srcOffsets[0].x / sw, b.srcOffsets[0].y / sh, b.srcOffsets[1].x / sw,
                       b.srcOffsets[1].y / sh};
        // Destination rectangle; a mirrored destination swaps the source edges instead.
        int x0 = b.dstOffsets[0].x, x1 = b.dstOffsets[1].x, y0 = b.dstOffsets[0].y,
            y1 = b.dstOffsets[1].y;
        if (x1 < x0) {
            std::swap(x0, x1);
            std::swap(uv[0], uv[2]);
        }
        if (y1 < y0) {
            std::swap(y0, y1);
            std::swap(uv[1], uv[3]);
        }
        if (x1 == x0 || y1 == y0) {
            continue;
        }
        MTLRenderPassDescriptor* rp = [MTLRenderPassDescriptor renderPassDescriptor];
        rp.colorAttachments[0].texture = dst;
        rp.colorAttachments[0].level = b.dstSubresource.mipLevel;
        rp.colorAttachments[0].slice = b.dstSubresource.baseArrayLayer;
        rp.colorAttachments[0].loadAction = MTLLoadActionLoad;
        rp.colorAttachments[0].storeAction = MTLStoreActionStore;
        id<MTLRenderCommandEncoder> enc = [o.cmd renderCommandEncoderWithDescriptor:rp];
        [enc setRenderPipelineState:pipeline];
        [enc setViewport:(MTLViewport){double(x0), double(y0), double(x1 - x0), double(y1 - y0),
                                       0.0, 1.0}];
        [enc setVertexBytes:uv length:sizeof uv atIndex:0];
        [enc setFragmentTexture:src atIndex:0];
        [enc setFragmentSamplerState:filter == VK_FILTER_NEAREST ? nearest_sampler : linear_sampler
                             atIndex:0];
        [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
        [enc endEncoding];
    }
}

/// Commits the Metal command buffers of the owned command buffers in `buffers`, in order, and
/// releases them; MoltenVK then gets the submit unchanged (its copies of them are empty). With
/// MoltenVK's single-queue semaphores ("Metal implicit guarantees within a single queue") order
/// on the queue is what synchronizes, and ours comes first.
template <typename Get>
void CommitOwned(uint32_t count, Get&& get) {
    if (!present_enabled) {
        return;
    }
    for (uint32_t i = 0; i < count; ++i) {
        const VkCommandBuffer cb = get(i);
        id<MTLCommandBuffer> cmd = nil;
        unsigned commands = 0;
        {
            std::scoped_lock lock{owned_mutex};
            auto it = owned.find(cb);
            if (it == owned.end()) {
                continue;
            }
            cmd = it->second.cmd;
            commands = it->second.commands;
            owned.erase(it);
            owned_count.store(unsigned(owned.size()), std::memory_order_release);
        }
        [cmd commit];
        static std::atomic<unsigned long long> committed{0};
        const auto n = ++committed;
        if (n == 1 || n % 1000 == 0) {
            std::printf("Metal layer: %llu command buffers encoded in Metal (this one: %u commands)\n",
                        n, commands);
        }
    }
}

/// One forwarding function per dispatcher member: counts, encodes Metal for an owned command
/// buffer (the commands the Metal path implements so far), else calls MoltenVK.
template <unsigned I, typename F>
struct Hook;
template <unsigned I, typename R, typename... A>
struct Hook<I, R (*)(A...)> {
    static inline R (*original)(A...) = nullptr;

    static R Call(A... args) {
        if (trace) {
            calls[I].fetch_add(1, std::memory_order_relaxed);
            if constexpr (I == Id_vkQueueSubmit || I == Id_vkQueueSubmit2 ||
                          I == Id_vkQueueSubmit2KHR) {
                Report();
            }
        }
        if constexpr (I == Id_vkQueueSubmit) {
            const auto [vkqueue, count, submits, fence] = std::tuple{args...};
            for (uint32_t s = 0; s < count; ++s) {
                CommitOwned(submits[s].commandBufferCount,
                            [&](uint32_t i) { return submits[s].pCommandBuffers[i]; });
            }
        } else if constexpr (I == Id_vkQueueSubmit2 || I == Id_vkQueueSubmit2KHR) {
            const auto [vkqueue, count, submits, fence] = std::tuple{args...};
            for (uint32_t s = 0; s < count; ++s) {
                CommitOwned(submits[s].commandBufferInfoCount, [&](uint32_t i) {
                    return submits[s].pCommandBufferInfos[i].commandBuffer;
                });
            }
        } else if constexpr (I != Id_vkBeginCommandBuffer && I != Id_vkEndCommandBuffer) {
            // vkCmd*: the first argument is the command buffer.
            if (Owned* o = FindOwned(std::get<0>(std::tuple{args...}))) {
                ++o->commands;
                Encode(*o, args...);
                if constexpr (!std::is_void_v<R>) {
                    return R{};
                } else {
                    return;
                }
            }
        }
        return original(args...);
    }

    /// The commands the Metal path implements for owned command buffers so far (step B: what the
    /// presenter records). Barriers need nothing: MoltenVK's resources are hazard-tracked and
    /// every command here is its own encoder.
    static void Encode(Owned& o, A... args) {
        if constexpr (I == Id_vkCmdPipelineBarrier || I == Id_vkCmdPipelineBarrier2 ||
                      I == Id_vkCmdPipelineBarrier2KHR || I == Id_vkCmdBeginDebugUtilsLabelEXT ||
                      I == Id_vkCmdEndDebugUtilsLabelEXT ||
                      I == Id_vkCmdInsertDebugUtilsLabelEXT) {
        } else if constexpr (I == Id_vkCmdClearColorImage) {
            const auto [cb, image, layout, color, count, ranges] = std::tuple{args...};
            ClearColor(o, image, color, count, ranges);
        } else if constexpr (I == Id_vkCmdBlitImage) {
            const auto [cb, src, src_layout, dst, dst_layout, count, regions, filter] =
                std::tuple{args...};
            Blit(o, src, dst, count, regions, filter);
        } else {
            Unsupported(I);
        }
    }

    static void Install(R (*&slot)(A...)) {
        if (slot) {
            original = slot;
            slot = &Call;
        }
    }
};

} // namespace

bool PresentEnabled() {
    return present_enabled;
}

void SetDevice(void* device, void* vkqueue) {
    if (!present_enabled) {
        return;
    }
    vk_device = static_cast<VkDevice>(device);
    export_objects = reinterpret_cast<PFN_vkExportMetalObjectsEXT>(
        VULKAN_HPP_DEFAULT_DISPATCHER.vkGetDeviceProcAddr(vk_device, "vkExportMetalObjectsEXT"));
    if (!export_objects) {
        std::printf("Metal layer: no VK_EXT_metal_objects; the Metal path stays off\n");
        present_enabled = false;
        return;
    }
    VkExportMetalCommandQueueInfoEXT q{VK_STRUCTURE_TYPE_EXPORT_METAL_COMMAND_QUEUE_INFO_EXT};
    q.queue = static_cast<VkQueue>(vkqueue);
    VkExportMetalObjectsInfoEXT info{VK_STRUCTURE_TYPE_EXPORT_METAL_OBJECTS_INFO_EXT, &q};
    export_objects(vk_device, &info);
    queue = q.mtlCommandQueue;
    if (!queue) {
        std::printf("Metal layer: MoltenVK's queue could not be exported; the Metal path stays off\n");
        present_enabled = false;
        return;
    }
    std::printf("Metal layer: the present pass is encoded in Metal (BB_METAL_PRESENT=1) on %s\n",
                queue.device.name.UTF8String);
}

void Own(void* command_buffer) {
    if (!present_enabled) {
        return;
    }
    Owned o;
    o.cmd = [queue commandBuffer];
    o.cmd.label = @"bbport Metal path";
    std::scoped_lock lock{owned_mutex};
    owned[static_cast<VkCommandBuffer>(command_buffer)] = o;
    owned_count.store(unsigned(owned.size()), std::memory_order_release);
}

void Install(void* dispatcher_ptr) {
    const char* env = std::getenv("BB_METAL_TRACE");
    trace = env && env[0] == '1';
    const char* present = std::getenv("BB_METAL_PRESENT");
    present_enabled = present && present[0] == '1';
    if (!trace && !present_enabled) {
        return;
    }
    auto& d = *static_cast<Dispatcher*>(dispatcher_ptr);
#define BB_INSTALL(name) Hook<Id_##name, decltype(d.name)>::Install(d.name);
    BB_LAYER_FUNCTIONS(BB_INSTALL)
#undef BB_INSTALL
    std::printf("Metal layer: installed over %u Vulkan command functions\n", unsigned(Count));
}

} // namespace MetalLayer
