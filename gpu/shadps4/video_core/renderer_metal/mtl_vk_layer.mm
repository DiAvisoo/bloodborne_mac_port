// SPDX-License-Identifier: GPL-2.0-or-later

#import <Metal/Metal.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>
#include "video_core/renderer_metal/mtl_vk_layer.h"
#include "video_core/renderer_vulkan/vk_common.h"
// The Metal export types (this directory builds without VK_USE_PLATFORM_METAL_EXT: under ARC its
// structs break vulkan.hpp; the C header alone is fine).
#include <vulkan/vulkan_metal.h>
#include "video_core/renderer_metal/mtl_frame.h"
#include "video_core/renderer_metal/mtl_shadow.h"

namespace MetalLayer {
namespace {

using Dispatcher = std::remove_reference_t<decltype(VULKAN_HPP_DEFAULT_DISPATCHER)>;
using MetalFrame::Owned;

// The dispatcher members the renderer's command recording and submission use (core names and the
// extension aliases vulkan.hpp may have filled instead), and the device functions whose objects
// the Metal path has to know about.
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
    X(vkCmdWriteTimestamp) X(vkCmdWriteTimestamp2) X(vkCmdWriteBufferMarkerAMD)                    \
    X(vkCreateSampler) X(vkDestroySampler) X(vkDestroyImageView) X(vkDestroyPipeline)              \
    X(vkGetBufferDeviceAddress) X(vkGetBufferDeviceAddressKHR) X(vkCreateShaderModule)            \
    X(vkCreateGraphicsPipelines) X(vkCreateComputePipelines) X(vkCreateImageView)               \
    X(vkSetDebugUtilsObjectNameEXT) X(vkUpdateDescriptorSets) X(vkGetDeviceProcAddr)             \
    X(vkGetInstanceProcAddr)

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
bool present_enabled; ///< BB_METAL_PRESENT=1: the present pass (step B)
bool frame_enabled;   ///< BB_METAL_FRAME=1: every command buffer (step C)
std::array<std::atomic<unsigned long long>, Count> calls{};
std::chrono::steady_clock::time_point since = std::chrono::steady_clock::now();
std::mutex report_mutex;
id<MTLCommandQueue> queue;

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
    if (frame_enabled) {
        auto& s = MetalFrame::GetStats();
        std::printf("Metal frame: per second: %.0f draws, %.0f dispatches; skipped: %.0f without "
                    "a Metal pipeline, %.0f tessellated, %.0f emulated topology, %.0f discarded; "
                    "%.0f unsupported commands; %zu device-address buffers declared per encoder; tessellated "
                    "draws %.0f, of them right after another %.0f\n",
                    s.draws.exchange(0) / seconds, s.dispatches.exchange(0) / seconds,
                    s.skipped_no_pipeline.exchange(0) / seconds, s.skipped_tess.exchange(0) / seconds,
                    s.skipped_emulated.exchange(0) / seconds, s.skipped_discard.exchange(0) / seconds,
                    s.unsupported.exchange(0) / seconds, MetalFrame::AddressBufferCount(),
                    s.tess_draws.exchange(0) / seconds, s.tess_after_tess.exchange(0) / seconds);
    }
    if (!trace) {
        return;
    }
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

/// Our entry points by Id (installed ones), for vkGet*ProcAddr: libraries that load their own
/// function pointers (FSR, the ImGui overlay) then record through the layer too.
std::array<PFN_vkVoidFunction, Count> hooked{};
PFN_vkVoidFunction HookedByName(const char* name) {
    if (!name || !(present_enabled || frame_enabled)) {
        return nullptr;
    }
    for (unsigned i = 0; i < Count; ++i) {
        if (hooked[i] && i != Id_vkGetDeviceProcAddr && i != Id_vkGetInstanceProcAddr &&
            std::strcmp(Names[i], name) == 0) {
            return hooked[i];
        }
    }
    return nullptr;
}

std::mutex owned_mutex;
std::unordered_map<VkCommandBuffer, Owned*> owned;
std::atomic<unsigned> owned_count{0}; ///< owned.size(): the lock is skipped while it is zero

/// Bumped whenever an owned command buffer appears or goes: the per-thread lookup cache below is
/// valid while it is unchanged (a command buffer is recorded by one thread at a time).
std::atomic<u64> owned_generation{0};

Owned* FindOwned(VkCommandBuffer cb) {
    if (owned_count.load(std::memory_order_acquire) == 0) {
        return nullptr;
    }
    thread_local VkCommandBuffer cached_cb = VK_NULL_HANDLE;
    thread_local Owned* cached = nullptr;
    thread_local u64 cached_generation = ~0ull;
    const u64 generation = owned_generation.load(std::memory_order_acquire);
    if (cb == cached_cb && generation == cached_generation) {
        return cached;
    }
    std::scoped_lock lock{owned_mutex};
    auto it = owned.find(cb);
    cached_cb = cb;
    cached = it == owned.end() ? nullptr : it->second;
    cached_generation = owned_generation.load(std::memory_order_acquire);
    return cached;
}

void OwnBuffer(VkCommandBuffer cb) {
    auto* o = new Owned;
    @autoreleasepool {
        o->cmd = [queue commandBufferWithUnretainedReferences];
    }
    o->cmd.label = @"bbport Metal path";
    std::scoped_lock lock{owned_mutex};
    if (auto it = owned.find(cb); it != owned.end()) {
        delete it->second; // begun again without a submit (reset)
        it->second = o;
    } else {
        owned[cb] = o;
    }
    owned_count.store(unsigned(owned.size()), std::memory_order_release);
    ++owned_generation;
}

/// Placeholder handles for the pipelines only the Metal path has (geometry shaders).
std::mutex placeholder_mutex;
std::vector<VkPipeline> placeholders;
std::atomic<uint64_t> next_placeholder{1};
VkPipeline PlaceholderPipeline() {
    const VkPipeline p = reinterpret_cast<VkPipeline>(0xBB00000000000000ull | next_placeholder++);
    std::scoped_lock lock{placeholder_mutex};
    placeholders.push_back(p);
    return p;
}
/// `forget`: the handle goes (vkDestroyPipeline).
bool IsPlaceholderPipeline(VkPipeline p, bool forget = true) {
    if ((reinterpret_cast<uint64_t>(p) >> 56) != 0xBB) {
        return false;
    }
    std::scoped_lock lock{placeholder_mutex};
    auto it = std::find(placeholders.begin(), placeholders.end(), p);
    if (it == placeholders.end()) {
        return false;
    }
    if (forget) {
        placeholders.erase(it);
    }
    return true;
}

void Unsupported(unsigned id) {
    ++MetalFrame::GetStats().unsupported;
    static std::array<std::atomic<bool>, Count> logged{};
    if (!logged[id].exchange(true)) {
        std::printf("Metal layer: %s is not implemented in the Metal path yet (skipped)\n",
                    Names[id]);
    }
}

/// Commits the Metal command buffers of the owned command buffers in `buffers`, in order, and
/// releases them; MoltenVK then gets the submit unchanged (its copies of them are empty). With
/// MoltenVK's single-queue semaphores ("Metal implicit guarantees within a single queue") order
/// on the queue is what synchronizes, and ours comes first.
template <typename Get>
void CommitOwned(uint32_t count, Get&& get) {
    if (owned_count.load(std::memory_order_acquire) == 0) {
        return;
    }
    for (uint32_t i = 0; i < count; ++i) {
        const VkCommandBuffer cb = get(i);
        Owned* o = nullptr;
        {
            std::scoped_lock lock{owned_mutex};
            auto it = owned.find(cb);
            if (it == owned.end()) {
                continue;
            }
            o = it->second;
            owned.erase(it);
            owned_count.store(unsigned(owned.size()), std::memory_order_release);
            ++owned_generation;
        }
        @autoreleasepool {
        MetalFrame::EndEncoders(*o);
        MetalFrame::ReturnScratchWhenDone(*o);
        if (!o->keep.empty()) {
            // Our temporary buffers live until the GPU is done with them.
            __block std::vector<id<MTLBuffer>> keep = std::move(o->keep);
            [o->cmd addCompletedHandler:^(id<MTLCommandBuffer>) {
              keep.clear();
            }];
        }
        [o->cmd commit];
        }
        static std::atomic<unsigned long long> committed{0};
        const auto n = ++committed;
        if (n == 1 || n % 1000 == 0) {
            std::printf("Metal layer: %llu command buffers encoded in Metal (this one: %u commands)\n",
                        n, o->commands);
        }
        delete o;
    }
}

/// One forwarding function per dispatcher member: counts, encodes Metal for an owned command
/// buffer, else calls MoltenVK.
template <unsigned I, typename F>
struct Hook;
template <unsigned I, typename R, typename... A>
struct Hook<I, R (*)(A...)> {
    static inline R (*original)(A...) = nullptr;

    static R Call(A... args) {
        if (trace) {
            calls[I].fetch_add(1, std::memory_order_relaxed);
        }
        if constexpr (I == Id_vkQueueSubmit) {
            Report();
            const auto [vkqueue, count, submits, fence] = std::tuple{args...};
            for (uint32_t s = 0; s < count; ++s) {
                CommitOwned(submits[s].commandBufferCount,
                            [&](uint32_t i) { return submits[s].pCommandBuffers[i]; });
            }
            return original(args...);
        } else if constexpr (I == Id_vkQueueSubmit2 || I == Id_vkQueueSubmit2KHR) {
            Report();
            const auto [vkqueue, count, submits, fence] = std::tuple{args...};
            for (uint32_t s = 0; s < count; ++s) {
                CommitOwned(submits[s].commandBufferInfoCount, [&](uint32_t i) {
                    return submits[s].pCommandBufferInfos[i].commandBuffer;
                });
            }
            return original(args...);
        } else if constexpr (I == Id_vkBeginCommandBuffer) {
            const R result = original(args...);
            if (frame_enabled) {
                OwnBuffer(std::get<0>(std::tuple{args...}));
            }
            return result;
        } else if constexpr (I == Id_vkEndCommandBuffer) {
            if (Owned* o = FindOwned(std::get<0>(std::tuple{args...}))) {
                @autoreleasepool {
                    MetalFrame::EndEncoders(*o);
                }
            }
            return original(args...);
        } else if constexpr (I == Id_vkCreateSampler) {
            const auto [dev, info, alloc, out] = std::tuple{args...};
            const R result = original(args...);
            if (result == VK_SUCCESS && (present_enabled || frame_enabled)) {
                @autoreleasepool {
                    MetalFrame::SamplerCreated(*out, info);
                }
            }
            return result;
        } else if constexpr (I == Id_vkCreateShaderModule) {
            const auto [dev, info, alloc, out] = std::tuple{args...};
            const R result = original(args...);
            if (result == VK_SUCCESS && frame_enabled) {
                MetalShadow::ModuleCreated(*out, info);
            }
            return result;
        } else if constexpr (I == Id_vkCreateGraphicsPipelines) {
            // Pipelines with a geometry shader exist only on the Metal path (MoltenVK has no
            // geometry stage): they get a placeholder handle MoltenVK never sees.
            const auto [dev, cache, count, infos, alloc, out] = std::tuple{args...};
            R result = VK_SUCCESS;
            for (uint32_t i = 0; i < count; ++i) {
                bool geometry = false;
                for (uint32_t st = 0; st < infos[i].stageCount; ++st) {
                    geometry |= infos[i].pStages[st].stage == VK_SHADER_STAGE_GEOMETRY_BIT;
                }
                if (geometry && frame_enabled) {
                    out[i] = PlaceholderPipeline();
                } else {
                    const R one = original(dev, cache, 1, &infos[i], alloc, &out[i]);
                    if (one != VK_SUCCESS) {
                        result = one;
                        continue;
                    }
                }
                if (frame_enabled && !geometry) {
                    MetalShadow::GraphicsCreated(&infos[i], out[i]);
                }
            }
            return result;
        } else if constexpr (I == Id_vkCreateComputePipelines) {
            const auto [dev, cache, count, infos, alloc, out] = std::tuple{args...};
            const R result = original(args...);
            if (frame_enabled) {
                for (uint32_t i = 0; i < count; ++i) {
                    @autoreleasepool {
                        MetalShadow::ComputeCreated(&infos[i], out[i]);
                    }
                }
            }
            return result;
        } else if constexpr (I == Id_vkCreateImageView) {
            const auto [dev, info, alloc, out] = std::tuple{args...};
            const R result = original(args...);
            if (result == VK_SUCCESS && (present_enabled || frame_enabled)) {
                MetalFrame::ImageViewCreated(*out, info);
            }
            return result;
        } else if constexpr (I == Id_vkSetDebugUtilsObjectNameEXT) {
            const auto info = std::get<1>(std::tuple{args...});
            if (info && info->objectType == VK_OBJECT_TYPE_PIPELINE &&
                IsPlaceholderPipeline(reinterpret_cast<VkPipeline>(info->objectHandle), false)) {
                return VK_SUCCESS; // MoltenVK does not know the placeholder
            }
            return original(args...);
        } else if constexpr (I == Id_vkGetDeviceProcAddr || I == Id_vkGetInstanceProcAddr) {
            const char* name = std::get<1>(std::tuple{args...});
            if (PFN_vkVoidFunction ours = HookedByName(name)) {
                return ours;
            }
            if (frame_enabled && name && std::strncmp(name, "vkCmd", 5) == 0) {
                // A command the layer does not cover: it would reach MoltenVK out of order.
                static std::mutex logged_mutex;
                static std::vector<std::string> logged;
                std::scoped_lock lock{logged_mutex};
                if (std::find(logged.begin(), logged.end(), name) == logged.end()) {
                    logged.emplace_back(name);
                    std::printf("Metal layer: %s requested but not covered by the Metal path\n", name);
                }
            }
            return original(args...);
        } else if constexpr (I == Id_vkUpdateDescriptorSets) {
            original(args...);
            if (frame_enabled) {
                MetalFrame::DescriptorSetsUpdated(std::get<1>(std::tuple{args...}),
                                                  std::get<2>(std::tuple{args...}));
            }
            return;
        } else if constexpr (I == Id_vkDestroySampler) {
            if (present_enabled || frame_enabled) {
                MetalFrame::SamplerDestroyed(std::get<1>(std::tuple{args...}));
            }
            return original(args...);
        } else if constexpr (I == Id_vkDestroyImageView) {
            if (present_enabled || frame_enabled) {
                MetalFrame::ImageViewDestroyed(std::get<1>(std::tuple{args...}));
            }
            return original(args...);
        } else if constexpr (I == Id_vkDestroyPipeline) {
            const VkPipeline pipeline = std::get<1>(std::tuple{args...});
            MetalShadow::Forget(pipeline);
            if (IsPlaceholderPipeline(pipeline)) {
                return;
            }
            return original(args...);
        } else if constexpr (I == Id_vkGetBufferDeviceAddress || I == Id_vkGetBufferDeviceAddressKHR) {
            const R address = original(args...);
            if (frame_enabled) {
                MetalFrame::DeviceAddressTaken(std::get<1>(std::tuple{args...})->buffer);
            }
            return address;
        } else {
            // vkCmd*: the first argument is the command buffer.
            if (Owned* o = FindOwned(std::get<0>(std::tuple{args...}))) {
                ++o->commands;
                // The recorder threads never drain an autorelease pool of their own: Metal's
                // autoreleased objects (pass descriptors, encoders) leaked ~17 KB per frame.
                @autoreleasepool {
                    Encode(*o, args...);
                }
                if constexpr (!std::is_void_v<R>) {
                    return R{};
                } else {
                    return;
                }
            }
            return original(args...);
        }
    }

    /// The Metal encoding of each command (mtl_frame.mm).
    static void Encode(Owned& o, A... args) {
        namespace F = MetalFrame;
        const auto t = std::tuple{args...};
        if constexpr (I == Id_vkCmdPipelineBarrier || I == Id_vkCmdPipelineBarrier2 ||
                      I == Id_vkCmdPipelineBarrier2KHR) {
            F::Barrier(o);
        } else if constexpr (I == Id_vkCmdBeginDebugUtilsLabelEXT ||
                             I == Id_vkCmdEndDebugUtilsLabelEXT ||
                             I == Id_vkCmdInsertDebugUtilsLabelEXT || I == Id_vkCmdSetLineWidth ||
                             I == Id_vkCmdSetPrimitiveRestartEnable ||
                             I == Id_vkCmdSetDepthBoundsTestEnable || I == Id_vkCmdSetDepthBounds ||
                             I == Id_vkCmdWriteTimestamp || I == Id_vkCmdWriteTimestamp2) {
            // Nothing in Metal (primitive restart is always on for strips; no depth bounds test).
        } else if constexpr (I == Id_vkCmdBeginRendering || I == Id_vkCmdBeginRenderingKHR) {
            F::BeginRendering(o, std::get<1>(t));
        } else if constexpr (I == Id_vkCmdEndRendering || I == Id_vkCmdEndRenderingKHR) {
            F::EndRendering(o);
        } else if constexpr (I == Id_vkCmdBindPipeline) {
            F::BindPipeline(o, std::get<1>(t), std::get<2>(t));
        } else if constexpr (I == Id_vkCmdPushDescriptorSetKHR) {
            F::PushDescriptorSet(o, std::get<1>(t), std::get<3>(t), std::get<4>(t), std::get<5>(t));
        } else if constexpr (I == Id_vkCmdBindDescriptorSets) {
            F::BindDescriptorSets(o, std::get<1>(t), std::get<3>(t), std::get<4>(t), std::get<5>(t));
        } else if constexpr (I == Id_vkCmdPushConstants) {
            F::PushConstants(o, std::get<3>(t), std::get<4>(t), std::get<5>(t));
        } else if constexpr (I == Id_vkCmdBindVertexBuffers) {
            F::BindVertexBuffers(o, std::get<1>(t), std::get<2>(t), std::get<3>(t),
                                 std::get<4>(t), nullptr);
        } else if constexpr (I == Id_vkCmdBindVertexBuffers2 || I == Id_vkCmdBindVertexBuffers2EXT) {
            F::BindVertexBuffers(o, std::get<1>(t), std::get<2>(t), std::get<3>(t),
                                 std::get<4>(t), std::get<6>(t));
        } else if constexpr (I == Id_vkCmdBindIndexBuffer) {
            F::BindIndexBuffer(o, std::get<1>(t), std::get<2>(t), std::get<3>(t));
        } else if constexpr (I == Id_vkCmdSetViewport) {
            F::SetViewports(o, std::get<2>(t), std::get<3>(t));
        } else if constexpr (I == Id_vkCmdSetViewportWithCount || I == Id_vkCmdSetViewportWithCountEXT) {
            F::SetViewports(o, std::get<1>(t), std::get<2>(t));
        } else if constexpr (I == Id_vkCmdSetScissor) {
            F::SetScissors(o, std::get<2>(t), std::get<3>(t));
        } else if constexpr (I == Id_vkCmdSetScissorWithCount || I == Id_vkCmdSetScissorWithCountEXT) {
            F::SetScissors(o, std::get<1>(t), std::get<2>(t));
        } else if constexpr (I == Id_vkCmdSetCullMode || I == Id_vkCmdSetCullModeEXT) {
            o.cull_mode = std::get<1>(t);
        } else if constexpr (I == Id_vkCmdSetFrontFace || I == Id_vkCmdSetFrontFaceEXT) {
            o.front_face = std::get<1>(t);
        } else if constexpr (I == Id_vkCmdSetDepthTestEnable || I == Id_vkCmdSetDepthTestEnableEXT) {
            o.depth_test = std::get<1>(t);
        } else if constexpr (I == Id_vkCmdSetDepthWriteEnable || I == Id_vkCmdSetDepthWriteEnableEXT) {
            o.depth_write = std::get<1>(t);
        } else if constexpr (I == Id_vkCmdSetDepthCompareOp || I == Id_vkCmdSetDepthCompareOpEXT) {
            o.depth_compare = std::get<1>(t);
        } else if constexpr (I == Id_vkCmdSetDepthBiasEnable) {
            o.depth_bias_enable = std::get<1>(t);
        } else if constexpr (I == Id_vkCmdSetDepthBias) {
            o.depth_bias[0] = std::get<1>(t);
            o.depth_bias[1] = std::get<2>(t);
            o.depth_bias[2] = std::get<3>(t);
        } else if constexpr (I == Id_vkCmdSetStencilTestEnable || I == Id_vkCmdSetStencilTestEnableEXT) {
            o.stencil_test = std::get<1>(t);
        } else if constexpr (I == Id_vkCmdSetStencilOp || I == Id_vkCmdSetStencilOpEXT) {
            const VkStencilFaceFlags faces = std::get<1>(t);
            for (auto* f : {&o.front, &o.back}) {
                if (faces & (f == &o.front ? VK_STENCIL_FACE_FRONT_BIT : VK_STENCIL_FACE_BACK_BIT)) {
                    f->fail = std::get<2>(t);
                    f->pass = std::get<3>(t);
                    f->depth_fail = std::get<4>(t);
                    f->compare = std::get<5>(t);
                }
            }
        } else if constexpr (I == Id_vkCmdSetStencilReference || I == Id_vkCmdSetStencilWriteMask ||
                             I == Id_vkCmdSetStencilCompareMask) {
            const VkStencilFaceFlags faces = std::get<1>(t);
            const uint32_t value = std::get<2>(t);
            for (auto* f : {&o.front, &o.back}) {
                if (faces & (f == &o.front ? VK_STENCIL_FACE_FRONT_BIT : VK_STENCIL_FACE_BACK_BIT)) {
                    if constexpr (I == Id_vkCmdSetStencilReference) {
                        f->reference = value;
                    } else if constexpr (I == Id_vkCmdSetStencilWriteMask) {
                        f->write_mask = value;
                    } else {
                        f->compare_mask = value;
                    }
                }
            }
        } else if constexpr (I == Id_vkCmdSetBlendConstants) {
            std::copy_n(std::get<1>(t), 4, o.blend_constants);
        } else if constexpr (I == Id_vkCmdSetRasterizerDiscardEnable) {
            o.rasterizer_discard = std::get<1>(t);
        } else if constexpr (I == Id_vkCmdDraw) {
            F::Draw(o, std::get<1>(t), std::get<2>(t), std::get<3>(t), std::get<4>(t));
        } else if constexpr (I == Id_vkCmdDrawIndexed) {
            F::DrawIndexed(o, std::get<1>(t), std::get<2>(t), std::get<3>(t), std::get<4>(t),
                           std::get<5>(t));
        } else if constexpr (I == Id_vkCmdDrawIndirect) {
            F::DrawIndirect(o, std::get<1>(t), std::get<2>(t), std::get<3>(t), std::get<4>(t), false);
        } else if constexpr (I == Id_vkCmdDrawIndexedIndirect) {
            F::DrawIndirect(o, std::get<1>(t), std::get<2>(t), std::get<3>(t), std::get<4>(t), true);
        } else if constexpr (I == Id_vkCmdDispatch) {
            F::Dispatch(o, std::get<1>(t), std::get<2>(t), std::get<3>(t));
        } else if constexpr (I == Id_vkCmdDispatchIndirect) {
            F::DispatchIndirect(o, std::get<1>(t), std::get<2>(t));
        } else if constexpr (I == Id_vkCmdCopyBuffer) {
            F::CopyBuffer(o, std::get<1>(t), std::get<2>(t), std::get<3>(t), std::get<4>(t));
        } else if constexpr (I == Id_vkCmdFillBuffer) {
            F::FillBuffer(o, std::get<1>(t), std::get<2>(t), std::get<3>(t), std::get<4>(t));
        } else if constexpr (I == Id_vkCmdUpdateBuffer) {
            F::UpdateBuffer(o, std::get<1>(t), std::get<2>(t), std::get<3>(t), std::get<4>(t));
        } else if constexpr (I == Id_vkCmdCopyImage) {
            F::CopyImage(o, std::get<1>(t), std::get<3>(t), std::get<5>(t), std::get<6>(t));
        } else if constexpr (I == Id_vkCmdCopyBufferToImage) {
            F::CopyBufferToImage(o, std::get<1>(t), std::get<2>(t), std::get<4>(t), std::get<5>(t));
        } else if constexpr (I == Id_vkCmdCopyImageToBuffer) {
            F::CopyImageToBuffer(o, std::get<1>(t), std::get<3>(t), std::get<4>(t), std::get<5>(t));
        } else if constexpr (I == Id_vkCmdClearColorImage) {
            F::ClearColorImage(o, std::get<1>(t), std::get<3>(t), std::get<4>(t), std::get<5>(t));
        } else if constexpr (I == Id_vkCmdClearDepthStencilImage) {
            F::ClearDepthStencilImage(o, std::get<1>(t), std::get<3>(t), std::get<4>(t),
                                      std::get<5>(t));
        } else if constexpr (I == Id_vkCmdBlitImage) {
            F::BlitImage(o, std::get<1>(t), std::get<3>(t), std::get<5>(t), std::get<6>(t),
                         std::get<7>(t));
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
    return present_enabled && !frame_enabled;
}

bool FrameEnabled() {
    return frame_enabled;
}

void* MetalCommandBuffer(void* command_buffer) {
    Owned* o = FindOwned(static_cast<VkCommandBuffer>(command_buffer));
    if (!o) {
        return nullptr;
    }
    MetalFrame::EndEncoders(*o);
    return (__bridge void*)o->cmd;
}

void* MtlDevice() {
    return queue ? (__bridge void*)queue.device : nullptr;
}

void* MtlQueue() {
    return (__bridge void*)queue;
}

void SetDevice(void* device, void* vkqueue) {
    if (!present_enabled && !frame_enabled) {
        return;
    }
    const VkDevice vk_device = static_cast<VkDevice>(device);
    auto export_objects = reinterpret_cast<PFN_vkExportMetalObjectsEXT>(
        VULKAN_HPP_DEFAULT_DISPATCHER.vkGetDeviceProcAddr(vk_device, "vkExportMetalObjectsEXT"));
    if (!export_objects) {
        std::printf("Metal layer: no VK_EXT_metal_objects; the Metal path stays off\n");
        present_enabled = frame_enabled = false;
        return;
    }
    VkExportMetalCommandQueueInfoEXT q{VK_STRUCTURE_TYPE_EXPORT_METAL_COMMAND_QUEUE_INFO_EXT};
    q.queue = static_cast<VkQueue>(vkqueue);
    VkExportMetalObjectsInfoEXT info{VK_STRUCTURE_TYPE_EXPORT_METAL_OBJECTS_INFO_EXT, &q};
    export_objects(vk_device, &info);
    queue = q.mtlCommandQueue;
    if (!queue) {
        std::printf("Metal layer: MoltenVK's queue could not be exported; the Metal path stays off\n");
        present_enabled = frame_enabled = false;
        return;
    }
    // The original (MoltenVK) entry: through the hooks the lookup would come back to the layer
    // (vkGetDeviceProcAddr answers with them) and recurse.
    using AddressHook = Hook<Id_vkGetBufferDeviceAddress,
                             decltype(VULKAN_HPP_DEFAULT_DISPATCHER.vkGetBufferDeviceAddress)>;
    auto get_address = AddressHook::original
                           ? AddressHook::original
                           : reinterpret_cast<PFN_vkGetBufferDeviceAddress>(
                                 VULKAN_HPP_DEFAULT_DISPATCHER.vkGetBufferDeviceAddress);
    MetalFrame::Init(vk_device, queue, export_objects, get_address);
    std::printf("Metal layer: %s encoded in Metal on %s\n",
                frame_enabled ? "every command buffer is (BB_METAL_FRAME=1)"
                              : "the present pass is (BB_METAL_PRESENT=1)",
                queue.device.name.UTF8String);
}

void Own(void* command_buffer) {
    if (!PresentEnabled()) {
        return; // frame mode owns every command buffer at vkBeginCommandBuffer
    }
    OwnBuffer(static_cast<VkCommandBuffer>(command_buffer));
}

void Install(void* dispatcher_ptr) {
    const char* env = std::getenv("BB_METAL_TRACE");
    trace = env && env[0] == '1';
    const char* present = std::getenv("BB_METAL_PRESENT");
    present_enabled = present && present[0] == '1';
    const char* frame = std::getenv("BB_METAL_FRAME");
    frame_enabled = frame && frame[0] == '1';
    if (!trace && !present_enabled && !frame_enabled) {
        return;
    }
    auto& d = *static_cast<Dispatcher*>(dispatcher_ptr);
#define BB_INSTALL(name)                                                                           \
    Hook<Id_##name, decltype(d.name)>::Install(d.name);                                            \
    hooked[Id_##name] = Hook<Id_##name, decltype(d.name)>::original                                \
                            ? reinterpret_cast<PFN_vkVoidFunction>(d.name)                         \
                            : nullptr;
    BB_LAYER_FUNCTIONS(BB_INSTALL)
#undef BB_INSTALL
    std::printf("Metal layer: installed over %u Vulkan functions\n", unsigned(Count));
}

} // namespace MetalLayer
