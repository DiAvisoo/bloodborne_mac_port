// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>
#include <utility>
#include <vector>
#include "video_core/renderer_metal/mtl_vk_layer.h"
#include "video_core/renderer_vulkan/vk_common.h"

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

/// One forwarding function per dispatcher member: counts, then calls the original (MoltenVK).
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
        return original(args...);
    }
    static void Install(R (*&slot)(A...)) {
        if (slot) {
            original = slot;
            slot = &Call;
        }
    }
};

} // namespace

void Install(void* dispatcher_ptr) {
    const char* env = std::getenv("BB_METAL_TRACE");
    trace = env && env[0] == '1';
    if (!trace) {
        return;
    }
    auto& d = *static_cast<Dispatcher*>(dispatcher_ptr);
#define BB_INSTALL(name) Hook<Id_##name, decltype(d.name)>::Install(d.name);
    BB_LAYER_FUNCTIONS(BB_INSTALL)
#undef BB_INSTALL
    std::printf("Metal layer: installed over %u Vulkan command functions (BB_METAL_TRACE=1)\n",
                unsigned(Count));
}

} // namespace MetalLayer
