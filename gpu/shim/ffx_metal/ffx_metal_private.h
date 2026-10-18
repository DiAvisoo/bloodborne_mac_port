// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: the FidelityFX Metal back end's state (shim/ffx_metal/ffx_metal.h), shared by its
// source files. Plain C++: Metal objects cross as void*.

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include "ffx_metal/ffx_metal.h"

namespace Metal {
struct CompiledShader;
}

namespace FfxMetal {

/// One effect context (an FSR 3 upscaler context is one).
struct EffectContext {
    bool active{};
    FfxEffect effect{};
};

/// The back end's state, at the start of the caller's scratch memory (ffx_vk.cpp's
/// BackendContext_VK); the effect contexts follow it.
struct BackendContext {
    Metal::Device* device{};
    uint32_t max_contexts{};
    uint32_t ref_count{};       ///< live effect contexts
    EffectContext* contexts{};  ///< max_contexts entries, in the scratch memory
};

inline BackendContext& Backend(FfxInterface* iface) {
    return *static_cast<BackendContext*>(iface->scratchBuffer);
}

/// A compiled FFX blob: the compute pipeline and its shader's reflection (MSL slots of the
/// blob's (set, binding) pairs, threads per threadgroup). Owned by the per-device cache.
struct Pipeline {
    void* pipeline{};                       ///< id<MTLComputePipelineState>, unretained
    const Metal::CompiledShader* shader{};
};

/// PipelineFor with the reflection; null on failure (cause in `error`).
const Pipeline* PipelineInfoFor(Metal::Device& device, const FfxShaderBlob& blob,
                                std::string* error, std::string_view name);

} // namespace FfxMetal
