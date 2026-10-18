// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: the FidelityFX Metal back end's state (shim/ffx_metal/ffx_metal.h), shared by its
// source files. Plain C++: Metal objects cross as void*.

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>
#include "ffx_metal/ffx_metal.h"
#include "video_core/renderer_metal/mtl_object.h"

namespace Metal {
struct CompiledShader;
}

namespace FfxMetal {

struct State;

/// One effect context (an FSR 3 upscaler context is one).
struct EffectContext {
    bool active{};
    FfxEffect effect{};
};

/// The back end's state, at the start of the caller's scratch memory (ffx_vk.cpp's
/// BackendContext_VK); the effect contexts follow it.
struct BackendContext {
    Metal::Device* device{};
    State* state{};      ///< resources and queued jobs (ffx_metal_resources.mm); null while
                                ///< no context is live
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

/// One resource slot: a texture the back end created (owned) or one the caller registered for a
/// dispatch (dynamic, unowned).
struct Slot {
    bool used{};
    bool dynamic{};      ///< registered per dispatch (fpRegisterResource), dropped by Unregister
    uint32_t context{};  ///< effect context that created or registered it
    FfxResourceDescription description{};
    FfxResourceStates state{};
    Metal::Object texture;      ///< owned textures (id<MTLTexture>)
    void* handle{};             ///< id<MTLTexture>, unretained (owned or the caller's)
    uint64_t bytes{};           ///< allocated size of an owned texture
    std::vector<Metal::Object> mip_views; ///< per-mip storage views, made on first use
    Metal::Object staging;      ///< init data (id<MTLBuffer>) until the first ExecuteGpuJobs
    std::string name;
};

/// The back end's resource table (slot 0 is the null resource) and the jobs scheduled since the
/// last fpExecuteGpuJobs. Behind BackendContext::state; the scratch memory holds only a pointer so
/// that the C++ containers are released with the last effect context.
struct State {
    std::vector<Slot> resources;
    std::vector<uint32_t> free_slots;
    std::vector<uint32_t> pending_uploads; ///< owned resources with `staging`
    std::vector<FfxGpuJobDescription> jobs;
    State() : resources(1) {}
};

/// The state of `backend`, made on first use.
State& StateOf(BackendContext& backend);

/// Sets the resource and job callbacks (ffx_metal_resources.mm) on `iface`.
void InstallResourceCallbacks(FfxInterface* iface);

/// Releases the resources of effect context `context` (all of them when the last context goes:
/// the state is deleted).
void ReleaseContext(BackendContext& backend, uint32_t context);

/// The id<MTLTexture> view of one mip of resource `resource` for UAV bindings (arrayIndex =
/// mip), made on first use and kept with the resource; null for a null resource or a mip past
/// the texture's.
void* StorageView(FfxInterface& iface, FfxResourceInternal resource, uint32_t mip);

} // namespace FfxMetal
