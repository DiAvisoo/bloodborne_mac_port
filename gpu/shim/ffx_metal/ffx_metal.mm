// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: the FidelityFX SDK 1.1.4 back end on native Metal (Metal-native M6): the FfxInterface
// callbacks (ffx_vk.cpp's role) and the pipeline cache of AMD's SPIR-V blobs.

#import <Metal/Metal.h>

#include <atomic>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <new>
#include <unordered_map>
#include <vector>
#include <xxhash.h>
#include "common/logging/log.h"
#include "ffx_metal/ffx_metal_private.h"
#include "ffx_shader_blobs.h" // AMD's blob accessor (ffx_fsr3_host)
#include "video_core/renderer_metal/mtl_device.h"
#include "video_core/renderer_metal/mtl_host_kernel.h"
#include "video_core/renderer_metal/mtl_object.h"
#include "video_core/renderer_metal/mtl_shader.h"


namespace FfxMetal {

namespace {

size_t ContextsOffset() {
    return (sizeof(BackendContext) + alignof(EffectContext) - 1) & ~(alignof(EffectContext) - 1);
}

// --- pipeline cache (decision 6: per device, keyed by the blob's data pointer) ---

struct CacheEntry {
    Metal::Object pipeline;
    std::unique_ptr<Metal::CompiledShader> shader;
    Pipeline info;
};

std::mutex cache_mutex;
std::unordered_map<Metal::Device*, std::unordered_map<const uint8_t*, std::unique_ptr<CacheEntry>>>
    cache;
std::atomic<uint32_t> compile_count{0};

// --- FfxInterface callbacks ---

FfxVersionNumber GetSDKVersion(FfxInterface*) {
    return FFX_SDK_MAKE_VERSION(FFX_SDK_VERSION_MAJOR, FFX_SDK_VERSION_MINOR,
                                FFX_SDK_VERSION_PATCH);
}

FfxErrorCode GetDeviceCapabilities(FfxInterface* iface, FfxDeviceCapabilities* out) {
    if (!iface || !out) {
        return FFX_ERROR_INVALID_POINTER;
    }
    *out = Capabilities(*Backend(iface).device);
    return FFX_OK;
}

FfxErrorCode CreateBackendContext(FfxInterface* iface, FfxEffect effect, FfxEffectBindlessConfig*,
                                  FfxUInt32* effect_context_id) {
    if (!iface || !effect_context_id) {
        return FFX_ERROR_INVALID_POINTER;
    }
    BackendContext& backend = Backend(iface);
    for (uint32_t i = 0; i < backend.max_contexts; ++i) {
        EffectContext& context = backend.contexts[i];
        if (!context.active) {
            context = EffectContext{.active = true, .effect = effect};
            ++backend.ref_count;
            *effect_context_id = i;
            return FFX_OK;
        }
    }
    return FFX_ERROR_OUT_OF_MEMORY;
}

FfxErrorCode DestroyBackendContext(FfxInterface* iface, FfxUInt32 effect_context_id) {
    if (!iface) {
        return FFX_ERROR_INVALID_POINTER;
    }
    BackendContext& backend = Backend(iface);
    if (effect_context_id >= backend.max_contexts || !backend.contexts[effect_context_id].active) {
        return FFX_ERROR_INVALID_ARGUMENT;
    }
    backend.contexts[effect_context_id] = EffectContext{};
    --backend.ref_count;
    return FFX_OK;
}

// Resources (Task 2) and pipelines / jobs (Task 3) are not there yet.
FfxErrorCode GetEffectGpuMemoryUsage(FfxInterface*, FfxUInt32, FfxEffectMemoryUsage*) {
    return FFX_ERROR_BACKEND_API_ERROR;
}
FfxErrorCode CreateResource(FfxInterface*, const FfxCreateResourceDescription*, FfxUInt32,
                            FfxResourceInternal*) {
    return FFX_ERROR_BACKEND_API_ERROR;
}
FfxErrorCode RegisterResource(FfxInterface*, const FfxResource*, FfxUInt32, FfxResourceInternal*) {
    return FFX_ERROR_BACKEND_API_ERROR;
}
FfxResource GetResource(FfxInterface*, FfxResourceInternal) {
    return FfxResource{};
}
FfxErrorCode UnregisterResources(FfxInterface*, FfxCommandList, FfxUInt32) {
    return FFX_ERROR_BACKEND_API_ERROR;
}
FfxErrorCode RegisterStaticResource(FfxInterface*, const FfxStaticResourceDescription*,
                                    FfxUInt32) {
    return FFX_ERROR_BACKEND_API_ERROR;
}
FfxResourceDescription GetResourceDescription(FfxInterface*, FfxResourceInternal) {
    return FfxResourceDescription{};
}
FfxErrorCode DestroyResource(FfxInterface*, FfxResourceInternal, FfxUInt32) {
    return FFX_ERROR_BACKEND_API_ERROR;
}
FfxErrorCode MapResource(FfxInterface*, FfxResourceInternal, void**) {
    return FFX_ERROR_BACKEND_API_ERROR;
}
FfxErrorCode UnmapResource(FfxInterface*, FfxResourceInternal) {
    return FFX_ERROR_BACKEND_API_ERROR;
}
FfxErrorCode StageConstantBufferData(FfxInterface*, void*, FfxUInt32, FfxConstantBuffer*) {
    return FFX_ERROR_BACKEND_API_ERROR;
}
FfxErrorCode CreatePipeline(FfxInterface*, FfxEffect, FfxPass, uint32_t,
                            const FfxPipelineDescription*, FfxUInt32, FfxPipelineState*) {
    return FFX_ERROR_BACKEND_API_ERROR;
}
FfxErrorCode DestroyPipeline(FfxInterface*, FfxPipelineState*, FfxUInt32) {
    return FFX_ERROR_BACKEND_API_ERROR;
}
FfxErrorCode ScheduleGpuJob(FfxInterface*, const FfxGpuJobDescription*) {
    return FFX_ERROR_BACKEND_API_ERROR;
}
FfxErrorCode ExecuteGpuJobs(FfxInterface*, FfxCommandList, FfxUInt32) {
    return FFX_ERROR_BACKEND_API_ERROR;
}

} // namespace

size_t ScratchSize(uint32_t max_contexts) {
    return ContextsOffset() + size_t(max_contexts) * sizeof(EffectContext);
}

FfxErrorCode GetInterface(FfxInterface* out, Metal::Device& device, void* scratch,
                          size_t scratch_size, uint32_t max_contexts) {
    if (!out || !scratch) {
        return FFX_ERROR_INVALID_POINTER;
    }
    if (max_contexts == 0 || scratch_size < ScratchSize(max_contexts)) {
        return FFX_ERROR_INSUFFICIENT_MEMORY;
    }
    if (reinterpret_cast<uintptr_t>(scratch) % alignof(BackendContext) != 0) {
        return FFX_ERROR_INVALID_ALIGNMENT;
    }
    *out = FfxInterface{};
    out->fpGetSDKVersion = GetSDKVersion;
    out->fpGetEffectGpuMemoryUsage = GetEffectGpuMemoryUsage;
    out->fpCreateBackendContext = CreateBackendContext;
    out->fpGetDeviceCapabilities = GetDeviceCapabilities;
    out->fpDestroyBackendContext = DestroyBackendContext;
    out->fpCreateResource = CreateResource;
    out->fpRegisterResource = RegisterResource;
    out->fpGetResource = GetResource;
    out->fpUnregisterResources = UnregisterResources;
    out->fpRegisterStaticResource = RegisterStaticResource;
    out->fpGetResourceDescription = GetResourceDescription;
    out->fpDestroyResource = DestroyResource;
    out->fpMapResource = MapResource;
    out->fpUnmapResource = UnmapResource;
    out->fpStageConstantBufferDataFunc = StageConstantBufferData;
    out->fpCreatePipeline = CreatePipeline;
    out->fpDestroyPipeline = DestroyPipeline;
    out->fpScheduleGpuJob = ScheduleGpuJob;
    out->fpExecuteGpuJobs = ExecuteGpuJobs;
    out->fpGetPermutationBlobByIndex = ffxGetPermutationBlobByIndex;
    // No breadcrumbs, frame generation swap chain or custom constant buffer allocator: FSR 3's
    // upscaler does not call them.
    out->scratchBuffer = scratch;
    out->scratchBufferSize = scratch_size;
    out->device = &device;

    auto* backend = new (scratch) BackendContext{};
    backend->device = &device;
    backend->max_contexts = max_contexts;
    backend->contexts = reinterpret_cast<EffectContext*>(static_cast<uint8_t*>(scratch) +
                                                         ContextsOffset());
    for (uint32_t i = 0; i < max_contexts; ++i) {
        new (&backend->contexts[i]) EffectContext{};
    }
    return FFX_OK;
}

FfxCommandList CommandList(Metal::Encoder& encoder) {
    return reinterpret_cast<FfxCommandList>(&encoder);
}

FfxDeviceCapabilities Capabilities(Metal::Device&) {
    FfxDeviceCapabilities caps{};
    caps.maximumSupportedShaderModel = FFX_SHADER_MODEL_6_2;
    // Apple GPUs run 32-wide SIMD groups: no wave64 blobs.
    caps.waveLaneCountMin = 32;
    caps.waveLaneCountMax = 32;
    const char* fp16 = std::getenv("BB_FFX_FP16");
    caps.fp16Supported = !(fp16 && std::strcmp(fp16, "0") == 0);
    return caps;
}

bool Supported(Metal::Device& device) {
    id<MTLDevice> mtl_device = (__bridge id<MTLDevice>)device.NativeDevice();
    return mtl_device && [mtl_device supportsFamily:MTLGPUFamilyApple7];
}

const Pipeline* PipelineInfoFor(Metal::Device& device, const FfxShaderBlob& blob,
                                std::string* error, std::string_view name) {
    auto fail = [&](const std::string& cause) -> const Pipeline* {
        LOG_ERROR(Render, "FidelityFX blob {}: {}", name, cause);
        if (error) {
            *error = fmt::format("{}: {}", name, cause);
        }
        return nullptr;
    };
    if (!blob.data || blob.size < 4 || blob.size % 4 != 0) {
        return fail("no SPIR-V");
    }
    // One lock over lookup and compilation: FFX creates its pipelines from one thread, and a
    // second request for a blob being compiled must not compile it again.
    std::scoped_lock lock{cache_mutex};
    auto& entries = cache[&device];
    if (const auto it = entries.find(blob.data); it != entries.end()) {
        return &it->second->info;
    }

    // The blobs are byte arrays: copied for alignment.
    std::vector<u32> spirv(blob.size / 4);
    std::memcpy(spirv.data(), blob.data, blob.size);
    const u64 hash = XXH3_64bits(blob.data, blob.size);
    Metal::CompileResult result = Metal::HostKernelCompiler(device).Compile(spirv, hash, name);
    if (result.status != Metal::CompileStatus::Ok || !result.shader) {
        return fail(result.error.empty() ? "SPIR-V -> MSL failed" : result.error);
    }
    if (result.shader->stage != Metal::ShaderStage::Compute) {
        return fail("not a compute shader");
    }

    id<MTLDevice> mtl_device = (__bridge id<MTLDevice>)device.NativeDevice();
    id<MTLFunction> function = (__bridge id<MTLFunction>)result.shader->function;
    NSError* ns_error = nil;
    id<MTLComputePipelineState> pipeline =
        [mtl_device newComputePipelineStateWithFunction:function error:&ns_error];
    if (!pipeline) {
        return fail(fmt::format("no compute pipeline state: {}",
                                ns_error ? ns_error.localizedDescription.UTF8String
                                         : "unknown error"));
    }
    compile_count.fetch_add(1, std::memory_order_relaxed);

    auto entry = std::make_unique<CacheEntry>();
    entry->pipeline = Metal::Object::Adopt((__bridge_retained void*)pipeline);
    entry->shader = std::move(result.shader);
    entry->info = Pipeline{.pipeline = entry->pipeline.Get(), .shader = entry->shader.get()};
    const Pipeline* info = &entry->info;
    entries.emplace(blob.data, std::move(entry));
    return info;
}

void* PipelineFor(Metal::Device& device, const FfxShaderBlob& blob, std::string* error,
                  std::string_view name) {
    const Pipeline* info = PipelineInfoFor(device, blob, error, name);
    return info ? info->pipeline : nullptr;
}

uint32_t PipelineCompileCount() {
    return compile_count.load(std::memory_order_relaxed);
}

} // namespace FfxMetal
