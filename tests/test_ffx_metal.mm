// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: unit test for the FidelityFX Metal back end (Metal-native M6 Tasks 1-2): every FSR 3.1
// upscaler blob AMD ships (SDK 1.1.4 SPIR-V, wave32, fp16 on and off) is reached through the
// back end's FfxInterface and compiles to a Metal compute pipeline; a second request for a blob
// is served from the pipeline cache. Links libbbgpu (the ShaderCompiler); the MSL disk cache goes
// to a temporary directory (SetHostKernelCacheDir).
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <bit>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <unistd.h>
#include <vector>
#include <FidelityFX/host/ffx_fsr3upscaler.h>
#include "fsr3upscaler/ffx_fsr3upscaler_private.h"
#include "ffx_metal/ffx_metal.h"
#include "video_core/renderer_metal/mtl_device.h"
#include "ffx_metal/ffx_metal_private.h"
#include "video_core/renderer_metal/mtl_encoder.h"
#include "video_core/renderer_metal/mtl_host_kernel.h"
#include "video_core/renderer_metal/mtl_texture.h"

namespace {

int failures = 0;

void Expect(bool ok, const char* what) {
    if (!ok) {
        std::printf("FAIL: %s\n", what);
        ++failures;
    }
}

const char* PassName(FfxFsr3UpscalerPass pass) {
    switch (pass) {
    case FFX_FSR3UPSCALER_PASS_PREPARE_INPUTS:
        return "prepare_inputs";
    case FFX_FSR3UPSCALER_PASS_LUMA_PYRAMID:
        return "luma_pyramid";
    case FFX_FSR3UPSCALER_PASS_SHADING_CHANGE_PYRAMID:
        return "shading_change_pyramid";
    case FFX_FSR3UPSCALER_PASS_SHADING_CHANGE:
        return "shading_change";
    case FFX_FSR3UPSCALER_PASS_PREPARE_REACTIVITY:
        return "prepare_reactivity";
    case FFX_FSR3UPSCALER_PASS_LUMA_INSTABILITY:
        return "luma_instability";
    case FFX_FSR3UPSCALER_PASS_ACCUMULATE:
        return "accumulate";
    case FFX_FSR3UPSCALER_PASS_ACCUMULATE_SHARPEN:
        return "accumulate_sharpen";
    case FFX_FSR3UPSCALER_PASS_RCAS:
        return "rcas";
    case FFX_FSR3UPSCALER_PASS_GENERATE_REACTIVE:
        return "generate_reactive";
    default:
        return "?";
    }
}

// The passes FSR 3.1 dispatches (DEBUG_VIEW is a debugging overlay; TCR_AUTOGENERATE is
// deprecated and has no blobs).
constexpr FfxFsr3UpscalerPass kPasses[] = {
    FFX_FSR3UPSCALER_PASS_PREPARE_INPUTS,     FFX_FSR3UPSCALER_PASS_LUMA_PYRAMID,
    FFX_FSR3UPSCALER_PASS_SHADING_CHANGE_PYRAMID, FFX_FSR3UPSCALER_PASS_SHADING_CHANGE,
    FFX_FSR3UPSCALER_PASS_PREPARE_REACTIVITY, FFX_FSR3UPSCALER_PASS_LUMA_INSTABILITY,
    FFX_FSR3UPSCALER_PASS_ACCUMULATE,         FFX_FSR3UPSCALER_PASS_ACCUMULATE_SHARPEN,
    FFX_FSR3UPSCALER_PASS_RCAS,               FFX_FSR3UPSCALER_PASS_GENERATE_REACTIVE,
};

// The six key bits of the upscaler's permutations (LANCZOS_TYPE .. ENABLE_SHARPENING).
constexpr uint32_t kKeyBits = 6;

void TestAllFsr3BlobsCompile(Metal::Device& device) {
    Expect(FfxMetal::Supported(device), "FSR 3 supported (Apple7+)");
    const FfxDeviceCapabilities caps = FfxMetal::Capabilities(device);
    Expect(caps.waveLaneCountMin == 32 && caps.waveLaneCountMax == 32, "wave32 only");
    Expect(caps.maximumSupportedShaderModel == FFX_SHADER_MODEL_6_2, "shader model 6.2");

    std::vector<uint8_t> scratch(FfxMetal::ScratchSize(1));
    FfxInterface iface{};
    Expect(FfxMetal::GetInterface(&iface, device, scratch.data(), scratch.size(), 1) == FFX_OK,
           "GetInterface");
    Expect(iface.fpGetPermutationBlobByIndex != nullptr, "fpGetPermutationBlobByIndex set");
    if (!iface.fpGetPermutationBlobByIndex) {
        return;
    }

    struct Seen {
        FfxShaderBlob blob;
        std::string name;
    };
    std::vector<Seen> blobs;
    for (const uint32_t fp16 : {0u, uint32_t(FSR3UPSCALER_SHADER_PERMUTATION_ALLOW_FP16)}) {
        for (uint32_t key = 0; key < (1u << kKeyBits); ++key) {
            for (const FfxFsr3UpscalerPass pass : kPasses) {
                const uint32_t options = key | fp16;
                FfxShaderBlob blob{};
                const FfxErrorCode code = iface.fpGetPermutationBlobByIndex(
                    FFX_EFFECT_FSR3UPSCALER, pass, FFX_BIND_COMPUTE_SHADER_STAGE, options, &blob);
                const std::string name = std::string("ffx ") + PassName(pass) +
                                         (fp16 ? " fp16" : "") + " key " + std::to_string(key);
                if (code != FFX_OK || !blob.data || blob.size < 4) {
                    std::printf("FAIL: %s: no blob (code 0x%x)\n", name.c_str(), unsigned(code));
                    ++failures;
                    continue;
                }
                uint32_t magic;
                std::memcpy(&magic, blob.data, sizeof(magic));
                if (magic != 0x07230203u) {
                    std::printf("FAIL: %s: not SPIR-V (0x%08x)\n", name.c_str(), magic);
                    ++failures;
                    continue;
                }
                std::string error;
                void* pipeline = FfxMetal::PipelineFor(device, blob, &error, name);
                if (!pipeline || !error.empty()) {
                    std::printf("FAIL: %s: %s\n", name.c_str(),
                                error.empty() ? "no pipeline" : error.c_str());
                    ++failures;
                    continue;
                }
                bool known = false;
                for (const Seen& seen : blobs) {
                    known |= seen.blob.data == blob.data;
                }
                if (!known) {
                    blobs.push_back({blob, name});
                }
            }
        }
    }

    // Every blob again: served from the cache, no compilation.
    const uint32_t compiled = FfxMetal::PipelineCompileCount();
    bool cached = true;
    for (const Seen& seen : blobs) {
        std::string error;
        cached &= FfxMetal::PipelineFor(device, seen.blob, &error, seen.name) != nullptr &&
                  error.empty();
    }
    Expect(cached, "second request returns a pipeline");
    Expect(FfxMetal::PipelineCompileCount() == compiled, "second request compiles nothing");
    Expect(compiled == blobs.size(), "one compilation per distinct blob");
    if (failures == 0) {
        std::printf("PASS AllFsr3BlobsCompile (%u pipelines)\n", compiled);
    }
}

// --- Task 2: resources, clears and copies ---

id<MTLBuffer> SharedBuffer(Metal::Device& device, NSUInteger length) {
    id<MTLBuffer> buffer = [(__bridge id<MTLDevice>)device.NativeDevice()
        newBufferWithLength:length
                    options:MTLResourceStorageModeShared];
    std::memset(buffer.contents, 0, length);
    return buffer;
}

Metal::BufferRef Ref(id<MTLBuffer> buffer) {
    return {(__bridge void*)buffer, 0};
}

void TestPixelFormatTable() {
    // The formats ffx_fsr3upscaler.cpp's resource list (and the shared ones) use.
    const FfxSurfaceFormat formats[] = {
        FFX_SURFACE_FORMAT_R16G16B16A16_FLOAT, FFX_SURFACE_FORMAT_R16G16_FLOAT,
        FFX_SURFACE_FORMAT_R16_FLOAT,          FFX_SURFACE_FORMAT_R32_FLOAT,
        FFX_SURFACE_FORMAT_R32_UINT,           FFX_SURFACE_FORMAT_R8_UNORM,
        FFX_SURFACE_FORMAT_R16_UNORM,          FFX_SURFACE_FORMAT_R16_SNORM,
        FFX_SURFACE_FORMAT_R32G32_FLOAT,       FFX_SURFACE_FORMAT_R8G8_UNORM,
        FFX_SURFACE_FORMAT_R8G8B8A8_UNORM,     FFX_SURFACE_FORMAT_R32G32B32A32_FLOAT,
    };
    bool all = true;
    for (const FfxSurfaceFormat format : formats) {
        all &= FfxMetal::PixelFormat(format) != uint32_t(MTLPixelFormatInvalid);
    }
    Expect(all, "every FSR 3 surface format has a Metal pixel format");
    Expect(FfxMetal::PixelFormat(FFX_SURFACE_FORMAT_R16G16_FLOAT) ==
               uint32_t(MTLPixelFormatRG16Float),
           "RG16F maps to RG16Float");
    Expect(FfxMetal::PixelFormat(FFX_SURFACE_FORMAT_UNKNOWN) == uint32_t(MTLPixelFormatInvalid),
           "unknown format is invalid");
    Expect(FfxMetal::PixelFormat(FFX_SURFACE_FORMAT_R16G16_TYPELESS) ==
               uint32_t(MTLPixelFormatInvalid),
           "typeless format is invalid");
    if (failures == 0) {
        std::printf("PASS PixelFormatTable\n");
    }
}

struct Backend {
    std::vector<uint8_t> scratch;
    FfxInterface iface{};
    FfxUInt32 context{};
    bool ok{};
    explicit Backend(Metal::Device& device) : scratch(FfxMetal::ScratchSize(1)) {
        ok = FfxMetal::GetInterface(&iface, device, scratch.data(), scratch.size(), 1) == FFX_OK &&
             iface.fpCreateBackendContext(&iface, FFX_EFFECT_FSR3UPSCALER, nullptr, &context) ==
                 FFX_OK;
    }
};

FfxCreateResourceDescription Describe(FfxSurfaceFormat format, uint32_t width, uint32_t height,
                                      uint32_t mips, FfxResourceInitData init) {
    FfxCreateResourceDescription desc{};
    desc.heapType = FFX_HEAP_TYPE_DEFAULT;
    desc.resourceDescription = {FFX_RESOURCE_TYPE_TEXTURE2D, format, width, height, 1, mips,
                                FFX_RESOURCE_FLAGS_NONE,
                                FfxResourceUsage(FFX_RESOURCE_USAGE_RENDERTARGET |
                                                 FFX_RESOURCE_USAGE_UAV)};
    desc.initialState = FFX_RESOURCE_STATE_COMPUTE_READ;
    desc.name = L"test";
    desc.initData = init;
    return desc;
}

// Reads `level` of `texture` (RG16F) back as floats.
std::vector<float> ReadRg16f(Metal::Device& device, void* texture, uint32_t level, uint32_t width,
                             uint32_t height) {
    Metal::Encoder encoder(device);
    id<MTLBuffer> readback = SharedBuffer(device, size_t(width) * height * 4);
    encoder.CopyTextureToBuffer(texture, level, 0, {}, {width, height, 1}, Ref(readback),
                                width * 4, width * height * 4);
    device.Wait(encoder.Submit());
    std::vector<float> out(size_t(width) * height * 2);
    const auto* halves = static_cast<const _Float16*>(readback.contents);
    for (size_t i = 0; i < out.size(); ++i) {
        out[i] = float(halves[i]);
    }
    return out;
}

void TestCreateClearCopy(Metal::Device& device) {
    constexpr uint32_t kW = 64, kH = 32, kMips = 3;
    const uint64_t start = FfxMetal::LiveTextureBytes();
    std::vector<float> ramp(kW * kH * 2);
    std::vector<_Float16> ramp_half(ramp.size());
    for (uint32_t i = 0; i < kW * kH; ++i) {
        ramp[i * 2] = float(i) / 2048.0f;
        ramp[i * 2 + 1] = float(2047 - i) / 2048.0f;
        ramp_half[i * 2] = _Float16(ramp[i * 2]);
        ramp_half[i * 2 + 1] = _Float16(ramp[i * 2 + 1]);
    }
    {
        Backend backend(device);
        Expect(backend.ok, "backend context created");
        FfxInterface& iface = backend.iface;

        FfxResourceInternal a{}, b{};
        const FfxCreateResourceDescription desc_a = Describe(
            FFX_SURFACE_FORMAT_R16G16_FLOAT, kW, kH, kMips,
            FfxResourceInitData::FfxResourceInitBuffer(ramp_half.size() * 2, ramp_half.data()));
        const FfxCreateResourceDescription desc_b =
            Describe(FFX_SURFACE_FORMAT_R16G16_FLOAT, kW, kH, kMips,
                     {FFX_RESOURCE_INIT_DATA_TYPE_UNINITIALIZED});
        Expect(iface.fpCreateResource(&iface, &desc_a, backend.context, &a) == FFX_OK, "create A");
        Expect(iface.fpCreateResource(&iface, &desc_b, backend.context, &b) == FFX_OK, "create B");
        Expect(FfxMetal::LiveTextureBytes() > start, "textures count towards LiveTextureBytes");

        const FfxResourceDescription got = iface.fpGetResourceDescription(&iface, a);
        Expect(got.width == kW && got.height == kH && got.mipCount == kMips &&
                   got.format == FFX_SURFACE_FORMAT_R16G16_FLOAT,
               "description of A");
        const FfxResource handle = iface.fpGetResource(&iface, a);
        Expect(handle.resource != nullptr, "A has a texture");
        id<MTLTexture> texture_a = (__bridge id<MTLTexture>)handle.resource;
        Expect(texture_a.mipmapLevelCount == kMips && texture_a.width == kW, "A texture shape");

        // The SDK's clear and copy jobs cover the whole resource (no mip selector): copy A into B
        // (all mips), then clear A (all mips), so A's mip 1 shows the clear and B's mip 0 the
        // ramp (A's init data) that the copy took before it.
        FfxGpuJobDescription copy{};
        copy.jobType = FFX_GPU_JOB_COPY;
        copy.copyJobDescriptor.src = a;
        copy.copyJobDescriptor.dst = b;
        FfxGpuJobDescription clear{};
        clear.jobType = FFX_GPU_JOB_CLEAR_FLOAT;
        clear.clearJobDescriptor = {{0.25f, 0.5f, 0.0f, 0.0f}, a};
        FfxGpuJobDescription barrier{};
        barrier.jobType = FFX_GPU_JOB_BARRIER;
        FfxGpuJobDescription discard{};
        discard.jobType = FFX_GPU_JOB_DISCARD;
        discard.discardJobDescriptor.target = b;
        Expect(iface.fpScheduleGpuJob(&iface, &copy) == FFX_OK, "schedule copy");
        Expect(iface.fpScheduleGpuJob(&iface, &barrier) == FFX_OK, "schedule barrier");
        Expect(iface.fpScheduleGpuJob(&iface, &clear) == FFX_OK, "schedule clear");
        Expect(iface.fpScheduleGpuJob(&iface, &discard) == FFX_OK, "schedule discard");
        {
            Metal::Encoder encoder(device);
            Expect(iface.fpExecuteGpuJobs(&iface, FfxMetal::CommandList(encoder),
                                          backend.context) == FFX_OK,
                   "execute jobs");
            device.Wait(encoder.Submit());
        }

        const std::vector<float> copied =
            ReadRg16f(device, iface.fpGetResource(&iface, b).resource, 0, kW, kH);
        bool ramp_ok = true;
        for (size_t i = 0; i < ramp.size(); ++i) {
            ramp_ok &= copied[i] == ramp[i];
        }
        Expect(ramp_ok, "B mip 0 equals the init ramp (upload and copy)");
        for (uint32_t mip = 0; mip < kMips; ++mip) {
            const uint32_t w = kW >> mip, h = kH >> mip;
            const std::vector<float> cleared = ReadRg16f(device, handle.resource, mip, w, h);
            bool clear_ok = true;
            for (size_t i = 0; i < size_t(w) * h; ++i) {
                clear_ok &= cleared[i * 2] == 0.25f && cleared[i * 2 + 1] == 0.5f;
            }
            Expect(clear_ok, mip == 1 ? "A mip 1 equals the clear colour" : "A mips cleared");
        }

        // Per-mip storage views (for UAV bindings).
        void* view = FfxMetal::StorageView(iface, a, 1);
        Expect(view != nullptr, "storage view of mip 1");
        if (view) {
            id<MTLTexture> mip_view = (__bridge id<MTLTexture>)view;
            Expect(mip_view.mipmapLevelCount == 1 && mip_view.width == kW / 2,
                   "storage view is the one mip");
            Expect(FfxMetal::StorageView(iface, a, 1) == view, "storage view is cached");
        }
        Expect(FfxMetal::StorageView(iface, a, kMips) == nullptr, "no storage view past the mips");

        Expect(iface.fpDestroyResource(&iface, a, backend.context) == FFX_OK, "destroy A");
        Expect(iface.fpDestroyResource(&iface, b, backend.context) == FFX_OK, "destroy B");
        Expect(FfxMetal::LiveTextureBytes() == start, "LiveTextureBytes back to its start");

        // A resource left alive goes with its context.
        FfxResourceInternal c{};
        Expect(iface.fpCreateResource(&iface, &desc_b, backend.context, &c) == FFX_OK, "create C");
        Expect(FfxMetal::LiveTextureBytes() > start, "C is live");
        Expect(iface.fpDestroyBackendContext(&iface, backend.context) == FFX_OK,
               "destroy the backend context");
        Expect(FfxMetal::LiveTextureBytes() == start, "destroying the context frees C");
    }
    device.Finish();
    if (failures == 0) {
        std::printf("PASS CreateClearCopy\n");
    }
}

void TestRegisterNull(Metal::Device& device) {
    Backend backend(device);
    FfxInterface& iface = backend.iface;
    FfxResourceInternal null_resource{-1};
    const FfxResource none = FfxMetal::Resource(nullptr, FFX_RESOURCE_USAGE_READ_ONLY, L"none");
    Expect(none.resource == nullptr, "Resource(nullptr) is a null FfxResource");
    Expect(iface.fpRegisterResource(&iface, &none, backend.context, &null_resource) == FFX_OK,
           "registering a null resource is OK");
    Expect(iface.fpGetResourceDescription(&iface, null_resource).width == 0,
           "a null resource has width 0");
    Expect(iface.fpGetResource(&iface, null_resource).resource == nullptr, "and no texture");

    // A real texture registered per dispatch and dropped by fpUnregisterResources.
    Metal::TextureDesc desc;
    desc.pixel_format = uint32_t(MTLPixelFormatRG16Float);
    desc.width = 16;
    desc.height = 8;
    desc.levels = 2;
    desc.render_target = desc.shader_write = true;
    Metal::Object texture = Metal::CreateTexture(device, desc, "registered", nullptr);
    const FfxResource described =
        FfxMetal::Resource(texture.Get(), FfxResourceUsage(FFX_RESOURCE_USAGE_UAV), L"registered");
    FfxResourceInternal registered{};
    Expect(iface.fpRegisterResource(&iface, &described, backend.context, &registered) == FFX_OK &&
               registered.internalIndex != 0,
           "register a texture");
    const FfxResourceDescription got = iface.fpGetResourceDescription(&iface, registered);
    Expect(got.width == 16 && got.height == 8 && got.mipCount == 2 &&
               got.format == FFX_SURFACE_FORMAT_R16G16_FLOAT,
           "registered description comes from the texture");
    Expect(iface.fpGetResource(&iface, registered).resource == texture.Get(),
           "registered texture is the caller's");
    // The dispatch pattern: jobs using the registered texture's views are encoded, the resources
    // are unregistered (freeing the views) and only then the command buffer is submitted.
    {
        FfxResourceInternal target{};
        const FfxCreateResourceDescription desc_t = Describe(
            FFX_SURFACE_FORMAT_R16G16_FLOAT, 16, 8, 2, {FFX_RESOURCE_INIT_DATA_TYPE_UNINITIALIZED});
        Expect(iface.fpCreateResource(&iface, &desc_t, backend.context, &target) == FFX_OK,
               "create copy target");
        Metal::Encoder encoder(device);
        Expect(FfxMetal::StorageView(iface, registered, 1) != nullptr, "view of the registered mip");
        FfxGpuJobDescription clear{};
        clear.jobType = FFX_GPU_JOB_CLEAR_FLOAT;
        clear.clearJobDescriptor = {{0.25f, 0.5f, 0.0f, 0.0f}, registered};
        FfxGpuJobDescription copy{};
        copy.jobType = FFX_GPU_JOB_COPY;
        copy.copyJobDescriptor.src = registered;
        copy.copyJobDescriptor.dst = target;
        iface.fpScheduleGpuJob(&iface, &clear);
        iface.fpScheduleGpuJob(&iface, &copy);
        Expect(iface.fpExecuteGpuJobs(&iface, FfxMetal::CommandList(encoder), backend.context) ==
                   FFX_OK,
               "execute on registered");
        Expect(iface.fpUnregisterResources(&iface, FfxMetal::CommandList(encoder),
                                           backend.context) == FFX_OK,
               "unregister before submit");
        device.Wait(encoder.Submit());
        const std::vector<float> out =
            ReadRg16f(device, iface.fpGetResource(&iface, target).resource, 1, 8, 4);
        Expect(out[0] == 0.25f && out[1] == 0.5f, "copy of the registered texture survived");
        iface.fpDestroyResource(&iface, target, backend.context);
        // re-register for the check below
        Expect(iface.fpRegisterResource(&iface, &described, backend.context, &registered) == FFX_OK,
               "register again");
    }
    {
        Metal::Encoder encoder(device);
        Expect(iface.fpUnregisterResources(&iface, FfxMetal::CommandList(encoder),
                                           backend.context) == FFX_OK,
               "unregister");
        device.Wait(encoder.Submit());
    }
    Expect(iface.fpGetResourceDescription(&iface, registered).width == 0,
           "unregistered resource is gone");
    Expect(iface.fpDestroyBackendContext(&iface, backend.context) == FFX_OK, "destroy context");
    if (failures == 0) {
        std::printf("PASS RegisterNull\n");
    }
}

} // namespace

int main() {
    @autoreleasepool {
        std::string dir_template =
            (std::filesystem::temp_directory_path() / "bb-ffx-metal-test.XXXXXX").string();
        const char* dir = mkdtemp(dir_template.data());
        if (!dir) {
            std::printf("FAIL: no temporary directory\n");
            return 1;
        }
        Metal::SetHostKernelCacheDir(dir);
        Metal::Device device;
        TestAllFsr3BlobsCompile(device);
        TestPixelFormatTable();
        TestCreateClearCopy(device);
        TestRegisterNull(device);
        Expect(device.ErrorCount() == 0, "no command buffer errors on the device queue");
        std::filesystem::remove_all(dir);
    }
    if (failures) {
        std::printf("FAILED: %d check(s)\n", failures);
        return 1;
    }
    return 0;
}
