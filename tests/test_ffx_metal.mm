// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: unit test for the FidelityFX Metal back end (Metal-native M6 Task 1): every FSR 3.1
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
#include "video_core/renderer_metal/mtl_host_kernel.h"

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
        Expect(device.ErrorCount() == 0, "no command buffer errors on the device queue");
        std::filesystem::remove_all(dir);
    }
    if (failures) {
        std::printf("FAILED: %d check(s)\n", failures);
        return 1;
    }
    return 0;
}
