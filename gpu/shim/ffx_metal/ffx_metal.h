// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: the FidelityFX SDK 1.1.4 back end on native Metal (Metal-native M6). AMD's FSR 3.1 host
// code (ffx_fsr3upscaler.cpp) talks to it through the FfxInterface callbacks, the role ffx_vk.cpp
// plays on Vulkan; AMD's SPIR-V blobs become Metal compute pipelines through the renderer's
// ShaderCompiler. Plain C++: Metal objects cross as void*.

#pragma once

#include <bit> // ffx_util.h uses std::popcount from C++20 on without including it
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <FidelityFX/host/ffx_interface.h>

namespace Metal {
class Device;
class Encoder;
} // namespace Metal

namespace FfxMetal {

/// Bytes of scratch memory GetInterface needs for `max_contexts` effect contexts.
size_t ScratchSize(uint32_t max_contexts);

/// Fills `out` with the Metal back end on `device`. `scratch` (ScratchSize bytes, owned by the
/// caller, outliving the interface) holds the back end's state.
FfxErrorCode GetInterface(FfxInterface* out, Metal::Device& device, void* scratch,
                          size_t scratch_size, uint32_t max_contexts);

/// The command list FFX dispatches record into: `encoder`, which must outlive the dispatch.
FfxCommandList CommandList(Metal::Encoder& encoder);

/// What the back end reports to FFX: shader model 6.2, wave32 only, fp16 (BB_FFX_FP16=0: off),
/// nothing else.
FfxDeviceCapabilities Capabilities(Metal::Device& device);

/// FSR 3 needs MTLGPUFamilyApple7 (texture atomics on r32uint, quad operations).
bool Supported(Metal::Device& device);

/// The compute pipeline of `blob` (id<MTLComputePipelineState>, unretained, alive as long as
/// `device`), compiled once per device and blob and then cached. Null on failure, with the cause
/// (and `name`, a label for logs and errors) in `error` when given.
void* PipelineFor(Metal::Device& device, const FfxShaderBlob& blob, std::string* error,
                  std::string_view name = "ffx");

/// Pipelines compiled by PipelineFor so far (cache misses), for tests.
uint32_t PipelineCompileCount();

} // namespace FfxMetal
