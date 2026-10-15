// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: the Metal shadow check (video_core/renderer_vulkan/vk_metal_shadow.h). Every pipeline the
// Vulkan path creates is built again as a Metal pipeline state from the same key and SPIR-V; the
// counts show how much of the game the Metal back end's pipelines cover. BB_METAL_SHADOW=1.

#import <Metal/Metal.h>

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>
#include "common/path_util.h"
#include "shader_recompiler/runtime_info.h"
#include "video_core/renderer_metal/mtl_device.h"
#include "video_core/renderer_metal/mtl_pso_cache.h"
#include "video_core/renderer_metal/mtl_shader.h"
#include "video_core/renderer_vulkan/vk_compute_pipeline.h"
#include "video_core/renderer_vulkan/vk_graphics_pipeline.h"
#include "video_core/renderer_vulkan/vk_metal_shadow.h"

namespace MetalShadow {
namespace {

struct ShaderEntry {
    std::once_flag once;
    std::unique_ptr<Metal::CompiledShader> compiled;
};

struct State {
    Metal::Device device;
    std::filesystem::path dir;
    Metal::ShaderCompiler compiler;
    Metal::PsoCache pso;
    std::mutex mutex;
    std::unordered_map<u64, std::shared_ptr<std::vector<u32>>> spirv; ///< by VkShaderModule
    std::unordered_map<u64, std::shared_ptr<ShaderEntry>> shaders;          ///< by VkShaderModule

    std::atomic<u32> render_ok{}, render_failed{}, compute_ok{}, compute_failed{};
    std::atomic<u32> shader_failed{}, no_spirv{};
    std::atomic<u32> skipped_tess{}, skipped_geometry{}, skipped_rect{}, skipped_quad{},
        skipped_aux_fragment{};
    std::atomic<u32> seen{};

    static std::filesystem::path Dir() {
        auto dir = Common::FS::GetUserPath(Common::FS::PathType::CacheDir) / "metal-shadow";
        std::error_code ec;
        std::filesystem::create_directories(dir, ec);
        return dir;
    }
    State() : dir{Dir()}, compiler{device, dir}, pso{device, dir} {
        std::printf("Metal shadow: building every pipeline as a Metal pipeline state too (%s, %s)\n",
                    device.Name().c_str(), dir.string().c_str());
    }

    const Metal::CompiledShader* Get(vk::ShaderModule module, u64 hash, const char* what) {
        const u64 handle = u64(VkShaderModule(module));
        std::shared_ptr<std::vector<u32>> words;
        std::shared_ptr<ShaderEntry> shader;
        {
            std::scoped_lock lock{mutex};
            auto it = spirv.find(handle);
            if (it == spirv.end()) {
                ++no_spirv;
                return nullptr;
            }
            words = it->second;
            auto& slot = shaders[handle];
            if (!slot) {
                slot = std::make_shared<ShaderEntry>();
            }
            shader = slot;
        }
        std::call_once(shader->once, [&] {
            auto result = compiler.Compile(*words, hash, what);
            if (result.status == Metal::CompileStatus::Ok) {
                shader->compiled = std::move(result.shader);
            } else {
                ++shader_failed;
            }
        });
        return shader->compiled.get();
    }

    void Report() {
        const u32 n = ++seen;
        if (n != 1 && n % 50 != 0) {
            return;
        }
        std::printf("Metal shadow: %u pipelines: render %u ok / %u failed, compute %u ok / %u "
                    "failed; shaders failed %u, no SPIR-V %u; not yet: tessellation %u, geometry "
                    "%u, rect lists %u, quad lists %u, clip-emulation fragment %u\n",
                    n, render_ok.load(), render_failed.load(), compute_ok.load(),
                    compute_failed.load(), shader_failed.load(), no_spirv.load(),
                    skipped_tess.load(), skipped_geometry.load(), skipped_rect.load(),
                    skipped_quad.load(), skipped_aux_fragment.load());
    }
};

State& Get() {
    static State state;
    return state;
}

} // namespace

bool Enabled() {
    static const bool enabled = [] {
        const char* env = std::getenv("BB_METAL_SHADOW");
        return env && env[0] == '1';
    }();
    return enabled;
}

void RegisterModule(vk::ShaderModule module, std::span<const u32> spirv) {
    auto& s = Get();
    auto words = std::make_shared<std::vector<u32>>(spirv.begin(), spirv.end());
    std::scoped_lock lock{s.mutex};
    s.spirv[u64(VkShaderModule(module))] = std::move(words);
}

void Graphics(const Vulkan::GraphicsPipelineKey& key, std::span<const vk::ShaderModule> modules,
              std::span<const bool> has_stage, std::span<const VertexInput> inputs,
              bool aux_fragment) {
    using Stage = ::Shader::SwStage;
    auto& s = Get();
    auto has = [&](Stage stage) { return has_stage[u32(stage)]; };
    if (has(Stage::TessellationControl) || has(Stage::TessellationEval)) {
        ++s.skipped_tess;
    } else if (has(Stage::Geometry)) {
        ++s.skipped_geometry;
    } else if (key.prim_type == AmdGpu::PrimitiveType::RectList) {
        ++s.skipped_rect;
    } else if (key.prim_type == AmdGpu::PrimitiveType::QuadList) {
        ++s.skipped_quad;
    } else if (aux_fragment) {
        ++s.skipped_aux_fragment;
    } else {
        const auto* vs = s.Get(modules[u32(Stage::Vertex)], key.stage_hashes[u32(Stage::Vertex)],
                               "vertex");
        const Metal::CompiledShader* fs = nullptr;
        bool ok = vs != nullptr;
        if (ok && has(Stage::Fragment)) {
            fs = s.Get(modules[u32(Stage::Fragment)], key.stage_hashes[u32(Stage::Fragment)],
                       "fragment");
            ok = fs != nullptr;
        }
        if (ok) {
            std::vector<Metal::VertexAttribute> attributes;
            for (const auto& in : inputs) {
                attributes.push_back({in.location, in.per_instance, in.divisor});
            }
            ok = s.pso.GetRenderPipeline(key, vs, fs, {.vertex_attributes = attributes}) != nullptr;
        }
        ++(ok ? s.render_ok : s.render_failed);
    }
    s.Report();
}

void Compute(const Vulkan::ComputePipelineKey& key, vk::ShaderModule module) {
    auto& s = Get();
    const auto* cs = s.Get(module, key.value, "compute");
    const bool ok = cs && s.pso.GetComputePipeline(key, cs) != nullptr;
    ++(ok ? s.compute_ok : s.compute_failed);
    s.Report();
}

} // namespace MetalShadow
