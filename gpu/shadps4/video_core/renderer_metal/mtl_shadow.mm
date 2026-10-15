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
#include "video_core/renderer_metal/mtl_spirv_rectlist.h"
#include "video_core/renderer_vulkan/vk_compute_pipeline.h"
#include "video_core/renderer_vulkan/vk_graphics_pipeline.h"
#include "video_core/renderer_vulkan/vk_metal_shadow.h"

namespace MetalShadow {
namespace {

struct ShaderEntry {
    std::once_flag once;
    std::unique_ptr<Metal::CompiledShader> compiled;
    std::once_flag rect_once; ///< the RectList rewrite of a vertex shader (mtl_spirv_rectlist.h)
    std::unique_ptr<Metal::CompiledShader> rect;
    std::mutex variant_mutex; ///< tessellation variants (Metal::CompileVariant)
    std::vector<std::pair<Metal::CompileVariant, std::unique_ptr<Metal::CompiledShader>>> variants;
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
    std::atomic<u32> rect_ok{}, rect_failed{};
    std::atomic<u32> tess_ok{}, tess_failed{}, tess_unsupported{};
    std::atomic<u32> skipped_geometry{}, skipped_quad{},
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

    const Metal::CompiledShader* Get(vk::ShaderModule module, u64 hash, const char* what,
                                     bool rect_list = false) {
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
        if (rect_list) {
            std::call_once(shader->rect_once, [&] {
                const auto rewrite = Metal::RewriteRectListVertex(*words);
                if (!rewrite.ok) {
                    std::printf("Metal shadow: rect list vertex shader %#llx not rewritable: %s\n",
                                (unsigned long long)hash, rewrite.error.c_str());
                    return;
                }
                auto result = compiler.Compile(rewrite.spirv, hash ^ 0x5265637443ull, "rect");
                if (result.status == Metal::CompileStatus::Ok) {
                    shader->rect = std::move(result.shader);
                }
            });
            return shader->rect.get();
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

    /// The module's SPIR-V and its cache entry, or false when it was never registered.
    bool Entry(vk::ShaderModule module, std::shared_ptr<std::vector<u32>>& words,
               std::shared_ptr<ShaderEntry>& entry) {
        const u64 handle = u64(VkShaderModule(module));
        std::scoped_lock lock{mutex};
        auto it = spirv.find(handle);
        if (it == spirv.end()) {
            ++no_spirv;
            return false;
        }
        words = it->second;
        auto& slot = shaders[handle];
        if (!slot) {
            slot = std::make_shared<ShaderEntry>();
        }
        entry = slot;
        return true;
    }

    const Metal::CompiledShader* Variant(vk::ShaderModule module, u64 hash,
                                         const Metal::CompileVariant& variant) {
        std::shared_ptr<std::vector<u32>> words;
        std::shared_ptr<ShaderEntry> entry;
        if (!Entry(module, words, entry)) {
            return nullptr;
        }
        std::scoped_lock lock{entry->variant_mutex};
        for (const auto& [known, shader] : entry->variants) {
            if (known == variant) {
                return shader.get();
            }
        }
        auto result = compiler.Compile(*words, hash, "variant", variant);
        auto& made = entry->variants.emplace_back(variant, std::move(result.shader));
        return made.second.get();
    }

    std::shared_ptr<std::vector<u32>> Words(vk::ShaderModule module) {
        std::shared_ptr<std::vector<u32>> words;
        std::shared_ptr<ShaderEntry> entry;
        return Entry(module, words, entry) ? words : nullptr;
    }

    void Report() {
        const u32 n = ++seen;
        if (n != 1 && n % 50 != 0) {
            return;
        }
        std::printf("Metal shadow: %u pipelines: render %u ok / %u failed, compute %u ok / %u "
                    "failed, rect lists %u ok / %u failed, tessellation %u ok / %u failed / %u not "
                    "possible in Metal; shaders failed %u, no SPIR-V %u; not yet: geometry %u, quad "
                    "lists %u, clip-emulation fragment %u\n",
                    n, render_ok.load(), render_failed.load(), compute_ok.load(),
                    compute_failed.load(), rect_ok.load(), rect_failed.load(), tess_ok.load(),
                    tess_failed.load(), tess_unsupported.load(), shader_failed.load(),
                    no_spirv.load(), skipped_geometry.load(),
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
        // As Huax-1839's PipelineCache::TessellationShaders: the vertex shader as a kernel per
        // index type, the control shader as a kernel, the evaluation shader as the
        // post-tessellation vertex function.
        using Kind = Metal::CompileVariant::Kind;
        const auto vs_module = modules[u32(Stage::Vertex)];
        const auto tcs_module = modules[u32(Stage::TessellationControl)];
        const auto tes_module = modules[u32(Stage::TessellationEval)];
        const auto tcs_words = s.Words(tcs_module);
        const auto tes_words = s.Words(tes_module);
        bool ok = has(Stage::Vertex) && has(Stage::TessellationControl) &&
                  has(Stage::TessellationEval) && tcs_words && tes_words;
        Metal::TessellationInput input;
        std::vector<Metal::VertexAttribute> attributes;
        for (const auto& in : inputs) {
            attributes.push_back({in.location, in.per_instance, in.divisor});
        }
        input.vertex_attributes = attributes;
        if (ok) {
            const auto tes_info = Metal::ReadTessellationInfo(*tes_words);
            const auto tcs_info = Metal::ReadTessellationInfo(*tcs_words);
            if ((tes_info.domain != Metal::TessDomain::Triangles &&
                 tes_info.domain != Metal::TessDomain::Quads) ||
                tes_info.point_mode) {
                ++s.tess_unsupported;
                s.Report();
                return;
            }
            ok = tcs_info.output_vertices != 0;
            for (u32 index_type = 0; ok && index_type < 3; ++index_type) {
                input.vertex[index_type] =
                    s.Variant(vs_module, key.stage_hashes[u32(Stage::Vertex)],
                              {.kind = Kind::TessVertex, .index_type = index_type});
                ok = input.vertex[index_type] != nullptr;
            }
            if (ok) {
                input.control = s.Variant(tcs_module, key.stage_hashes[u32(Stage::TessellationControl)],
                                          {.kind = Kind::TessControl, .domain = tes_info.domain});
                input.eval = s.Variant(tes_module, key.stage_hashes[u32(Stage::TessellationEval)],
                                       {.kind = Kind::TessEval,
                                        .output_vertices = tcs_info.output_vertices});
                ok = input.control && input.eval;
            }
        }
        const Metal::CompiledShader* fs = nullptr;
        if (ok && has(Stage::Fragment)) {
            fs = s.Get(modules[u32(Stage::Fragment)], key.stage_hashes[u32(Stage::Fragment)],
                       "fragment");
            ok = fs != nullptr;
        }
        ok = ok && s.pso.GetTessellationPipeline(key, input, fs) != nullptr;
        ++(ok ? s.tess_ok : s.tess_failed);
    } else if (has(Stage::Geometry)) {
        ++s.skipped_geometry;
    } else if (key.prim_type == AmdGpu::PrimitiveType::QuadList) {
        ++s.skipped_quad;
    } else if (aux_fragment) {
        ++s.skipped_aux_fragment;
    } else {
        // RectList: the vertex shader rewritten to emit two triangles per rectangle (MoltenVK
        // emulates it with a tessellation control shader instead).
        const bool rect_list = key.prim_type == AmdGpu::PrimitiveType::RectList;
        const auto* vs = s.Get(modules[u32(Stage::Vertex)], key.stage_hashes[u32(Stage::Vertex)],
                               "vertex", rect_list);
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
            ok = s.pso.GetRenderPipeline(key, vs, fs,
                                         {.vertex_attributes = attributes,
                                          .rect_list_triangles = rect_list}) != nullptr;
        }
        if (rect_list) {
            ++(ok ? s.rect_ok : s.rect_failed);
        } else {
            ++(ok ? s.render_ok : s.render_failed);
        }
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
