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
#include "video_core/renderer_metal/mtl_shadow.h"
#include "video_core/renderer_metal/mtl_vk_layer.h"
#include "video_core/renderer_metal/mtl_spirv_rectlist.h"
#include "video_core/renderer_metal/mtl_geometry.h"
#include "video_core/renderer_metal/mtl_spirv_geometry.h"
#include "video_core/renderer_metal/vk_to_mtl.h"
#include <xxhash.h>
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
    /// MoltenVK's device and queue when the Metal path is on (its pipelines must be that device's).
    Metal::Device device{MetalLayer::MtlDevice(), MetalLayer::MtlQueue()};
    std::filesystem::path dir;
    Metal::ShaderCompiler compiler;
    Metal::PsoCache pso;
    Metal::Geometry::PassThroughCache pass_through{device};
    std::mutex mutex;
    std::unordered_map<u64, std::shared_ptr<std::vector<u32>>> spirv; ///< by VkShaderModule
    std::unordered_map<u64, std::shared_ptr<ShaderEntry>> shaders;          ///< by VkShaderModule

    std::atomic<u32> render_ok{}, render_failed{}, compute_ok{}, compute_failed{};
    std::atomic<u32> shader_failed{}, no_spirv{};
    std::atomic<u32> rect_ok{}, rect_failed{};
    std::atomic<u32> tess_ok{}, tess_failed{}, tess_unsupported{};
    std::atomic<u32> geometry_ok{}, geometry_failed{}, geometry_unsupported{};
    std::atomic<u32> skipped_geometry{}, skipped_quad{},
        skipped_aux_fragment{};
    std::atomic<u32> seen{};

    std::mutex pipelines_mutex;
    std::unordered_map<VkPipeline, GraphicsEntry> graphics;
    std::unordered_map<VkPipeline, const Metal::ComputePipeline*> compute;

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
                    "failed, geometry %u ok / %u failed / %u unsupported, rect lists %u ok / %u failed, tessellation %u ok / %u failed / %u not "
                    "possible in Metal; shaders failed %u, no SPIR-V %u; not yet: geometry %u, quad "
                    "lists %u, clip-emulation fragment %u\n",
                    n, render_ok.load(), render_failed.load(), compute_ok.load(),
                    compute_failed.load(), geometry_ok.load(), geometry_failed.load(),
                    geometry_unsupported.load(), rect_ok.load(), rect_failed.load(), tess_ok.load(),
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
        const char* shadow = std::getenv("BB_METAL_SHADOW");
        const char* frame = std::getenv("BB_METAL_FRAME");
        return (shadow && shadow[0] == '1') || (frame && frame[0] == '1');
    }();
    return enabled;
}

const GraphicsEntry* FindGraphics(VkPipeline pipeline) {
    auto& s = Get();
    std::scoped_lock lock{s.pipelines_mutex};
    auto it = s.graphics.find(pipeline);
    return it == s.graphics.end() ? nullptr : &it->second;
}

const Metal::ComputePipeline* FindCompute(VkPipeline pipeline) {
    auto& s = Get();
    std::scoped_lock lock{s.pipelines_mutex};
    auto it = s.compute.find(pipeline);
    return it == s.compute.end() ? nullptr : it->second;
}

void* DepthStencil(const Metal::DepthStencilKey& key) {
    return Get().pso.GetDepthStencilState(key);
}

const Metal::Geometry::IndirectKernels* GeometryIndirectKernels() {
    static const auto kernels = [] {
        auto k = std::make_unique<Metal::Geometry::IndirectKernels>(Get().device);
        if (!k->Ok()) {
            std::printf("Metal shadow: indirect geometry kernels: %s\n", k->Error().c_str());
        }
        return k;
    }();
    return kernels->Ok() ? kernels.get() : nullptr;
}

void Forget(VkPipeline pipeline) {
    if (!Enabled()) {
        return;
    }
    auto& s = Get();
    std::scoped_lock lock{s.pipelines_mutex};
    s.graphics.erase(pipeline);
    s.compute.erase(pipeline);
}

void RegisterModule(vk::ShaderModule module, std::span<const u32> spirv) {
    auto& s = Get();
    auto words = std::make_shared<std::vector<u32>>(spirv.begin(), spirv.end());
    std::scoped_lock lock{s.mutex};
    s.spirv[u64(VkShaderModule(module))] = std::move(words);
}

void Graphics(const Vulkan::GraphicsPipelineKey& key, std::span<const vk::ShaderModule> modules,
              std::span<const bool> has_stage, std::span<const VertexInput> inputs,
              bool aux_fragment, vk::Pipeline vk_pipeline) {
    @autoreleasepool { // pipeline-creation threads drain no pool of their own
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
        const Metal::TessellationPipeline* tess =
            ok ? s.pso.GetTessellationPipeline(key, input, fs) : nullptr;
        ok = tess != nullptr;
        ++(ok ? s.tess_ok : s.tess_failed);
        if (ok) {
            std::scoped_lock lock{s.pipelines_mutex};
            s.graphics[VkPipeline(vk_pipeline)] = {.render = tess->render, .tess = tess};
        }
    } else if (has(Stage::Geometry)) {
        // As Huax-1839's PipelineCache::GeometryShaders: the export shader as a kernel per index
        // type, the geometry shader as a kernel, a generated pass-through vertex function.
        using Kind = Metal::CompileVariant::Kind;
        const auto es_module = modules[u32(Stage::Vertex)];
        const auto gs_module = modules[u32(Stage::Geometry)];
        const auto gs_words = s.Words(gs_module);
        std::string why;
        const Metal::GeometryPipeline* geo = nullptr;
        if (!has(Stage::Vertex) || !gs_words) {
            why = "a stage is missing";
        } else {
            const auto analysis = Metal::AnalyzeGeometryShader(*gs_words);
            const bool matches =
                analysis.ok &&
                ((analysis.info.input == Metal::GsInputPrimitive::Points &&
                  key.prim_type == AmdGpu::PrimitiveType::PointList) ||
                 (analysis.info.input == Metal::GsInputPrimitive::Lines &&
                  key.prim_type == AmdGpu::PrimitiveType::LineList) ||
                 (analysis.info.input == Metal::GsInputPrimitive::Triangles &&
                  key.prim_type == AmdGpu::PrimitiveType::TriangleList));
            if (!analysis.ok || !matches) {
                ++s.geometry_unsupported;
                why = analysis.ok ? "primitive type differs from the shader's input"
                                  : analysis.error;
            } else {
                Metal::GeometryPipelineInput input;
                std::vector<Metal::VertexAttribute> attributes;
                for (const auto& in : inputs) {
                    attributes.push_back({in.location, in.per_instance, in.divisor});
                }
                input.vertex_attributes = attributes;
                bool ok = true;
                for (u32 index_type = 0; ok && index_type < 3; ++index_type) {
                    input.es[index_type] =
                        s.Variant(es_module, key.stage_hashes[u32(Stage::Vertex)],
                                  {.kind = Kind::TessVertex, .index_type = index_type});
                    ok = input.es[index_type] != nullptr;
                }
                if (ok) {
                    Metal::CompileVariant variant{.kind = Kind::GeometryKernel};
                    variant.es.stride = input.es[0]->tess_layout.output;
                    variant.es.members = input.es[0]->tess_layout.members;
                    input.gs = s.Variant(gs_module, key.stage_hashes[u32(Stage::Geometry)], variant);
                    ok = input.gs != nullptr;
                }
                const Metal::CompiledShader* fs = nullptr;
                if (ok && has(Stage::Fragment)) {
                    fs = s.Get(modules[u32(Stage::Fragment)],
                               key.stage_hashes[u32(Stage::Fragment)], "fragment");
                    ok = fs != nullptr;
                }
                if (ok) {
                    std::span<const Metal::StageInput> fs_inputs;
                    if (fs) {
                        fs_inputs = fs->stage_inputs;
                    }
                    input.pass_through = s.pass_through.Get(
                        input.gs->geometry, fs_inputs,
                        key.clip_space == AmdGpu::ClipSpace::MinusWToW, why);
                    ok = input.pass_through != nullptr;
                }
                geo = ok ? s.pso.GetGeometryPipeline(key, input, fs) : nullptr;
                if (!geo && why.empty()) {
                    why = "a shader variant or the pipeline state failed";
                }
            }
        }
        ++(geo ? s.geometry_ok : s.geometry_failed);
        if (geo) {
            std::scoped_lock lock{s.pipelines_mutex};
            s.graphics[VkPipeline(vk_pipeline)] = {.render = geo->render, .geometry = geo};
        } else {
            static std::atomic<u32> logged{0};
            if (logged++ < 10) {
                std::printf("Metal shadow: geometry pipeline not built: %s\n", why.c_str());
            }
        }
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
            const Metal::RenderPipeline* render = s.pso.GetRenderPipeline(
                key, vs, fs, {.vertex_attributes = attributes, .rect_list_triangles = rect_list});
            ok = render != nullptr;
            if (ok) {
                std::scoped_lock lock{s.pipelines_mutex};
                s.graphics[VkPipeline(vk_pipeline)] = {.render = render, .rect_list = rect_list};
            }
        }
        if (rect_list) {
            ++(ok ? s.rect_ok : s.rect_failed);
        } else {
            ++(ok ? s.render_ok : s.render_failed);
        }
    }
    s.Report();
    }
}

void Compute(const Vulkan::ComputePipelineKey& key, vk::ShaderModule module,
             vk::Pipeline vk_pipeline) {
    @autoreleasepool { // pipeline-creation threads drain no pool of their own
    auto& s = Get();
    const auto* cs = s.Get(module, key.value, "compute");
    const Metal::ComputePipeline* pipeline = cs ? s.pso.GetComputePipeline(key, cs) : nullptr;
    const bool ok = pipeline != nullptr;
    if (ok) {
        std::scoped_lock lock{s.pipelines_mutex};
        s.compute[VkPipeline(vk_pipeline)] = pipeline;
    }
    ++(ok ? s.compute_ok : s.compute_failed);
    s.Report();
    }
}

// ---- Generic pipelines (host passes): built from their Vulkan create infos ----

namespace {
thread_local int game_build_depth = 0;

/// Fixes the SPIR-V specialization constants at the values given (or their defaults) and turns
/// them into plain constants, so SPIRV-Cross emits no Metal function constants (a pipeline state
/// would then need a specialized function): OpSpecConstant's literal, OpSpecConstantTrue/False by
/// opcode, the SpecId decorations dropped. OpSpecConstantOp stays (none in the host shaders).
std::vector<u32> Specialize(const std::vector<u32>& words, const VkSpecializationInfo* info) {
    if (words.size() < 5) {
        return words;
    }
    std::unordered_map<u32, u32> spec_ids; // result id -> SpecId
    for (size_t i = 5; i < words.size();) {
        const u32 count = words[i] >> 16, op = words[i] & 0xffff;
        if (!count) {
            break;
        }
        if (op == 71 && count >= 4 && words[i + 2] == 1) { // OpDecorate id SpecId n
            spec_ids[words[i + 1]] = words[i + 3];
        }
        i += count;
    }
    std::vector<u32> out(words.begin(), words.begin() + 5);
    out.reserve(words.size());
    for (size_t i = 5; i < words.size();) {
        const u32 count = words[i] >> 16, op = words[i] & 0xffff;
        if (!count || i + count > words.size()) {
            break;
        }
        if (op == 71 && count >= 4 && words[i + 2] == 1) {
            i += count; // SpecId on what is now a plain constant
            continue;
        }
        const size_t at = out.size();
        out.insert(out.end(), words.begin() + i, words.begin() + i + count);
        if ((op == 48 || op == 49 || op == 50 || op == 51) && count >= 3) {
            u32 new_op = op == 48 ? 41u : op == 49 ? 42u : op == 50 ? 43u : 44u;
            auto it = spec_ids.find(words[i + 2]);
            if (info && it != spec_ids.end()) {
                for (u32 e = 0; e < info->mapEntryCount; ++e) {
                    const auto& entry = info->pMapEntries[e];
                    if (entry.constantID != it->second ||
                        entry.offset + entry.size > info->dataSize) {
                        continue;
                    }
                    const u8* data = static_cast<const u8*>(info->pData) + entry.offset;
                    if (op == 50) {
                        for (u32 w = 0; w < count - 3 && w * 4 < entry.size; ++w) {
                            u32 v = 0;
                            std::memcpy(&v, data + w * 4, std::min<size_t>(4, entry.size - w * 4));
                            out[at + 3 + w] = v;
                        }
                    } else if (op == 48 || op == 49) {
                        bool value = false;
                        for (size_t b = 0; b < entry.size; ++b) {
                            value |= data[b] != 0;
                        }
                        new_op = value ? 41u : 42u;
                    }
                }
            }
            out[at] = (count << 16) | new_op;
        }
        i += count;
    }
    return out;
}

std::mutex generic_mutex;
std::vector<std::unique_ptr<Metal::CompiledShader>> generic_shaders;
std::vector<std::unique_ptr<Metal::RenderPipeline>> generic_pipelines;
std::atomic<u32> generic_ok{0}, generic_failed{0};

const Metal::CompiledShader* CompileStage(State& s, const VkPipelineShaderStageCreateInfo& stage,
                                          const char* what) {
    std::shared_ptr<std::vector<u32>> words;
    {
        std::scoped_lock lock{s.mutex};
        auto it = s.spirv.find(u64(stage.module));
        if (it == s.spirv.end()) {
            return nullptr;
        }
        words = it->second;
    }
    const std::vector<u32> spec = Specialize(*words, stage.pSpecializationInfo);
    const u64 hash = XXH3_64bits(spec.data(), spec.size() * 4);
    auto result = s.compiler.Compile(spec, hash, what);
    if (result.status != Metal::CompileStatus::Ok) {
        return nullptr;
    }
    std::scoped_lock lock{generic_mutex};
    return generic_shaders.emplace_back(std::move(result.shader)).get();
}

void NoteGeneric(bool ok, const char* why) {
    ++(ok ? generic_ok : generic_failed);
    if (!ok) {
        std::printf("Metal shadow: generic pipeline not built: %s\n", why);
    }
}
} // namespace

GameBuildScope::GameBuildScope() {
    ++game_build_depth;
}
GameBuildScope::~GameBuildScope() {
    --game_build_depth;
}

void ModuleCreated(VkShaderModule module, const VkShaderModuleCreateInfo* info) {
    RegisterModule(vk::ShaderModule(module),
                   std::span<const u32>(info->pCode, info->codeSize / sizeof(u32)));
}

void ComputeCreated(const VkComputePipelineCreateInfo* info, VkPipeline pipeline) {
    if (game_build_depth || !pipeline) {
        return;
    }
    auto& s = Get();
    const auto* cs = CompileStage(s, info->stage, "host compute");
    if (!cs) {
        NoteGeneric(false, "compute shader did not translate");
        return;
    }
    Vulkan::ComputePipelineKey key{};
    key.value = size_t(XXH3_64bits(&cs, sizeof cs));
    const Metal::ComputePipeline* p = s.pso.GetComputePipeline(key, cs);
    NoteGeneric(p != nullptr, "compute pipeline state");
    if (p) {
        std::scoped_lock lock{s.pipelines_mutex};
        s.compute[pipeline] = p;
    }
}

void GraphicsCreated(const VkGraphicsPipelineCreateInfo* info, VkPipeline pipeline) {
    if (game_build_depth || !pipeline) {
        return;
    }
    using namespace Metal::VkToMtl;
    auto& s = Get();
    const Metal::CompiledShader* vs = nullptr;
    const Metal::CompiledShader* fs = nullptr;
    for (u32 i = 0; i < info->stageCount; ++i) {
        const auto& stage = info->pStages[i];
        if (stage.stage == VK_SHADER_STAGE_VERTEX_BIT) {
            vs = CompileStage(s, stage, "host vertex");
        } else if (stage.stage == VK_SHADER_STAGE_FRAGMENT_BIT) {
            fs = CompileStage(s, stage, "host fragment");
        } else {
            NoteGeneric(false, "a stage other than vertex and fragment");
            return;
        }
    }
    if (!vs) {
        NoteGeneric(false, "vertex shader did not translate");
        return;
    }
    auto entry = std::make_unique<Metal::RenderPipeline>();
    GraphicsEntry ge;
    ge.generic = true;
    @autoreleasepool {
        MTLRenderPipelineDescriptor* d = [MTLRenderPipelineDescriptor new];
        d.vertexFunction = (__bridge id<MTLFunction>)vs->function;
        if (fs) {
            d.fragmentFunction = (__bridge id<MTLFunction>)fs->function;
        }
        // Attachments (dynamic rendering).
        const VkPipelineRenderingCreateInfo* rendering = nullptr;
        for (auto* n = static_cast<const VkBaseInStructure*>(info->pNext); n; n = n->pNext) {
            if (n->sType == VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO) {
                rendering = reinterpret_cast<const VkPipelineRenderingCreateInfo*>(n);
            }
        }
        const auto* blend = info->pColorBlendState;
        if (rendering) {
            for (u32 i = 0; i < rendering->colorAttachmentCount; ++i) {
                const u32 format = PixelFormat(vk::Format(rendering->pColorAttachmentFormats[i]));
                if (format && IsDepthPixelFormat(format)) {
                    // Metal asserts on a depth format as a color target (never valid Vulkan
                    // either); the pipeline is left without a Metal counterpart.
                    const std::string why =
                        "color attachment " + std::to_string(i) + " has the depth format " +
                        vk::to_string(vk::Format(rendering->pColorAttachmentFormats[i]));
                    NoteGeneric(false, why.c_str());
                    return;
                }
                entry->color_pixel_formats[i] = format;
                d.colorAttachments[i].pixelFormat = MTLPixelFormat(format);
                if (blend && i < blend->attachmentCount) {
                    const auto& a = blend->pAttachments[i];
                    d.colorAttachments[i].writeMask =
                        MTLColorWriteMask(ColorWriteMask(vk::ColorComponentFlags(a.colorWriteMask)));
                    d.colorAttachments[i].blendingEnabled = a.blendEnable;
                    if (a.blendEnable) {
                        d.colorAttachments[i].sourceRGBBlendFactor =
                            MTLBlendFactor(BlendFactor(vk::BlendFactor(a.srcColorBlendFactor)));
                        d.colorAttachments[i].destinationRGBBlendFactor =
                            MTLBlendFactor(BlendFactor(vk::BlendFactor(a.dstColorBlendFactor)));
                        d.colorAttachments[i].rgbBlendOperation =
                            MTLBlendOperation(BlendOp(vk::BlendOp(a.colorBlendOp)));
                        d.colorAttachments[i].sourceAlphaBlendFactor =
                            MTLBlendFactor(BlendFactor(vk::BlendFactor(a.srcAlphaBlendFactor)));
                        d.colorAttachments[i].destinationAlphaBlendFactor =
                            MTLBlendFactor(BlendFactor(vk::BlendFactor(a.dstAlphaBlendFactor)));
                        d.colorAttachments[i].alphaBlendOperation =
                            MTLBlendOperation(BlendOp(vk::BlendOp(a.alphaBlendOp)));
                    }
                }
            }
            entry->num_color_attachments = rendering->colorAttachmentCount;
            if (rendering->depthAttachmentFormat) {
                entry->depth_pixel_format = PixelFormat(vk::Format(rendering->depthAttachmentFormat));
                d.depthAttachmentPixelFormat = MTLPixelFormat(entry->depth_pixel_format);
            }
            if (rendering->stencilAttachmentFormat) {
                entry->stencil_pixel_format =
                    PixelFormat(vk::Format(rendering->stencilAttachmentFormat));
                d.stencilAttachmentPixelFormat = MTLPixelFormat(entry->stencil_pixel_format);
            }
        }
        if (info->pMultisampleState) {
            d.rasterSampleCount = info->pMultisampleState->rasterizationSamples;
            entry->sample_count = info->pMultisampleState->rasterizationSamples;
        }
        // Vertex input: binding b at buffer slot kVertexBufferBase - b, dynamic stride.
        if (const auto* vi = info->pVertexInputState; vi && vi->vertexAttributeDescriptionCount) {
            MTLVertexDescriptor* vd = [MTLVertexDescriptor vertexDescriptor];
            for (u32 i = 0; i < vi->vertexAttributeDescriptionCount; ++i) {
                const auto& a = vi->pVertexAttributeDescriptions[i];
                vd.attributes[a.location].format = MTLVertexFormat(VertexFormat(vk::Format(a.format)));
                vd.attributes[a.location].offset = a.offset;
                vd.attributes[a.location].bufferIndex = Metal::kVertexBufferBase - a.binding;
            }
            for (u32 i = 0; i < vi->vertexBindingDescriptionCount; ++i) {
                const auto& b = vi->pVertexBindingDescriptions[i];
                const u32 slot = Metal::kVertexBufferBase - b.binding;
                vd.layouts[slot].stride = MTLBufferLayoutStrideDynamic;
                vd.layouts[slot].stepFunction = b.inputRate == VK_VERTEX_INPUT_RATE_INSTANCE
                                                    ? MTLVertexStepFunctionPerInstance
                                                    : MTLVertexStepFunctionPerVertex;
                if (b.binding < ge.strides.size()) {
                    ge.strides[b.binding] = b.stride;
                }
                auto& vslot = entry->vertex_slots[entry->num_vertex_slots++];
                vslot.location = b.binding;
                vslot.slot = slot;
            }
            d.vertexDescriptor = vd;
        }
        const auto topology = info->pInputAssemblyState
                                  ? PrimitiveType(vk::PrimitiveTopology(info->pInputAssemblyState->topology))
                                  : PrimitiveType(vk::PrimitiveTopology::eTriangleList);
        entry->primitive_type = topology.type;
        entry->primitive_emulated = topology.needs_emulation;
        d.inputPrimitiveTopology = topology.type == MTLPrimitiveTypePoint ? MTLPrimitiveTopologyClassPoint
                                   : topology.type == MTLPrimitiveTypeLine ||
                                           topology.type == MTLPrimitiveTypeLineStrip
                                       ? MTLPrimitiveTopologyClassLine
                                       : MTLPrimitiveTopologyClassTriangle;
        NSError* error = nil;
        id<MTLRenderPipelineState> state =
            [(__bridge id<MTLDevice>)MetalLayer::MtlDevice() newRenderPipelineStateWithDescriptor:d
                                                                                           error:&error];
        if (!state) {
            std::string why = "Metal: ";
            why += error ? error.localizedDescription.UTF8String : "unknown";
            NoteGeneric(false, why.c_str());
            return;
        }
        entry->state = (__bridge_retained void*)state;
    }
    entry->vs = vs;
    entry->fs = fs;
    entry->fill_mode = MTLTriangleFillModeFill;
    entry->depth_clip = true;
    // The state Vulkan fixes unless it is dynamic.
    std::vector<VkDynamicState> dynamic;
    if (info->pDynamicState) {
        dynamic.assign(info->pDynamicState->pDynamicStates,
                       info->pDynamicState->pDynamicStates + info->pDynamicState->dynamicStateCount);
    }
    const auto is_dynamic = [&](VkDynamicState d) {
        return std::find(dynamic.begin(), dynamic.end(), d) != dynamic.end();
    };
    if (const auto* r = info->pRasterizationState) {
        ge.fixed.cull = !is_dynamic(VK_DYNAMIC_STATE_CULL_MODE);
        ge.fixed.cull_mode = r->cullMode;
        ge.fixed.front_face = !is_dynamic(VK_DYNAMIC_STATE_FRONT_FACE);
        ge.fixed.front = r->frontFace;
        ge.fixed.depth_bias = !is_dynamic(VK_DYNAMIC_STATE_DEPTH_BIAS_ENABLE);
        ge.fixed.depth_bias_enable = r->depthBiasEnable;
        entry->depth_clip = !r->depthClampEnable;
    }
    if (const auto* ds = info->pDepthStencilState) {
        ge.fixed.depth = !is_dynamic(VK_DYNAMIC_STATE_DEPTH_TEST_ENABLE);
        ge.fixed.depth_test = ds->depthTestEnable;
        ge.fixed.depth_write = ds->depthWriteEnable;
        ge.fixed.depth_compare = ds->depthCompareOp;
        ge.fixed.stencil = !is_dynamic(VK_DYNAMIC_STATE_STENCIL_TEST_ENABLE);
        ge.fixed.stencil_test = ds->stencilTestEnable;
    } else {
        ge.fixed.depth = ge.fixed.stencil = true; // no depth/stencil state: both off
    }
    ge.render = entry.get();
    {
        std::scoped_lock lock{generic_mutex};
        generic_pipelines.push_back(std::move(entry));
    }
    std::scoped_lock lock{s.pipelines_mutex};
    s.graphics[pipeline] = ge;
    NoteGeneric(true, "");
}

} // namespace MetalShadow
