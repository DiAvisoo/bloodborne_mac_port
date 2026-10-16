// SPDX-License-Identifier: GPL-2.0-or-later
#include "video_core/renderer_vulkan/vk_scene_resolution.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <fmt/format.h>
#include "bbport_toggles.h"
#include "common/assert.h"
#include "video_core/renderer_metal/mtl_blit_shaders.h"
#include "video_core/renderer_metal/mtl_depth_stencil.h"
#include "video_core/renderer_metal/mtl_encoder.h"
#include "video_core/renderer_metal/mtl_render_pipeline.h"
#include "video_core/renderer_metal/mtl_texture.h"
#include "video_core/renderer_metal/vk_to_mtl.h"
#include "video_core/renderer_vulkan/vk_runtime.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/texture_cache/texture_cache.h"

#include <vulkan/vulkan_format_traits.hpp>

namespace Vulkan {
namespace {

constexpr std::array<u8, 4> IdentitySwizzle{
    Metal::VkToMtl::SwizzleRed, Metal::VkToMtl::SwizzleGreen, Metal::VkToMtl::SwizzleBlue,
    Metal::VkToMtl::SwizzleAlpha};

// Scaled colour copies: the texel nearest to the target pixel's centre, as the Vulkan path's
// blitImage with eNearest (floor((x + 0.5) * src / dst)). Metal blits cannot scale. Nearest
// preserves integer G-buffer formats; FSR reconstructs colour at the HDR scene boundary.
// Float, uint and int flavours (the shader's type must match the format's).
constexpr const char* ScaleSource = R"msl(
#define SCENE_SCALE(T, SUFFIX)                                                                    \
    fragment vec<T, 4> scene_scale##SUFFIX##_fs(FsTriOut in [[stage_in]],                        \
                                                texture2d<T> src [[texture(0)]]) {               \
        return src.read(TexelOf(in.uv, uint2(src.get_width(), src.get_height())), 0);           \
    }
SCENE_SCALE(float, )
SCENE_SCALE(uint, _uint)
SCENE_SCALE(int, _int)
)msl";
constexpr const char* ScaleFunctions[3] = {"scene_scale_fs", "scene_scale_uint_fs",
                                           "scene_scale_int_fs"};

/// Which flavour of scene_scale reads and writes the format: 0 float, 1 uint, 2 int.
u32 NumericFlavour(vk::Format format) {
    const char* numeric = vk::componentNumericFormat(format, 0);
    if (std::strcmp(numeric, "UINT") == 0) {
        return 1;
    }
    if (std::strcmp(numeric, "SINT") == 0) {
        return 2;
    }
    return 0;
}

constexpr u8 StencilOpReplace = 2; // VK_STENCIL_OP_REPLACE
constexpr u8 CompareAlways = 7;    // VK_COMPARE_OP_ALWAYS

Metal::DepthStencilKey ResampleState(bool stencil, u8 write_mask) {
    Metal::DepthStencilKey key{};
    key.depth_test = true;
    key.depth_write = true;
    key.depth_compare = CompareAlways;
    key.stencil_test = stencil;
    if (stencil) {
        // Reference and compare are fixed; an exported stencil value replaces the reference.
        const Metal::StencilFace face{.fail = StencilOpReplace,
                                      .pass = StencilOpReplace,
                                      .depth_fail = StencilOpReplace,
                                      .compare = CompareAlways,
                                      .read_mask = 0xff,
                                      .write_mask = write_mask};
        key.front = face;
        key.back = face;
    }
    return key;
}

/// Fullscreen-triangle render state: what the rasterizer may have left is reset.
void SetFullscreenState(Metal::Encoder& e, void* pipeline, u32 width, u32 height) {
    e.SetRenderPipeline(pipeline);
    e.SetCullMode(Metal::CullMode::None);
    e.SetDepthBias(0.0f, 0.0f, 0.0f);
    e.SetFillMode(Metal::FillMode::Fill);
    e.SetDepthClip(Metal::DepthClip::Clip);
    e.SetViewport({0.0f, 0.0f, float(width), float(height), 0.0f, 1.0f});
    e.SetScissor({0, 0, width, height});
}

} // namespace

SceneTargets::SceneTargets(Metal::Device& d, Scheduler& s, Runtime& r,
                           VideoCore::TextureCache& t)
    : SceneTargets(d, s, r, [&t](VideoCore::ImageId id, u64 uid) {
        return t.TryGetImage(id, uid);
    }) {}
SceneTargets::SceneTargets(Metal::Device& d, Scheduler& s, Runtime& r, Lookup get)
    : device{d}, scheduler{s}, runtime{r}, lookup{std::move(get)} {
    runtime.scene_targets = this;
    const char* force = std::getenv("BB_SCENE_STENCIL_BITS");
    force_stencil_bits = force && force[0] == '1';
    library = Metal::CompileLibrary(device, std::string{Metal::BlitShaders::Source} + ScaleSource,
                                    "Scene resolution");
    auto& states = scheduler.GetDepthStencilCache();
    no_depth_state = states.Get(Metal::DepthStencilKey{});
    depth_write_state = states.Get(ResampleState(false, 0));
    depth_stencil_state = states.Get(ResampleState(true, 0xff));
    stencil_bit_states[0] = states.Get(ResampleState(true, 0));
    for (u32 i = 1; i < stencil_bit_states.size(); ++i) {
        stencil_bit_states[i] = states.Get(ResampleState(true, u8(1u << (i - 1))));
    }
}
SceneTargets::~SceneTargets() {
    scheduler.Finish();
    runtime.scene_targets = nullptr;
}
bool SceneTargets::Blittable(const VideoCore::Image& image) const {
    // Every non-block colour texture is a render target (Image's texture description), so a
    // nearest-texel draw can scale it; same-size copies are plain texture copies.
    const auto& desc = image.backing->desc;
    return !image.info.props.is_depth && !image.info.props.is_block && desc.render_target &&
           desc.pixel_format != 0;
}
bool SceneTargets::ShaderResampled(const VideoCore::Image& image) const {
    // Depth formats (Stencil8 has no depth to read). Apple GPUs export stencil from fragment
    // shaders; BB_SCENE_STENCIL_BITS=1 tests the per-bit resampler instead.
    const auto& desc = image.backing->desc;
    return image.info.props.is_depth && desc.render_target &&
           Metal::VkToMtl::IsDepthPixelFormat(desc.pixel_format);
}
// 1 for the scene size, 2 for half resolution (the game's post-processing chain), else 0.
// Half resolution needs a scene size that halves exactly: the proxy keeps the scene's factor,
// so the shaders' FragCoord conversion (1920 / scene width) stays correct for it.
static u32 SizeDivisor(const VideoCore::ImageInfo& i, SceneResolution::Size scene) {
    if (i.size.width == 1920 && i.size.height == 1080) return 1;
    if (i.size.width == 960 && i.size.height == 540 && scene.width % 2 == 0 &&
        scene.height % 2 == 0 && !BbToggle::Disabled(BbToggle::SceneHalfRes)) {
        return 2;
    }
    return 0;
}
bool SceneTargets::Eligible(const VideoCore::Image& image) const {
    const auto& i = image.info;
    const u32 div = SizeDivisor(i, size);
    if (!div || i.size.depth != 1 || i.resources.layers != 1 || i.num_samples != 1 ||
        i.props.is_block || !image.backing || image.backing->num_samples != 1) {
        return false;
    }
    if (i.resources.levels != 1) {
        // Mip chains (the bloom pyramid at half resolution): one blitted proxy per level.
        return div == 2 && i.resources.levels <= 16 && !i.props.is_depth &&
               Blittable(image);
    }
    return Blittable(image) || ShaderResampled(image);
}
bool SceneTargets::EligibleScene(const VideoCore::Image& image) const {
    return image.info.size.width == 1920 && image.info.size.height == 1080 && Eligible(image);
}
SceneResolution::Size SceneTargets::ProxySize(const VideoCore::Image& image, u32 level) const {
    const u32 div = std::max(1u, SizeDivisor(image.info, size));
    return {std::max(1u, size.width / div >> level), std::max(1u, size.height / div >> level)};
}
bool SceneTargets::SetSize(SceneResolution::Size next) {
    if (next == size) return false;
    ResolveAll();
    scheduler.Finish();
    entries.clear(); // no command buffer can still reference these images/views
    tracked.clear();
    ++generation;
    recent = {};
    size = next;
    std::printf("Scene resolution: raster %ux%u, half-resolution targets %ux%u, UI 1920x1080 "
                "(live)\n", size.width, size.height, size.width / 2, size.height / 2);
    return true;
}
void SceneTargets::ResolveAll() {
    for (auto& [key, entry] : entries) {
        if (entry->state.dirty) {
            if (auto* image = lookup(entry->source, entry->uid)) Copy(*entry, *image, true);
        }
    }
}
void SceneTargets::NativeAccess(VideoCore::Image& image, bool write) {
    if (copying || !image.scene_proxy) return; // most images never had a proxy
    const u32 levels = std::min(16u, image.info.resources.levels);
    for (u32 level = 0; level < levels; ++level) {
        const auto it = entries.find(Key(image.image_uid, level));
        if (it == entries.end()) continue;
        auto& entry = *it->second;
        if (entry.state.dirty) Copy(entry, image, true);
        if (write) entry.state.NativeWrite();
    }
}
void SceneTargets::Copy(Entry& e, VideoCore::Image& original, bool to_native) {
    copying = true;
    scheduler.EndRendering();
    const Runtime::TransferMark mark{runtime, to_native ? "scene proxy resolve" : "scene proxy fill",
                                     original};
    if (debug) {
        std::printf("Scene %s: %s %ux%u level %u\n", to_native ? "resolve" : "fill",
                    vk::to_string(original.info.pixel_format).c_str(), original.info.size.width,
                    original.info.size.height, e.level);
    }
    // Joins the texture binding helper; NativeAccess skips the image (copying).
    runtime.AccessImage(&original, to_native);
    void* const proxy = e.image.Get();
    void* const native = original.GetImage();
    void* const src = to_native ? proxy : native;
    void* const dst = to_native ? native : proxy;
    const u32 src_level = to_native ? 0u : e.level;
    const u32 dst_level = to_native ? e.level : 0u;
    const u32 native_width = std::max(1u, original.info.size.width >> e.level);
    const u32 native_height = std::max(1u, original.info.size.height >> e.level);
    const u32 dst_width = to_native ? native_width : e.width;
    const u32 dst_height = to_native ? native_height : e.height;
    if (native_width == e.width && native_height == e.height) {
        // Same size (all aspects of depth/stencil formats included).
        scheduler.Record([src, dst, src_level, dst_level, dst_width, dst_height](Metal::Encoder& enc) {
            enc.CopyTexture(src, src_level, 0, {}, dst, dst_level, 0, {},
                            {dst_width, dst_height, 1});
        });
    } else if (ShaderResampled(original)) {
        // (Depth proxies are single-level: Eligible.)
        Resample(src, dst, original, dst_width, dst_height);
    } else {
        Scale(src, src_level, dst, dst_level, dst_width, dst_height, original);
    }
    if (to_native) e.state.Resolved(); else e.state.CopiedToProxy();
    copying = false;
}
void* SceneTargets::Pipeline(const PipelineKey& key) {
    for (const auto& [k, pipeline] : pipelines) if (k == key) return pipeline.Get();
    const Metal::RenderPipelineDesc desc{
        .library = library.Get(),
        .vertex_function = Metal::BlitShaders::FullscreenTriangle,
        .fragment_function = key.fragment,
        .color_format = key.color_format,
        .depth_format = key.depth_format,
        .stencil_format = key.stencil_format,
    };
    const std::string label = fmt::format("scene {} {}/{}/{}", key.fragment, key.color_format,
                                          key.depth_format, key.stencil_format);
    pipelines.emplace_back(key, Metal::CreateRenderPipeline(device, desc, label));
    return pipelines.back().second.Get();
}
void SceneTargets::Scale(void* src, u32 src_level, void* dst, u32 dst_level, u32 width,
                         u32 height, const VideoCore::Image& original) {
    const u32 format = Metal::TexturePixelFormat(dst);
    Metal::Object source = Metal::CreateTextureView(src, Metal::TexturePixelFormat(src),
                                                    Metal::TextureType::Tex2D, src_level, 1, 0, 1,
                                                    IdentitySwizzle);
    ASSERT_MSG(source, "Scene proxy copy: no 2D view of level {}", src_level);
    void* const pipeline =
        Pipeline({.fragment = ScaleFunctions[NumericFlavour(original.info.pixel_format)],
                  .color_format = format});
    Metal::RenderPassDesc pass{};
    pass.colors[0].texture = dst;
    pass.colors[0].level = dst_level;
    pass.width = width;
    pass.height = height;
    scheduler.Record([pass, pipeline, state = no_depth_state,
                      source = std::move(source)](Metal::Encoder& e) mutable {
        void* const texture = source.Get();
        e.KeepAlive(std::move(source));
        e.BeginRenderPass(pass);
        SetFullscreenState(e, pipeline, pass.width, pass.height);
        e.SetDepthStencil(state);
        e.SetTexture(Metal::Stage::Fragment, 0, texture);
        e.Draw(Metal::Primitive::Triangle, 3);
        e.EndRenderPass();
    });
    // The guest pipelines' dynamic state must be emitted again after this pipeline.
    scheduler.GetDynamicState().Invalidate();
}
void SceneTargets::Resample(void* src, void* dst, const VideoCore::Image& original, u32 width,
                            u32 height) {
    const u32 format = Metal::TexturePixelFormat(dst);
    const bool stencil = Metal::VkToMtl::HasStencilPixelFormat(format);
    const bool stencil_bits = stencil && force_stencil_bits;
    Metal::Object stencil_view;
    if (stencil) {
        stencil_view = Metal::CreateTextureView(
            src, Metal::StencilViewPixelFormat(Metal::TexturePixelFormat(src)),
            Metal::TextureType::Tex2D, 0, 1, 0, 1, IdentitySwizzle);
        ASSERT_MSG(stencil_view, "Scene proxy resample: no stencil view of {}",
                   vk::to_string(original.info.pixel_format));
    }
    const char* fragment = !stencil       ? Metal::BlitShaders::DepthResample
                           : stencil_bits ? Metal::BlitShaders::DepthResampleStencilBits
                                          : Metal::BlitShaders::DepthResampleStencil;
    void* const pipeline = Pipeline(
        {.fragment = fragment, .depth_format = format, .stencil_format = stencil ? format : 0});
    Metal::RenderPassDesc pass{};
    pass.depth.texture = dst;
    pass.depth.has_stencil = stencil;
    pass.depth.clear_depth = true; // every pixel is written (compare always)
    pass.depth.clear_stencil = stencil; // the per-bit path ORs bits into zero
    pass.depth.stencil = 0;
    pass.width = width;
    pass.height = height;
    void* const state = !stencil ? depth_write_state : depth_stencil_state;
    scheduler.Record([pass, pipeline, state, stencil_bits, bit_states = stencil_bit_states, src,
                      stencil_view = std::move(stencil_view)](Metal::Encoder& e) mutable {
        void* const stencil_texture = stencil_view.Get();
        if (stencil_view) e.KeepAlive(std::move(stencil_view));
        e.BeginRenderPass(pass);
        SetFullscreenState(e, pipeline, pass.width, pass.height);
        e.SetTexture(Metal::Stage::Fragment, 0, src);
        if (stencil_texture) e.SetTexture(Metal::Stage::Fragment, 1, stencil_texture);
        if (stencil_bits) {
            // Copy depth first (bit 0, write mask 0), then reconstruct all eight stencil bits
            // with fixed-function replacement; discard keeps unset bits clear.
            for (u32 i = 0; i < bit_states.size(); ++i) {
                const u32 bit = i ? 1u << (i - 1) : 0u;
                e.SetBytes(Metal::Stage::Fragment, 0, &bit, sizeof(bit));
                e.SetDepthStencil(bit_states[i]);
                e.SetStencilReference(bit, bit);
                e.Draw(Metal::Primitive::Triangle, 3);
            }
        } else {
            e.SetDepthStencil(state);
            e.Draw(Metal::Primitive::Triangle, 3);
        }
        e.EndRenderPass();
    });
    // The guest pipelines' dynamic state must be emitted again after this pipeline.
    scheduler.GetDynamicState().Invalidate();
}
SceneTargets::Entry& SceneTargets::Get(VideoCore::ImageId id, u32 level) {
    auto& original = *lookup(id, 0);
    const u64 key = Key(original.image_uid, level);
    // A draw has up to six targets, mostly the same as the previous draw's.
    for (const auto& [recent_key, recent_entry] : recent) {
        if (recent_entry && recent_key == key) {
            if (!recent_entry->state.valid) Copy(*recent_entry, original, false);
            return *recent_entry;
        }
    }
    auto& entry = entries[key];
    original.scene_proxy = true;
    if (!entry) {
        entry = std::make_unique<Entry>();
        entry->source = id;
        entry->uid = original.image_uid;
        entry->level = level;
        const auto proxy = ProxySize(original, level);
        entry->width = proxy.width;
        entry->height = proxy.height;
        auto desc = original.backing->desc;
        desc.type = Metal::TextureType::Tex2D;
        desc.width = proxy.width;
        desc.height = proxy.height;
        desc.depth = 1;
        desc.levels = 1;
        desc.layers = 1;
        desc.samples = 1;
        desc.render_target = true;
        const auto label = fmt::format("scene proxy {:#x} level {}", original.info.guest_address,
                                       level);
        entry->image = Metal::CreateTexture(device, desc, label, nullptr);
        ASSERT_MSG(entry->image, "Scene proxy {}x{} of {} not created", proxy.width,
                   proxy.height, vk::to_string(original.info.pixel_format));
        tracked.insert(original.image_uid);
    }
    recent[recent_next++ % recent.size()] = {key, entry.get()};
    if (!entry->state.valid) Copy(*entry, original, false);
    return *entry;
}
void* SceneTargets::View(Entry& e, const VideoCore::Image& original,
                         const VideoCore::ImageViewInfo& info) {
    for (const auto& [key, view] : e.views) if (key == info) return view.Get();
    // Depth proxies are viewed whole (depth and stencil; samplers read the depth). Colour views
    // follow ImageView: the view's format where Metal can make it, with its swizzle.
    const u32 texture_format = original.backing->desc.pixel_format;
    u32 pixel_format = texture_format;
    vk::Format format = original.info.pixel_format;
    if (!original.info.props.is_depth) {
        const u32 requested = Metal::VkToMtl::PixelFormat(info.format);
        const u32 safe = requested ? Metal::SafeViewPixelFormat(texture_format, requested,
                                                                original.backing->desc.format_views)
                                   : texture_format;
        if (safe == requested) {
            pixel_format = requested;
            format = info.format;
        }
    }
    const auto swizzle = original.info.props.is_depth
                             ? Metal::VkToMtl::IdentitySwizzle
                             : Metal::VkToMtl::ComposeSwizzle(info.mapping, format);
    Metal::Object view;
    if (pixel_format == texture_format && swizzle == Metal::VkToMtl::IdentitySwizzle) {
        view = Metal::Object::Retain(e.image.Get());
    } else {
        view = Metal::CreateTextureView(e.image.Get(), pixel_format, Metal::TextureType::Tex2D, 0,
                                        1, 0, 1, {swizzle.r, swizzle.g, swizzle.b, swizzle.a});
        ASSERT_MSG(view, "Scene proxy view of {} as {} not created",
                   vk::to_string(original.info.pixel_format), vk::to_string(info.format));
    }
    void* const result = view.Get();
    e.views.emplace_back(info, std::move(view));
    return result;
}
SceneTargets::Target SceneTargets::Attachment(VideoCore::ImageId id,
                                               const VideoCore::ImageViewInfo& info) {
    auto& original = *lookup(id, 0);
    auto& e = Get(id, info.range.base.level);
    e.state.ProxyWrite();
    return {View(e, original, info)};
}
bool SceneTargets::ProxyCurrent(const VideoCore::Image& image, u32 level, u32 levels) const {
    if (copying || !image.scene_proxy || image.info.props.is_depth || levels != 1) {
        return false;
    }
    const auto it = entries.find(Key(image.image_uid, level));
    return it != entries.end() && it->second->state.valid;
}
void SceneTargets::PrepareSample(const VideoCore::Image&, u32) {}
std::optional<SceneTargets::Target> SceneTargets::SampleProxy(
    const VideoCore::Image& image, const VideoCore::ImageViewInfo& info) {
    const u32 level = info.range.base.level;
    if (!ProxyCurrent(image, level, info.range.extent.levels)) {
        return std::nullopt;
    }
    auto& e = *entries.find(Key(image.image_uid, level))->second;
    return Target{View(e, image, info)};
}
SceneTargets::Target SceneTargets::Read(VideoCore::ImageId id,
                                       const VideoCore::ImageViewInfo& info) {
    auto& original = *lookup(id, 0);
    auto& e = Get(id);
    return {View(e, original, info)};
}
} // namespace Vulkan
