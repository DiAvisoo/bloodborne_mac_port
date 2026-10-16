// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once
#include <array>
#include <memory>
#include <optional>
#include <functional>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "video_core/renderer_vulkan/scene_resolution.h"
#include "video_core/renderer_metal/mtl_object.h"
#include "video_core/texture_cache/image.h"

namespace VideoCore { class TextureCache; }
namespace Metal { class Device; }
namespace Vulkan {
class Runtime;
class Scheduler;

// Reduced raster targets, with native-size images retained for unmodified guest compute,
// integer texture loads, copies and CPU readbacks. All native access passes Runtime::AccessImage.
class SceneTargets {
public:
    SceneTargets(Metal::Device&, Scheduler&, Runtime&, VideoCore::TextureCache&);
    using Lookup = std::function<VideoCore::Image*(VideoCore::ImageId, u64)>;
    SceneTargets(Metal::Device&, Scheduler&, Runtime&, Lookup);
    ~SceneTargets();
    bool SetSize(SceneResolution::Size);
    SceneResolution::Size Size() const { return size; }
    bool Reduced() const { return size != SceneResolution::Size{}; }
    /// A reduced proxy can stand in for the image: a 1920x1080 scene target or, scaled by
    /// the same factor, a half-resolution (960x540) one (toggle SceneHalfRes).
    bool Eligible(const VideoCore::Image&) const;
    /// Eligible at the full scene size (1920x1080): the upscaler's inputs, the scene start.
    bool EligibleScene(const VideoCore::Image&) const;
    /// The proxy size of an eligible image: the scene size divided like the native size.
    SceneResolution::Size ProxySize(const VideoCore::Image&, u32 level = 0) const;
    struct Target {
        void* texture{}; ///< id<MTLTexture>: the proxy's view for the requested range
    };
    Target Attachment(VideoCore::ImageId, const VideoCore::ImageViewInfo&);
    /// The proxy (level 0) for reading or writing outside a render pass (compute, copies).
    Target Read(VideoCore::ImageId, const VideoCore::ImageViewInfo&);
    /// Before commands on the native image: resolves dirty proxies into it; `write` makes them
    /// stale (the native image becomes current).
    void NativeAccess(VideoCore::Image&, bool write);
    /// The proxy of `image` for sampling when it holds the current content (else nullopt: the
    /// native image is current). Mipmapped targets have one proxy per level: views of one level
    /// only.
    std::optional<Target> SampleProxy(const VideoCore::Image& image,
                                      const VideoCore::ImageViewInfo& info);
    /// Whether SampleProxy would return the proxy of `image` for these levels.
    [[nodiscard]] bool ProxyCurrent(const VideoCore::Image& image, u32 level = 0,
                                    u32 levels = 1) const;
    /// SampleProxy for a view obtained before (valid while Generation() is unchanged). Nothing to
    /// do on Metal (no layouts; Metal tracks the proxies' hazards): kept for the callers.
    void PrepareSample(const VideoCore::Image& image, u32 level = 0);
    /// Changes when proxies and their views are destroyed (SetSize).
    [[nodiscard]] u64 Generation() const noexcept {
        return generation;
    }
    /// Whether NativeAccess has work for this image (a reduced-size proxy exists).
    [[nodiscard]] bool Tracks(u64 image_uid) const {
        return !copying && tracked.contains(image_uid);
    }
    void ResolveAll();
    bool debug = false; ///< BB_SCENE_DEBUG frame: print resolves and fills
private:
    struct Entry {
        VideoCore::ImageId source{};
        u64 uid = 0;
        u32 level = 0; ///< mip level of the native image this proxy stands for
        u32 width = 0, height = 0; ///< proxy size
        Metal::Object image; ///< id<MTLTexture> of the proxy
        std::vector<std::pair<VideoCore::ImageViewInfo, Metal::Object>> views;
        SceneResolution::Coherence state;
    };
    struct PipelineKey {
        const char* fragment{};
        u32 color_format{}, depth_format{}, stencil_format{};
        bool operator==(const PipelineKey&) const = default;
    };
    static constexpr u64 Key(u64 uid, u32 level) {
        return uid << 4 | level;
    }
    Entry& Get(VideoCore::ImageId, u32 level = 0);
    void* View(Entry&, const VideoCore::Image&, const VideoCore::ImageViewInfo&);
    void Copy(Entry&, VideoCore::Image&, bool to_native);
    /// Colour images: copied by CopyTexture at the same size, else by a nearest-texel draw.
    bool Blittable(const VideoCore::Image&) const;
    /// Depth (and stencil) images: resampled by a fullscreen draw writing depth and, with
    /// stencil export, the stencil value (or one draw per stencil bit, BB_SCENE_STENCIL_BITS=1).
    bool ShaderResampled(const VideoCore::Image&) const;
    void Scale(void* src, u32 src_level, void* dst, u32 dst_level, u32 width, u32 height,
               const VideoCore::Image& original);
    void Resample(void* src, void* dst, const VideoCore::Image& original, u32 width, u32 height);
    void* Pipeline(const PipelineKey&);
    Metal::Device& device;
    Scheduler& scheduler;
    Runtime& runtime;
    Lookup lookup;
    SceneResolution::Size size;
    std::unordered_map<u64, std::unique_ptr<Entry>> entries; ///< by Key(uid, level)
    std::unordered_set<u64> tracked; ///< uids with proxies
    bool copying = false;
    bool force_stencil_bits = false; ///< test the portable resampler on any driver
    u64 generation = 0;
    std::array<std::pair<u64, Entry*>, 8> recent{}; ///< last entries by Key
    u32 recent_next = 0;
    Metal::Object library; ///< Metal::BlitShaders plus the nearest colour scale
    std::vector<std::pair<PipelineKey, Metal::Object>> pipelines;
    void* no_depth_state{};      ///< id<MTLDepthStencilState> (owned by the scheduler's cache)
    void* depth_write_state{};   ///< depth always/write, no stencil
    void* depth_stencil_state{}; ///< plus stencil replace (exported value), write mask 0xff
    std::array<void*, 9> stencil_bit_states{}; ///< write masks 0, 1, 2, 4, ..., 128
};
} // namespace Vulkan
