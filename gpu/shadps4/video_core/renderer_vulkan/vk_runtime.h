// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <memory>
#include <unordered_map>

#include "common/types.h"
#include "video_core/buffer_cache/buffer.h"
#include "video_core/buffer_cache/buffer_copy.h"
#include "video_core/renderer_vulkan/vk_staging_buffer_pool.h"
#include "video_core/texture_cache/image.h"
#include "video_core/texture_cache/types.h"

namespace VideoCore {
class BlitHelper;
} // namespace VideoCore

namespace Vulkan {

class GpuProfiler;
class SceneTargets;

class Scheduler;

class Runtime {
public:
    explicit Runtime(Metal::Device& device, Scheduler& scheduler);
    ~Runtime() = default;

    /// bbport: upload hoisting. A copy issued inside a render pass may run ahead of the whole
    /// submission (Scheduler::RecordHoistedCopy) when nothing in this submission has used its
    /// destination yet; then the pass stays open. Tracked per 1 KiB page of registered buffers
    /// (the buffer cache arenas), with the submission's tick.
    void RegisterHoistTarget(const VideoCore::Buffer* buffer, u64 size);
    void NoteSubmissionUse(const VideoCore::Buffer* buffer, u64 offset, u64 size);
    /// Records the copies ahead of the submission and returns true, or false (nothing done)
    /// when that is not safe or not useful; the caller then uses CopyBuffer.
    /// `shader_copy`: the game's copy shader, between guest ranges. A source in a registered
    /// buffer must be unused in this submission too, and the copy runs after the copies hoisted
    /// before it (they may have uploaded its ranges). The caller marks both ranges used.
    bool HoistCopy(const VideoCore::Buffer* src, const VideoCore::Buffer* dst,
                   std::span<const VideoCore::BufferCopy> copies, bool shader_copy = false);

    Scheduler& GetScheduler() {
        return scheduler;
    }

    Metal::Device& GetDevice() {
        return device;
    }

    StagingBufferPool& GetStagingPool() {
        return staging_pool;
    }

    void TickFrame();
    SceneTargets* scene_targets = nullptr;

    /// bbport: `where` names the caller in BB_PASS_STATS.
    void CopyBuffer(const VideoCore::Buffer& src, const VideoCore::Buffer& dst,
                    std::span<const VideoCore::BufferCopy> copies,
                    std::source_location where = std::source_location::current());

    /// Fills `size` bytes at `offset` with the 32-bit `value` (offset and size multiples of 4).
    void FillBuffer(const VideoCore::Buffer& dst, u64 offset, u64 size, u32 value);

    bool Transit(VideoCore::Image* image, vk::ImageLayout dst_layout,
                 vk::PipelineStageFlags2 dst_stage, vk::AccessFlags2 dst_access,
                 std::optional<VideoCore::SubresourceRange> subres_range = {});

    /// BB_GPU_PROFILE: a profiler segment for a transfer of `image` while it lives; the
    /// segment it interrupted continues afterwards.
    class TransferMark {
    public:
        TransferMark(Runtime& runtime, const char* what, const VideoCore::Image& image);
        ~TransferMark();
        TransferMark(const TransferMark&) = delete;
        TransferMark& operator=(const TransferMark&) = delete;

    private:
        GpuProfiler* profiler = nullptr;
        u64 resume = 0;
    };
    void UploadImage(VideoCore::Image* dst, const VideoCore::Buffer* src,
                     std::span<const vk::BufferImageCopy> upload_copies);
    void DownloadImage(VideoCore::Image* src, const VideoCore::Buffer* dst,
                       std::span<const vk::BufferImageCopy> download_copies);

    void CopyImage(VideoCore::Image* src, VideoCore::Image* dst);
    void CopyImageWithBuffer(VideoCore::Image* src, VideoCore::Image* dst,
                             const VideoCore::Buffer* buffer, u64 offset);
    void CopyMip(VideoCore::Image* src, VideoCore::Image* dst, u32 mip, u32 slice);

    void CopyColorAndDepth(VideoCore::Image* src, VideoCore::Image* dst);

    void CopyDepthStencil(VideoCore::Image* src, VideoCore::Image* dst,
                          const VideoCore::SubresourceRange& sub_range);

    void ResolveImage(VideoCore::Image* src, VideoCore::Image* dst,
                      const VideoCore::SubresourceRange& src_range,
                      const VideoCore::SubresourceRange& dst_range);
    void ClearImage(VideoCore::Image* dst, const VideoCore::SubresourceRange& range,
                    const vk::ClearValue& clear_value);

    void SetBackingSamples(VideoCore::Image* image, u32 num_samples, bool copy_backing = true);

    /// Metal hazard-tracks buffers: only the hoisting bookkeeping remains. The stage and access
    /// arguments are unused (callers lose them in M3).
    void AccessBuffer(const VideoCore::Buffer* handle, u64 offset, u64 size,
                      vk::PipelineStageFlags2 src_stage = {}, vk::AccessFlags2 src_access = {});

    /// No-op: there are no barriers to flush (callers are removed in M3).
    void FlushBarriers(std::source_location where = std::source_location::current()) {}

    /// bbport: runs before this thread changes image state (layouts, pending image
    /// barriers): the rasterizer joins its texture binding helper there (BindHelper).
    void SetImageAccessHook(void (*hook)(void*), void* context) {
        image_access_hook = hook;
        image_access_context = context;
    }
    void BeforeImageAccess() {
        if (image_access_hook) {
            image_access_hook(image_access_context);
        }
    }

private:
    struct HoistPages {
        std::unique_ptr<u16[]> tick; ///< low 16 bits of the submission tick that used the page
        u64 pages;
    };
    std::unordered_map<const VideoCore::Buffer*, HoistPages> hoist_targets;

    void (*image_access_hook)(void*) = nullptr;
    void* image_access_context = nullptr;
    Metal::Device& device;
    Scheduler& scheduler;
    std::unique_ptr<VideoCore::BlitHelper> blit_helper;
    StagingBufferPool staging_pool;
    VideoCore::Image::Barriers image_barriers;
};

} // namespace Vulkan
