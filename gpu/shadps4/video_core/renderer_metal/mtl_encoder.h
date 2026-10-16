// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: Metal::Encoder records the renderer's commands into Metal command buffers (Metal-native
// renderer, spec component 2). It is the state machine of mtl_frame.mm (metal-backend) with
// native arguments: a render pass opens its Metal encoder at the first draw (a pass with neither
// draws nor clears costs nothing), blits and dispatches inside a pass end the render encoder and
// the pass reopens with Load actions, and state reaches the Metal encoder only when it changed.
// One thread uses an Encoder at a time.

#pragma once

#include <array>
#include <memory>
#include "common/types.h"
#include "video_core/renderer_metal/mtl_object.h"

namespace Metal {

class Device;

enum class Stage : u8 { Vertex, Fragment, Compute };
enum class Primitive : u8 { Point, Line, LineStrip, Triangle, TriangleStrip };
enum class IndexType : u8 { U16, U32 };
enum class CullMode : u8 { None, Front, Back };
enum class Winding : u8 { Clockwise, CounterClockwise };

struct Origin3 {
    u32 x{}, y{}, z{};
};
struct Size3 {
    u32 width{1}, height{1}, depth{1};
};

struct ColorAttachment {
    void* texture{}; ///< id<MTLTexture>; null: slot unused
    u32 level{}, slice{};
    bool clear{};
    std::array<double, 4> clear_color{};
};

struct DepthAttachment {
    void* texture{}; ///< id<MTLTexture>; null: no depth
    u32 level{}, slice{};
    bool has_stencil{}; ///< the format has a stencil aspect (bound as the stencil attachment too)
    bool clear_depth{}, clear_stencil{};
    double depth{1.0};
    u32 stencil{};
};

struct RenderPassDesc {
    static constexpr u32 MaxColors = 8;
    std::array<ColorAttachment, MaxColors> colors{};
    DepthAttachment depth{};
    u32 width{}, height{}; ///< render area, used when there are no attachments
};

struct Viewport {
    float x{}, y{}, width{}, height{}, min_depth{}, max_depth{1.0f};
};
struct Scissor {
    u32 x{}, y{}, width{}, height{};
};

class Encoder {
public:
    explicit Encoder(Device& device);
    ~Encoder();

    Encoder(const Encoder&) = delete;
    Encoder& operator=(const Encoder&) = delete;

    // Render passes. A pass must be ended before Submit.
    void BeginRenderPass(const RenderPassDesc& desc);
    void EndRenderPass();
    [[nodiscard]] bool InRenderPass() const noexcept;

    // Render state: kept across passes, applied at the next draw.
    void SetRenderPipeline(void* pipeline); ///< id<MTLRenderPipelineState>
    void SetDepthStencil(void* state);      ///< id<MTLDepthStencilState>
    void SetViewport(const Viewport& viewport);
    void SetScissor(const Scissor& scissor);
    void SetCullMode(CullMode mode);
    void SetFrontFace(Winding winding);
    void SetDepthBias(float constant, float slope, float clamp);
    void SetStencilReference(u32 front, u32 back);
    void SetBlendColor(float r, float g, float b, float a);

    // Resources by stage and slot: render stages apply at the next draw, compute at the next
    // dispatch. Slots: buffers 0-30, textures 0-31, samplers 0-15.
    void SetBuffer(Stage stage, u32 slot, BufferRef buffer);
    /// Copies `size` bytes now. Up to 4 KiB they go inline (setBytes); more go through a
    /// temporary shared buffer kept until the command buffer completes.
    void SetBytes(Stage stage, u32 slot, const void* data, u32 size);
    void SetTexture(Stage stage, u32 slot, void* texture); ///< id<MTLTexture>
    void SetSampler(Stage stage, u32 slot, void* sampler); ///< id<MTLSamplerState>

    void Draw(Primitive primitive, u32 vertex_count, u32 instance_count = 1, u32 first_vertex = 0,
              u32 first_instance = 0);
    void DrawIndexed(Primitive primitive, u32 index_count, IndexType type, BufferRef indices,
                     u32 instance_count = 1, s32 base_vertex = 0, u32 first_instance = 0);

    // Compute.
    void SetComputePipeline(void* pipeline); ///< id<MTLComputePipelineState>
    void Dispatch(Size3 groups, Size3 threads_per_group);

    // Blits. Inside a render pass they end the render encoder; the pass continues afterwards.
    void CopyBuffer(BufferRef src, BufferRef dst, u64 size);
    /// Fills with the low byte of `pattern` (as mtl_frame did; logged once if its bytes differ).
    void FillBuffer(BufferRef dst, u64 size, u32 pattern);
    void CopyBufferToTexture(BufferRef src, u32 bytes_per_row, u32 bytes_per_image, void* texture,
                             u32 level, u32 slice, Origin3 origin, Size3 size);
    void CopyTextureToBuffer(void* texture, u32 level, u32 slice, Origin3 origin, Size3 size,
                             BufferRef dst, u32 bytes_per_row, u32 bytes_per_image);
    void CopyTexture(void* src, u32 src_level, u32 src_slice, Origin3 src_origin, void* dst,
                     u32 dst_level, u32 dst_slice, Origin3 dst_origin, Size3 size);

    /// Keeps `object` alive until the current command buffer completes.
    void KeepAlive(Object object);
    /// Ends the open render, compute or blit encoder (a pass in progress reopens with Load).
    void EndEncoding();
    /// The command buffer being recorded (id<MTLCommandBuffer>, created if needed), for work
    /// encoded outside the Encoder (MetalFX). Call EndEncoding() first.
    void* CommandBuffer();
    /// Ends encoding, commits with the device's next tick and returns it. Starts a new command
    /// buffer on the next command. Asserts that no render pass is in progress.
    u64 Submit();

private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

} // namespace Metal
