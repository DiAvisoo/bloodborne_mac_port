// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: Metal::Encoder records the renderer's commands into Metal command buffers (Metal-native
// renderer, spec component 2). It is the state machine of mtl_frame.mm (metal-backend) with
// native arguments: a render pass opens its Metal encoder at the first draw (a pass with no draws,
// clears or resolves costs nothing), blits and dispatches inside a pass end the render encoder
// and the pass reopens with Load actions, and state reaches the Metal encoder only when it changed.
// One thread uses an Encoder at a time.

#pragma once

#include <array>
#include <memory>
#include <string_view>
#include "common/types.h"
#include "video_core/renderer_metal/mtl_object.h"

namespace Metal {

class Device;

enum class Stage : u8 { Vertex, Fragment, Compute };
enum class Primitive : u8 { Point, Line, LineStrip, Triangle, TriangleStrip };
enum class IndexType : u8 { U16, U32 };
enum class CullMode : u8 { None, Front, Back };
enum class BlitAspect : u8 { All, Depth, Stencil };
enum class Winding : u8 { Clockwise, CounterClockwise };
enum class FillMode : u8 { Fill, Lines };
enum class DepthClip : u8 { Clip, Clamp };
/// How render encoders use a buffer reached by GPU address (Encoder::DeclareIndirect).
enum class IndirectUse : u8 {
    Read,        ///< read by vertex and fragment shaders (guest memory, page table)
    VertexWrite, ///< read and written by vertex shaders (object motion history)
};

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
    u32 depth_plane{};         ///< 3D targets
    void* resolve_texture{};   ///< MSAA resolve target (StoreAndMultisampleResolve)
    u32 resolve_level{}, resolve_slice{};
};

struct DepthAttachment {
    void* texture{}; ///< id<MTLTexture>; null: no depth
    u32 level{}, slice{};
    bool has_stencil{}; ///< the format has a stencil aspect (bound as the stencil attachment too)
    bool clear_depth{}, clear_stencil{};
    double depth{1.0};
    u32 stencil{};
    bool stencil_only{}; ///< Stencil8-type formats: bound as the stencil attachment only
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
    void SetDepthStencil(void* state);      ///< id<MTLDepthStencilState>; null: no depth/stencil test
    void SetViewport(const Viewport& viewport);
    void SetScissor(const Scissor& scissor);
    void SetCullMode(CullMode mode);
    void SetFrontFace(Winding winding);
    void SetDepthBias(float constant, float slope, float clamp);
    void SetStencilReference(u32 front, u32 back);
    void SetBlendColor(float r, float g, float b, float a);
    void SetFillMode(FillMode mode);
    void SetDepthClip(DepthClip mode);
    /// Tessellation factors for DrawPatches (setTessellationFactorBuffer:offset:instanceStride:).
    void SetTessellationFactorBuffer(BufferRef factors, u32 instance_stride);

    // Resources by stage and slot: render stages apply at the next draw, compute at the next
    // dispatch. Slots: buffers 0-30, textures 0-127, samplers 0-15.
    // Bindings are unretained pointers that stay bound until replaced, into new Metal encoders
    // (which get every bound slot again) but not across Submit: a new command buffer starts
    // unbound, as in Vulkan, so objects may be freed once their command buffer was submitted.
    void SetBuffer(Stage stage, u32 slot, BufferRef buffer);
    /// Copies `size` bytes now. Up to 4 KiB they go inline (setBytes); more go through a
    /// temporary shared buffer kept until the command buffer completes.
    void SetBytes(Stage stage, u32 slot, const void* data, u32 size);
    /// A vertex buffer with an attribute stride (the pipeline's vertex layout uses
    /// MTLBufferLayoutStrideDynamic): setVertexBuffer:offset:attributeStride:. Occupies the
    /// vertex-stage buffer `slot`, like SetBuffer(Stage::Vertex, ...), and is replaced by it.
    /// A stride of 0 reads element 0 for every vertex (Vulkan's stride 0).
    void SetVertexBuffer(u32 slot, BufferRef buffer, u32 stride);
    void SetTexture(Stage stage, u32 slot, void* texture); ///< id<MTLTexture>
    void SetSampler(Stage stage, u32 slot, void* sampler); ///< id<MTLSamplerState>

    void Draw(Primitive primitive, u32 vertex_count, u32 instance_count = 1, u32 first_vertex = 0,
              u32 first_instance = 0);
    void DrawIndexed(Primitive primitive, u32 index_count, IndexType type, BufferRef indices,
                     u32 instance_count = 1, s32 base_vertex = 0, u32 first_instance = 0);
    /// Draws with GPU-side arguments: MTLDrawPrimitivesIndirectArguments at `args`.
    void DrawIndirect(Primitive primitive, BufferRef args);
    /// MTLDrawIndexedPrimitivesIndirectArguments at `args`.
    void DrawIndexedIndirect(Primitive primitive, IndexType type, BufferRef indices, BufferRef args);
    /// Multi-draw indirect with a GPU count: clamps a copy of the `max_count` argument records at `args`
    /// (stride `stride` bytes) so records at index >= *count draw nothing (vertex/index count 0),
    /// with a compute pass before the render pass reopens, then issues max_count indirect draws.
    /// The clamp works on a scratch copy of the records (`args` is only read); the caller's
    /// compute state is untouched.
    void DrawIndirectCount(Primitive primitive, BufferRef args, u32 stride, BufferRef count,
                           u32 max_count, bool indexed, IndexType type = IndexType::U16,
                           BufferRef indices = {});
    /// Patch draw (drawPatches:...): `patch_index_buffer` may be empty. Needs a tessellation
    /// render pipeline and SetTessellationFactorBuffer.
    void DrawPatches(u32 control_points, u32 patch_start, u32 patch_count,
                     BufferRef patch_index_buffer, u32 instance_count, u32 base_instance);

    // Compute.
    void SetComputePipeline(void* pipeline); ///< id<MTLComputePipelineState>
    void Dispatch(Size3 groups, Size3 threads_per_group);
    /// Threadgroup counts (3 u32) are read from `args` on the GPU.
    void DispatchIndirect(BufferRef args, Size3 threads_per_group);
    /// The open compute encoder (id<MTLComputeCommandEncoder>, opened if needed) for kernels
    /// encoded outside the Encoder (tessellation and geometry emulation). The caller's pipeline
    /// and slots are re-applied by its next Dispatch.
    void* ComputeEncoder();

    // Blits. Inside a render pass they end the render encoder; the pass continues afterwards.
    void CopyBuffer(BufferRef src, BufferRef dst, u64 size);
    /// Fills `size` bytes with the 32-bit `pattern`: a blit when all four bytes are equal, else a
    /// compute kernel over whole words (an unaligned size is rounded down, an unaligned offset
    /// skips the fill; both log once). Size 0 does nothing (also for CopyBuffer).
    void FillBuffer(BufferRef dst, u64 size, u32 pattern);
    void CopyBufferToTexture(BufferRef src, u32 bytes_per_row, u32 bytes_per_image, void* texture,
                             u32 level, u32 slice, Origin3 origin, Size3 size,
                             BlitAspect aspect = BlitAspect::All);
    void CopyTextureToBuffer(void* texture, u32 level, u32 slice, Origin3 origin, Size3 size,
                             BufferRef dst, u32 bytes_per_row, u32 bytes_per_image,
                             BlitAspect aspect = BlitAspect::All);
    void CopyTexture(void* src, u32 src_level, u32 src_slice, Origin3 src_origin, void* dst,
                     u32 dst_level, u32 dst_slice, Origin3 dst_origin, Size3 size);

    /// Buffers shaders reach by GPU address, declared on every Metal encoder this Encoder opens
    /// from now on (useResource) so they are resident and Metal's hazard tracking orders the
    /// encoders that use them (the buffers must be tracked: the default for device buffers).
    /// Compute encoders: read/write. Render encoders: read (vertex and fragment stages), or
    /// with VertexWrite read/write at the vertex stage. Declaring a buffer again changes its use.
    void DeclareIndirect(void* buffer, IndirectUse use = IndirectUse::Read);
    /// Stops declaring `buffer` on encoders opened from now on (before it is released; the
    /// open encoder keeps it until it ends).
    void RemoveIndirect(void* buffer);
    /// True while any slot, pipeline or render state is set (Submit clears all of it).
    [[nodiscard]] bool HasBindings() const noexcept;

    /// Makes the GPU wait, before the commands recorded after this call, until the device has
    /// signalled `device_tick` (Device::Commit; Timeline::DeviceTickOf for another submitter's
    /// logical tick). Ends the open Metal encoder (a pass in progress reopens with Load).
    /// 0: no-op (already complete). `device_tick` must already be committed (asserted): the queue
    /// starts command buffers in commit order, so waiting for a later commit never ends.
    void WaitForDeviceTick(u64 device_tick);
    /// Debug groups: on the command buffer when no encoder is open, else on the open encoder
    /// (closed with the encoder if still open then). Pops must match pushes.
    void PushDebugGroup(std::string_view label);
    void PopDebugGroup();
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
    /// Ends encoding and commits with no tick (Device::CommitUntracked); keep-alive objects are
    /// released when it completes. For the scheduler's hoisted-upload command buffer.
    void SubmitUntracked();

private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

} // namespace Metal
