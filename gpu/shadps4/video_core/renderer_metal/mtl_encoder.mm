// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: Metal::Encoder (Metal-native renderer, spec component 2).

#import <Metal/Metal.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cstring>
#include <vector>
#include "common/assert.h"
#include "common/logging/log.h"
#include "video_core/renderer_metal/mtl_device.h"
#include "video_core/renderer_metal/mtl_encoder.h"

namespace Metal {

namespace {

constexpr u32 MaxBuffers = 31, MaxTextures = 32, MaxSamplers = 16;
constexpr u32 MaxInlineBytes = 4096;

/// One stage's resources: what the caller set, and which slots changed since the Metal encoder
/// last saw them (bit per slot). A new Metal encoder starts empty, so it gets every used slot.
struct StageState {
    struct Slot {
        void* buffer{};
        u64 offset{};
        std::vector<u8> bytes; ///< non-empty: inline bytes instead of the buffer
    };
    std::array<Slot, MaxBuffers> buffers{};
    std::array<void*, MaxTextures> textures{};
    std::array<void*, MaxSamplers> samplers{};
    u32 buffers_dirty{}, buffers_used{};
    u32 textures_dirty{}, textures_used{};
    u32 samplers_dirty{}, samplers_used{};

    void Reapply() {
        buffers_dirty = buffers_used;
        textures_dirty = textures_used;
        samplers_dirty = samplers_used;
    }
};

/// Calls `f(slot)` for every set bit of `mask`.
template <typename F>
void ForEachBit(u32 mask, F&& f) {
    while (mask) {
        const u32 slot = u32(std::countr_zero(mask));
        f(slot);
        mask &= mask - 1;
    }
}

MTLPrimitiveType ToMtl(Primitive primitive) {
    switch (primitive) {
    case Primitive::Point:
        return MTLPrimitiveTypePoint;
    case Primitive::Line:
        return MTLPrimitiveTypeLine;
    case Primitive::LineStrip:
        return MTLPrimitiveTypeLineStrip;
    case Primitive::Triangle:
        return MTLPrimitiveTypeTriangle;
    case Primitive::TriangleStrip:
        return MTLPrimitiveTypeTriangleStrip;
    }
    UNREACHABLE();
}

MTLCullMode ToMtl(CullMode mode) {
    switch (mode) {
    case CullMode::None:
        return MTLCullModeNone;
    case CullMode::Front:
        return MTLCullModeFront;
    case CullMode::Back:
        return MTLCullModeBack;
    }
    UNREACHABLE();
}

MTLBlitOption BlitOptions(BlitAspect aspect) {
    switch (aspect) {
    case BlitAspect::All:
        return MTLBlitOptionNone;
    case BlitAspect::Depth:
        return MTLBlitOptionDepthFromDepthStencil;
    case BlitAspect::Stencil:
        return MTLBlitOptionStencilFromDepthStencil;
    }
    UNREACHABLE();
}

} // namespace

struct Encoder::Impl {
    explicit Impl(Device& device_)
        : device{device_}, dev{(__bridge id<MTLDevice>)device_.NativeDevice()},
          queue{(__bridge id<MTLCommandQueue>)device_.NativeQueue()} {}

    Device& device;
    id<MTLDevice> dev;
    id<MTLCommandQueue> queue;
    id<MTLCommandBuffer> cmd;
    id<MTLRenderCommandEncoder> render;
    id<MTLComputeCommandEncoder> compute;
    id<MTLBlitCommandEncoder> blit;
    std::vector<Object> keep;

    // Render pass (Task 5).
    bool in_pass{}, pass_opened{}, pass_clears{};
    bool pass_resolves{}; ///< a colour attachment resolves at the end of the pass
    MTLRenderPassDescriptor* pass_first; ///< as begun, with its clears
    MTLRenderPassDescriptor* pass_load;  ///< the same with Load actions (reopening)
    u32 target_width{}, target_height{}; ///< of the pass (the scissor is clamped to it)

    // Render state (Task 5): what the caller set, and what changed since the encoder saw it.
    enum RenderBit : u32 {
        BitPipeline = 1u << 0,
        BitDepthStencil = 1u << 1,
        BitViewport = 1u << 2,
        BitScissor = 1u << 3,
        BitCull = 1u << 4,
        BitWinding = 1u << 5,
        BitBias = 1u << 6,
        BitStencilRef = 1u << 7,
        BitBlend = 1u << 8,
    };
    u32 render_dirty{}, render_used{};
    void* render_pipeline{};
    void* depth_stencil{};
    Viewport viewport{};
    Scissor scissor{};
    CullMode cull{CullMode::None};
    Winding winding{Winding::CounterClockwise};
    float bias[3]{}; ///< constant, slope, clamp
    u32 stencil_front{}, stencil_back{};
    float blend[4]{};
    StageState vertex, fragment;

    // Compute state (Task 4).
    void* compute_pipeline{};
    bool compute_pipeline_dirty{};
    StageState compute_stage;

    std::vector<void*> indirect; ///< declared on every encoder opened (survives Submit)
    id<MTLComputePipelineState> fill_u32;

    void DeclareOn(id<MTLRenderCommandEncoder> e) {
        for (void* b : indirect) {
            [e useResource:(__bridge id<MTLBuffer>)b
                     usage:MTLResourceUsageRead
                    stages:MTLRenderStageVertex | MTLRenderStageFragment];
        }
    }
    void DeclareOn(id<MTLComputeCommandEncoder> e) {
        for (void* b : indirect) {
            [e useResource:(__bridge id<MTLBuffer>)b
                     usage:MTLResourceUsageRead | MTLResourceUsageWrite];
        }
    }

    /// A new command buffer starts unbound (as in Vulkan).
    void ResetBindings() {
        vertex = {};
        fragment = {};
        compute_stage = {};
        render_pipeline = depth_stencil = compute_pipeline = nullptr;
        render_used = render_dirty = 0;
        compute_pipeline_dirty = false;
        viewport = {};
        scissor = {};
        cull = CullMode::None;
        winding = Winding::CounterClockwise;
        bias[0] = bias[1] = bias[2] = 0;
        stencil_front = stencil_back = 0;
        blend[0] = blend[1] = blend[2] = blend[3] = 0;
    }

    id<MTLComputePipelineState> FillPipeline() {
        if (!fill_u32) {
            @autoreleasepool {
                NSError* error = nil;
                id<MTLLibrary> lib = [dev newLibraryWithSource:@R"(
#include <metal_stdlib>
using namespace metal;
kernel void fill_u32(device uint* dst [[buffer(0)]], constant uint2& args [[buffer(1)]],
                     uint i [[thread_position_in_grid]]) {
    if (i < args.y) dst[i] = args.x;
}
)" options:nil error:&error];
                ASSERT_MSG(lib, "fill_u32 kernel: {}",
                           error ? error.localizedDescription.UTF8String : "unknown");
                fill_u32 = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"fill_u32"]
                                                              error:&error];
                ASSERT_MSG(fill_u32, "fill_u32 pipeline: {}",
                           error ? error.localizedDescription.UTF8String : "unknown");
            }
        }
        return fill_u32;
    }

    /// Ends encoding and commits; `tracked` takes a tick. Returns 0 for an untracked commit.
    u64 Finish(bool tracked) {
        ASSERT_MSG(!in_pass, "Submit inside a render pass");
        EndRender();
        EndCompute();
        EndBlit();
        id<MTLCommandBuffer> c = Cmd();
        auto* held = new std::vector<Object>(std::move(keep));
        keep.clear();
        [c addCompletedHandler:^(id<MTLCommandBuffer>) {
            delete held;
        }];
        u64 tick = 0;
        if (tracked) {
            tick = device.Commit((__bridge void*)c);
        } else {
            device.CommitUntracked((__bridge void*)c);
        }
        cmd = nil;
        ResetBindings();
        return tick;
    }

    id<MTLCommandBuffer> Cmd() {
        if (!cmd) {
            @autoreleasepool {
                cmd = [queue commandBufferWithUnretainedReferences];
            }
        }
        return cmd;
    }

    void EndRender() {
        if (render) {
            [render endEncoding];
            render = nil;
        }
    }
    void EndCompute() {
        if (compute) {
            [compute endEncoding];
            compute = nil;
        }
    }
    void EndBlit() {
        if (blit) {
            [blit endEncoding];
            blit = nil;
        }
    }

    /// Work recorded inside a pass that has not opened yet must come after the pass's clears.
    void FlushPendingClear() {
        if (in_pass && !pass_opened && pass_clears) {
            Render();
        }
    }

    id<MTLBlitCommandEncoder> Blit() {
        if (!blit) {
            FlushPendingClear();
            EndRender();
            EndCompute();
            @autoreleasepool {
                blit = [Cmd() blitCommandEncoder];
            }
        }
        return blit;
    }

    id<MTLComputeCommandEncoder> Compute() {
        if (!compute) {
            FlushPendingClear();
            EndRender();
            EndBlit();
            @autoreleasepool {
                compute = [Cmd() computeCommandEncoder];
            }
            DeclareOn(compute);
            compute_stage.Reapply();
            compute_pipeline_dirty = compute_pipeline != nullptr;
        }
        return compute;
    }

    void ApplyCompute() {
        id<MTLComputeCommandEncoder> e = Compute();
        if (compute_pipeline_dirty) {
            [e setComputePipelineState:(__bridge id<MTLComputePipelineState>)compute_pipeline];
            compute_pipeline_dirty = false;
        }
        StageState& s = compute_stage;
        ForEachBit(s.buffers_dirty, [&](u32 slot) {
            const StageState::Slot& b = s.buffers[slot];
            if (!b.bytes.empty()) {
                [e setBytes:b.bytes.data() length:b.bytes.size() atIndex:slot];
            } else {
                [e setBuffer:(__bridge id<MTLBuffer>)b.buffer offset:b.offset atIndex:slot];
            }
        });
        ForEachBit(s.textures_dirty, [&](u32 slot) {
            [e setTexture:(__bridge id<MTLTexture>)s.textures[slot] atIndex:slot];
        });
        ForEachBit(s.samplers_dirty, [&](u32 slot) {
            [e setSamplerState:(__bridge id<MTLSamplerState>)s.samplers[slot] atIndex:slot];
        });
        s.buffers_dirty = s.textures_dirty = s.samplers_dirty = 0;
    }

    id<MTLRenderCommandEncoder> Render() {
        if (!render) {
            ASSERT_MSG(in_pass, "Draw outside a render pass");
            EndCompute();
            EndBlit();
            @autoreleasepool {
                render = [Cmd() renderCommandEncoderWithDescriptor:pass_opened ? pass_load
                                                                               : pass_first];
            }
            pass_opened = true;
            DeclareOn(render);
            render_dirty = render_used; // incl. the scissor, clamped to this pass's target
            vertex.Reapply();
            fragment.Reapply();
        }
        return render;
    }

    void ApplyStage(id<MTLRenderCommandEncoder> e, StageState& s, bool is_vertex) {
        ForEachBit(s.buffers_dirty, [&](u32 slot) {
            const StageState::Slot& b = s.buffers[slot];
            if (!b.bytes.empty()) {
                if (is_vertex) {
                    [e setVertexBytes:b.bytes.data() length:b.bytes.size() atIndex:slot];
                } else {
                    [e setFragmentBytes:b.bytes.data() length:b.bytes.size() atIndex:slot];
                }
            } else if (is_vertex) {
                [e setVertexBuffer:(__bridge id<MTLBuffer>)b.buffer offset:b.offset atIndex:slot];
            } else {
                [e setFragmentBuffer:(__bridge id<MTLBuffer>)b.buffer offset:b.offset atIndex:slot];
            }
        });
        ForEachBit(s.textures_dirty, [&](u32 slot) {
            id<MTLTexture> t = (__bridge id<MTLTexture>)s.textures[slot];
            if (is_vertex) {
                [e setVertexTexture:t atIndex:slot];
            } else {
                [e setFragmentTexture:t atIndex:slot];
            }
        });
        ForEachBit(s.samplers_dirty, [&](u32 slot) {
            id<MTLSamplerState> st = (__bridge id<MTLSamplerState>)s.samplers[slot];
            if (is_vertex) {
                [e setVertexSamplerState:st atIndex:slot];
            } else {
                [e setFragmentSamplerState:st atIndex:slot];
            }
        });
        s.buffers_dirty = s.textures_dirty = s.samplers_dirty = 0;
    }

    void ApplyRender() {
        id<MTLRenderCommandEncoder> e = Render();
        const u32 d = render_dirty;
        if (d & BitPipeline) {
            [e setRenderPipelineState:(__bridge id<MTLRenderPipelineState>)render_pipeline];
        }
        if (d & BitDepthStencil) {
            [e setDepthStencilState:(__bridge id<MTLDepthStencilState>)depth_stencil];
        }
        if (d & BitViewport) {
            [e setViewport:(MTLViewport){viewport.x, viewport.y, viewport.width, viewport.height,
                                         viewport.min_depth, viewport.max_depth}];
        }
        if (d & BitScissor) {
            // Metal requires the scissor inside the target (Vulkan does not); state carried from
            // a bigger pass is common.
            const u32 x = std::min(scissor.x, target_width);
            const u32 y = std::min(scissor.y, target_height);
            [e setScissorRect:(MTLScissorRect){x, y, std::min(scissor.width, target_width - x),
                                               std::min(scissor.height, target_height - y)}];
        }
        if (d & BitCull) {
            [e setCullMode:ToMtl(cull)];
        }
        if (d & BitWinding) {
            [e setFrontFacingWinding:winding == Winding::Clockwise ? MTLWindingClockwise
                                                                    : MTLWindingCounterClockwise];
        }
        if (d & BitBias) {
            [e setDepthBias:bias[0] slopeScale:bias[1] clamp:bias[2]];
        }
        if (d & BitStencilRef) {
            [e setStencilFrontReferenceValue:stencil_front backReferenceValue:stencil_back];
        }
        if (d & BitBlend) {
            [e setBlendColorRed:blend[0] green:blend[1] blue:blend[2] alpha:blend[3]];
        }
        render_dirty = 0;
        ApplyStage(e, vertex, true);
        ApplyStage(e, fragment, false);
    }

    void MarkRender(u32 bit) {
        render_dirty |= bit;
        render_used |= bit;
    }

    StageState& StateOf(Stage stage) {
        switch (stage) {
        case Stage::Vertex:
            return vertex;
        case Stage::Fragment:
            return fragment;
        case Stage::Compute:
            return compute_stage;
        }
        UNREACHABLE();
    }
};

Encoder::Encoder(Device& device) : impl{std::make_unique<Impl>(device)} {}

Encoder::~Encoder() {
    if (impl->in_pass) {
        EndRenderPass();
    }
    if (impl->cmd) {
        Submit(); // nothing recorded is lost, and the keep-alive list is released after the GPU
    }
}

// ---- Blits ----

void Encoder::CopyBuffer(BufferRef src, BufferRef dst, u64 size) {
    [impl->Blit() copyFromBuffer:(__bridge id<MTLBuffer>)src.buffer
                    sourceOffset:src.offset
                        toBuffer:(__bridge id<MTLBuffer>)dst.buffer
               destinationOffset:dst.offset
                            size:size];
}

void Encoder::FillBuffer(BufferRef dst, u64 size, u32 pattern) {
    if (pattern == u32(u8(pattern)) * 0x01010101u) {
        [impl->Blit() fillBuffer:(__bridge id<MTLBuffer>)dst.buffer
                           range:NSMakeRange(dst.offset, size)
                           value:u8(pattern)];
        return;
    }
    ASSERT((dst.offset | size) % 4 == 0);
    // A dedicated compute encoder, so the caller's compute state is untouched.
    impl->FlushPendingClear();
    impl->EndRender();
    impl->EndCompute();
    impl->EndBlit();
    id<MTLComputePipelineState> pipeline = impl->FillPipeline();
    const u32 args[2] = {pattern, u32(size / 4)};
    @autoreleasepool {
        id<MTLComputeCommandEncoder> e = [impl->Cmd() computeCommandEncoder];
        [e setComputePipelineState:pipeline];
        [e setBuffer:(__bridge id<MTLBuffer>)dst.buffer offset:dst.offset atIndex:0];
        [e setBytes:args length:sizeof(args) atIndex:1];
        [e dispatchThreads:MTLSizeMake(size / 4, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        [e endEncoding];
    }
}

void Encoder::CopyBufferToTexture(BufferRef src, u32 bytes_per_row, u32 bytes_per_image,
                                  void* texture, u32 level, u32 slice, Origin3 origin, Size3 size,
                                  BlitAspect aspect) {
    [impl->Blit() copyFromBuffer:(__bridge id<MTLBuffer>)src.buffer
                    sourceOffset:src.offset
               sourceBytesPerRow:bytes_per_row
             sourceBytesPerImage:bytes_per_image
                      sourceSize:MTLSizeMake(size.width, size.height, size.depth)
                       toTexture:(__bridge id<MTLTexture>)texture
                destinationSlice:slice
                destinationLevel:level
               destinationOrigin:MTLOriginMake(origin.x, origin.y, origin.z)
                        options:BlitOptions(aspect)];
}

void Encoder::CopyTextureToBuffer(void* texture, u32 level, u32 slice, Origin3 origin, Size3 size,
                                  BufferRef dst, u32 bytes_per_row, u32 bytes_per_image,
                                  BlitAspect aspect) {
    [impl->Blit() copyFromTexture:(__bridge id<MTLTexture>)texture
                      sourceSlice:slice
                      sourceLevel:level
                     sourceOrigin:MTLOriginMake(origin.x, origin.y, origin.z)
                       sourceSize:MTLSizeMake(size.width, size.height, size.depth)
                         toBuffer:(__bridge id<MTLBuffer>)dst.buffer
                destinationOffset:dst.offset
           destinationBytesPerRow:bytes_per_row
         destinationBytesPerImage:bytes_per_image
                          options:BlitOptions(aspect)];
}

void Encoder::CopyTexture(void* src, u32 src_level, u32 src_slice, Origin3 src_origin, void* dst,
                          u32 dst_level, u32 dst_slice, Origin3 dst_origin, Size3 size) {
    [impl->Blit() copyFromTexture:(__bridge id<MTLTexture>)src
                      sourceSlice:src_slice
                      sourceLevel:src_level
                     sourceOrigin:MTLOriginMake(src_origin.x, src_origin.y, src_origin.z)
                       sourceSize:MTLSizeMake(size.width, size.height, size.depth)
                        toTexture:(__bridge id<MTLTexture>)dst
                 destinationSlice:dst_slice
                 destinationLevel:dst_level
                destinationOrigin:MTLOriginMake(dst_origin.x, dst_origin.y, dst_origin.z)];
}

// ---- Command buffer ----

void Encoder::KeepAlive(Object object) {
    impl->keep.push_back(std::move(object));
}

void Encoder::EndEncoding() {
    impl->EndRender();
    impl->EndCompute();
    impl->EndBlit();
}

void* Encoder::CommandBuffer() {
    return (__bridge void*)impl->Cmd();
}

u64 Encoder::Submit() {
    return impl->Finish(true);
}

void Encoder::SubmitUntracked() {
    impl->Finish(false);
}

void Encoder::DeclareIndirect(void* buffer) {
    if (std::find(impl->indirect.begin(), impl->indirect.end(), buffer) != impl->indirect.end()) {
        return;
    }
    impl->indirect.push_back(buffer);
    id<MTLBuffer> b = (__bridge id<MTLBuffer>)buffer;
    if (impl->render) {
        [impl->render useResource:b
                            usage:MTLResourceUsageRead
                           stages:MTLRenderStageVertex | MTLRenderStageFragment];
    }
    if (impl->compute) {
        [impl->compute useResource:b usage:MTLResourceUsageRead | MTLResourceUsageWrite];
    }
}

bool Encoder::HasBindings() const noexcept {
    const auto used = [](const StageState& s) {
        return s.buffers_used | s.textures_used | s.samplers_used;
    };
    return impl->render_used || impl->compute_pipeline || used(impl->vertex) ||
           used(impl->fragment) || used(impl->compute_stage);
}

// ---- Compute (Task 4) ----

void Encoder::SetBuffer(Stage stage, u32 slot, BufferRef buffer) {
    ASSERT(slot < MaxBuffers);
    StageState& s = impl->StateOf(stage);
    StageState::Slot& b = s.buffers[slot];
    b.buffer = buffer.buffer;
    b.offset = buffer.offset;
    b.bytes.clear();
    s.buffers_dirty |= 1u << slot;
    s.buffers_used |= 1u << slot;
}

void Encoder::SetBytes(Stage stage, u32 slot, const void* data, u32 size) {
    ASSERT(slot < MaxBuffers && size > 0);
    if (size > MaxInlineBytes) {
        id<MTLBuffer> temp;
        @autoreleasepool {
            temp = [impl->dev newBufferWithBytes:data
                                          length:size
                                         options:MTLResourceStorageModeShared];
        }
        SetBuffer(stage, slot, {(__bridge void*)temp, 0});
        KeepAlive(Object::Retain((__bridge void*)temp));
        return;
    }
    StageState& s = impl->StateOf(stage);
    StageState::Slot& b = s.buffers[slot];
    b.buffer = nullptr;
    b.offset = 0;
    b.bytes.assign(static_cast<const u8*>(data), static_cast<const u8*>(data) + size);
    s.buffers_dirty |= 1u << slot;
    s.buffers_used |= 1u << slot;
}

void Encoder::SetTexture(Stage stage, u32 slot, void* texture) {
    ASSERT(slot < MaxTextures);
    StageState& s = impl->StateOf(stage);
    s.textures[slot] = texture;
    s.textures_dirty |= 1u << slot;
    s.textures_used |= 1u << slot;
}

void Encoder::SetSampler(Stage stage, u32 slot, void* sampler) {
    ASSERT(slot < MaxSamplers);
    StageState& s = impl->StateOf(stage);
    s.samplers[slot] = sampler;
    s.samplers_dirty |= 1u << slot;
    s.samplers_used |= 1u << slot;
}

void Encoder::SetComputePipeline(void* pipeline) {
    if (pipeline != impl->compute_pipeline) {
        impl->compute_pipeline = pipeline;
        impl->compute_pipeline_dirty = true;
    }
}

void Encoder::Dispatch(Size3 groups, Size3 threads_per_group) {
    ASSERT_MSG(impl->compute_pipeline, "Dispatch without a compute pipeline");
    impl->ApplyCompute();
    [impl->compute dispatchThreadgroups:MTLSizeMake(groups.width, groups.height, groups.depth)
                  threadsPerThreadgroup:MTLSizeMake(threads_per_group.width,
                                                    threads_per_group.height,
                                                    threads_per_group.depth)];
}

// ---- Render passes and draws (Task 5) ----

void Encoder::BeginRenderPass(const RenderPassDesc& desc) {
    ASSERT_MSG(!impl->in_pass, "BeginRenderPass inside a render pass");
    EndEncoding();
    MTLRenderPassDescriptor* first = [MTLRenderPassDescriptor renderPassDescriptor];
    bool clears = false, attachments = false, resolves = false;
    for (u32 i = 0; i < RenderPassDesc::MaxColors; ++i) {
        const ColorAttachment& c = desc.colors[i];
        if (!c.texture) {
            continue;
        }
        attachments = true;
        MTLRenderPassColorAttachmentDescriptor* a = first.colorAttachments[i];
        a.texture = (__bridge id<MTLTexture>)c.texture;
        a.level = c.level;
        a.slice = c.slice;
        a.depthPlane = c.depth_plane;
        a.loadAction = c.clear ? MTLLoadActionClear : MTLLoadActionLoad;
        a.storeAction = MTLStoreActionStore;
        a.clearColor = MTLClearColorMake(c.clear_color[0], c.clear_color[1], c.clear_color[2],
                                         c.clear_color[3]);
        clears |= c.clear;
        if (c.resolve_texture) {
            a.resolveTexture = (__bridge id<MTLTexture>)c.resolve_texture;
            a.resolveLevel = c.resolve_level;
            a.resolveSlice = c.resolve_slice;
            a.storeAction = MTLStoreActionStoreAndMultisampleResolve;
            resolves = true;
        }
    }
    const DepthAttachment& d = desc.depth;
    if (d.texture) {
        attachments = true;
        id<MTLTexture> dt = (__bridge id<MTLTexture>)d.texture;
        if (!d.stencil_only) {
            MTLRenderPassDepthAttachmentDescriptor* a = first.depthAttachment;
            a.texture = dt;
            a.level = d.level;
            a.slice = d.slice;
            a.loadAction = d.clear_depth ? MTLLoadActionClear : MTLLoadActionLoad;
            a.storeAction = MTLStoreActionStore;
            a.clearDepth = d.depth;
            clears |= d.clear_depth;
        }
        if (d.has_stencil || d.stencil_only) {
            MTLRenderPassStencilAttachmentDescriptor* s = first.stencilAttachment;
            s.texture = dt;
            s.level = d.level;
            s.slice = d.slice;
            s.loadAction = d.clear_stencil ? MTLLoadActionClear : MTLLoadActionLoad;
            s.storeAction = MTLStoreActionStore;
            s.clearStencil = d.stencil;
            clears |= d.clear_stencil;
        }
    }
    if (!attachments) {
        first.renderTargetWidth = desc.width;
        first.renderTargetHeight = desc.height;
        first.defaultRasterSampleCount = 1;
    }
    // Target size: the first color attachment, else depth, else the declared render area.
    u32 width = desc.width, height = desc.height;
    const auto size_of = [&](void* texture, u32 level) {
        id<MTLTexture> t = (__bridge id<MTLTexture>)texture;
        width = std::max<u32>(1, u32(t.width) >> level);
        height = std::max<u32>(1, u32(t.height) >> level);
    };
    const auto color = std::find_if(desc.colors.begin(), desc.colors.end(),
                                    [](const ColorAttachment& c) { return c.texture != nullptr; });
    if (color != desc.colors.end()) {
        size_of(color->texture, color->level);
    } else if (desc.depth.texture) {
        size_of(desc.depth.texture, desc.depth.level);
    }
    impl->target_width = width;
    impl->target_height = height;
    MTLRenderPassDescriptor* load = [first copy];
    for (u32 i = 0; i < RenderPassDesc::MaxColors; ++i) {
        load.colorAttachments[i].loadAction = MTLLoadActionLoad;
    }
    load.depthAttachment.loadAction = MTLLoadActionLoad;
    load.stencilAttachment.loadAction = MTLLoadActionLoad;
    impl->pass_first = first;
    impl->pass_load = load;
    impl->pass_clears = clears;
    impl->pass_resolves = resolves;
    impl->pass_opened = false;
    impl->in_pass = true;
}

void Encoder::EndRenderPass() {
    ASSERT_MSG(impl->in_pass, "EndRenderPass outside a render pass");
    if (!impl->pass_opened && (impl->pass_clears || impl->pass_resolves)) {
        impl->Render(); // a pass that only clears or resolves still has to do it
    }
    impl->EndRender();
    impl->in_pass = false;
    impl->pass_first = nil;
    impl->pass_load = nil;
}

void Encoder::SetRenderPipeline(void* pipeline) {
    if (pipeline != impl->render_pipeline) {
        impl->render_pipeline = pipeline;
        impl->MarkRender(Impl::BitPipeline);
    }
}

void Encoder::SetDepthStencil(void* state) {
    if (state != impl->depth_stencil) {
        impl->depth_stencil = state;
        impl->MarkRender(Impl::BitDepthStencil);
    }
}

void Encoder::SetViewport(const Viewport& viewport) {
    impl->viewport = viewport;
    impl->MarkRender(Impl::BitViewport);
}

void Encoder::SetScissor(const Scissor& scissor) {
    impl->scissor = scissor;
    impl->MarkRender(Impl::BitScissor);
}

void Encoder::SetCullMode(CullMode mode) {
    impl->cull = mode;
    impl->MarkRender(Impl::BitCull);
}

void Encoder::SetFrontFace(Winding winding) {
    impl->winding = winding;
    impl->MarkRender(Impl::BitWinding);
}

void Encoder::SetDepthBias(float constant, float slope, float clamp) {
    impl->bias[0] = constant;
    impl->bias[1] = slope;
    impl->bias[2] = clamp;
    impl->MarkRender(Impl::BitBias);
}

void Encoder::SetStencilReference(u32 front, u32 back) {
    impl->stencil_front = front;
    impl->stencil_back = back;
    impl->MarkRender(Impl::BitStencilRef);
}

void Encoder::SetBlendColor(float r, float g, float b, float a) {
    impl->blend[0] = r;
    impl->blend[1] = g;
    impl->blend[2] = b;
    impl->blend[3] = a;
    impl->MarkRender(Impl::BitBlend);
}

void Encoder::Draw(Primitive primitive, u32 vertex_count, u32 instance_count, u32 first_vertex,
                   u32 first_instance) {
    ASSERT_MSG(impl->render_pipeline, "Draw without a render pipeline");
    impl->ApplyRender();
    [impl->render drawPrimitives:ToMtl(primitive)
                     vertexStart:first_vertex
                     vertexCount:vertex_count
                   instanceCount:instance_count
                    baseInstance:first_instance];
}

void Encoder::DrawIndexed(Primitive primitive, u32 index_count, IndexType type, BufferRef indices,
                          u32 instance_count, s32 base_vertex, u32 first_instance) {
    ASSERT_MSG(impl->render_pipeline, "Draw without a render pipeline");
    impl->ApplyRender();
    [impl->render drawIndexedPrimitives:ToMtl(primitive)
                             indexCount:index_count
                              indexType:type == IndexType::U32 ? MTLIndexTypeUInt32
                                                               : MTLIndexTypeUInt16
                            indexBuffer:(__bridge id<MTLBuffer>)indices.buffer
                      indexBufferOffset:indices.offset
                          instanceCount:instance_count
                             baseVertex:base_vertex
                           baseInstance:first_instance];
}

bool Encoder::InRenderPass() const noexcept {
    return impl->in_pass;
}

} // namespace Metal
