// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: Metal::Encoder (Metal-native renderer, spec component 2).

#import <Metal/Metal.h>

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
    MTLRenderPassDescriptor* pass_first; ///< as begun, with its clears
    MTLRenderPassDescriptor* pass_load;  ///< the same with Load actions (reopening)

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

    std::atomic<bool> fill_pattern_logged{};

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

    id<MTLBlitCommandEncoder> Blit() {
        if (!blit) {
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
            EndRender();
            EndBlit();
            @autoreleasepool {
                compute = [Cmd() computeCommandEncoder];
            }
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
    if (impl->cmd) {
        if (impl->in_pass) {
            EndRenderPass();
        }
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
    const u8 byte = u8(pattern);
    if (pattern != u32(byte) * 0x01010101u && !impl->fill_pattern_logged.exchange(true)) {
        LOG_WARNING(Render, "FillBuffer pattern {:#x} has different bytes; filled with its low byte",
                    pattern);
    }
    [impl->Blit() fillBuffer:(__bridge id<MTLBuffer>)dst.buffer
                       range:NSMakeRange(dst.offset, size)
                       value:byte];
}

void Encoder::CopyBufferToTexture(BufferRef src, u32 bytes_per_row, u32 bytes_per_image,
                                  void* texture, u32 level, u32 slice, Origin3 origin, Size3 size) {
    [impl->Blit() copyFromBuffer:(__bridge id<MTLBuffer>)src.buffer
                    sourceOffset:src.offset
               sourceBytesPerRow:bytes_per_row
             sourceBytesPerImage:bytes_per_image
                      sourceSize:MTLSizeMake(size.width, size.height, size.depth)
                       toTexture:(__bridge id<MTLTexture>)texture
                destinationSlice:slice
                destinationLevel:level
               destinationOrigin:MTLOriginMake(origin.x, origin.y, origin.z)];
}

void Encoder::CopyTextureToBuffer(void* texture, u32 level, u32 slice, Origin3 origin, Size3 size,
                                  BufferRef dst, u32 bytes_per_row, u32 bytes_per_image) {
    [impl->Blit() copyFromTexture:(__bridge id<MTLTexture>)texture
                      sourceSlice:slice
                      sourceLevel:level
                     sourceOrigin:MTLOriginMake(origin.x, origin.y, origin.z)
                       sourceSize:MTLSizeMake(size.width, size.height, size.depth)
                         toBuffer:(__bridge id<MTLBuffer>)dst.buffer
                destinationOffset:dst.offset
           destinationBytesPerRow:bytes_per_row
         destinationBytesPerImage:bytes_per_image];
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
    ASSERT_MSG(!impl->in_pass, "Submit inside a render pass");
    EndEncoding();
    id<MTLCommandBuffer> cmd = impl->Cmd();
    auto* held = new std::vector<Object>(std::move(impl->keep));
    impl->keep.clear();
    [cmd addCompletedHandler:^(id<MTLCommandBuffer>) {
        delete held;
    }];
    const u64 tick = impl->device.Commit((__bridge void*)cmd);
    impl->cmd = nil;
    return tick;
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

void Encoder::BeginRenderPass(const RenderPassDesc&) {
    UNREACHABLE_MSG("Metal::Encoder render passes: Task 5");
}
void Encoder::EndRenderPass() {
    UNREACHABLE_MSG("Metal::Encoder render passes: Task 5");
}
bool Encoder::InRenderPass() const noexcept {
    return impl->in_pass;
}
void Encoder::SetRenderPipeline(void*) {
    UNREACHABLE_MSG("Metal::Encoder render state: Task 5");
}
void Encoder::SetDepthStencil(void*) {
    UNREACHABLE_MSG("Metal::Encoder render state: Task 5");
}
void Encoder::SetViewport(const Viewport&) {
    UNREACHABLE_MSG("Metal::Encoder render state: Task 5");
}
void Encoder::SetScissor(const Scissor&) {
    UNREACHABLE_MSG("Metal::Encoder render state: Task 5");
}
void Encoder::SetCullMode(CullMode) {
    UNREACHABLE_MSG("Metal::Encoder render state: Task 5");
}
void Encoder::SetFrontFace(Winding) {
    UNREACHABLE_MSG("Metal::Encoder render state: Task 5");
}
void Encoder::SetDepthBias(float, float, float) {
    UNREACHABLE_MSG("Metal::Encoder render state: Task 5");
}
void Encoder::SetStencilReference(u32, u32) {
    UNREACHABLE_MSG("Metal::Encoder render state: Task 5");
}
void Encoder::SetBlendColor(float, float, float, float) {
    UNREACHABLE_MSG("Metal::Encoder render state: Task 5");
}
void Encoder::Draw(Primitive, u32, u32, u32, u32) {
    UNREACHABLE_MSG("Metal::Encoder draws: Task 5");
}
void Encoder::DrawIndexed(Primitive, u32, IndexType, BufferRef, u32, s32, u32) {
    UNREACHABLE_MSG("Metal::Encoder draws: Task 5");
}

} // namespace Metal
