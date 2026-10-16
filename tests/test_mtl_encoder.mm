// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: unit test for the Metal-native foundation (docs/superpowers/plans/
// 2026-10-16-metal-native-m1-device-encoder.md): Metal::Object, Metal::Device ticks and limits,
// Metal::Encoder blits, dispatches, render passes and draws, on the system's GPU. No window.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <map>
#include <thread>
#include <vector>
#include "video_core/renderer_metal/mtl_buffer.h"
#include "video_core/renderer_metal/mtl_object.h"
#include "video_core/renderer_metal/mtl_device.h"
#include "video_core/renderer_metal/mtl_encoder.h"
#include "video_core/renderer_metal/mtl_depth_stencil.h"
#include "video_core/renderer_metal/mtl_specialize.h"
#include "video_core/renderer_metal/mtl_texture.h"
#include "video_core/renderer_metal/mtl_timeline.h"

namespace {

int failures = 0;

void Expect(bool ok, const char* what) {
    if (!ok) {
        std::printf("FAIL: %s\n", what);
        ++failures;
    }
}

std::atomic<int> probes_alive{0};

std::vector<u32> ReadWords(const char* path) {
    FILE* f = std::fopen(path, "rb");
    std::vector<u32> words;
    if (!f) return words;
    u32 w;
    while (std::fread(&w, 4, 1, f) == 1) words.push_back(w);
    std::fclose(f);
    return words;
}

/// OpConstant (43) result values of the module, by result id.
std::map<u32, u32> Constants(const std::vector<u32>& w) {
    std::map<u32, u32> out;
    for (size_t i = 5; i < w.size();) {
        const u32 count = w[i] >> 16, op = w[i] & 0xffff;
        if (op == 43 && count == 4) out[w[i + 2]] = w[i + 3];
        Expect(op != 50, "no OpSpecConstant left after Specialize");
        i += count ? count : 1;
    }
    return out;
}

void TestSpecialize() {
    const std::vector<u32> spirv = ReadWords("out/spec_const.spv");
    Expect(!spirv.empty(), "out/spec_const.spv exists");
    const Metal::SpecConstant constants[] = {{3, 77}};
    const std::vector<u32> out = Metal::Specialize(spirv, constants);
    bool has77 = false, has1 = false;
    for (const auto& [id, value] : Constants(out)) {
        has77 |= value == 77;
        has1 |= value == 1;
    }
    Expect(has77 && has1, "SpecId 3 takes the given value, SpecId 0 keeps its default");
}

void TestDepthStencilCache() {
    Metal::Device device;
    Metal::DepthStencilCache cache{device};
    Metal::DepthStencilKey a;
    a.depth_test = a.depth_write = true;
    a.depth_compare = 1; // Less
    Metal::DepthStencilKey b = a;
    b.depth_write = false;
    void* sa = cache.Get(a);
    Expect(sa && cache.Get(a) == sa, "same key, same state");
    Expect(cache.Get(b) && cache.Get(b) != sa, "different key, different state");
}

} // namespace

/// Counts live instances, to see when an Object releases what it holds.
@interface Probe : NSObject
@end
@implementation Probe
- (instancetype)init {
    if ((self = [super init])) {
        probes_alive.fetch_add(1);
    }
    return self;
}
- (void)dealloc {
    probes_alive.fetch_sub(1);
}
@end

namespace {

void TestObject() {
    @autoreleasepool {
        {
            Metal::Object a = Metal::Object::Adopt((__bridge_retained void*)[Probe new]);
            Expect(bool(a) && probes_alive == 1, "Object::Adopt holds the object");
            Metal::Object b = std::move(a);
            Expect(!a && bool(b) && probes_alive == 1, "moving an Object moves the reference");
            Metal::Object c = Metal::Object::Retain(b.Get());
            b.Reset();
            Expect(!b && probes_alive == 1, "Object::Retain keeps the object after the first Reset");
        }
        Expect(probes_alive == 0, "the last Object releases the object");
    }
}

void TestTicks() {
    Metal::Device device;
    const Metal::Limits& limits = device.GetLimits();
    Expect(limits.max_buffer_length >= (u64(1) << 30), "max buffer length read from the device");
    Expect(limits.uniform_alignment == 16 && limits.storage_alignment == 16 &&
               limits.max_samplers == 500000,
           "limits keep MoltenVK's values");
    Expect(device.CurrentTick() == 1 && device.KnownGpuTick() == 0, "ticks start at 1");

    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)device.NativeQueue();
    id<MTLCommandBuffer> empty = [queue commandBuffer];
    const u64 first = device.Commit((__bridge void*)empty);
    Expect(first == 1 && device.CurrentTick() == 2, "Commit signals the current tick and advances");
    device.Wait(first);
    Expect(device.IsFree(first), "an empty command buffer completes");
    Expect(!device.IsFree(2), "an uncommitted tick is not free");

    for (int i = 0; i < 3; ++i) {
        device.Commit((__bridge void*)[queue commandBuffer]);
    }
    device.Finish();
    Expect(device.IsFree(4) && device.KnownGpuTick() >= 4, "Finish waits for every commit");
    Expect(device.ErrorCount() == 0, "no command buffer failed");
}

id<MTLBuffer> SharedBuffer(Metal::Device& device, NSUInteger length) {
    id<MTLDevice> dev = (__bridge id<MTLDevice>)device.NativeDevice();
    id<MTLBuffer> buffer = [dev newBufferWithLength:length options:MTLResourceStorageModeShared];
    std::memset(buffer.contents, 0, length);
    return buffer;
}

Metal::BufferRef Ref(id<MTLBuffer> buffer, u64 offset = 0) {
    return {(__bridge void*)buffer, offset};
}

id<MTLTexture> Texture(Metal::Device& device, u32 width, u32 height) {
    id<MTLDevice> dev = (__bridge id<MTLDevice>)device.NativeDevice();
    MTLTextureDescriptor* desc =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                           width:width
                                                          height:height
                                                       mipmapped:NO];
    desc.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
    desc.storageMode = MTLStorageModePrivate;
    return [dev newTextureWithDescriptor:desc];
}

u32 Pixel(id<MTLBuffer> readback, u32 width, u32 x, u32 y) {
    u32 value;
    std::memcpy(&value, static_cast<const u8*>(readback.contents) + (y * width + x) * 4, 4);
    return value;
}

void TestBlits() {
    Metal::Device device;
    Metal::Encoder encoder{device};
    id<MTLBuffer> src = SharedBuffer(device, 256);
    id<MTLBuffer> dst = SharedBuffer(device, 256);
    for (u32 i = 0; i < 256; ++i) {
        static_cast<u8*>(src.contents)[i] = u8(i);
    }
    encoder.FillBuffer(Ref(dst), 64, 0x7f7f7f7fu);
    encoder.CopyBuffer(Ref(src, 16), Ref(dst, 128), 32);
    // Buffer -> texture -> texture -> buffer round trip, 4x4 RGBA8.
    id<MTLTexture> a = Texture(device, 4, 4);
    id<MTLTexture> b = Texture(device, 4, 4);
    id<MTLBuffer> pixels = SharedBuffer(device, 64);
    for (u32 i = 0; i < 64; ++i) {
        static_cast<u8*>(pixels.contents)[i] = u8(200 - i);
    }
    id<MTLBuffer> back = SharedBuffer(device, 64);
    encoder.CopyBufferToTexture(Ref(pixels), 16, 64, (__bridge void*)a, 0, 0, {}, {4, 4, 1});
    encoder.CopyTexture((__bridge void*)a, 0, 0, {}, (__bridge void*)b, 0, 0, {}, {4, 4, 1});
    encoder.CopyTextureToBuffer((__bridge void*)b, 0, 0, {}, {4, 4, 1}, Ref(back), 16, 64);
    const u64 tick = encoder.Submit();
    device.Wait(tick);

    const u8* d = static_cast<const u8*>(dst.contents);
    bool filled = true, copied = true;
    for (u32 i = 0; i < 64; ++i) {
        filled &= d[i] == 0x7f;
    }
    for (u32 i = 0; i < 32; ++i) {
        copied &= d[128 + i] == u8(16 + i);
    }
    Expect(filled, "FillBuffer fills the range");
    Expect(copied && d[127] == 0 && d[160] == 0, "CopyBuffer copies exactly the range");
    Expect(std::memcmp(pixels.contents, back.contents, 64) == 0,
           "buffer -> texture -> texture -> buffer keeps the bytes");
    Expect(device.ErrorCount() == 0, "blits ran without errors");
}

void TestKeepAlive() {
    Metal::Device device;
    Metal::Encoder encoder{device};
    @autoreleasepool {
        encoder.KeepAlive(Metal::Object::Adopt((__bridge_retained void*)[Probe new]));
    }
    Expect(probes_alive == 1, "KeepAlive holds the object while recording");
    const u64 empty = encoder.Submit(); // nothing recorded: still a tick that completes
    device.Wait(empty);
    for (int i = 0; i < 100 && probes_alive != 0; ++i) {
        [NSThread sleepForTimeInterval:0.01]; // the completion handler may run just after the event
    }
    Expect(probes_alive == 0, "KeepAlive releases the object once the GPU is done");
    Expect(encoder.Submit() == empty + 1, "every Submit takes the next tick");
}


/// Completion handlers must never write into a Device that is gone: a new Device (often at the
/// same address) must start with no known GPU work.
void TestDeviceLifetime() {
    bool clean = true;
    for (int round = 0; round < 50 && clean; ++round) {
        {
            Metal::Device old;
            id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)old.NativeQueue();
            for (int i = 0; i < 4; ++i) {
                old.Commit((__bridge void*)[queue commandBuffer]);
            }
        }
        Metal::Device fresh;
        [NSThread sleepForTimeInterval:0.002]; // let late handlers of `old` run
        clean = fresh.KnownGpuTick() == 0 && !fresh.IsFree(1);
    }
    Expect(clean, "a destroyed Device's completion handlers do not touch a new Device");
}

constexpr const char* kTestShaders = R"(
#include <metal_stdlib>
using namespace metal;

kernel void doubles(device uint* out [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    out[i] = i * 2u;
}

kernel void copy_constants(device uint* out [[buffer(0)]], constant uint* in [[buffer(1)]],
                           uint i [[thread_position_in_grid]]) {
    out[i] = in[i];
}

struct VOut { float4 pos [[position]]; };

vertex VOut fullscreen(uint vid [[vertex_id]]) {
    const float2 uv = float2((vid << 1) & 2, vid & 2);
    VOut o;
    o.pos = float4(uv * 2.0 - 1.0, 0.0, 1.0);
    return o;
}

vertex VOut indexed_quad(uint vid [[vertex_id]], constant float2* corners [[buffer(0)]]) {
    VOut o;
    o.pos = float4(corners[vid], 0.0, 1.0);
    return o;
}

fragment float4 solid(constant float4& color [[buffer(0)]]) {
    return color;
}
)";

id<MTLLibrary> TestLibrary(Metal::Device& device) {
    id<MTLDevice> dev = (__bridge id<MTLDevice>)device.NativeDevice();
    NSError* error = nil;
    id<MTLLibrary> library = [dev newLibraryWithSource:@(kTestShaders) options:nil error:&error];
    if (!library) {
        std::printf("FAIL: test shaders: %s\n", error.localizedDescription.UTF8String);
        std::exit(1);
    }
    return library;
}

id<MTLComputePipelineState> Kernel(Metal::Device& device, id<MTLLibrary> library, NSString* name) {
    id<MTLDevice> dev = (__bridge id<MTLDevice>)device.NativeDevice();
    NSError* error = nil;
    id<MTLComputePipelineState> pipeline =
        [dev newComputePipelineStateWithFunction:[library newFunctionWithName:name] error:&error];
    if (!pipeline) {
        std::printf("FAIL: kernel: %s\n", error.localizedDescription.UTF8String);
        std::exit(1);
    }
    return pipeline;
}

void TestCompute() {
    Metal::Device device;
    Metal::Encoder encoder{device};
    id<MTLLibrary> library = TestLibrary(device);
    id<MTLComputePipelineState> doubles = Kernel(device, library, @"doubles");
    id<MTLBuffer> out = SharedBuffer(device, 64 * 4);
    encoder.SetComputePipeline((__bridge void*)doubles);
    encoder.SetBuffer(Metal::Stage::Compute, 0, Ref(out));
    encoder.Dispatch({4, 1, 1}, {16, 1, 1});
    // A blit between two dispatches ends the compute encoder; the second dispatch must get its
    // pipeline and buffer again on the new encoder.
    id<MTLBuffer> scratch = SharedBuffer(device, 16);
    encoder.FillBuffer(Ref(scratch), 16, 0);
    id<MTLBuffer> out2 = SharedBuffer(device, 64 * 4);
    encoder.SetBuffer(Metal::Stage::Compute, 0, Ref(out2));
    encoder.Dispatch({4, 1, 1}, {16, 1, 1});
    device.Wait(encoder.Submit());

    bool ok = true, ok2 = true;
    for (u32 i = 0; i < 64; ++i) {
        ok &= static_cast<const u32*>(out.contents)[i] == i * 2;
        ok2 &= static_cast<const u32*>(out2.contents)[i] == i * 2;
    }
    Expect(ok, "Dispatch runs the kernel over the grid");
    Expect(ok2, "state is reapplied on a compute encoder opened after a blit");
}

void TestLargeBytes() {
    Metal::Device device;
    Metal::Encoder encoder{device};
    id<MTLLibrary> library = TestLibrary(device);
    id<MTLComputePipelineState> copy = Kernel(device, library, @"copy_constants");
    std::vector<u32> small(64), large(2048); // 256 B inline; 8 KiB over the setBytes limit
    for (u32 i = 0; i < small.size(); ++i) {
        small[i] = 1000 + i;
    }
    for (u32 i = 0; i < large.size(); ++i) {
        large[i] = 50000 + i;
    }
    id<MTLBuffer> out_small = SharedBuffer(device, small.size() * 4);
    id<MTLBuffer> out_large = SharedBuffer(device, large.size() * 4);
    encoder.SetComputePipeline((__bridge void*)copy);
    encoder.SetBuffer(Metal::Stage::Compute, 0, Ref(out_small));
    encoder.SetBytes(Metal::Stage::Compute, 1, small.data(), u32(small.size() * 4));
    encoder.Dispatch({1, 1, 1}, {64, 1, 1});
    encoder.SetBuffer(Metal::Stage::Compute, 0, Ref(out_large));
    encoder.SetBytes(Metal::Stage::Compute, 1, large.data(), u32(large.size() * 4));
    std::fill(large.begin(), large.end(), 0u); // the encoder must have copied the bytes already
    encoder.Dispatch({32, 1, 1}, {64, 1, 1});
    device.Wait(encoder.Submit());

    Expect(std::memcmp(out_small.contents, small.data(), small.size() * 4) == 0,
           "SetBytes up to 4 KiB arrives");
    bool ok = true;
    for (u32 i = 0; i < 2048; ++i) {
        ok &= static_cast<const u32*>(out_large.contents)[i] == 50000 + i;
    }
    Expect(ok, "SetBytes over 4 KiB arrives (temporary buffer)");
    Expect(device.ErrorCount() == 0, "compute ran without errors");
}

constexpr u32 kSize = 64;
constexpr u32 kRed = 0xff0000ffu, kGreen = 0xff00ff00u, kBlue = 0xffff0000u, kBlack = 0xff000000u,
              kYellow = 0xff00ffffu; // RGBA8 bytes read as a little-endian u32

id<MTLRenderPipelineState> SolidPipeline(Metal::Device& device, id<MTLLibrary> library,
                                         NSString* vertex) {
    id<MTLDevice> dev = (__bridge id<MTLDevice>)device.NativeDevice();
    MTLRenderPipelineDescriptor* desc = [MTLRenderPipelineDescriptor new];
    desc.vertexFunction = [library newFunctionWithName:vertex];
    desc.fragmentFunction = [library newFunctionWithName:@"solid"];
    desc.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm;
    NSError* error = nil;
    id<MTLRenderPipelineState> pipeline = [dev newRenderPipelineStateWithDescriptor:desc
                                                                              error:&error];
    if (!pipeline) {
        std::printf("FAIL: render pipeline: %s\n", error.localizedDescription.UTF8String);
        std::exit(1);
    }
    return pipeline;
}

Metal::RenderPassDesc ColorPass(id<MTLTexture> target, bool clear, u32 rgba) {
    Metal::RenderPassDesc desc;
    desc.colors[0].texture = (__bridge void*)target;
    desc.colors[0].clear = clear;
    desc.colors[0].clear_color = {double(rgba & 0xff) / 255.0, double((rgba >> 8) & 0xff) / 255.0,
                                  double((rgba >> 16) & 0xff) / 255.0, 1.0};
    desc.width = kSize;
    desc.height = kSize;
    return desc;
}

/// Records a copy of `texture` into a new shared buffer (kSize x kSize RGBA8).
id<MTLBuffer> Readback(Metal::Device& device, Metal::Encoder& encoder, id<MTLTexture> texture) {
    id<MTLBuffer> out = SharedBuffer(device, kSize * kSize * 4);
    encoder.CopyTextureToBuffer((__bridge void*)texture, 0, 0, {}, {kSize, kSize, 1}, Ref(out),
                                kSize * 4, kSize * kSize * 4);
    return out;
}

void FillTexture(Metal::Device& device, Metal::Encoder& encoder, id<MTLTexture> texture, u32 rgba) {
    id<MTLBuffer> pixels = SharedBuffer(device, kSize * kSize * 4);
    for (u32 i = 0; i < kSize * kSize; ++i) {
        static_cast<u32*>(pixels.contents)[i] = rgba;
    }
    encoder.CopyBufferToTexture(Ref(pixels), kSize * 4, kSize * kSize * 4, (__bridge void*)texture,
                                0, 0, {}, {kSize, kSize, 1});
    encoder.KeepAlive(Metal::Object::Retain((__bridge void*)pixels));
}

void SetColor(Metal::Encoder& encoder, u32 rgba) {
    const float color[4] = {float(rgba & 0xff) / 255.0f, float((rgba >> 8) & 0xff) / 255.0f,
                            float((rgba >> 16) & 0xff) / 255.0f, 1.0f};
    encoder.SetBytes(Metal::Stage::Fragment, 0, color, sizeof(color));
}

void TestClearOnlyPass() {
    Metal::Device device;
    Metal::Encoder encoder{device};
    id<MTLTexture> cleared = Texture(device, kSize, kSize);
    id<MTLTexture> untouched = Texture(device, kSize, kSize);
    FillTexture(device, encoder, cleared, kBlue);
    FillTexture(device, encoder, untouched, kBlue);
    encoder.BeginRenderPass(ColorPass(cleared, true, kYellow));
    encoder.EndRenderPass();
    encoder.BeginRenderPass(ColorPass(untouched, false, 0));
    encoder.EndRenderPass();
    id<MTLBuffer> a = Readback(device, encoder, cleared);
    id<MTLBuffer> b = Readback(device, encoder, untouched);
    device.Wait(encoder.Submit());
    Expect(Pixel(a, kSize, 5, 5) == kYellow && Pixel(a, kSize, 63, 63) == kYellow,
           "a pass that only clears still clears");
    Expect(Pixel(b, kSize, 5, 5) == kBlue, "a pass with no draws and no clears leaves the target");
}

void TestPassReopenKeepsDrawing() {
    Metal::Device device;
    Metal::Encoder encoder{device};
    id<MTLLibrary> library = TestLibrary(device);
    id<MTLRenderPipelineState> solid = SolidPipeline(device, library, @"fullscreen");
    id<MTLTexture> target = Texture(device, kSize, kSize);
    id<MTLBuffer> scratch = SharedBuffer(device, 16);

    encoder.BeginRenderPass(ColorPass(target, true, kBlack));
    encoder.SetRenderPipeline((__bridge void*)solid);
    encoder.SetViewport({0, 0, float(kSize), float(kSize), 0, 1});
    encoder.SetScissor({0, 0, kSize / 2, kSize});
    SetColor(encoder, kRed);
    encoder.Draw(Metal::Primitive::Triangle, 3);
    encoder.FillBuffer(Ref(scratch), 16, 0); // interrupts the pass
    encoder.SetScissor({kSize / 2, 0, kSize / 2, kSize});
    SetColor(encoder, kGreen);
    encoder.Draw(Metal::Primitive::Triangle, 3); // reopens the pass
    encoder.EndRenderPass();
    id<MTLBuffer> out = Readback(device, encoder, target);
    device.Wait(encoder.Submit());
    Expect(Pixel(out, kSize, 1, 1) == kRed, "the draw before the interruption is kept (Load)");
    Expect(Pixel(out, kSize, kSize - 2, 1) == kGreen, "the draw after the interruption lands");
    Expect(device.ErrorCount() == 0, "no errors reopening a pass");
}

void TestStatePersistsAcrossPasses() {
    Metal::Device device;
    Metal::Encoder encoder{device};
    id<MTLLibrary> library = TestLibrary(device);
    id<MTLRenderPipelineState> solid = SolidPipeline(device, library, @"fullscreen");
    id<MTLTexture> first = Texture(device, kSize, kSize);
    id<MTLTexture> second = Texture(device, kSize, kSize);

    encoder.BeginRenderPass(ColorPass(first, true, kBlack));
    encoder.SetRenderPipeline((__bridge void*)solid);
    encoder.SetViewport({0, 0, float(kSize), float(kSize), 0, 1});
    encoder.SetScissor({0, 0, kSize, kSize / 2}); // top half only
    SetColor(encoder, kGreen);
    encoder.Draw(Metal::Primitive::Triangle, 3);
    encoder.EndRenderPass();
    // Second pass: nothing set again; pipeline, viewport, scissor and the color must carry over.
    encoder.BeginRenderPass(ColorPass(second, true, kBlack));
    encoder.Draw(Metal::Primitive::Triangle, 3);
    encoder.EndRenderPass();
    id<MTLBuffer> out = Readback(device, encoder, second);
    device.Wait(encoder.Submit());
    Expect(Pixel(out, kSize, 3, 3) == kGreen, "state set in one pass applies in the next");
    Expect(Pixel(out, kSize, 3, kSize - 3) == kBlack, "including the scissor");
}

void TestIndexedDraw() {
    Metal::Device device;
    Metal::Encoder encoder{device};
    id<MTLLibrary> library = TestLibrary(device);
    id<MTLRenderPipelineState> quad = SolidPipeline(device, library, @"indexed_quad");
    id<MTLTexture> target = Texture(device, kSize, kSize);
    // Left half of clip space as two triangles from four corners.
    const float corners[8] = {-1, -1, 0, -1, -1, 1, 0, 1};
    id<MTLBuffer> indices = SharedBuffer(device, 12);
    const u16 index_data[6] = {0, 1, 2, 2, 1, 3};
    std::memcpy(indices.contents, index_data, sizeof(index_data));

    encoder.BeginRenderPass(ColorPass(target, true, kBlack));
    encoder.SetRenderPipeline((__bridge void*)quad);
    encoder.SetViewport({0, 0, float(kSize), float(kSize), 0, 1});
    encoder.SetScissor({0, 0, kSize, kSize});
    encoder.SetBytes(Metal::Stage::Vertex, 0, corners, sizeof(corners));
    SetColor(encoder, kBlue);
    encoder.DrawIndexed(Metal::Primitive::Triangle, 6, Metal::IndexType::U16, Ref(indices));
    encoder.EndRenderPass();
    id<MTLBuffer> out = Readback(device, encoder, target);
    device.Wait(encoder.Submit());
    Expect(Pixel(out, kSize, 2, kSize / 2) == kBlue, "DrawIndexed draws the left half");
    Expect(Pixel(out, kSize, kSize - 2, kSize / 2) == kBlack, "and not the right half");
}


/// IsFree/Wait mean the command buffer completed (its handlers ran), not only that the GPU
/// reached the signal: owners free resources at that point, and Metal (validation layer) counts
/// a command buffer as using them until it completes.
void TestWaitMeansCompleted() {
    Metal::Device device;
    Metal::Encoder encoder{device};
    id<MTLBuffer> buffer = SharedBuffer(device, 4096);
    bool completed = true;
    for (int i = 0; i < 200 && completed; ++i) {
        encoder.FillBuffer(Ref(buffer), 4096, u32(i & 0xff) * 0x01010101u);
        id<MTLCommandBuffer> cmd = (__bridge id<MTLCommandBuffer>)encoder.CommandBuffer();
        const u64 tick = encoder.Submit();
        device.Wait(tick);
        completed = cmd.status == MTLCommandBufferStatusCompleted;
    }
    Expect(completed, "after Wait(tick) the tick's command buffer has completed");
}


/// Bindings end with their command buffer (as in Vulkan): a slot left bound must not be applied
/// again after Submit, when its object may be gone (here an 8 KiB SetBytes temporary).
void TestBindingsEndAtSubmit() {
    Metal::Device device;
    Metal::Encoder encoder{device};
    id<MTLLibrary> library = TestLibrary(device);
    id<MTLComputePipelineState> copy = Kernel(device, library, @"copy_constants");
    std::vector<u32> large(2048, 7u), small(64, 9u);
    id<MTLBuffer> out = SharedBuffer(device, 64 * 4);
    encoder.SetComputePipeline((__bridge void*)copy);
    encoder.SetBuffer(Metal::Stage::Compute, 0, Ref(out));
    encoder.SetBytes(Metal::Stage::Compute, 1, small.data(), u32(small.size() * 4));
    encoder.SetBytes(Metal::Stage::Compute, 2, large.data(), u32(large.size() * 4)); // unused slot
    encoder.Dispatch({1, 1, 1}, {64, 1, 1});
    device.Wait(encoder.Submit());
    for (int i = 0; i < 100; ++i) {
        [NSThread sleepForTimeInterval:0.001]; // the temporary is released after completion
    }
    id<MTLBuffer> out2 = SharedBuffer(device, 64 * 4);
    encoder.SetComputePipeline((__bridge void*)copy);
    encoder.SetBuffer(Metal::Stage::Compute, 0, Ref(out2));
    encoder.SetBytes(Metal::Stage::Compute, 1, small.data(), u32(small.size() * 4));
    encoder.Dispatch({1, 1, 1}, {64, 1, 1}); // must not touch slot 2's freed buffer
    device.Wait(encoder.Submit());
    Expect(static_cast<const u32*>(out2.contents)[63] == 9u, "a new command buffer starts unbound");
}

/// The recorder is a plain thread with no autorelease pool: recording must not leave
/// autoreleased objects behind on it (they would pile up until the thread exits). A pass
/// descriptor holds its attachment textures, so a leaked descriptor keeps the texture alive.
void TestNoLeakOnPoollessThread() {
    Metal::Device device;
    bool released = false;
    std::thread recorder([&] {
        Metal::Encoder encoder{device};
        __weak id<MTLTexture> weak;
        {
            id<MTLTexture> target = Texture(device, kSize, kSize);
            weak = target;
            encoder.BeginRenderPass(ColorPass(target, false, 0));
            encoder.EndRenderPass();
        }
        released = weak == nil; // still on the thread: its implicit pool has not drained
        device.Wait(encoder.Submit());
    });
    recorder.join();
    Expect(released, "recording on a thread without a pool leaves nothing autoreleased");
}

/// A copy recorded inside a clearing pass before its first draw sees the cleared target.
void TestCopyInPassSeesClear() {
    Metal::Device device;
    Metal::Encoder encoder{device};
    id<MTLTexture> target = Texture(device, kSize, kSize);
    FillTexture(device, encoder, target, kBlue);
    encoder.BeginRenderPass(ColorPass(target, true, kYellow));
    id<MTLBuffer> out = Readback(device, encoder, target); // before any draw
    encoder.EndRenderPass();
    device.Wait(encoder.Submit());
    Expect(Pixel(out, kSize, 4, 4) == kYellow, "the pass's clear happens before a copy inside it");
}

/// A scissor carried from a bigger pass is clamped to the next, smaller target (Metal requires
/// the scissor inside the render target; Vulkan does not).
void TestScissorClampedToTarget() {
    Metal::Device device;
    Metal::Encoder encoder{device};
    id<MTLLibrary> library = TestLibrary(device);
    id<MTLRenderPipelineState> solid = SolidPipeline(device, library, @"fullscreen");
    id<MTLTexture> big = Texture(device, kSize, kSize);
    id<MTLTexture> small = Texture(device, kSize / 2, kSize / 2);
    encoder.BeginRenderPass(ColorPass(big, true, kBlack));
    encoder.SetRenderPipeline((__bridge void*)solid);
    encoder.SetViewport({0, 0, float(kSize), float(kSize), 0, 1});
    encoder.SetScissor({0, 0, kSize, kSize});
    SetColor(encoder, kRed);
    encoder.Draw(Metal::Primitive::Triangle, 3);
    encoder.EndRenderPass();
    Metal::RenderPassDesc second = ColorPass(small, true, kBlack);
    second.width = second.height = kSize / 2;
    encoder.BeginRenderPass(second);
    encoder.SetViewport({0, 0, float(kSize / 2), float(kSize / 2), 0, 1});
    encoder.Draw(Metal::Primitive::Triangle, 3); // scissor 64x64 carried into a 32x32 target
    encoder.EndRenderPass();
    id<MTLBuffer> out = SharedBuffer(device, kSize * kSize);
    encoder.CopyTextureToBuffer((__bridge void*)small, 0, 0, {}, {kSize / 2, kSize / 2, 1}, Ref(out),
                                kSize / 2 * 4, kSize * kSize);
    device.Wait(encoder.Submit());
    Expect(Pixel(out, kSize / 2, kSize / 2 - 1, kSize / 2 - 1) == kRed,
           "a draw with a carried, too-large scissor fills the smaller target");
}

void TestFillPattern() {
    Metal::Device device;
    Metal::Encoder encoder{device};
    id<MTLBuffer> b = SharedBuffer(device, 64);
    encoder.FillBuffer(Ref(b), 32, 0x0000000Fu);       // HTile-style pattern: bytes differ
    encoder.FillBuffer(Ref(b, 32), 32, 0x5a5a5a5au);   // uniform bytes: blit
    device.Wait(encoder.Submit());
    const u32* w = static_cast<const u32*>(b.contents);
    bool ok = true;
    for (int i = 0; i < 8; ++i) ok &= w[i] == 0x0000000Fu;
    for (int i = 8; i < 16; ++i) ok &= w[i] == 0x5a5a5a5au;
    Expect(ok, "FillBuffer writes the exact 32-bit pattern");
}

void TestSubmitResetsBindings() {
    Metal::Device device;
    Metal::Encoder encoder{device};
    id<MTLBuffer> b = SharedBuffer(device, 16);
    encoder.SetBuffer(Metal::Stage::Compute, 3, Ref(b));
    Expect(encoder.HasBindings(), "a binding is recorded");
    device.Wait(encoder.Submit());
    Expect(!encoder.HasBindings(), "Submit clears bindings (objects may be freed after it)");
}

void TestUntrackedRunsFirst() {
    Metal::Device device;
    Metal::Encoder upload{device}, main_encoder{device};
    id<MTLBuffer> src = SharedBuffer(device, 64), mid = SharedBuffer(device, 64),
                  dst = SharedBuffer(device, 64);
    std::memset(src.contents, 0x11, 64);
    upload.CopyBuffer(Ref(src), Ref(mid), 64);   // "hoisted" upload
    main_encoder.CopyBuffer(Ref(mid), Ref(dst), 64);
    upload.SubmitUntracked();
    const u64 tick = main_encoder.Submit();
    Expect(tick == 1, "an untracked commit takes no tick");
    device.Wait(tick);
    Expect(static_cast<const u8*>(dst.contents)[63] == 0x11,
           "the untracked command buffer ran before the tracked one");
}

void TestIndirectResidency() {
    // A kernel reads a buffer only through its GPU address (as shaders read the arena).
    Metal::Device device;
    Metal::Encoder encoder{device};
    id<MTLDevice> dev = (__bridge id<MTLDevice>)device.NativeDevice();
    NSError* error = nil;
    id<MTLLibrary> lib = [dev newLibraryWithSource:@R"(
#include <metal_stdlib>
using namespace metal;
kernel void through_address(device uint* out [[buffer(0)]], constant ulong& address [[buffer(1)]],
                            uint i [[thread_position_in_grid]]) {
    out[i] = ((device const uint*)address)[i];
}
)" options:nil error:&error];
    id<MTLComputePipelineState> k =
        [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"through_address"] error:&error];
    id<MTLBuffer> hidden = [dev newBufferWithLength:256 options:MTLResourceStorageModePrivate];
    id<MTLBuffer> source = SharedBuffer(device, 256), out = SharedBuffer(device, 256);
    for (u32 i = 0; i < 64; ++i) static_cast<u32*>(source.contents)[i] = 300 + i;
    device.MakeResident((__bridge void*)hidden);
    encoder.DeclareIndirect((__bridge void*)hidden);
    encoder.CopyBuffer(Ref(source), Ref(hidden), 256);  // written by a blit, read by address
    const u64 address = hidden.gpuAddress;
    encoder.SetComputePipeline((__bridge void*)k);
    encoder.SetBuffer(Metal::Stage::Compute, 0, Ref(out));
    encoder.SetBytes(Metal::Stage::Compute, 1, &address, sizeof(address));
    encoder.Dispatch({1, 1, 1}, {64, 1, 1});
    device.Wait(encoder.Submit());
    Expect(static_cast<const u32*>(out.contents)[63] == 363, "a resident buffer reads by address");
    device.Evict((__bridge void*)hidden);
}

void TestDepthAspectCopy() {
    Metal::Device device;
    Metal::Encoder encoder{device};
    id<MTLDevice> dev = (__bridge id<MTLDevice>)device.NativeDevice();
    MTLTextureDescriptor* d = [MTLTextureDescriptor
        texture2DDescriptorWithPixelFormat:MTLPixelFormatDepth32Float_Stencil8 width:4 height:4 mipmapped:NO];
    d.usage = MTLTextureUsageRenderTarget;
    d.storageMode = MTLStorageModePrivate;
    id<MTLTexture> ds = [dev newTextureWithDescriptor:d];
    Metal::RenderPassDesc pass;
    pass.depth.texture = (__bridge void*)ds;
    pass.depth.has_stencil = true;
    pass.depth.clear_depth = pass.depth.clear_stencil = true;
    pass.depth.depth = 0.5;
    pass.depth.stencil = 7;
    encoder.BeginRenderPass(pass);
    encoder.EndRenderPass();
    id<MTLBuffer> depth = SharedBuffer(device, 64), stencil = SharedBuffer(device, 16);
    encoder.CopyTextureToBuffer((__bridge void*)ds, 0, 0, {}, {4, 4, 1}, Ref(depth), 16, 64,
                                Metal::BlitAspect::Depth);
    encoder.CopyTextureToBuffer((__bridge void*)ds, 0, 0, {}, {4, 4, 1}, Ref(stencil), 4, 16,
                                Metal::BlitAspect::Stencil);
    device.Wait(encoder.Submit());
    Expect(static_cast<const float*>(depth.contents)[5] == 0.5f, "depth aspect copied");
    Expect(static_cast<const u8*>(stencil.contents)[5] == 7, "stencil aspect copied");
}

} // namespace

/// Two schedulers' timelines on one device: each counts its own commits 1, 2, ... and a tick of
/// one timeline means that timeline's command buffer, whatever the other committed in between.
void TestTimelines() {
    Metal::Device device;
    Metal::Encoder enc_a{device}, enc_b{device};
    Metal::Timeline a{device}, b{device};
    Expect(a.CurrentTick() == 1 && b.CurrentTick() == 1 && a.KnownGpuTick() == 0,
           "timelines start at 1");
    Expect(a.IsFree(0), "tick 0 is always free");
    Expect(!a.IsFree(1), "an uncommitted logical tick is not free");

    id<MTLBuffer> buffer = SharedBuffer(device, 4096);
    enc_a.FillBuffer(Ref(buffer), 4096, 0x01010101u);
    const u64 a1 = a.Committed(enc_a.Submit());
    enc_b.FillBuffer(Ref(buffer), 4096, 0x02020202u);
    const u64 b1 = b.Committed(enc_b.Submit());
    enc_b.FillBuffer(Ref(buffer), 4096, 0x03030303u);
    const u64 b2 = b.Committed(enc_b.Submit());

    // A's second command buffer waits for an event the CPU signals later.
    id<MTLDevice> dev = (__bridge id<MTLDevice>)device.NativeDevice();
    id<MTLSharedEvent> gate = [dev newSharedEvent];
    id<MTLCommandBuffer> gated = (__bridge id<MTLCommandBuffer>)enc_a.CommandBuffer();
    [gated encodeWaitForEvent:gate value:1];
    enc_a.FillBuffer(Ref(buffer), 4096, 0x04040404u);
    const u64 a2 = a.Committed(enc_a.Submit());
    Expect(a1 == 1 && a2 == 2 && b1 == 1 && b2 == 2, "each timeline numbers its own commits");
    Expect(a.CurrentTick() == 3 && b.CurrentTick() == 3, "each timeline advances on its own");

    b.Wait(b2);
    Expect(b.IsFree(2) && b.KnownGpuTick() == 2, "B's tick 2 is its own second command buffer");
    a.Wait(a1);
    Expect(a.IsFree(1), "A's tick 1 done");
    Expect(!a.IsFree(2), "A's tick 2 is A's gated command buffer, not B's completed one");
    Expect(a.KnownGpuTick() == 1, "A knows only its own completed ticks");
    Expect(!a.IsFree(3), "A's uncommitted tick is not free");

    std::atomic<bool> waited{false};
    std::thread waiter([&] {
        a.Wait(a2);
        waited.store(true);
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    Expect(!waited.load(), "Wait on A's tick 2 blocks while A's command buffer is gated");
    gate.signaledValue = 1;
    waiter.join();
    Expect(gated.status == MTLCommandBufferStatusCompleted, "Wait returned after A's own buffer");
    Expect(a.IsFree(2) && a.KnownGpuTick() == 2, "A's tick 2 done after the gate opened");

    // Many commits, waited on out of order: pruning keeps every answer right and the map small.
    u64 last_known = a.KnownGpuTick();
    bool monotonic = true;
    for (int i = 0; i < 300; ++i) {
        enc_a.FillBuffer(Ref(buffer), 4096, u32(i & 0xff) * 0x01010101u);
        a.Committed(enc_a.Submit());
        if (i % 3 == 0) {
            enc_b.FillBuffer(Ref(buffer), 4096, 0);
            b.Committed(enc_b.Submit());
        }
        const u64 known = a.KnownGpuTick();
        monotonic = monotonic && known >= last_known;
        last_known = known;
    }
    Expect(monotonic, "KnownGpuTick never goes back");
    Expect(a.CurrentTick() == 303 && b.CurrentTick() == 103, "ticks after many commits");
    a.Wait(150);
    Expect(a.IsFree(150), "a tick in the middle waited for");
    a.Wait(302);
    bool all_free = true;
    for (u64 t = 0; t <= 302; ++t) {
        all_free = all_free && a.IsFree(t);
    }
    Expect(all_free && a.KnownGpuTick() == 302, "every committed tick is free after the last");
    Expect(!a.IsFree(303), "the next tick is still not free");
    Expect(a.PendingCount() == 0, "completed mappings are dropped");
    device.Finish();
    Expect(b.KnownGpuTick() == 102 && b.PendingCount() == 0, "B's mappings pruned too");
}

void TestBufferCreation() {
    Metal::Device device;
    u64 stream_address = 0, private_address = 0;
    u8 *stream_contents = nullptr, *private_contents = nullptr;
    Metal::Object stream = Metal::CreateBuffer(device, 64, VideoCore::MemoryType::Stream, "test",
                                               &stream_address, &stream_contents);
    Metal::Object priv = Metal::CreateBuffer(device, 64, VideoCore::MemoryType::DeviceLocal, "",
                                             &private_address, &private_contents);
    Expect(stream && priv, "buffers created");
    Expect(stream_contents != nullptr, "Stream buffer has CPU contents");
    Expect(private_contents == nullptr, "DeviceLocal buffer has no CPU contents");
    Expect(stream_address != 0 && private_address != 0, "both buffers have a GPU address");
}

void TestTextures() {
    Metal::Device device;
    u64 rgba_size = 0, bc1_size = 0;
    Metal::TextureDesc rgba{};
    rgba.pixel_format = MTLPixelFormatRGBA8Unorm;
    rgba.width = rgba.height = 4;
    rgba.render_target = rgba.shader_write = rgba.format_views = true;
    Metal::Object rgba_tex = Metal::CreateTexture(device, rgba, "rgba", &rgba_size);
    Metal::TextureDesc bc1{};
    bc1.pixel_format = MTLPixelFormatBC1_RGBA;
    bc1.width = bc1.height = 8;
    Metal::Object bc1_tex = Metal::CreateTexture(device, bc1, "", &bc1_size);
    Expect(rgba_tex && bc1_tex, "4x4 RGBA8 and 8x8 BC1 textures created");
    Expect(rgba_size > 0 && bc1_size > 0, "textures report an allocated size");
    if (rgba_tex) {
        id<MTLTexture> t = (__bridge id<MTLTexture>)rgba_tex.Get();
        Expect(t.storageMode == MTLStorageModePrivate, "texture storage is private");
        Expect((t.usage & (MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget |
                           MTLTextureUsageShaderWrite | MTLTextureUsagePixelFormatView)) ==
                   (MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget |
                    MTLTextureUsageShaderWrite | MTLTextureUsagePixelFormatView),
               "texture usage follows the desc");
    }
    if (bc1_tex) {
        id<MTLTexture> t = (__bridge id<MTLTexture>)bc1_tex.Get();
        Expect(t.usage == MTLTextureUsageShaderRead, "BC1 texture is shader-read only");
    }
    Expect(Metal::BytesPerBlock(MTLPixelFormatBC1_RGBA) == 8, "BC1 block is 8 bytes");
    Expect(Metal::BlockWidth(MTLPixelFormatBC1_RGBA) == 4 &&
               Metal::BlockHeight(MTLPixelFormatBC1_RGBA) == 4,
           "BC1 block is 4x4");
    Expect(Metal::BytesPerBlock(MTLPixelFormatRGBA8Unorm) == 4 &&
               Metal::BlockWidth(MTLPixelFormatRGBA8Unorm) == 1,
           "RGBA8 texel is 4 bytes");
    Expect(Metal::BytesPerBlock(MTLPixelFormatInvalid) == 0, "unknown format has no size");
    Expect(Metal::StencilViewPixelFormat(MTLPixelFormatDepth32Float_Stencil8) ==
                   MTLPixelFormatX32_Stencil8 &&
               Metal::StencilViewPixelFormat(MTLPixelFormatDepth32Float) == 0,
           "stencil view formats");

    Metal::Object view = Metal::CreateTextureView(
        rgba_tex.Get(), MTLPixelFormatRGBA8Unorm_sRGB, Metal::TextureType::Tex2D, 0, 1, 0, 1,
        {MTLTextureSwizzleBlue, MTLTextureSwizzleGreen, MTLTextureSwizzleRed,
         MTLTextureSwizzleOne});
    Expect(bool(view), "swizzled sRGB view created");
    if (view) {
        id<MTLTexture> v = (__bridge id<MTLTexture>)view.Get();
        Expect(v.swizzle.red == MTLTextureSwizzleBlue && v.swizzle.alpha == MTLTextureSwizzleOne,
               "view carries the swizzle");
        Expect(v.pixelFormat == MTLPixelFormatRGBA8Unorm_sRGB, "view has the requested format");
    }

    Metal::SamplerDesc sd{};
    sd.min_filter = sd.mag_filter = MTLSamplerMinMagFilterLinear;
    sd.mip_filter = MTLSamplerMipFilterLinear;
    sd.address_u = sd.address_v = sd.address_w = MTLSamplerAddressModeClampToEdge;
    sd.max_lod = 4.0f;
    sd.max_anisotropy = 64; // clamped to 16
    sd.compare = true;
    sd.compare_function = MTLCompareFunctionLessEqual;
    Metal::Object sampler = Metal::CreateSampler(device, sd);
    Expect(bool(sampler), "sampler created");
    Expect(Metal::SupportsSampleCount(device, 1) && Metal::SupportsSampleCount(device, 4),
           "1x and 4x samples supported");
}

int main() {
    @autoreleasepool {
        TestObject();
        TestTicks();
        TestDeviceLifetime();
        TestBlits();
        TestKeepAlive();
        TestWaitMeansCompleted();
        TestCompute();
        TestLargeBytes();
        TestClearOnlyPass();
        TestPassReopenKeepsDrawing();
        TestStatePersistsAcrossPasses();
        TestIndexedDraw();
        TestBindingsEndAtSubmit();
        TestNoLeakOnPoollessThread();
        TestCopyInPassSeesClear();
        TestScissorClampedToTarget();
        TestFillPattern();
        TestSubmitResetsBindings();
        TestUntrackedRunsFirst();
        TestIndirectResidency();
        TestDepthAspectCopy();
        TestSpecialize();
        TestDepthStencilCache();
        TestTimelines();
        TestBufferCreation();
        TestTextures();
    }
    if (failures) {
        std::printf("FAILED: %d check(s)\n", failures);
        return 1;
    }
    std::printf("PASS: Metal-native foundation (object, device ticks, encoder)\n");
    return 0;
}
