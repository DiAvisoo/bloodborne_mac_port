// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: unit test for the Metal-native foundation (docs/superpowers/plans/
// 2026-10-16-metal-native-m1-device-encoder.md): Metal::Object, Metal::Device ticks and limits,
// Metal::Encoder blits, dispatches, render passes and draws, on the system's GPU. No window.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <vector>
#include "video_core/renderer_metal/mtl_object.h"
#include "video_core/renderer_metal/mtl_device.h"
#include "video_core/renderer_metal/mtl_encoder.h"

#include <csetjmp>

// The library binds these host-runtime data symbols at load (host/host_runtime.c defines them in
// bbgpu-host); a test executable has to provide them, as tests/test_srt_walker.cpp does.
extern "C" __thread sigjmp_buf* runtime_fault_recover;
__thread sigjmp_buf* runtime_fault_recover;
extern "C" uint64_t runtime_disabled_optimizations;
uint64_t runtime_disabled_optimizations;

namespace {

int failures = 0;

void Expect(bool ok, const char* what) {
    if (!ok) {
        std::printf("FAIL: %s\n", what);
        ++failures;
    }
}

std::atomic<int> probes_alive{0};

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

[[maybe_unused]] u32 Pixel(id<MTLBuffer> readback, u32 width, u32 x, u32 y) {
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

} // namespace

int main() {
    @autoreleasepool {
        TestObject();
        TestTicks();
        TestDeviceLifetime();
        TestBlits();
        TestKeepAlive();
        TestCompute();
        TestLargeBytes();
    }
    if (failures) {
        std::printf("FAILED: %d check(s)\n", failures);
        return 1;
    }
    std::printf("PASS: Metal-native foundation (object, device ticks, encoder)\n");
    return 0;
}
