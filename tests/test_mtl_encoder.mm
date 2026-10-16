// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: unit test for the Metal-native foundation (docs/superpowers/plans/
// 2026-10-16-metal-native-m1-device-encoder.md): Metal::Object, Metal::Device ticks and limits,
// Metal::Encoder blits, dispatches, render passes and draws, on the system's GPU. No window.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <atomic>
#include <cstdio>
#include <cstring>
#include <vector>
#include "video_core/renderer_metal/mtl_object.h"
#include "video_core/renderer_metal/mtl_device.h"

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

} // namespace

int main() {
    @autoreleasepool {
        TestObject();
        TestTicks();
    }
    if (failures) {
        std::printf("FAILED: %d check(s)\n", failures);
        return 1;
    }
    std::printf("PASS: Metal-native foundation (object, device ticks, encoder)\n");
    return 0;
}
