// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: Metal::Device (Metal-native renderer, spec component 1).

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "common/assert.h"
#include "common/logging/log.h"
#include "video_core/renderer_metal/mtl_device_impl.h"

namespace Metal {

namespace {

void RaiseTo(std::atomic<u64>& value, u64 tick) {
    u64 known = value.load(std::memory_order_acquire);
    while (known < tick &&
           !value.compare_exchange_weak(known, tick, std::memory_order_acq_rel)) {
    }
}

/// BB_METAL_HAZARD_TRACKING=1: tracked buffers, no fences (read once, logged once).
bool HazardTrackingRequested() {
    static const bool requested = [] {
        const char* env = std::getenv("BB_METAL_HAZARD_TRACKING");
        const bool on = env && std::strcmp(env, "1") == 0;
        if (on) {
            std::printf("Metal: hazard tracking on every buffer (BB_METAL_HAZARD_TRACKING=1); "
                        "no explicit ordering\n");
        }
        return on;
    }();
    return requested;
}

} // namespace

Device::Device(void* mtl_device, void* mtl_queue) : impl{std::make_unique<Impl>()} {
    @autoreleasepool {
        impl->device = mtl_device ? (__bridge id<MTLDevice>)mtl_device : MTLCreateSystemDefaultDevice();
        ASSERT_MSG(impl->device, "No Metal device");
        impl->queue = mtl_queue ? (__bridge id<MTLCommandQueue>)mtl_queue : [impl->device newCommandQueue];
        ASSERT_MSG(impl->queue, "No Metal command queue");
        impl->event = [impl->device newSharedEvent];
        ASSERT_MSG(impl->event, "No Metal shared event");
        MTLResidencySetDescriptor* rd = [MTLResidencySetDescriptor new];
        rd.label = @"bbport arena and address-reached buffers";
        NSError* error = nil;
        impl->residency = [impl->device newResidencySetWithDescriptor:rd error:&error];
        ASSERT_MSG(impl->residency, "No Metal residency set: {}",
                   error ? error.localizedDescription.UTF8String : "unknown");
        [impl->queue addResidencySet:impl->residency];
        name = impl->device.name.UTF8String;
        limits.max_buffer_length = impl->device.maxBufferLength;
        limits.recommended_working_set = impl->device.recommendedMaxWorkingSetSize;
        hazard_tracking = HazardTrackingRequested();
        if (!hazard_tracking) {
            // Every fence gets one update before any wait, so waits never depend on whether a
            // fence was used yet (OrderTracker::OpenAhead waits on all of them).
            id<MTLCommandBuffer> prime = [impl->queue commandBuffer];
            id<MTLBlitCommandEncoder> blit = [prime blitCommandEncoder];
            for (u32 i = 0; i < OrderTracker::NumFences; ++i) {
                impl->fences[i] = [impl->device newFence];
                ASSERT_MSG(impl->fences[i], "No Metal fence");
                [blit updateFence:impl->fences[i]];
            }
            [blit endEncoding];
            [prime commit];
            [prime waitUntilCompleted];
        }
    }
}

Device::~Device() {
    Finish();
}

u64 Device::AllocatedSize() const {
    return impl->device.currentAllocatedSize;
}

void* Device::NativeDevice() const noexcept {
    return (__bridge void*)impl->device;
}

void* Device::NativeQueue() const noexcept {
    return (__bridge void*)impl->queue;
}

bool Device::IsFree(u64 tick) const {
    return KnownGpuTick() >= tick;
}

void Device::Wait(u64 tick) {
    ASSERT_MSG(tick < CurrentTick(), "Waiting for tick {} that was never committed", tick);
    // Completion handlers raise the known tick and wake waiters (also for failed command buffers).
    for (u64 known = KnownGpuTick(); known < tick; known = KnownGpuTick()) {
        counters->gpu_tick.wait(known, std::memory_order_acquire);
    }
}

void Device::Finish() {
    const u64 last = CurrentTick() - 1;
    if (last != 0) {
        Wait(last);
    }
}

void Device::CommitUntracked(void* command_buffer) {
    TrackGpuTime(command_buffer);
    [(__bridge id<MTLCommandBuffer>)command_buffer commit];
}

void Device::TrackGpuTime(void* command_buffer) {
    std::shared_ptr<Counters> shared = counters;
    [(__bridge id<MTLCommandBuffer>)command_buffer addCompletedHandler:^(id<MTLCommandBuffer> done) {
      const double start = done.GPUStartTime, end = done.GPUEndTime;
      if (!(end > start) || start <= 0) {
          return; // never ran (error) or no work
      }
      std::scoped_lock lock{shared->busy_mutex};
      shared->gpu_busy_ns.fetch_add(u64(shared->busy.Add(start, end) * 1e9),
                                    std::memory_order_relaxed);
    }];
}

void Device::MakeResident(void* buffer) {
    std::scoped_lock lock{residency_mutex};
    id<MTLBuffer> b = (__bridge id<MTLBuffer>)buffer;
    [impl->residency addAllocation:b];
    [impl->residency commit];
    if (resident.emplace(buffer, b.length).second) {
        resident_bytes += b.length;
    }
}

void Device::Evict(void* buffer) {
    std::scoped_lock lock{residency_mutex};
    [impl->residency removeAllocation:(__bridge id<MTLBuffer>)buffer];
    [impl->residency commit];
    if (const auto it = resident.find(buffer); it != resident.end()) {
        resident_bytes -= it->second;
        resident.erase(it);
    }
}

u64 Device::ResidentBytes() const {
    std::scoped_lock lock{residency_mutex};
    return resident_bytes;
}

u64 Device::Commit(void* command_buffer) {
    id<MTLCommandBuffer> cmd = (__bridge id<MTLCommandBuffer>)command_buffer;
    const u64 tick = current_tick.fetch_add(1, std::memory_order_acq_rel);
    [cmd encodeSignalEvent:impl->event value:tick];
    // Before the tick handler: when Wait(tick) returns, this command buffer's time is counted.
    TrackGpuTime(command_buffer);
    std::shared_ptr<Counters> shared = counters;
    [cmd addCompletedHandler:^(id<MTLCommandBuffer> done) {
        if (done.status == MTLCommandBufferStatusError) {
            shared->errors.fetch_add(1, std::memory_order_relaxed);
            LOG_ERROR(Render, "Metal command buffer of tick {} failed: {}", tick,
                      done.error ? done.error.localizedDescription.UTF8String : "unknown error");
        }
        RaiseTo(shared->gpu_tick, tick);
        shared->gpu_tick.notify_all();
    }];
    [cmd commit];
    return tick;
}

} // namespace Metal
