// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: Metal::Device (Metal-native renderer, spec component 1).

#include <atomic>
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

} // namespace

Device::Device(void* mtl_device, void* mtl_queue) : impl{std::make_unique<Impl>()} {
    @autoreleasepool {
        impl->device = mtl_device ? (__bridge id<MTLDevice>)mtl_device : MTLCreateSystemDefaultDevice();
        ASSERT_MSG(impl->device, "No Metal device");
        impl->queue = mtl_queue ? (__bridge id<MTLCommandQueue>)mtl_queue : [impl->device newCommandQueue];
        ASSERT_MSG(impl->queue, "No Metal command queue");
        impl->event = [impl->device newSharedEvent];
        ASSERT_MSG(impl->event, "No Metal shared event");
        name = impl->device.name.UTF8String;
        limits.max_buffer_length = impl->device.maxBufferLength;
        limits.recommended_working_set = impl->device.recommendedMaxWorkingSetSize;
    }
}

Device::~Device() {
    Finish();
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

u64 Device::Commit(void* command_buffer) {
    id<MTLCommandBuffer> cmd = (__bridge id<MTLCommandBuffer>)command_buffer;
    const u64 tick = current_tick.fetch_add(1, std::memory_order_acq_rel);
    [cmd encodeSignalEvent:impl->event value:tick];
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
