// SPDX-License-Identifier: GPL-2.0-or-later

#include <cstdio>
#include "common/assert.h"
#include "video_core/renderer_metal/mtl_device_impl.h"

namespace Metal {

Device::Device(void* mtl_device, void* mtl_queue) : impl{std::make_unique<Impl>()} {
    @autoreleasepool {
        impl->device = mtl_device ? (__bridge id<MTLDevice>)mtl_device : MTLCreateSystemDefaultDevice();
        ASSERT_MSG(impl->device, "No Metal device");
        impl->queue = mtl_queue ? (__bridge id<MTLCommandQueue>)mtl_queue : [impl->device newCommandQueue];
        ASSERT_MSG(impl->queue, "No Metal command queue");
        name = impl->device.name.UTF8String;
    }
}

Device::~Device() = default;

} // namespace Metal
