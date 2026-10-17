// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: MTLBuffer creation for the buffer cache (Metal-native renderer).

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "video_core/renderer_metal/mtl_buffer.h"
#include "video_core/renderer_metal/mtl_device.h"

namespace Metal {

namespace {

MTLResourceOptions OptionsFor(const Device& device, VideoCore::MemoryType type) {
    switch (type) {
    case VideoCore::MemoryType::DeviceLocal:
        // Untracked: the Encoder orders them (mtl_order.h); shared buffers stay tracked.
        return MTLResourceStorageModePrivate |
               (device.HazardTracking() ? 0 : MTLResourceHazardTrackingModeUntracked);
    case VideoCore::MemoryType::HostUncached:
        return MTLResourceStorageModeShared | MTLResourceCPUCacheModeWriteCombined;
    case VideoCore::MemoryType::HostCached:
    case VideoCore::MemoryType::Stream:
    default:
        return MTLResourceStorageModeShared;
    }
}

} // namespace

Object CreateBuffer(Device& device, u64 size, VideoCore::MemoryType type, std::string_view label,
                    u64* gpu_address, u8** contents) {
    *gpu_address = 0;
    *contents = nullptr;
    id<MTLDevice> mtl_device = (__bridge id<MTLDevice>)device.NativeDevice();
    id<MTLBuffer> buffer = [mtl_device newBufferWithLength:size options:OptionsFor(device, type)];
    if (buffer == nil) {
        return {};
    }
    if (!label.empty()) {
        buffer.label = [[NSString alloc] initWithBytes:label.data()
                                                length:label.size()
                                              encoding:NSUTF8StringEncoding];
    }
    *gpu_address = buffer.gpuAddress;
    if (type != VideoCore::MemoryType::DeviceLocal) {
        *contents = static_cast<u8*>(buffer.contents);
    }
    return Object::Adopt((__bridge_retained void*)buffer);
}

} // namespace Metal
