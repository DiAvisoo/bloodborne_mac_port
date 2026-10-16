// SPDX-License-Identifier: GPL-2.0-or-later
#include "bbport_metalfx.h"

#import <Metal/Metal.h>
#import <MetalFX/MetalFX.h>
#define VK_USE_PLATFORM_METAL_EXT
#include <vulkan/vulkan.h>
#include <cstdio>
#include <cstdlib>

namespace BbMetalFx {
namespace {

id<MTLFXTemporalScaler> scaler;
MTLPixelFormat scaler_formats[4];
uint32_t scaler_size[4];
bool scaler_hdr;

void LogOnce(bool& done, const char* message) {
    if (!done) {
        done = true;
        std::printf("MetalFX: %s\n", message);
    }
}

PFN_vkExportMetalObjectsEXT Export(const Frame& f) {
    static PFN_vkExportMetalObjectsEXT export_objects = reinterpret_cast<PFN_vkExportMetalObjectsEXT>(
        f.get_device_proc_addr(f.device, "vkExportMetalObjectsEXT"));
    return export_objects;
}

id<MTLTexture> TextureOf(const Frame& f, VkImage image) {
    VkExportMetalTextureInfoEXT texture{
        .sType = VK_STRUCTURE_TYPE_EXPORT_METAL_TEXTURE_INFO_EXT,
        .image = image,
        .plane = VK_IMAGE_ASPECT_PLANE_0_BIT,
    };
    VkExportMetalObjectsInfoEXT info{.sType = VK_STRUCTURE_TYPE_EXPORT_METAL_OBJECTS_INFO_EXT,
                                     .pNext = &texture};
    Export(f)(f.device, &info);
    return texture.mtlTexture;
}

id<MTLCommandQueue> QueueOf(const Frame& f) {
    VkExportMetalCommandQueueInfoEXT queue{
        .sType = VK_STRUCTURE_TYPE_EXPORT_METAL_COMMAND_QUEUE_INFO_EXT,
        .queue = f.queue,
    };
    VkExportMetalObjectsInfoEXT info{.sType = VK_STRUCTURE_TYPE_EXPORT_METAL_OBJECTS_INFO_EXT,
                                     .pNext = &queue};
    Export(f)(f.device, &info);
    return queue.mtlCommandQueue;
}

} // namespace

bool Supported() {
    if (@available(macOS 13.0, *)) {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        return device && [MTLFXTemporalScalerDescriptor supportsDevice:device];
    }
    return false;
}

void Release() {
    scaler = nil;
}

bool Encode(const Frame& f) {
    static bool no_export, no_textures, no_scaler, usage_warned, encoded_once;
    if (!Export(f)) {
        LogOnce(no_export, "vkExportMetalObjectsEXT unavailable (VK_EXT_metal_objects)");
        return false;
    }
    @autoreleasepool {
        id<MTLTexture> color = TextureOf(f, f.color);
        id<MTLTexture> depth = TextureOf(f, f.depth);
        id<MTLTexture> motion = TextureOf(f, f.motion);
        id<MTLTexture> output = TextureOf(f, f.output);
        id<MTLCommandQueue> queue = QueueOf(f);
        if (!color || !depth || !motion || !output || !queue) {
            LogOnce(no_textures, "could not get the Metal textures or queue behind the Vulkan objects");
            return false;
        }
        const MTLPixelFormat formats[4] = {color.pixelFormat, depth.pixelFormat, motion.pixelFormat,
                                           output.pixelFormat};
        const uint32_t size[4] = {f.width, f.height, f.out_width, f.out_height};
        bool same = scaler != nil && scaler_hdr == f.hdr;
        for (int i = 0; same && i < 4; ++i) {
            same = formats[i] == scaler_formats[i] && size[i] == scaler_size[i];
        }
        if (!same) {
            scaler = nil;
            MTLFXTemporalScalerDescriptor* d = [[MTLFXTemporalScalerDescriptor alloc] init];
            d.colorTextureFormat = formats[0];
            d.depthTextureFormat = formats[1];
            d.motionTextureFormat = formats[2];
            d.outputTextureFormat = formats[3];
            d.inputWidth = f.width;
            d.inputHeight = f.height;
            d.outputWidth = f.out_width;
            d.outputHeight = f.out_height;
            d.autoExposureEnabled = f.hdr; // HDR scene color, as FSR's AUTO_EXPOSURE
            scaler = [d newTemporalScalerWithDevice:queue.device];
            if (!scaler) {
                std::printf("MetalFX: scaler creation failed (color %lu depth %lu motion %lu output "
                            "%lu, %ux%u -> %ux%u)\n",
                            (unsigned long)formats[0], (unsigned long)formats[1],
                            (unsigned long)formats[2], (unsigned long)formats[3], f.width,
                            f.height, f.out_width, f.out_height);
                LogOnce(no_scaler, "temporal scaler unavailable for these inputs");
                return false;
            }
            scaler_hdr = f.hdr;
            for (int i = 0; i < 4; ++i) {
                scaler_formats[i] = formats[i];
                scaler_size[i] = size[i];
            }
            std::printf("MetalFX: temporal scaler %ux%u -> %ux%u\n", f.width, f.height, f.out_width,
                        f.out_height);
        }
        if (!usage_warned && ((color.usage & scaler.colorTextureUsage) != scaler.colorTextureUsage ||
                              (output.usage & scaler.outputTextureUsage) != scaler.outputTextureUsage)) {
            usage_warned = true;
            std::printf("MetalFX: texture usage short of the scaler's (color %lu/%lu, output %lu/%lu)\n",
                        (unsigned long)color.usage, (unsigned long)scaler.colorTextureUsage,
                        (unsigned long)output.usage, (unsigned long)scaler.outputTextureUsage);
        }
        scaler.colorTexture = color;
        scaler.depthTexture = depth;
        scaler.motionTexture = motion;
        scaler.outputTexture = output;
        scaler.inputContentWidth = f.width;
        scaler.inputContentHeight = f.height;
        scaler.jitterOffsetX = f.jitter_x;
        scaler.jitterOffsetY = f.jitter_y;
        scaler.motionVectorScaleX = f.motion_scale_x;
        scaler.motionVectorScaleY = f.motion_scale_y;
        scaler.reset = f.reset;
        scaler.depthReversed = NO;
        if (f.command_buffer) {
            // The Metal frame path's command buffer: in order with the frame, no extra submit.
            [scaler encodeToCommandBuffer:(__bridge id<MTLCommandBuffer>)f.command_buffer];
        } else {
            id<MTLCommandBuffer> cmdbuf = [queue commandBuffer];
            cmdbuf.label = @"bbport MetalFX";
            [scaler encodeToCommandBuffer:cmdbuf];
            [cmdbuf commit];
        }
        if (!encoded_once) {
            encoded_once = true;
            std::printf("MetalFX: first frame encoded\n");
        }
    }
    return true;
}

} // namespace BbMetalFx
