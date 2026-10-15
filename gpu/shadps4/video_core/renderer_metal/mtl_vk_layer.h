// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: the Metal back end's entry into the Vulkan renderer (docs/DESIGN_METAL_BACKEND.md, M2).
// The renderer records its GPU work as vk::CommandBuffer calls, which go through vulkan.hpp's
// dynamic dispatcher. Install() replaces the command-buffer functions in that dispatcher with ours;
// each forwards to MoltenVK unless the Metal path owns the command buffer (later steps).
// Step A: forwarding plus counts (BB_METAL_TRACE=1 prints the command stream every 5 s).

#pragma once

namespace MetalLayer {

/// `dispatcher`: VULKAN_HPP_DEFAULT_DISPATCHER, right after its init(device). Installs only when
/// BB_METAL_TRACE=1 or a Metal path is on (BB_METAL_PRESENT=1); otherwise it stays untouched.
void Install(void* dispatcher);

/// The device and queue (handles as void*: VkDevice, VkQueue) whose Metal objects the layer uses.
void SetDevice(void* device, void* queue);

/// id<MTLDevice> / id<MTLCommandQueue> of MoltenVK's queue (null while the Metal path is off).
void* MtlDevice();
void* MtlQueue();

/// Step B (BB_METAL_PRESENT=1): the presenter's command buffer is encoded in Metal.
bool PresentEnabled();

/// From now until it is submitted, `command_buffer` (a VkCommandBuffer, already begun) is encoded
/// by the Metal path; its commands do not reach MoltenVK.
void Own(void* command_buffer);

} // namespace MetalLayer
