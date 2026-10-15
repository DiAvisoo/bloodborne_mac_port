// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: the Metal back end's entry into the Vulkan renderer (docs/DESIGN_METAL_BACKEND.md, M2).
// The renderer records its GPU work as vk::CommandBuffer calls, which go through vulkan.hpp's
// dynamic dispatcher. Install() replaces the command-buffer functions in that dispatcher with ours;
// each forwards to MoltenVK unless the Metal path owns the command buffer (later steps).
// Step A: forwarding plus counts (BB_METAL_TRACE=1 prints the command stream every 5 s).

#pragma once

namespace MetalLayer {

/// `dispatcher`: VULKAN_HPP_DEFAULT_DISPATCHER, right after its init(device). Installs only when
/// BB_METAL_TRACE=1 or the Metal path is on; otherwise the dispatcher is left untouched.
void Install(void* dispatcher);

} // namespace MetalLayer
