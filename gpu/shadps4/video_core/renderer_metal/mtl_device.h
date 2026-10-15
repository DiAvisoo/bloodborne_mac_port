// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: the Metal device and queue our own encoders use (docs/DESIGN_METAL_BACKEND.md). The
// interface follows Huax-1839/bloodborne-macos's Metal::Device (GPL-2.0-or-later); here the
// device and queue are normally MoltenVK's (VK_EXT_metal_objects), so our command buffers and
// MoltenVK's are ordered on one queue.

#pragma once

#include <memory>
#include <string>

namespace Metal {

class Device {
public:
    /// `mtl_device`/`mtl_queue`: id<MTLDevice>/id<MTLCommandQueue> to use (MoltenVK's); null: the
    /// system default device and a queue of our own (tools and tests).
    explicit Device(void* mtl_device = nullptr, void* mtl_queue = nullptr);
    ~Device();

    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;

    struct Impl;
    Impl& GetImpl() const {
        return *impl;
    }

    const std::string& Name() const {
        return name;
    }

private:
    std::unique_ptr<Impl> impl;
    std::string name;
};

} // namespace Metal
