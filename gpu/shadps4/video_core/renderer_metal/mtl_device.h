// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: the Metal device, queue and submission ticks (Metal-native renderer, spec component 1).
// The interface began as Huax-1839/bloodborne-macos's Metal::Device (GPL-2.0-or-later). On the
// metal-backend path the device and queue are MoltenVK's (VK_EXT_metal_objects).

#pragma once

#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include "common/types.h"

namespace Metal {

/// Limits the caches use. MoltenVK's values, so the caches behave as with MoltenVK.
struct Limits {
    u64 max_buffer_length{};       ///< MTLDevice.maxBufferLength
    u64 recommended_working_set{}; ///< MTLDevice.recommendedMaxWorkingSetSize
    u32 uniform_alignment{16};
    u32 storage_alignment{16};
    u32 texel_buffer_alignment{16};
    u32 non_coherent_atom{16};
    float max_anisotropy{16.0f};
    float max_lod_bias{16.0f};
    u32 max_samplers{500000};
};

class Device {
public:
    /// `mtl_device`/`mtl_queue`: id<MTLDevice>/id<MTLCommandQueue> to use (MoltenVK's); null: the
    /// system default device and a queue of our own.
    explicit Device(void* mtl_device = nullptr, void* mtl_queue = nullptr);
    /// Waits for every committed command buffer.
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
    const Limits& GetLimits() const noexcept {
        return limits;
    }
    /// MTLDevice.currentAllocatedSize: bytes the device has allocated for this process.
    [[nodiscard]] u64 AllocatedSize() const;
    /// id<MTLDevice> and id<MTLCommandQueue>, unretained.
    void* NativeDevice() const noexcept;
    void* NativeQueue() const noexcept;

    /// The tick the next Commit() signals. Ticks start at 1 and grow by one per commit.
    [[nodiscard]] u64 CurrentTick() const noexcept {
        return current_tick.load(std::memory_order_acquire);
    }
    /// The highest tick whose command buffer is known to have completed.
    [[nodiscard]] u64 KnownGpuTick() const noexcept {
        return counters->gpu_tick.load(std::memory_order_acquire);
    }
    /// True once every command buffer up to `tick` completed (its completion handlers ran, so
    /// resources it used may be freed; the event is signalled slightly earlier).
    [[nodiscard]] bool IsFree(u64 tick) const;
    /// Blocks until `tick` is done; also returns for command buffers that failed. `tick` must have
    /// been committed (asserted).
    void Wait(u64 tick);
    /// Waits for every command buffer committed so far.
    void Finish();
    /// Commits `command_buffer` (id<MTLCommandBuffer>) so that it signals CurrentTick() when done,
    /// and returns that tick. One thread commits at a time, so ticks follow commit order.
    u64 Commit(void* command_buffer);
    /// Commits `command_buffer` with no tick of its own. It runs before every later commit on the
    /// queue and is covered by the next tracked tick (one queue completes in order).
    void CommitUntracked(void* command_buffer);
    /// Buffers shaders reach only by GPU address (arena, page table, fault buffer): kept resident
    /// through the queue's residency set.
    void MakeResident(void* buffer);
    void Evict(void* buffer);
    /// Command buffers that completed with an error (each one is logged).
    [[nodiscard]] u64 ErrorCount() const noexcept {
        return counters->errors.load(std::memory_order_relaxed);
    }

private:
    std::unique_ptr<Impl> impl;
    std::string name;
    Limits limits;
    std::mutex residency_mutex;
    std::atomic<u64> current_tick{1};
    /// What completion handlers write. Shared with each handler, so a handler that runs after
    /// the Device is gone writes into memory it still owns.
    struct Counters {
        std::atomic<u64> gpu_tick{0};
        std::atomic<u64> errors{0};
    };
    std::shared_ptr<Counters> counters{std::make_shared<Counters>()};
};

} // namespace Metal
