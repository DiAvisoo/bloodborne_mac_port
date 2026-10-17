// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: explicit ordering of untracked buffers (Metal-native renderer). Pure C++: which
// MTLFences a Metal encoder waits on when it opens and updates when it ends.
//
// Why: Metal's automatic hazard tracking on the guest arena and the fault buffer (bound writable
// to nearly every encoder) made each encoder wait for the one before it (docs/FINDINGS.md,
// 2026-10-17). Device-local buffers are therefore created untracked (mtl_buffer.mm) and ordered
// with fences. Textures and shared (CPU-visible) buffers stay tracked: Metal orders those.
//
// The rule: barriers between groups of encoders. Every Metal encoder that touches untracked
// memory of a class takes a slot: fences it updates when it ends, and for a render encoder one
// more after its vertex stage. Encoders do not wait for each other unless a barrier comes
// between them. A barrier names what it waits for (BarrierScope): the render encoders, their
// vertex stages only, the copy encoders, the compute encoders or the uploads ahead, opened since
// that scope was last waited for; every encoder opened after it waits for those fences (before the vertex
// stage of render encoders). Encoders of kinds the barrier does not name keep overlapping.
// Waits carry over to later encoders until an encoder that already waited for them is itself
// waited for (it stands for them).
// Who decides where barriers go:
//   - the renderer, for the buffers it orders itself (the guest arena: it knows every range a
//     command reads and writes, by which kind of encoder and at which stage,
//     mtl_buffer_barriers.h, and records the barrier a conflicting command needs: RAW, WAR and
//     WAW, not RAR);
//   - this tracker, for every other untracked buffer: a write there (Access::Write) is isolated,
//     with full barriers before and after it, so it is ordered after every earlier access and
//     before every later one. Reads there need nothing more (writes are isolated).
// Object motion's position history is its own class (one fence per encoder, after its vertex
// stage): its passes do not conflict within a frame, and a full barrier at frame start orders
// one frame's after the last one's.
//
// Fences are global to one command queue and are evaluated in commit order ("at kernel
// submission", MTLComputeCommandEncoder.h), so the rule holds across command buffers as long as
// the in-order stream commits in the order it opens encoders (one Encoder, one thread).
// A slot is reused only when no barrier still to come or still waited on needs it (slots of
// command buffers known complete are dropped); when every slot is, the reused one is waited on
// first (its end), so its newest updates still stand for every encoder that used it.
//
// Uploads hoisted ahead of the in-order stream's command buffer (OpenAhead) are chained through
// one fence; a barrier with ScopeAhead (the renderer's, before a later user of an upload's
// ranges) makes the encoders after it wait on it. Uploads whose ranges may still be in use by
// work in flight are not hoisted (Runtime::HoistCopy).
// Within one Metal encoder nothing is ordered (as with hazard tracking).

#pragma once

#include <array>
#include <atomic>
#include <bit>
#include <mutex>
#include "common/types.h"
#include "video_core/renderer_metal/mtl_buffer_barriers.h"

namespace Metal {

/// Untracked resources ordered as a whole.
enum class HazardClass : u8 {
    Memory,        ///< every untracked buffer (guest arena, page table, staging, fault buffer)
    MotionHistory, ///< object motion's position history (vertex stage; ordered per frame)
};
constexpr u32 NumHazardClasses = 2;

enum class Access : u8 { None, Read, Write };

/// The kind of a Metal encoder, for scoped barriers (BarrierScope, EncoderKinds).
enum class Lane : u8 {
    Render,  ///< render encoders
    Copy,    ///< blit encoders, and compute encoders of copies (the copy shader, fills)
    Compute, ///< compute encoders of dispatches
};

/// Slots per hazard class: encoders that may still need waiting for. A frame and a half in flight
/// is ~450 Metal encoders; 64 slots ran out about 150 times a frame (each reuse makes the new
/// encoder wait for an unrelated one), BB_FRAME_STATS "busy-slot reuses".
constexpr u32 OrderSlots = 256;

/// A set of slots, a bit each.
struct SlotSet {
    static constexpr u32 Words = OrderSlots / 64;
    std::array<u64, Words> w{};

    void Set(u32 k) noexcept { w[k / 64] |= 1ull << (k % 64); }
    [[nodiscard]] bool Test(u32 k) const noexcept { return (w[k / 64] >> (k % 64)) & 1; }
    [[nodiscard]] bool Any() const noexcept {
        for (u64 x : w) {
            if (x) {
                return true;
            }
        }
        return false;
    }
    SlotSet& operator|=(const SlotSet& o) noexcept {
        for (u32 i = 0; i < Words; ++i) w[i] |= o.w[i];
        return *this;
    }
    /// Removes the slots of `o`.
    SlotSet& Remove(const SlotSet& o) noexcept {
        for (u32 i = 0; i < Words; ++i) w[i] &= ~o.w[i];
        return *this;
    }
    void Clear() noexcept { w = {}; }
    template <typename F>
    void ForEach(F&& f) const {
        for (u32 i = 0; i < Words; ++i) {
            for (u64 x = w[i]; x; x &= x - 1) {
                f(i * 64 + u32(std::countr_zero(x)));
            }
        }
    }
    /// The first slot from `from` (wrapping around) not in the set; OrderSlots: none.
    [[nodiscard]] u32 FirstFree(u32 from) const noexcept {
        for (u32 n = 0; n < OrderSlots; n += 64) {
            const u32 k = (from + n) % OrderSlots;
            // The free bits of the 64 slots from k (crossing a word boundary when unaligned).
            const u32 i = k / 64, b = k % 64;
            u64 free = ~w[i] >> b;
            if (b) {
                free |= ~w[(i + 1) % Words] << (64 - b);
            }
            if (free) {
                return (k + u32(std::countr_zero(free))) % OrderSlots;
            }
        }
        return OrderSlots;
    }
};

/// The fence waits to encode when an encoder opens and the updates to encode when it ends
/// (indices into the device's fence table, OrderTracker::NumFences).
struct OrderPlan {
    /// Per class, at most one fence per slot (the end and vertex wait sets are disjoint) plus,
    /// when every slot is busy, the reused slot's end (whose vertex fence may be waited too);
    /// then the uploads' fence: NumHazardClasses * (OrderSlots + 1) + 1. Updates: a render
    /// slot's two fences per class.
    static constexpr u32 MaxWaits = NumHazardClasses * (OrderSlots + 1) + 1, MaxUpdates = 4;
    std::array<u16, MaxWaits> waits{};
    std::array<u16, MaxUpdates> updates{};
    u32 num_waits{}, num_updates{};

    [[nodiscard]] bool Waits(u32 fence) const noexcept;
    [[nodiscard]] bool Updates(u32 fence) const noexcept;
    [[nodiscard]] bool Empty() const noexcept {
        return num_waits == 0 && num_updates == 0;
    }
};

class OrderTracker {
public:
    static constexpr u32 Slots = OrderSlots; ///< per class
    enum FenceStage : u32 { VertexDone, EncoderDone };
    /// Updated by every encoder committed ahead of the in-order stream.
    static constexpr u32 AheadFence = NumHazardClasses * Slots * 2;
    static constexpr u32 NumFences = AheadFence + 1;

    static constexpr u32 SlotFence(HazardClass c, u32 k, FenceStage s = EncoderDone) {
        return (u32(c) * Slots + k) * 2 + s;
    }
    /// The class of a fence (AheadFence: Memory), and whether it is a vertex-stage fence.
    static constexpr HazardClass ClassOf(u32 fence) {
        return fence >= AheadFence ? HazardClass::Memory : HazardClass(fence / (Slots * 2));
    }
    static constexpr bool IsVertexFence(u32 fence) {
        return fence < AheadFence && (fence & 1) == VertexDone;
    }

    using Accesses = std::array<Access, NumHazardClasses>;

    /// An encoder of the in-order stream opens with `accesses` (per class): Read takes a slot,
    /// Write is isolated (full barriers before and after it), None takes no fence. A render
    /// encoder also updates a fence after its vertex stage (Memory class). `tick`: the device
    /// tick of the command buffer it is in; `completed`: the highest tick known complete (their
    /// slots need no waiting; 0: unknown).
    OrderPlan Open(const Accesses& accesses, Lane lane = Lane::Compute, u64 tick = 0,
                   u64 completed = 0);
    /// What the encoders opened from now on wait for (ScopeNone: nothing).
    void Barrier(HazardClass c, BarrierScopes scopes = ScopeAll);
    /// An encoder of a command buffer committed before the in-order stream's current one (the
    /// scheduler's hoisted uploads, recorded while that one is): waits for the uploads before it,
    /// then updates AheadFence. (Its ranges are not in use by work in flight: Runtime::HoistCopy.)
    /// Every fence must have been updated once before (Device primes them).
    static OrderPlan OpenAhead();
    /// Encoders that found every slot of class `c` busy and reused one, waiting on it first
    /// (a false dependency; BB_FRAME_STATS).
    [[nodiscard]] u64 BusyReuses(HazardClass c) const noexcept {
        return busy_reuses[u32(c)].load(std::memory_order_relaxed);
    }

private:
    struct ClassState {
        // Slots not waited for yet by a barrier of the scope.
        SlotSet render_vertex; ///< render slots whose vertex stage no barrier waited for
        SlotSet render_end;    ///< render slots whose end no barrier waited for
        SlotSet copy_end;      ///< copy slots no barrier waited for
        SlotSet compute_end;   ///< compute slots no barrier waited for
        // What every new encoder waits for, and the barrier generation that installed it.
        SlotSet wait_vertex, wait_end;
        std::array<u32, Slots> vertex_gen{}, end_gen{};
        std::array<u32, Slots> open_gen{}; ///< the generation a slot's encoder opened in
        std::array<u64, Slots> slot_tick{}; ///< its command buffer's device tick (0: unknown)
        SlotSet render;     ///< slots of render encoders (they have a vertex fence)
        u32 gen{};          ///< barriers so far
        u32 next{};         ///< round-robin start of the slot search
        bool isolate{}; ///< the last encoder wrote unordered memory: full barrier before the next
        bool wait_ahead{}; ///< new Memory encoders wait for the uploads ahead (since ahead_gen)
        u32 ahead_gen{};
    };
    void BarrierLocked(ClassState& s, HazardClass c, BarrierScopes scopes);
    std::mutex mutex;
    std::array<ClassState, NumHazardClasses> classes{};
    std::array<std::atomic<u64>, NumHazardClasses> busy_reuses{};
};

} // namespace Metal
