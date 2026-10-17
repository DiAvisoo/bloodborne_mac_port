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

/// The fence waits to encode when an encoder opens and the updates to encode when it ends
/// (indices into the device's fence table, OrderTracker::NumFences).
struct OrderPlan {
    static constexpr u32 MaxWaits = NumHazardClasses * 64 + 2, MaxUpdates = 4;
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
    static constexpr u32 Slots = 64; ///< per class (64-bit masks)
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

private:
    struct ClassState {
        // Slots not waited for yet by a barrier of the scope.
        u64 render_vertex{}; ///< render slots whose vertex stage no barrier waited for
        u64 render_end{};    ///< render slots whose end no barrier waited for
        u64 copy_end{};      ///< copy slots no barrier waited for
        u64 compute_end{};   ///< compute slots no barrier waited for
        // What every new encoder waits for, and the barrier generation that installed it.
        u64 wait_vertex{}, wait_end{};
        std::array<u32, Slots> vertex_gen{}, end_gen{};
        std::array<u32, Slots> open_gen{}; ///< the generation a slot's encoder opened in
        std::array<u64, Slots> slot_tick{}; ///< its command buffer's device tick (0: unknown)
        u64 render{};       ///< slots of render encoders (they have a vertex fence)
        u32 gen{};          ///< barriers so far
        u32 next{};         ///< round-robin start of the slot search
        bool isolate{}; ///< the last encoder wrote unordered memory: full barrier before the next
        bool wait_ahead{}; ///< new Memory encoders wait for the uploads ahead (since ahead_gen)
        u32 ahead_gen{};
    };
    void BarrierLocked(ClassState& s, HazardClass c, BarrierScopes scopes);
    std::mutex mutex;
    std::array<ClassState, NumHazardClasses> classes{};
};

} // namespace Metal
