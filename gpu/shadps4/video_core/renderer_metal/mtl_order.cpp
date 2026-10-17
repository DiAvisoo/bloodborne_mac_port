// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: explicit ordering of untracked buffers (mtl_order.h).

#include <algorithm>
#include <bit>
#include "video_core/renderer_metal/mtl_order.h"

namespace Metal {

namespace {

void AddWait(OrderPlan& p, u32 fence) {
    if (!p.Waits(fence)) {
        p.waits[p.num_waits++] = u16(fence);
    }
}

void AddUpdate(OrderPlan& p, u32 fence) {
    p.updates[p.num_updates++] = u16(fence);
}

/// Calls `f(k)` for every set bit of `mask`.
template <typename F>
void ForEachSlot(u64 mask, F&& f) {
    for (; mask; mask &= mask - 1) {
        f(u32(std::countr_zero(mask)));
    }
}

static_assert(OrderTracker::Slots == 64, "slot sets are 64-bit masks");
static_assert(OrderPlan::MaxWaits >= NumHazardClasses * OrderTracker::Slots + 1);
static_assert(OrderPlan::MaxUpdates >= NumHazardClasses * 2);

} // namespace

bool OrderPlan::Waits(u32 fence) const noexcept {
    return std::find(waits.begin(), waits.begin() + num_waits, u16(fence)) !=
           waits.begin() + num_waits;
}

bool OrderPlan::Updates(u32 fence) const noexcept {
    return std::find(updates.begin(), updates.begin() + num_updates, u16(fence)) !=
           updates.begin() + num_updates;
}

void OrderTracker::BarrierLocked(ClassState& s, HazardClass c, BarrierScopes scopes) {
    u64 ends = 0, vertices = 0;
    if (scopes & ScopeRenderEnd) {
        ends |= s.render_end;
        s.render_end = s.render_vertex = 0;
    } else if (scopes & ScopeRenderVertex) {
        vertices |= s.render_vertex;
        s.render_vertex = 0;
    }
    if (scopes & ScopeCopyEnd) {
        ends |= s.copy_end;
        s.copy_end = 0;
    }
    if (scopes & ScopeComputeEnd) {
        ends |= s.compute_end;
        s.compute_end = 0;
    }
    vertices &= ~ends;
    const bool ahead = (scopes & ScopeAhead) && c == HazardClass::Memory;
    if (!(ends | vertices) && !ahead) {
        return; // nothing since: the current waits still order everything they did
    }
    // A slot waited for now waited, when it opened, for everything installed up to its
    // generation: those waits are implied.
    u32 newest = 0;
    ForEachSlot(ends | vertices, [&](u32 k) { newest = std::max(newest, s.open_gen[k] + 1); });
    ForEachSlot(s.wait_end, [&](u32 k) {
        if (s.end_gen[k] < newest) {
            s.wait_end &= ~(1ull << k);
        }
    });
    ForEachSlot(s.wait_vertex, [&](u32 k) {
        if (s.vertex_gen[k] < newest) {
            s.wait_vertex &= ~(1ull << k);
        }
    });
    if (s.wait_ahead && s.ahead_gen < newest) {
        s.wait_ahead = false;
    }
    ++s.gen;
    ForEachSlot(ends, [&](u32 k) { s.end_gen[k] = s.gen; });
    ForEachSlot(vertices, [&](u32 k) { s.vertex_gen[k] = s.gen; });
    s.wait_end |= ends;
    s.wait_vertex = (s.wait_vertex | vertices) & ~s.wait_end;
    if (ahead) {
        s.wait_ahead = true;
        s.ahead_gen = s.gen;
    }
}

OrderPlan OrderTracker::Open(const Accesses& accesses, Lane lane, u64 tick, u64 completed) {
    std::scoped_lock lock{mutex};
    OrderPlan plan;
    for (u32 i = 0; i < NumHazardClasses; ++i) {
        const Access access = accesses[i];
        if (access == Access::None) {
            continue;
        }
        const HazardClass c = HazardClass(i);
        ClassState& s = classes[i];
        // Encoders of completed command buffers need no waiting: drop their slots.
        if (completed) {
            u64 done = 0;
            for (u32 k = 0; k < Slots; ++k) {
                if (s.slot_tick[k] && s.slot_tick[k] <= completed) {
                    done |= 1ull << k;
                    s.slot_tick[k] = 0;
                }
            }
            s.render_vertex &= ~done;
            s.render_end &= ~done;
            s.copy_end &= ~done;
            s.compute_end &= ~done;
            s.wait_vertex &= ~done;
            s.wait_end &= ~done;
        }
        if (s.isolate || access == Access::Write) {
            s.isolate = false;
            BarrierLocked(s, c, ScopeAll); // after the isolated write / before this one
        }
        s.isolate = access == Access::Write;
        ForEachSlot(s.wait_end, [&](u32 k) { AddWait(plan, SlotFence(c, k)); });
        ForEachSlot(s.wait_vertex, [&](u32 k) {
            AddWait(plan, SlotFence(c, k, (s.render >> k) & 1 ? VertexDone : EncoderDone));
        });
        if (s.wait_ahead) {
            AddWait(plan, AheadFence);
        }
        const bool memory = c == HazardClass::Memory;
        // A slot no barrier needs, searching from `next`; else reuse `next`, waiting on it.
        const u64 busy = s.render_vertex | s.render_end | s.copy_end | s.compute_end |
                         s.wait_vertex | s.wait_end;
        u32 k = s.next;
        if (~busy) {
            const u64 rotated = std::rotr(~busy, int(s.next));
            k = (s.next + u32(std::countr_zero(rotated))) % Slots;
        } else {
            AddWait(plan, SlotFence(c, k));
        }
        s.next = (k + 1) % Slots;
        s.open_gen[k] = s.gen;
        s.slot_tick[k] = tick;
        // Motion history is read and written by vertex shaders only: one fence, after them.
        const bool render_slot = lane == Lane::Render && memory;
        const u64 bit = 1ull << k;
        s.render = (s.render & ~bit) | (render_slot ? bit : 0);
        if (render_slot) {
            s.render_vertex |= bit;
            s.render_end |= bit;
            AddUpdate(plan, SlotFence(c, k, VertexDone));
        } else if (lane == Lane::Copy && memory) {
            s.copy_end |= bit;
        } else {
            s.compute_end |= bit; // (motion history: every encoder, one fence)
        }
        AddUpdate(plan, SlotFence(c, k));
    }
    return plan;
}

void OrderTracker::Barrier(HazardClass c, BarrierScopes scopes) {
    std::scoped_lock lock{mutex};
    BarrierLocked(classes[u32(c)], c, scopes);
}

OrderPlan OrderTracker::OpenAhead() {
    OrderPlan plan;
    AddWait(plan, AheadFence);
    AddUpdate(plan, AheadFence);
    return plan;
}

} // namespace Metal
