// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: explicit ordering of untracked buffers (mtl_order.h).

#include <algorithm>
#include <bit>
#include "video_core/renderer_metal/mtl_order.h"

namespace Metal {

namespace {

/// A fence not added before (slot sets give distinct fences; only the reused slot's end and
/// the uploads' fence are added beside them).
void AddWait(OrderPlan& p, u32 fence) {
    p.waits[p.num_waits++] = u16(fence);
}

void AddWaitOnce(OrderPlan& p, u32 fence) {
    if (!p.Waits(fence)) {
        AddWait(p, fence);
    }
}

void AddUpdate(OrderPlan& p, u32 fence) {
    p.updates[p.num_updates++] = u16(fence);
}

// The bound derived at OrderPlan::MaxWaits.
static_assert(OrderPlan::MaxWaits == NumHazardClasses * (OrderTracker::Slots + 1) + 1);
static_assert(OrderPlan::MaxUpdates >= NumHazardClasses * 2);
static_assert(OrderTracker::NumFences <= 0x10000, "plans store fence indices as u16");
static_assert(OrderSlots % 64 == 0);

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
    SlotSet ends, vertices;
    if (scopes & ScopeRenderEnd) {
        ends |= s.render_end;
        s.render_end.Clear();
        s.render_vertex.Clear();
    } else if (scopes & ScopeRenderVertex) {
        vertices |= s.render_vertex;
        s.render_vertex.Clear();
    }
    if (scopes & ScopeCopyEnd) {
        ends |= s.copy_end;
        s.copy_end.Clear();
    }
    if (scopes & ScopeComputeEnd) {
        ends |= s.compute_end;
        s.compute_end.Clear();
    }
    vertices.Remove(ends);
    const bool ahead = (scopes & ScopeAhead) && c == HazardClass::Memory;
    if (!ends.Any() && !vertices.Any() && !ahead) {
        return; // nothing since: the current waits still order everything they did
    }
    // A slot waited for now waited, when it opened, for everything installed up to its
    // generation: those waits are implied.
    u32 newest = 0;
    const auto newer = [&](u32 k) { newest = std::max(newest, s.open_gen[k] + 1); };
    ends.ForEach(newer);
    vertices.ForEach(newer);
    SlotSet implied;
    s.wait_end.ForEach([&](u32 k) {
        if (s.end_gen[k] < newest) {
            implied.Set(k);
        }
    });
    s.wait_vertex.ForEach([&](u32 k) {
        if (s.vertex_gen[k] < newest) {
            implied.Set(k);
        }
    });
    s.wait_end.Remove(implied);
    s.wait_vertex.Remove(implied);
    if (s.wait_ahead && s.ahead_gen < newest) {
        s.wait_ahead = false;
    }
    ++s.gen;
    ends.ForEach([&](u32 k) { s.end_gen[k] = s.gen; });
    vertices.ForEach([&](u32 k) { s.vertex_gen[k] = s.gen; });
    s.wait_end |= ends;
    s.wait_vertex |= vertices;
    s.wait_vertex.Remove(s.wait_end);
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
            SlotSet done;
            for (u32 k = 0; k < Slots; ++k) {
                if (s.slot_tick[k] && s.slot_tick[k] <= completed) {
                    done.Set(k);
                    s.slot_tick[k] = 0;
                }
            }
            if (done.Any()) {
                for (SlotSet* set : {&s.render_vertex, &s.render_end, &s.copy_end,
                                     &s.compute_end, &s.wait_vertex, &s.wait_end}) {
                    set->Remove(done);
                }
            }
        }
        if (s.isolate || access == Access::Write) {
            s.isolate = false;
            BarrierLocked(s, c, ScopeAll); // after the isolated write / before this one
        }
        s.isolate = access == Access::Write;
        s.wait_end.ForEach([&](u32 k) { AddWait(plan, SlotFence(c, k)); });
        s.wait_vertex.ForEach([&](u32 k) {
            AddWait(plan, SlotFence(c, k, s.render.Test(k) ? VertexDone : EncoderDone));
        });
        if (s.wait_ahead) {
            AddWait(plan, AheadFence);
        }
        const bool memory = c == HazardClass::Memory;
        // A slot no barrier needs, searching from `next`; else reuse `next`, waiting on it.
        SlotSet busy = s.render_vertex;
        for (const SlotSet* set :
             {&s.render_end, &s.copy_end, &s.compute_end, &s.wait_vertex, &s.wait_end}) {
            busy |= *set;
        }
        u32 k = busy.FirstFree(s.next);
        if (k == Slots) {
            k = s.next;
            AddWaitOnce(plan, SlotFence(c, k));
            busy_reuses[i].fetch_add(1, std::memory_order_relaxed);
        }
        s.next = (k + 1) % Slots;
        s.open_gen[k] = s.gen;
        s.slot_tick[k] = tick;
        // Motion history is read and written by vertex shaders only: one fence, after them.
        const bool render_slot = lane == Lane::Render && memory;
        SlotSet bit;
        bit.Set(k);
        s.render.Remove(bit);
        if (render_slot) {
            s.render.Set(k);
            s.render_vertex.Set(k);
            s.render_end.Set(k);
            AddUpdate(plan, SlotFence(c, k, VertexDone));
        } else if (lane == Lane::Copy && memory) {
            s.copy_end.Set(k);
        } else {
            s.compute_end.Set(k); // (motion history: every encoder, one fence)
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
