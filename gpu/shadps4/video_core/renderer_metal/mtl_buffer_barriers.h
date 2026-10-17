// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: which accesses to the guest arena need a barrier, and what it waits for
// (Metal-native renderer). Pure C++.
//
// The arena is untracked (mtl_buffer.mm) and nearly every command reads it, so ordering it per
// Metal encoder serialises the frame (docs/FINDINGS.md, 2026-10-17). The renderer knows the
// ranges each command touches instead, and orders them the way the Vulkan renderer's barrier
// tracker did (0.4.0), with scoped barriers. Accesses since the last barriers are recorded per
// 256-byte block with the kind of encoder that makes them (render passes, copies, compute
// dispatches, the uploads hoisted ahead of the command buffer) and, for reads in render passes,
// the stage. A command whose access conflicts with one of them gets a barrier first
// (Metal::Encoder::OrderBarrier) that waits for the encoders of the conflicting kind only:
//   - a read conflicts with a write (RAW): the writers' ends;
//   - a write conflicts with a write (WAW) or a read (WAR): their ends, or for reads made only
//     at the vertex stage of render passes (vertex, index and indirect-argument fetches,
//     pre-rasterisation shaders) their vertex stages;
//   - reads never conflict with reads (RAR).
// A barrier retires the accesses it waits for (they are ordered before everything after it);
// the others stay current for later conflicts. Encoders of the kinds a barrier does not wait
// for keep running alongside (Metal::OrderTracker).
// Use, on the thread that records the commands (accesses are in recording order):
//   Conflicts(...) for every access of the next command (OR the scopes); Barrier(scopes) (record
//   the Encoder barrier with what it returns, when not None); record the command; Mark(...) its
//   accesses.
// Granularity: 256-byte blocks (a mask per 4 KiB page: constant tables of neighbouring draws
// share pages, and whole-page tracking made their reads conflict with the copies between them);
// ranges of whole 1 MiB chunks are marked per chunk (vertex buffers, big copies).

#pragma once

#include <memory>
#include <unordered_map>
#include "common/types.h"

namespace Metal {

/// What a barrier waits for (a mask; 0 = no barrier).
enum BarrierScope : u8 {
    ScopeNone = 0,
    ScopeRenderVertex = 1, ///< the vertex stage of render encoders
    ScopeRenderEnd = 2,    ///< render encoders (implies ScopeRenderVertex)
    ScopeCopyEnd = 4,      ///< copy encoders (blits, the copy-shader and fill kernels)
    ScopeComputeEnd = 8,   ///< compute dispatches
    ScopeAhead = 16,       ///< the uploads hoisted ahead of the command buffer
    ScopeAll = ScopeRenderEnd | ScopeCopyEnd | ScopeComputeEnd | ScopeAhead,
};
using BarrierScopes = u8;

/// When a read happens in the render pass that makes it.
enum class ReadStage : u8 {
    Vertex, ///< vertex fetch, index fetch, indirect arguments, pre-rasterisation shaders
    Late,   ///< fragment shaders (and everything in compute and blit encoders)
};

/// Which kinds of Metal encoder make an access (a mask: a tessellated draw reads its inputs in
/// compute kernels and in its render pass).
enum EncoderKinds : u8 {
    EncoderRender = 1,
    EncoderCopy = 2,    ///< blits, and compute encoders of copies (the copy shader, fills)
    EncoderCompute = 4, ///< compute dispatches (guest and host kernels)
    EncoderAhead = 8,   ///< uploads hoisted ahead of the command buffer (Encoder Ordering::Ahead)
};

class BufferBarriers {
public:
    static constexpr u32 PageBits = 12;
    static constexpr u32 BlockBits = 8; ///< 16 blocks per page: a u16 mask
    static constexpr u32 ChunkBits = 20;

    /// Tracks `buffer` (its first `size` bytes; accesses past them are clamped).
    void Register(const void* buffer, u64 size);
    void Unregister(const void* buffer) {
        buffers.erase(buffer);
    }
    [[nodiscard]] bool Registered(const void* buffer) const {
        return buffers.contains(buffer);
    }
    /// What a barrier before the access must wait for (unregistered buffers: nothing; the
    /// Encoder orders those itself).
    [[nodiscard]] BarrierScopes Conflicts(const void* buffer, u64 offset, u64 size,
                                          bool write) const;
    /// Records the access, made by `encoders` (reads in render passes at `stage`).
    void Mark(const void* buffer, u64 offset, u64 size, bool write, u8 encoders,
              ReadStage stage = ReadStage::Late);
    /// Retires the accesses `scopes` waits for; returns the scopes that had any (the barrier
    /// to record; ScopeNone: nothing to order).
    BarrierScopes Barrier(BarrierScopes scopes);
    [[nodiscard]] u64 Barriers() const noexcept {
        return barriers;
    }

private:
    enum Kind : u32 {
        RenderVertexRead,
        RenderLateRead,
        RenderWrite,
        CopyRead,
        CopyWrite,
        ComputeRead,
        ComputeWrite,
        AheadRead,
        AheadWrite,
        NumKinds
    };
    /// What a barrier waits for, per kind (BarrierScope bit order).
    enum Group : u32 {
        GroupRenderVertex,
        GroupRenderEnd,
        GroupCopy,
        GroupCompute,
        GroupAhead,
        NumGroups
    };
    static constexpr Group GroupOf(Kind kind) {
        return kind == RenderVertexRead ? GroupRenderVertex
               : kind <= RenderWrite    ? GroupRenderEnd
               : kind <= CopyWrite      ? GroupCopy
               : kind <= ComputeWrite   ? GroupCompute
                                        : GroupAhead;
    }
    struct Tags {
        std::unique_ptr<u32[]> page[NumKinds];   ///< epoch of the last access, per page
        std::unique_ptr<u16[]> blocks[NumKinds]; ///< the page's blocks accessed in that epoch
        std::unique_ptr<u32[]> any[NumKinds];    ///< ... of some page of the chunk
        std::unique_ptr<u32[]> whole[NumKinds];  ///< ... of the whole chunk
        u64 size{};
    };
    /// Whether a `kind` access of the current epoch overlaps [offset, end).
    bool Overlaps(const Tags& t, Kind kind, u64 offset, u64 end) const;
    void MarkKind(Tags& t, Kind kind, u64 offset, u64 end);

    std::unordered_map<const void*, Tags> buffers;
    u32 epoch[NumGroups]{1, 1, 1, 1, 1};
    bool marked[NumGroups]{}; ///< accesses since the epoch began
    u64 barriers{};
};

} // namespace Metal
