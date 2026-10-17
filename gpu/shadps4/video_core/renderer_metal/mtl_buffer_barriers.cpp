// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: which accesses to the guest arena need a barrier (mtl_buffer_barriers.h).

#include <algorithm>
#include <cstdlib>
#include "common/logging/log.h"
#include "video_core/renderer_metal/mtl_buffer_barriers.h"

namespace Metal {

namespace {
constexpr u64 PageSize = 1ull << BufferBarriers::PageBits;
constexpr u64 ChunkSize = 1ull << BufferBarriers::ChunkBits;
constexpr u64 PagesPerChunkBits = BufferBarriers::ChunkBits - BufferBarriers::PageBits;
static_assert(PageSize >> BufferBarriers::BlockBits == 16, "a u16 block mask per page");

/// The blocks of page `page` that [offset, end) covers.
u16 BlockMask(u64 page, u64 offset, u64 end) {
    const u64 page_begin = page << BufferBarriers::PageBits;
    const u64 lo = (std::max(offset, page_begin) - page_begin) >> BufferBarriers::BlockBits;
    const u64 hi = (std::min(end, page_begin + PageSize) - 1 - page_begin) >> BufferBarriers::BlockBits;
    return u16(((2u << hi) - 1) & ~((1u << lo) - 1));
}
} // namespace

void BufferBarriers::FreeDeleter::operator()(void* p) const noexcept {
    std::free(p);
}

void BufferBarriers::Register(const void* buffer, u64 size) {
    const u64 pages = (size + PageSize - 1) >> PageBits;
    const u64 chunks = (size + ChunkSize - 1) >> ChunkBits;
    Tags t;
    t.size = size;
    // calloc: zero (epoch 0, never accessed); arrays this large are fresh zero-filled pages
    // the OS commits only as accesses touch them (make_unique<T[]> would write every byte).
    const auto zero = []<typename T>(ZeroArray<T>& array, u64 count) {
        array.reset(static_cast<T*>(std::calloc(count, sizeof(T))));
    };
    bool allocated = true;
    for (u32 k = 0; k < NumKinds && allocated; ++k) {
        zero(t.page[k], pages);
        zero(t.blocks[k], pages);
        zero(t.any[k], chunks);
        zero(t.whole[k], chunks);
        allocated = t.page[k] && t.blocks[k] && t.any[k] && t.whole[k];
    }
    if (!allocated) {
        LOG_ERROR(Render, "Buffer ordering: no memory for the access tags of a {:#x}-byte buffer; "
                          "every access to it waits for all work before it",
                  size);
        t = Tags{};
        t.size = size;
        t.conservative = true;
    }
    buffers[buffer] = std::move(t);
}

bool BufferBarriers::Overlaps(const Tags& t, Kind kind, u64 offset, u64 end) const {
    const u32 current = epoch[GroupOf(kind)];
    if (t.whole_buffer[kind] == current) {
        return true;
    }
    if (offset == 0 && end >= t.size) {
        return t.any_buffer[kind] == current; // a whole-buffer query: no walk either
    }
    const u32* page = t.page[kind].get();
    const u16* blocks = t.blocks[kind].get();
    const u64 first_page = offset >> PageBits, last_page = (end - 1) >> PageBits;
    for (u64 chunk = offset >> ChunkBits; chunk <= (end - 1) >> ChunkBits; ++chunk) {
        if (t.whole[kind][chunk] == current) {
            return true;
        }
        if (t.any[kind][chunk] != current) {
            continue;
        }
        const u64 lo = std::max(first_page, chunk << PagesPerChunkBits);
        const u64 hi = std::min(last_page, ((chunk + 1) << PagesPerChunkBits) - 1);
        for (u64 p = lo; p <= hi; ++p) {
            if (page[p] == current && (blocks[p] & BlockMask(p, offset, end))) {
                return true;
            }
        }
    }
    return false;
}

BarrierScopes BufferBarriers::Conflicts(const void* buffer, u64 offset, u64 size,
                                        bool write) const {
    const auto it = buffers.find(buffer);
    if (it == buffers.end() || size == 0 || offset >= it->second.size) {
        return ScopeNone;
    }
    const Tags& t = it->second;
    if (t.conservative) {
        return ScopeAll;
    }
    const u64 end = std::min(offset + size, t.size);
    const auto has = [&](Kind kind) { return marked[GroupOf(kind)] && Overlaps(t, kind, offset, end); };
    BarrierScopes scopes = ScopeNone;
    // RAW and WAW: the writers' ends.
    if (has(RenderWrite)) {
        scopes |= ScopeRenderEnd;
    }
    if (has(CopyWrite)) {
        scopes |= ScopeCopyEnd;
    }
    if (has(ComputeWrite)) {
        scopes |= ScopeComputeEnd;
    }
    if (has(AheadWrite)) {
        scopes |= ScopeAhead;
    }
    if (!write) {
        return scopes; // RAR: nothing more
    }
    // WAR: the readers' ends, or the vertex stage of render passes that read only there.
    if (!(scopes & ScopeRenderEnd)) {
        if (has(RenderLateRead)) {
            scopes |= ScopeRenderEnd;
        } else if (has(RenderVertexRead)) {
            scopes |= ScopeRenderVertex;
        }
    }
    if (!(scopes & ScopeCopyEnd) && has(CopyRead)) {
        scopes |= ScopeCopyEnd;
    }
    if (!(scopes & ScopeComputeEnd) && has(ComputeRead)) {
        scopes |= ScopeComputeEnd;
    }
    if (!(scopes & ScopeAhead) && has(AheadRead)) {
        scopes |= ScopeAhead;
    }
    return scopes;
}

void BufferBarriers::MarkKind(Tags& t, Kind kind, u64 offset, u64 end) {
    const Group g = GroupOf(kind);
    marked[g] = true;
    const u32 current = epoch[g];
    t.any_buffer[kind] = current;
    if (offset == 0 && end >= t.size) {
        t.whole_buffer[kind] = current; // one tag, not a walk over every chunk
        return;
    }
    u32* page = t.page[kind].get();
    u16* blocks = t.blocks[kind].get();
    const u64 first_page = offset >> PageBits, last_page = (end - 1) >> PageBits;
    for (u64 chunk = offset >> ChunkBits; chunk <= (end - 1) >> ChunkBits; ++chunk) {
        t.any[kind][chunk] = current;
        const u64 chunk_begin = chunk << ChunkBits;
        if (offset <= chunk_begin && end >= std::min(chunk_begin + ChunkSize, t.size)) {
            t.whole[kind][chunk] = current;
            continue;
        }
        const u64 lo = std::max(first_page, chunk << PagesPerChunkBits);
        const u64 hi = std::min(last_page, ((chunk + 1) << PagesPerChunkBits) - 1);
        for (u64 p = lo; p <= hi; ++p) {
            const u16 mask = BlockMask(p, offset, end);
            blocks[p] = page[p] == current ? u16(blocks[p] | mask) : mask;
            page[p] = current;
        }
    }
}

void BufferBarriers::Mark(const void* buffer, u64 offset, u64 size, bool write, u8 encoders,
                          ReadStage stage) {
    const auto it = buffers.find(buffer);
    if (it == buffers.end() || size == 0 || offset >= it->second.size) {
        return;
    }
    Tags& t = it->second;
    if (t.conservative) {
        for (bool& m : marked) {
            m = true; // the next barrier has something to order
        }
        return;
    }
    const u64 end = std::min(offset + size, t.size);
    if (encoders & EncoderRender) {
        MarkKind(t,
                 write                        ? RenderWrite
                 : stage == ReadStage::Vertex ? RenderVertexRead
                                              : RenderLateRead,
                 offset, end);
    }
    if (encoders & EncoderCopy) {
        MarkKind(t, write ? CopyWrite : CopyRead, offset, end);
    }
    if (encoders & EncoderCompute) {
        MarkKind(t, write ? ComputeWrite : ComputeRead, offset, end);
    }
    if (encoders & EncoderAhead) {
        MarkKind(t, write ? AheadWrite : AheadRead, offset, end);
    }
}

BarrierScopes BufferBarriers::Barrier(BarrierScopes scopes) {
    if (scopes & ScopeRenderEnd) {
        scopes |= ScopeRenderVertex;
    }
    BarrierScopes done = ScopeNone;
    for (u32 g = 0; g < NumGroups; ++g) {
        if ((scopes & (1u << g)) && marked[g]) {
            marked[g] = false;
            // Epoch 0 means "never"; after 2^32 barriers (days of play) a stale tag could match
            // again, which only adds a barrier.
            if (++epoch[g] == 0) {
                epoch[g] = 1;
            }
            done |= BarrierScopes(1u << g);
        }
    }
    if (done) {
        ++barriers;
    }
    return done;
}

} // namespace Metal
