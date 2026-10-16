# Task 5 report
Implemented: buffer.{h,cpp} (UniqueBuffer/Buffer/StreamBuffer on MTLBuffer; Flush/Invalidate/is_coherent/Sparse/VMA removed), vk_staging_buffer_pool.{h,cpp} and vk_constant_ring.h take Metal::Device& + Metal::Timeline& (per Task 4 ruling), new core files renderer_metal/mtl_buffer.{h,mm} (CreateBuffer; added to METAL_CORE_SOURCES and REMOVE_ITEM), unit test TestBufferCreation in tests/test_mtl_encoder.mm.
Tests: core command PASS twice (incl. MTL_DEBUG_LAYER); objects of buffer.cpp and vk_staging_buffer_pool.cpp compile; vk_constant_ring.h syntax-checked via buffer.cpp flags; handle check empty.
Decisions: mtl_buffer.h includes buffer.h for MemoryType (plain C++ header, no deps on the cache); UniqueBuffer also caches Contents(). Callers (buffer_cache.cpp, rasterizer, etc: Flush(), ConstantRing::Flush, Buffer ctor signature, Sparse) intentionally left broken for later tasks.

## Fix round 1
Changes: StreamBuffer/StagingBufferPool/ConstantRing take Vulkan::Scheduler& again for tick reads/waits (Scheduler::Wait flushes the current tick; fixes the assert), Metal::Device& kept for creation. MemoryType moved to renderer_metal/mtl_memory_type.h (included by buffer.h; mtl_buffer.h no longer includes buffer.h). `deps` symlink untracked (git rm --cached, `deps` added to .gitignore, link kept on disk).
Tests: core command -> PASS x2 (incl. MTL_DEBUG_LAYER=1); object compiles of buffer.cpp and vk_staging_buffer_pool.cpp: no errors; vk_constant_ring.h syntax check: no errors; handle check: no output.
