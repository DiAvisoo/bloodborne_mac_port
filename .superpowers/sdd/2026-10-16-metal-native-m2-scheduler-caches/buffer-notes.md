# Buffer-side inventory (read-only survey) — paths under gpu/shadps4/video_core/
## Buffer classes (buffer_cache/buffer.{h,cpp})
- buffer.h:15,22-25 vk_common/VMA handles -> delete; include renderer_metal/mtl_object.h.
- buffer.h:30-36 MemoryType: DeviceLocal->Private; HostUncached->Shared|WriteCombined; HostCached,Stream->Shared; Sparse->delete.
- buffer.h:38-44 ReadFlags/AllFlags delete. buffer.h:46-81 UniqueBuffer -> Metal::Object + gpu_address (move ctor/assign can go).
- buffer.cpp:34-64 VMA flag mapping -> MemoryType->MTLResourceOptions (in renderer_metal/mtl_buffer.mm, core).
- buffer.cpp:66-121 Create/Destroy (vmaCreateBuffer, sparse branch :107-111, getBufferAddress :113-120) -> newBufferWithLength via mtl_buffer.mm; keep "Failed allocating buffer" assert.
- buffer.cpp:123-152 ctor: takes Metal::Device&; SetObjectName (:137-141) -> label; mapped_data = span(contents,size) for shared, empty for private; is_coherent always true -> drop.
- buffer.h:109-116 Handle() -> void*; BufferDeviceAddress() -> GpuAddress() (no assert).
- buffer.cpp:154-166 Flush/Invalidate -> delete (+ StagingBufferRef::Flush/Invalidate vk_staging_buffer_pool.h:25-31 and callers).
- StreamBuffer buffer.h:132-190, buffer.cpp:171-281: ticks; delete non_coherent_atom_size (:174,:182-186); Commit (:224-231) just AdvanceAndWatch.
## Arena/page table (buffer_cache.{h,cpp})
- Probe :63-86 (+FindMemoryType :39-51, arena_memory_type_index) delete -> block_size = MIN_BLOCK_SIZE (16 KiB, shift 14; MoltenVK gave the same). block_shift feeds recompiler (vk_rasterizer.cpp:50 GetSparsePageShift) and fault shader.
- Page table :88-94: private buffer blocks_per_arena_page*NUM_ARENA_PAGES*8 bytes, zero fill via runtime.FillBuffer; element u64.
- Mac arena :95-136: drop #ifdef __APPLE__; max_buffer from maxBufferSize (:113-117) -> device.GetLimits().max_buffer_length; keep BB_ARENA_GIB (:103-107), BB_MAX_BUFFER_GIB (:118-120), assert (:122), printf, MacArenaBase/MacArenaPages (buffer_cache.h:45-49); arena DeviceLocal; RegisterHoistTarget (:135) unchanged.
- EnsureResident :503-580 keep Apple half (:516-534) with arena->GpuAddress() (:526), drop staging.Flush (:533), delete :535-579.
- Backing (buffer_cache.h:181-191) drop vk::DeviceMemory/offset. GetArena :443-501 keep Apple assert (:451-456), delete :457-466 and :470-500.
## Uploads/readback/streams
- SynchronizeMemory :582-602, UploadCopies :604-680: vk::BufferCopy -> VideoCore::BufferCopy (buffer_cache/buffer_copy.h, made in Task 4; positional emplace_back keeps order); remove staging.Flush (:644,:678); BbCopy paths, RunGuestCopy (:153-159, drop Flush :157-158), CopySparseMemory are CPU code: keep.
- ObtainBuffer :340-366 UniformMinAlignment -> limits; ObtainBufferForImage :394 StorageMinAlignment -> limits; drop Flush :418.
- DownloadMemory :209-247: CopyBuffer to HostCached staging, scheduler.Finish(); delete Invalidate (:240).
- SynchronizeMemoryFromImage :682-741: HTile FillBuffer(...,0xF) (:686) now exact; vk::BufferImageCopy -> texture side.
- SubmitPendingArenaBinds :743-780, pending_binds, ArenaBinds/BindsForArena (h:122-133), memory_semaphore (h:16,179): delete (+ caller vk_rasterizer.cpp:87 is M3).
## Runtime + staging (vk_runtime, vk_staging_buffer_pool)
- CopyBuffer vk_runtime.cpp:166-215: keep EndRendering; delete IsBufferAccessed/FlushBarriers/AccessBuffer bookkeeping (:172-195,:202-214); keep NoteSubmissionUse; loop enc.CopyBuffer.
- FillBuffer :217-230; InlineData :232-245 delete. RegisterHoistTarget :104, NoteSubmissionUse :109 keep; HoistCopy :121-160 keep logic -> scheduler.RecordHoistedCopy(src.Ref(), dst.Ref(), copies, ordered).
- Barrier code: IsBufferAccessed :809, AccessBuffer :823-878, FlushBarriers memory half :880-939, memory_barrier (ctor :95-97), BarrierTracker/access_memo (vk_runtime.h:144-157).
- StagingBufferPool: ticks (staging_buffer_pool.cpp:117,135,158,178,196) — RULING: take Metal::Timeline& (scheduler.GetTimeline()), NOT Metal::Device ticks; DeviceLocal requests (vk_runtime.cpp:565, tile_manager.cpp:252,314) give private rings mapped==nullptr; prewarm memset (:52-53) ok.
- vk_constant_ring.h: Stamp :75 / Retire :92 (uses GetWorkSemaphore) -> Timeline; Handle() -> Metal::BufferRef; created rasterizer.cpp:80.
## Fault manager
- fault_buffer_process.comp: local_size 64; spec 0 CACHING_PAGEBITS (default 12; pass block_shift-based value as today), 1 MAX_PAGE_FAULTS (1024); binding0 SSBO fault_buffer rw; binding1 SSBO (u64[] and uint[] aliased; atomicAdd word0); no push constants.
- fault_manager.cpp :19-24 fault_buffer DeviceLocal (num_pages/8 bytes), download_buffer HostCached (8x8 KiB); :28-84 layout/spec/CompileSPV/pipeline -> BuildHostKernel; ProcessFaultBuffer :87-182 delete barriers (:97-114,:145-149,:158-162) and descriptor writes (:115-142); keep Wait, PopPendingOperations, DeferOperation.
- mtl_shader.h: ShaderCompiler(Device&, cache_dir); Compile(span<const u32>, u64 hash, string_view name, CompileVariant={}) -> CompileResult{shader,status,error}; CompiledShader{function (+1 id<MTLFunction>), bindings, Find(set,binding,kind), local_size,...}; buffer 0 reserved for push constants, SSBOs from slot 1 in (set,binding) order, aliased views share a slot. Use Find, don't hard-code.
## External consumers (M3, do not convert now)
- Buffer::Handle() ~27 sites (vk_rasterizer 13, constant_ring, camera_motion, shader_hle, tile_manager (Task 10), runtime image transfers (Task 9)); mapped_data in constant_ring, rasterizer; Buffer constructions in camera_motion, constant_ring, rasterizer; AccessBuffer users rasterizer/shader_hle/tile_manager.
## Risks
- copyFromBuffer/fillBuffer offsets/sizes multiples of 4 (DownloadMemory, gpu_modified_ranges may be byte-granular) — round outward to 4 where safe or assert.
