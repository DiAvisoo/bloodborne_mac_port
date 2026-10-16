# SDD ledger — plan: docs/superpowers/plans/2026-10-16-metal-native-m2-scheduler-caches.md
Spec: docs/superpowers/specs/2026-10-16-metal-native-renderer-design.md (reachable)
Pre-flight table:
| T1→T2,T3 | core lib target METAL_CORE_SOURCES consumed by T2/T3/T5/T8/T12 additions | ok: each adds to the list |
| T2→T4 | SubmitUntracked/DeclareIndirect/HasBindings produced; scheduler consumes | ok |
| T3→T4,T6,T10,T11 | DepthStencilCache, Specialize | ok |
| T4→T5..T12 | Scheduler(Metal::Device&), Record(Func(Encoder&)), BufferCopy in buffer_cache/buffer_copy.h, RecordHoistedCopy(BufferRef,BufferRef,span<BufferCopy>,bool) | ok |
| T5→T6,T7 | Buffer(Metal::Device&,...), Ref(), GpuAddress(), mtl_buffer.mm in core | ok |
| T6 mtl_host_kernel → T10 | BuildHostKernel | ok |
| T7 vs T9 | same file vk_runtime.cpp; T7 gate tolerates image-half errors | ok |
| T8 mtl_texture → T9,T11 | CreateTextureView/BytesPerBlock | ok |
| T9 ResolveImage extends T2's Encoder (resolve-only pass) | ok, T9 adds test |
| T12 Device::AllocatedSize added to core | ok |
Self-consistency: T1 test command needs log.cpp/assert.cpp + fmt — consistent with Task 1 Step 5. T5 offers two options for buffer.cpp then chooses mtl_buffer.mm — consistent.
Ruling (preflight): Task 2 FillBuffer non-uniform path opens its own compute encoder (does not disturb caller compute state) — plan text says so — cost if wrong: one extra encoder per such fill.
Ruling (preflight): execution is subagent-driven (user away; plan large) — cost: more tokens than inline.
Task 1: complete (commits 47e379a..17b3aeb, review clean)
Task 1: minor (deferred): link warning "(arm64) unable to open object file" (debug-map paths of the archive)
Task 1: minor (deferred): core filenames duplicated between METAL_CORE_SOURCES and REMOVE_ITEM
Task 1: minor (deferred): bbmetal_core compiles without bbgpu's definitions (ODR hazard if a core header ever depends on them)
Task 2: complete (commits 17b3aeb..87ec5be, review clean)
Task 2: minor (deferred): ~Device does not removeResidencySet from the queue
Task 2: minor (deferred): Evict / DeclareIndirect lifetime contracts undocumented (buffer must outlive in-flight work / the Encoder)
Task 2: minor (deferred): FillBuffer size==0 not early-returned
Task 2: minor (deferred): CommitUntracked has no error-logging handler
Task 2: minor (deferred): no tests for depth_plane/resolve/stencil_only attachments, fill after pending clear, SubmitUntracked keep-alive
Task 3: Ruling: Specialize takes 32-bit values only; implementer flagged 64-bit spec constants lost in mtl_shadow — every SpecializationMapEntry in the renderer is 4 bytes (fault_manager.cpp:51, tile_manager.cpp:155) — cost if wrong: a 64-bit constant would keep its default
Task 3: complete (commits 87ec5be..ff19203, review clean)
Task 3: minor (deferred): Specialize test lacks bool-flip and composite coverage; stale "none in the host shaders" comment
Task 3: minor (deferred): DepthStencilCache caches a nil state silently (no assert/log)
Task 4: Ruling: accept implementer decisions — RenderAttachment gains clear_stencil (separate depth/stencil clears); num_layers dropped (M3 to handle layered passes if used); viewport converted with RasterUtil::ToMetalViewport (proven on metal-backend); upload_mutex held through the main Submit (old ordering); priority-ops thread waits for commit before Device::Wait — cost if wrong: M3 rework of RenderState/viewport handling
Task 4: Ruling: review Important 1 (device-global ticks shared by the presenter's 3 schedulers break Finish/Wait/deferred ops) — real; plan defect (brief mandated device ticks). Fix: per-scheduler logical ticks mapped to device ticks at commit, in a core class Metal::Timeline with unit tests; Scheduler CurrentTick/IsFree/Wait/KnownGpuTick go through it. Consequence for Task 5+: staging pool, stream buffers and constant ring take ticks from the owning Scheduler (timeline), not Metal::Device — cost if wrong: one indirection per tick query
Task 4: minor (deferred): ~Encoder commits without submit_mutex at shutdown (upload_encoder commits tracked)
Task 4: minor (deferred): BB_PASS_STATS SameAttachments ignores level/slice/depth_plane
Task 4: minor (deferred): RenderAttachment clear_value memcmp may see stale bytes (M3 must value-init)
Task 4: minor (deferred): DepthStencilKey keeps depth_write when test off (duplicate states)
Task 4: minor (deferred): BbToggle::PendingPollLimit now unused
Task 4: minor (deferred): FrontAndBack cull maps to Back (M3 could discard such draws)
Task 4: fix round 1/5 (1 addressed, 0 open — per-scheduler Timeline; commits 8eec341..9de9df8)
Task 4: complete (commits ff19203..9de9df8, review clean after round 1)
Task 4: minor (deferred): Timeline::Wait asserts for ticks more than one ahead (as before)
Task 4: minor (deferred): GetWorkSemaphore residue in pipeline_cache/constant_ring/presenter (M3/M4 move them to the timeline)
Task 5: Ruling: amends the Task 4 consequence — StreamBuffer/StagingBufferPool/ConstantRing take Vulkan::Scheduler& (as originally) for ticks and waits, because Scheduler::Wait flushes when the tick is current while Timeline::Wait asserts (review Important 1); MemoryType moves to a core header (renderer_metal/mtl_memory_type.h) so the core never includes cache headers — cost if wrong: caches depend on the scheduler header (as before)
Task 5: minor (deferred): UniqueBuffer move leaves stale contents/gpu_address in the moved-from object
Task 5: minor (deferred): TestBufferCreation covers only Stream/DeviceLocal
