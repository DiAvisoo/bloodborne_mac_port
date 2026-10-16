# Metal-native M2 — decision record (rulings and deferred minors)

Copied from the execution ledger when M2 finished (2026-10-16). Each ruling: what was decided — why — cost if wrong.

## Rulings
- Task 3: Ruling: Specialize takes 32-bit values only; implementer flagged 64-bit spec constants lost in mtl_shadow — every SpecializationMapEntry in the renderer is 4 bytes (fault_manager.cpp:51, tile_manager.cpp:155) — cost if wrong: a 64-bit constant would keep its default
- Task 4: Ruling: accept implementer decisions — RenderAttachment gains clear_stencil (separate depth/stencil clears); num_layers dropped (M3 to handle layered passes if used); viewport converted with RasterUtil::ToMetalViewport (proven on metal-backend); upload_mutex held through the main Submit (old ordering); priority-ops thread waits for commit before Device::Wait — cost if wrong: M3 rework of RenderState/viewport handling
- Task 4: Ruling: review Important 1 (device-global ticks shared by the presenter's 3 schedulers break Finish/Wait/deferred ops) — real; plan defect (brief mandated device ticks). Fix: per-scheduler logical ticks mapped to device ticks at commit, in a core class Metal::Timeline with unit tests; Scheduler CurrentTick/IsFree/Wait/KnownGpuTick go through it. Consequence for Task 5+: staging pool, stream buffers and constant ring take ticks from the owning Scheduler (timeline), not Metal::Device — cost if wrong: one indirection per tick query
- Task 5: Ruling: amends the Task 4 consequence — StreamBuffer/StagingBufferPool/ConstantRing take Vulkan::Scheduler& (as originally) for ticks and waits, because Scheduler::Wait flushes when the tick is current while Timeline::Wait asserts (review Important 1); MemoryType moves to a core header (renderer_metal/mtl_memory_type.h) so the core never includes cache headers — cost if wrong: caches depend on the scheduler header (as before)
- Task 5: Ruling: implementer committed .superpowers/ (workspace scratch) in 1b6e825 (and earlier?); controller untracked it and added .superpowers/ to .gitignore (hygiene, not code) — cost if wrong: none
- Task 6: Ruling: accept implementer decisions — Runtime::CopyBuffer/FillBuffer/HoistCopy take const Buffer& (arenas are const Buffer*; Task 7 implements these shapes); old texture_cache/types.h BufferCopy removed; BufferCache keeps TextureCache& (SynchronizeMemoryFromImage); unbound host-kernel slots ~0u; DownloadMemory widens to 4-byte words but writes back exact ranges; staging aligned; fault count capped at 1023 — cost if wrong: small rework in Task 7/9
- Task 6: Ruling: render encoders declare indirect buffers read-only (concern 4) — same as metal-backend's proven UseAddressBuffers (render reads guest memory by pointer, writes come from compute; BB_METAL_RENDER_WRITES existed for the other case); fault buffer and page table are also bound directly (vk_rasterizer.cpp:2128-2132) so Metal tracks them — cost if wrong: a render-stage pointer write not ordered before a later compute read (would show as missing faults/geometry at M4)
- Task 8: Ruling: accept — Image::CreateBacking(num_samples) for Task 9; StencilViewPixelFormat; image_barriers removed from vk_runtime.h; 2D views of 3D images bind the whole 3D texture (log once); swizzled views not render targets (M3) — cost if wrong: M3 rework for 3D-slice/BGRA targets
- Task 8: Ruling: review Minor 2 (incompatible-format views of depth images abort in Metal) re-graded Important by effect (process abort) and folded into fix round 1 with Important 1 (same guard)
- Task 8: Ruling: plan premise "no Metal sampler LOD bias" is false on macOS 26 (MTLSamplerDescriptor.lodBias API_AVAILABLE(macos(26.0))); fix sets it under @available, logs once on older systems — cost if wrong: none
- Task 9: Ruling: accept — Transit replaced by public AccessImage(image, write) (keeps access hook + scene-resolution proxy sync; M3 callers switch); copies clamped to both textures; format mismatch -> view, else buffer bounce, else log+skip; clears remap BGRA-like colours, skip depth/BC; handle-check hit on Runtime::GetDevice() is the accessor (false positive) — cost if wrong: M3 adjusts callers
- Task 9: Ruling: review minors "no MSAA guard in Upload/Download buffer copies" and "integer-format resolves unguarded" re-graded Important by effect (Metal may abort the process on invalid blits/resolves, as views did in Task 8) — fix round 1 adds log-once-and-skip guards
- Task 10: Ruling: TileManager(Metal::Device&, Scheduler&, Runtime&) (StreamBuffer dropped: params via SetBytes); Task 12 updates texture_cache.cpp's construction — cost if wrong: none
- Task 11: Ruling: accept — mtl_render_pipeline.{h,mm} (core) so plain C++ can build MSL pipelines; TexturePixelFormat/TextureSampleCount query the real texture (backings clamp samples); texture_cache.cpp's BlitHelper construction left for Task 12; blit leaves its pipeline/depth-stencil/texture 0 bound (M3 rasterizer sets its own per draw) — cost if wrong: M3 must rebind per draw (it already does)
- Task 12: Ruling: accept — BlitHelper member dropped from TextureCache (Runtime owns one); aspect_mask uses replaced by props.is_depth in buffer_cache/texture_cache; vk_runtime.cpp's only remaining error is the M3 header vk_scene_resolution.h (known exception) — cost if wrong: none
- Task 12: Ruling: review Important "AllocatedSize counts the 16 GiB arena -> GC may thrash on 16 GB Macs" needs a runtime measurement — recorded as an explicit M4 verification item in the roadmap (final fix wave), not changed blind — cost if wrong: constant texture GC on 16 GB Macs until fixed
- Final: Ruling: fix wave includes I1, I2 (option b: Timeline::DeviceTickOf + Encoder::WaitForDeviceTick so M3 can order presenter after draw on the GPU), M1 zero-size fill/copy early return, M2 unaligned pattern fill re-graded Important by effect (asserts the process on guest-sized HTile fills) -> round down + log once, M3 ordered hoisted copies separated by EndEncoding, M4 sampler GC erase via DeferOperation, M5 ~Scheduler Finish() after SyncRecording; M6 HasBindings kept (used by tests) — cost if wrong: small
- Final: Ruling: accept fixer corrections — WaitForDeviceTick requires the producer's tick to be committed already (one queue starts command buffers in commit order; waiting on a later commit hangs — my test spec was wrong); ordered hoisted copies end the blit encoder before every copy; ~Scheduler clears on_submit before Finish — cost if wrong: none

## Deferred minors
- Task 1: minor (deferred): link warning "(arm64) unable to open object file" (debug-map paths of the archive)
- Task 1: minor (deferred): core filenames duplicated between METAL_CORE_SOURCES and REMOVE_ITEM
- Task 1: minor (deferred): bbmetal_core compiles without bbgpu's definitions (ODR hazard if a core header ever depends on them)
- Task 2: minor (deferred): ~Device does not removeResidencySet from the queue
- Task 2: minor (deferred): Evict / DeclareIndirect lifetime contracts undocumented (buffer must outlive in-flight work / the Encoder)
- Task 2: minor (deferred): FillBuffer size==0 not early-returned
- Task 2: minor (deferred): CommitUntracked has no error-logging handler
- Task 2: minor (deferred): no tests for depth_plane/resolve/stencil_only attachments, fill after pending clear, SubmitUntracked keep-alive
- Task 3: minor (deferred): Specialize test lacks bool-flip and composite coverage; stale "none in the host shaders" comment
- Task 3: minor (deferred): DepthStencilCache caches a nil state silently (no assert/log)
- Task 4: minor (deferred): ~Encoder commits without submit_mutex at shutdown (upload_encoder commits tracked)
- Task 4: minor (deferred): BB_PASS_STATS SameAttachments ignores level/slice/depth_plane
- Task 4: minor (deferred): RenderAttachment clear_value memcmp may see stale bytes (M3 must value-init)
- Task 4: minor (deferred): DepthStencilKey keeps depth_write when test off (duplicate states)
- Task 4: minor (deferred): BbToggle::PendingPollLimit now unused
- Task 4: minor (deferred): FrontAndBack cull maps to Back (M3 could discard such draws)
- Task 4: minor (deferred): Timeline::Wait asserts for ticks more than one ahead (as before)
- Task 4: minor (deferred): GetWorkSemaphore residue in pipeline_cache/constant_ring/presenter (M3/M4 move them to the timeline)
- Task 5: minor (deferred): UniqueBuffer move leaves stale contents/gpu_address in the moved-from object
- Task 5: minor (deferred): TestBufferCreation covers only Stream/DeviceLocal
- Task 5: minor (deferred): .gitignore `deps` line unanchored (prefer /deps)
- Task 6: minor (deferred): page table / fault buffer comments say "reached by address" but both are bound directly (MakeResident/DeclareIndirect redundant there; fault buffer's DeclareIndirect is plan-mandated)
- Task 6: minor (deferred): process-global ShaderCompiler holds Device& (static destruction order unchecked)
- Task 6: minor (deferred): no unit test for BuildHostKernel slot map / DownloadMemory widening
- Task 7: minor (deferred): M3 must drop external IsBufferAccessed callers (vk_rasterizer 1249,1255,1411,1844,1907,2097,2192; vk_shader_hle 107-108) and adapt CopyBuffer/FillBuffer pointer->reference callers; AccessBuffer's unused stage/access params removed with it
- Task 7: minor (deferred): FlushBarriers no longer runs BeforeImageAccess/image barriers — Task 9/M3 must make sure no image path relied on it
- Task 8: minor (deferred): out-of-range view requests clamped silently; border_color_base unused; no texture-cache-level test of view selection
- Task 8: minor (deferred): SafeViewPixelFormat's sRGB-twin early return ignores texture_has_format_views (unreachable today: every non-block image has format views)
- Task 9: minor (deferred): integer-format MSAA resolves unguarded (Metal cannot resolve integer formats); D16S8/D24S8 guest layouts differ from Metal D32S8 (M5 conversion)
- Task 9: minor (deferred): same-format combined depth-stencil CopyImage now copies stencil too (old code masked it)
- Task 9: minor (deferred): BC copies with mismatched non-block-multiple mip edges may produce a size Metal rejects
- Task 10: minor (deferred): Dispatch threads {64,1,1} literal (could use kernel.local_size); no static_assert on TilingInfo size; buffer bindings without range (check at M4)
- Task 11: minor (deferred): MSAA copy test does not distinguish per-sample reads (source samples identical)
- Task 11: minor (deferred): blit NumericFlavour derived from guest vk::Format instead of the Metal texture format; HasStencilPixelFormat true for stencil-only formats (assert dst has depth); color_to_ms_depth samples texture2d<float> only
- Task 12: minor (deferred): formatting after edits in texture_cache (indentation, GetSampler continuation, dead scope block, stale Steam Deck comment, duplicate budget reads)
- Final: minor (deferred): HasBindings unused outside tests

## Fix rounds
- Task 4: fix round 1/5 (1 addressed, 0 open — per-scheduler Timeline; commits 8eec341..9de9df8)
- Task 5: fix round 1/5 (2 addressed, 0 open — scheduler waits, deps untracked; commits 1143430..1b6e825)
- Task 8: Ruling: review Minor 2 (incompatible-format views of depth images abort in Metal) re-graded Important by effect (process abort) and folded into fix round 1 with Important 1 (same guard)
- Task 8: fix round 1/5 (3 addressed, 0 open — SafeViewPixelFormat guard, sampler lodBias on macOS 26; commits 3ee456f..b6ac09b)
- Task 9: Ruling: review minors "no MSAA guard in Upload/Download buffer copies" and "integer-format resolves unguarded" re-graded Important by effect (Metal may abort the process on invalid blits/resolves, as views did in Task 8) — fix round 1 adds log-once-and-skip guards
- Task 9: fix round 1/5 (2 addressed, 0 open — MSAA buffer-copy and integer-resolve guards; commits 13e09a3..0c1ba13)
- Final: fixed I1,I2,M1-M5 — re-review all addressed (commit 39a43fc); M2 complete
