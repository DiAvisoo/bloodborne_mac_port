# Metal-native M3 — decision record (rulings and deferred minors)

Copied from the execution ledger when M3 finished (2026-10-16). Each ruling: what was decided — why — cost if wrong.

## Rulings
- Task 1: Ruling: plan-ambiguous wording ("clamps the records at args") resolved as copy-and-clamp into scratch — in-place mutation of guest-mirrored argument records corrupts cache/guest data (review Important 1)
- Task 2: Ruling: no VertexAttribute struct exists to move; Task 3 defines the plain VertexAttribute it needs — cost if wrong: none
- Task 3: Ruling: depth convention confirmed by review — converted exactly once in the vertex-stage epilogue; Tasks 4/6 must NOT convert on RenderPipeline::clip_minus_w_to_w and must keep vk_rasterizer.cpp:3341 depth range + ToMetalViewport unchanged (carried to Task 4/6 dispatch)
- Task 3: Ruling: Task 6 dispatch must WaitReady() then skip if Failed()/Compute()==nullptr for compute, and re-check Failed() after waiting in shader_compile=wait mode
- Task 4: Ruling: accept split — binding/preparation/plain+indirect draws/dispatch/run queue/scratch in core mtl_draw_core.mm (tested), PSO-type and kernel paths in bbgpu mtl_draw.mm (object-compiled; exercised in game); CompiledShader dtor/Find moved to core mtl_compiled_shader.mm; VertexBufferSlot moved to mtl_shader.h; fill mode/depth clip from the pipeline — cost if wrong: tess/geometry kernel paths first tested in game (M4)
- Task 4: Ruling: review Important (Flush contract unenforced -> silent drop) fixed by Runs::EndRenderPass(e)/Runs::Submit(e) wrappers that flush (+ Submitted(tick)) and an assert in Flush when pending and no pass is open; minor "forgetting Submitted leaks" folded in (wrapper calls it); patch-draw vertex slots checked in the same round
- Task 5: Ruling: prepared vertex-input path stays enabled (equivalent by construction per review); Task 6 adds a vertex-only toggle so M4 can A/B it; Task 6 binds null vertex streams with stride 0 (constant) instead of 16 where the helper allows
- Task 7: Ruling: accept deviation — scaled colour copies use new scene_scale_{float,uint,int}_fs reading texel floor((x+0.5)*src/dst) (= Vulkan nearest blit; ms_image_blit cannot scale); same-size depth/stencil copies via CopyTexture; Read(id, info) without Vulkan flags (upscaler M5 adapts); test_scene_resolution ported, runs in Task 8 (also BB_SCENE_STENCIL_BITS=1)
- Task 8: Ruling: accept — x86-64 library compiles the Metal renderer as an interim (needs x86 SPIRV-Cross from bootstrap_deps.sh; -no_fixup_chains for lazy binding); M7 trims the x86 library to the remote shim per spec — cost if wrong: extra x86 build deps until M7
- Task 8: Ruling: presenter stand-in flushes in PrepareFrame (not Present) so it never flushes from the swap thread — cost if wrong: none (stub replaced in M4)
- Final: Ruling: fix wave = I1, I2, I3 + M1 (MultiCopy null kernel -> fall back to CopyBuffer), M2 (DynamicState::Invalidate in Scheduler ctor), msl-check load-ratio gate, run.sh BB_METAL_FRAME echo removed, stale clip_minus_w_to_w marked informational/removed — cost: small

## Deferred minors
- Task 1: minor (deferred): debug labels drop across encoder breaks (no re-push); SetTessellationFactorBuffer({}) cannot unbind; ComputeEncoder() pointer staleness undocumented; one clamp encoder per DrawIndirectCount (pass break each); test noise (unused index binding, blank line)
- Task 1: minor (deferred): scratch allocated per DrawIndirectCount call; kernel reads max_count full strides from args (caller must size the table accordingly)
- Task 2: minor (deferred): pso_cache Impl comment says "three maps" (four); vk_pipeline_key.h includes vk_common.h (vocabulary)
- Task 3: minor (deferred): stale RenderPipeline::clip_minus_w_to_w flag (informational only); no SerializationSupport round-trip test; Flush serializes the archive on a pool thread under archive_mutex; PipelineCache::scheduler unused
- Task 4: minor (deferred): single viewport/scissor only (Encoder gap; backlog); indirect test gaps (records unchanged, partial count); BB_DRAW_WARN_ONCE defined twice; tess skip double-counted in stats; dispatch now ends a tessellation run (closer to Vulkan order)
- Task 4: minor (deferred): blit/clear inside a pass still needs a caller Flush (documented in mtl_draw.h)
- Task 5: minor (deferred): GetVertexInputs attributes computed and discarded on the fallback path; hard-coded 16384/restart constants (move to Metal::Limits); scene_resolution.h half-converted (Task 7 finishes)
- Task 6: minor (deferred): Constant-step layout at stride 0 not exercised under the debug layer; InsertDebugLabel pushes/pops empty groups; RecordDraw draw_run_open contract undocumented
- Task 7: minor (deferred): Resample binds raw src as depth2d (array-typed depth image would fail validation; build a 2D view); scale texel float rounding at exact boundaries; NumericFlavour from guest format; unused mtl_shader.h include; orphaned SPIR-V host shaders; BlitShaders compiled twice
- Task 8: minor (deferred): stages 4-7 cannot build until each milestone converts its files (roadmap note); run.sh still echoes BB_METAL_FRAME; unsupported-reason strings duplicated between msl-check and pipeline build; msl-check passes even if most keys are not loadable; undefined stage macro silently selects stubs (#error); LoadStored copies the archive per blob; stray blank line
- Final: minor (deferred): PopDebugGroup on command buffer while an encoder is open (markers off by default); no-output pass 16x16 scissor drops draws with scissor offset >=16; render pipeline depth format vs image creation format (check at first light)
