# Native Metal back end (plan stage 3)

Branch `metal-backend`. Status: design; M0 standalone test passes (2026-10-15).

## Why

Measured on the M4 Max in split mode (FINDINGS, "sizing a native Metal back end"):

| | 1440p native | 1080p native |
|---|---|---|
| Frame (uncapped) | ~17.7 ms (55–58 FPS) | ~13.7 ms (71–77 FPS) |
| GPU busy | 14.6 ms (vertex 5.3, fragment 9.3, compute 1.7) | 10.8 ms (vertex 5.3, fragment 4.9, compute 1.7) |
| GPU idle | ~3.1 ms, mostly 4 stalls inside MoltenVK's command buffer | ~2.9 ms |
| Recorder thread | 47% busy, about half of it MoltenVK + Apple's driver | 54% |

What MoltenVK costs us, and what owning the Metal encoding would change:

1. **GPU idle stalls (~2.5–3 ms/frame):** MoltenVK's tessellation emulation, the encoder after an
   image copy, the command-buffer boundary, between compute encoders. Our own encoding orders
   these itself.
2. **Render-pass structure (~409 encoders, ~190 render passes per frame; ~13–16 µs GPU each, plus
   re-binning geometry per pass):** we decide where passes start and end and their load/store
   actions; merge the passes split by copies, uploads and the copy shader; compute in one encoder.
3. **CPU encoding:** half of the recorder thread is MoltenVK's state tracking and command
   translation. Matters most on slower Macs (M1/M2) and is the rival port's limit.

Not changed: the fragment shading cost (same shaders, same MSL from SPIRV-Cross).
Estimate: 1440p ~17.7 → ~13–14 ms, 1080p ~13.7 → ~10–11 ms on the M4 Max.

## Shape: replace the command path, not the whole renderer

The Vulkan renderer is ~25k lines (`renderer_vulkan/`) plus ~10k lines in the texture and buffer
caches that touch Vulkan. Most of the latter is resource management (create, upload, track),
which is not where the time goes. So, in stages:

- **Keep MoltenVK for resources** at first: `VkImage`, `VkBuffer`, memory, samplers, the guest
  arena buffer. `VK_EXT_metal_objects` gives their `MTLTexture`/`MTLBuffer` (MetalFX already
  does this, `gpu/shim/bbport_metalfx.mm`), and `MTLBuffer.gpuAddress` matches the buffer device
  addresses the shaders use.
- **Own the frame's command encoding.** The seam already exists: stage B (`bb:DrawRec`) queues
  deferred commands (`Scheduler::Record`), and `bb:VkRecorder` executes them as `vkCmd*` calls.
  A Metal command path replaces those calls with our own `MTLCommandBuffer` encoders on MoltenVK's
  `MTLCommandQueue` (exported), keeping submission order with MoltenVK's own work.
- **Own the pipelines.** The shader recompiler's SPIR-V goes through SPIRV-Cross (the version
  MoltenVK uses, same options) to MSL, then `MTLRenderPipelineState`/`MTLComputePipelineState`,
  compiled on background threads and kept in an `MTLBinaryArchive`. Bindings: push descriptors map
  directly to `setVertexBuffer`/`setFragmentTexture`… per draw (no descriptor sets to emulate).
- **A switch:** `BB_METAL=1` selects the Metal command path; MoltenVK stays the default and the
  fallback until the Metal path wins on speed and correctness.

## Milestones (each one A/B measured with `prof.sh`-style runs at 1080p and 1440p)

- **M0, interop spike (now):** in the arm64 host, export MoltenVK's queue, an image's texture and a
  buffer; encode our own pass between MoltenVK command buffers, with prefill on; check ordering and
  contents; find MoltenVK's hazard-tracking mode for its resources (tracked: Metal orders our
  encoders for us; untracked: we need fences). Standalone test first (`tests/`), then in the game
  on the present pass.
  *Result (2026-10-15, `tests/test_metal_interop.mm`, in `build_macos.sh --test`):* MoltenVK
  exports its queue, an image's `MTLTexture` and a memory's `MTLBuffer`; a Vulkan clear, our render
  pass + blit, and a Vulkan copy run in queue order, with prefill off and on (2), whether the
  Vulkan copy is recorded before or after our commit (order follows commit). MoltenVK's resources
  are **hazard-tracked** (texture private, buffer shared, `gpuAddress` available), so Metal orders
  our encoders against them without fences at first. *Next:* the same in the game (the present
  pass through our encoder, `BB_METAL=1`).
- **M1, pipelines:** SPIRV-Cross → MSL with our binding model, background compiles, binary archive;
  validated by building every cached pipeline and comparing MSL with MoltenVK's
  (`MVK_CONFIG_SHADER_DUMP_DIR`).
- **M2, draws and dispatches:** graphics draws, compute dispatches, copies and clears of the main
  frame through Metal encoders, our own render-pass boundaries; barriers become our fences.
  Correctness by screenshots on the test routes; then speed.
- **M3, pass merging:** load/store actions from what we know is read later; merge passes split by
  copies/uploads; one compute encoder for consecutive dispatches.
- **M4, tessellation:** Metal tessellation (control stage as a compute kernel, post-tessellation
  vertex function) scheduled without MoltenVK's stalls; rect/quad lists as plain triangles where
  the shader allows.
- **M5, presenter and upscalers:** present through the `CAMetalLayer` directly, MetalFX without the
  commit-order workaround, FSR 3.1 as compute.
- **M6 (optional):** resources created in Metal too; MoltenVK dropped.

## Risks and questions

- MoltenVK's view of resource state: images we write must not be in a state MoltenVK will
  "transition" afterwards (layout changes are no-ops on Metal; check it emits nothing).
- Prefilled command buffers: MoltenVK commits its prefilled command buffer when the Vulkan one
  ends; our command buffers must be committed in order with it (MetalFX met this: it queued behind
  the next prefilled buffer). Owning the whole frame's encoding removes the problem.
- SPIRV-Cross option parity with MoltenVK (the `NoContraction`/precise handling, argument buffers
  off, the same vertex attribute and tessellation conventions).
- The texture cache's copies, blits and tile manager still record Vulkan commands; they move one
  by one (M2/M3), or stay on MoltenVK's command buffers in order.
