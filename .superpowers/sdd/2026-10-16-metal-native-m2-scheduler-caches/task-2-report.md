# Task 2 report
Implemented per brief: Device::CommitUntracked/MakeResident/Evict (residency set on queue), Encoder 32-bit FillBuffer (blit or dedicated compute kernel), BlitAspect copies, depth_plane/resolve/stencil_only attachments, DeclareIndirect, HasBindings, SubmitUntracked, Submit clears bindings (shared Impl::Finish), ~Encoder ends pass first.
Commit: 87ec5be.
## TDD
RED: core command gave 8 compile errors (HasBindings, SubmitUntracked, MakeResident, DeclareIndirect, Evict, BlitAspect missing).
GREEN: core command prints PASS twice (normal + MTL_DEBUG_LAYER=1). 10 consecutive runs: 10/10 PASS.
## Notes
Existing tests already rebind after Submit; no changes needed. Residency calls are macOS 15 APIs (deployment target 15.0). Pre-existing linker warning about a missing arm64 object file is unrelated.
