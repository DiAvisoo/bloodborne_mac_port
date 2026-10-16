# Task 3 report
Implemented mtl_depth_stencil.{h,mm}, mtl_specialize.{h,cpp} (moved from mtl_shadow.mm, which now builds a SpecConstant list), tests + tests/data/spec_const.comp. vk_to_mtl.mm moved into bbmetal_core (no renderer deps; StencilOp helper already existed) and removed from bbgpu glob. Hash is FNV over fields (no xxhash in core).
RED: test compile failed "mtl_depth_stencil.h file not found". GREEN: PASS twice (plain + MTL_DEBUG_LAYER=1), 5 further runs PASS. Full libbbgpu.dylib builds and links (build_gpu_arm64.sh rc=0).
Note: Specialize only applies 4-byte spec entries (as the brief says); 64-bit spec constants keep defaults.
