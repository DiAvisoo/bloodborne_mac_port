## Global Constraints

- Worktree `~/Developer/GitHub/bb-metal-wt`, branch `metal-native`. arm64 build dir `out/gpu-arm64`
  (configure once with `bash scripts/macos/build_gpu_arm64.sh`; it fails at link from Task 4 on —
  expected).
- Plain C++ headers never import Objective-C; Metal objects cross as `void*` / `Metal::Object` /
  `Metal::BufferRef`. `.mm` files are ARC.
- Vulkan-Headers stay as vocabulary only (`vk::Format`, `vk::ClearValue`, `vk::CompareOp`, enums);
  no `vk::Device`, `vk::Buffer`, `vk::Image`, `vk::ImageView`, `vk::Sampler`, `vk::CommandBuffer`,
  `vk::Semaphore`, `vk::Fence`, `vk::DeviceMemory`, `Vma*` in converted files.
- Limits keep MoltenVK's values (`Metal::Limits`): alignments 16, anisotropy 16, LOD bias 16,
  samplers 500000. Arena block size stays 16 KiB (`MIN_BLOCK_SIZE`, shift 14 — what MoltenVK's
  256-byte alignment gave).
- Ticks: `Metal::Device` (start 1, +1 per tracked commit; untracked commits get no tick and are
  covered by the next tracked one on the same queue).
- Resources are hazard-tracked by Metal: all barrier/layout code is deleted, not translated.
- File names stay (`vk_scheduler.*`, `vk_runtime.*`, …) until M7 renames them; namespaces stay.
- Commits end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

**Commands** (`cd ~/Developer/GitHub/bb-metal-wt` first):

- Core + unit test (Tasks 1–3; `$A=deps/arm64`):

```bash
ninja -C out/gpu-arm64 bbmetal_core >/tmp/m2-core.log 2>&1 || { grep -n "error:" /tmp/m2-core.log | head; false; } \
 && /usr/bin/clang++ -arch arm64 -mmacosx-version-min=15.0 -std=c++20 -fobjc-arc -O2 -g -Wall -Wextra -Werror \
    -Igpu/shim -Igpu/shadps4 -Ideps/arm64/include tests/test_mtl_encoder.mm gpu/shim/common/logging/log.cpp \
    gpu/shim/common/assert.cpp out/gpu-arm64/libbbmetal_core.a -Ldeps/arm64/lib -lfmt \
    -framework Metal -framework Foundation -o out/mtl-encoder-test \
 && out/mtl-encoder-test && MTL_DEBUG_LAYER=1 out/mtl-encoder-test
```

- Object compile check (Tasks 4–12), for each listed file `F` (path under `gpu/`):
  `ninja -C out/gpu-arm64 CMakeFiles/bbgpu.dir/F.o` (e.g.
  `CMakeFiles/bbgpu.dir/shadps4/video_core/buffer_cache/buffer.cpp.o`). Expected: no errors.
- Handle check for converted files: `grep -nE "vk::(Device|Buffer|Image|ImageView|Sampler|CommandBuffer|Semaphore|Fence|DeviceMemory|Unique)\b|vma[A-Z]|Vma[A-Z]|pipelineBarrier|GetDevice\(\)" <files>` → no output.

