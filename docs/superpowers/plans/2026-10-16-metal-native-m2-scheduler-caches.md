# Metal-native M2: Scheduler and caches on Metal — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The scheduler records into `Metal::Encoder`, and the buffer cache, texture cache and
their GPU helpers create and use Metal objects only — no Vulkan handle, VMA call or barrier left
in them.

**Architecture:** Conversion in place (spec approach R1). New Metal core pieces (fills, blit
aspects, attachments, residency, untracked commits, depth-stencil states, SPIR-V specialisation)
go into a small static library `bbmetal_core` that unit tests link on their own, because from
Task 4 on the full `bbgpu` library stops linking until M3 converts its consumers (rasterizer,
presenter, upscalers). Converted files are checked by compiling their object files one by one.

**Tech Stack:** C++20, Objective-C++ (ARC) for `.mm`, Metal (macOS 15: `MTLResidencySet`),
SPIR-V (glslang at build time), the project's `shim/common` log and assert.

**Spec:** `docs/superpowers/specs/2026-10-16-metal-native-renderer-design.md` (components 2, 3).
**Roadmap:** `docs/superpowers/plans/2026-10-16-metal-native-roadmap.md` (incl. "Open items carried
from M1"). **M1 code:** `gpu/shadps4/video_core/renderer_metal/mtl_{object,device,encoder}.*`.

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

## Review Focus

1. A 32-bit fill whose bytes differ (the HTile `0xF` fill, guest DMA fills) must write the exact
   pattern — Task 2, `TestFillPattern`.
2. Buffers reached only through GPU addresses (arena via the page table) must be resident and
   ordered against the compute that writes them — Task 2, `TestIndirectResidency`.
3. Bindings must not survive a Submit (objects are freed at `IsFree(tick)` after it) — Task 2,
   `TestSubmitResetsBindings`.
4. Hoisted uploads committed untracked must run before the submission they belong to and be
   covered by its tick — Task 2, `TestUntrackedRunsFirst`.
5. Depth-only and stencil-bearing formats must copy the right aspect (texture cache downloads
   depth only) — Task 2, `TestDepthAspectCopy`.

---

### Task 1: `bbmetal_core` library; the encoder test stops depending on `bbgpu`

**Files:**
- Modify: `gpu/CMakeLists.txt` (arm64 Metal block, ~lines 186-206)
- Modify: `tests/test_mtl_encoder.mm` (remove the host-runtime symbol stubs)
- Modify: `build_macos.sh` (`--test` block added in M1)

**Interfaces:**
- Produces: `out/gpu-arm64/libbbmetal_core.a` with `mtl_object.cpp`, `mtl_device.mm`,
  `mtl_encoder.mm` and every later core file; `bbgpu` links it.

- [ ] **Step 1: Write the failing check.** Run the core command above.
  Expected: `ninja: error: unknown target 'bbmetal_core'`.

- [ ] **Step 2: Add the target.** In `gpu/CMakeLists.txt`, inside `if (BB_ARM64)` of the Metal
  block, before `file(GLOB METAL_SOURCES …)`, add:

```cmake
        # Metal-native core (docs/superpowers/plans/2026-10-16-metal-native-m2-scheduler-caches.md):
        # device, encoder and helpers with no renderer dependencies; unit tests link it alone.
        set(METAL_CORE_SOURCES
            shadps4/video_core/renderer_metal/mtl_object.cpp
            shadps4/video_core/renderer_metal/mtl_device.mm
            shadps4/video_core/renderer_metal/mtl_encoder.mm)
        add_library(bbmetal_core STATIC ${METAL_CORE_SOURCES})
        target_include_directories(bbmetal_core PRIVATE shim shadps4 ${SHADER_INCLUDE})
        target_compile_features(bbmetal_core PRIVATE cxx_std_20)
        target_link_libraries(bbmetal_core PRIVATE fmt::fmt)
        set_target_properties(bbmetal_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
        foreach(source IN LISTS METAL_CORE_SOURCES)
            if (source MATCHES "\\.mm$")
                set_source_files_properties(${source} TARGET_DIRECTORY bbmetal_core PROPERTIES
                    COMPILE_OPTIONS "-fobjc-arc;-UVK_USE_PLATFORM_METAL_EXT;-Wno-unguarded-availability-new")
            endif()
        endforeach()
```

  After the `file(GLOB METAL_SOURCES …)` line add
  `list(REMOVE_ITEM METAL_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/shadps4/video_core/renderer_metal/mtl_object.cpp ${CMAKE_CURRENT_SOURCE_DIR}/shadps4/video_core/renderer_metal/mtl_device.mm ${CMAKE_CURRENT_SOURCE_DIR}/shadps4/video_core/renderer_metal/mtl_encoder.mm)`
  and after `target_sources(bbgpu PRIVATE ${METAL_SOURCES})` add
  `target_link_libraries(bbgpu PRIVATE bbmetal_core "-framework Metal" "-framework Foundation")`.
  If `fmt::fmt` is not the target name used elsewhere in the file, use the one `bbgpu` links
  (`grep -n "fmt" gpu/CMakeLists.txt`).

- [ ] **Step 3: Drop the stubs from the test.** Delete the `#include <csetjmp>` block with the
  `runtime_fault_recover` / `runtime_disabled_optimizations` definitions from
  `tests/test_mtl_encoder.mm` (the core library has no such references).

- [ ] **Step 4: Run.** Reconfigure (`bash scripts/macos/build_gpu_arm64.sh`; it still links at this
  task) then the core command. Expected: `PASS: Metal-native foundation …` twice.

- [ ] **Step 5: Suite.** In `build_macos.sh`, replace the M1 encoder-test compile line's
  `-Igpu/shadps4 tests/test_mtl_encoder.mm -Lout/gpu-arm64 -lbbgpu -Wl,-rpath,"$ROOT/out/gpu-arm64"`
  with `-Igpu/shim -Igpu/shadps4 -I"$ARM64_PREFIX/include" tests/test_mtl_encoder.mm gpu/shim/common/logging/log.cpp gpu/shim/common/assert.cpp out/gpu-arm64/libbbmetal_core.a -L"$ARM64_PREFIX/lib" -lfmt`
  and the guard `[[ -f out/gpu-arm64/libbbgpu.dylib ]]` with `[[ -f out/gpu-arm64/libbbmetal_core.a ]]`.

- [ ] **Step 6: Commit** (`gpu/CMakeLists.txt`, `tests/test_mtl_encoder.mm`, `build_macos.sh`):
  "Metal-native M2: bbmetal_core static library; the encoder test links it alone".

---

### Task 2: Encoder and Device additions the caches need

**Files:**
- Modify: `gpu/shadps4/video_core/renderer_metal/mtl_device.{h,mm}`, `mtl_device_impl.h`,
  `mtl_encoder.{h,mm}`
- Test: `tests/test_mtl_encoder.mm`

**Interfaces (produces — exact):**

```cpp
// mtl_device.h, class Device (public):
    /// Commits `command_buffer` with no tick of its own. It runs before every later commit on the
    /// queue and is covered by the next tracked tick (one queue completes in order).
    void CommitUntracked(void* command_buffer);
    /// Buffers shaders reach only by GPU address (arena, page table, fault buffer): kept resident
    /// through the queue's residency set.
    void MakeResident(void* buffer);
    void Evict(void* buffer);

// mtl_encoder.h:
enum class BlitAspect : u8 { All, Depth, Stencil };
struct ColorAttachment {           // fields added (existing ones unchanged)
    u32 depth_plane{};             ///< 3D targets
    void* resolve_texture{};       ///< MSAA resolve target (StoreAndMultisampleResolve)
    u32 resolve_level{}, resolve_slice{};
};
struct DepthAttachment {           // field added
    bool stencil_only{};           ///< Stencil8-type formats: bound as the stencil attachment only
};
class Encoder {                     // members added / changed
    /// Fills `size` bytes (offset and size multiples of 4) with the 32-bit `pattern`: a blit when
    /// all four bytes are equal, else a compute kernel. Replaces the M1 low-byte fill.
    void FillBuffer(BufferRef dst, u64 size, u32 pattern);
    void CopyBufferToTexture(BufferRef src, u32 bytes_per_row, u32 bytes_per_image, void* texture,
                             u32 level, u32 slice, Origin3 origin, Size3 size,
                             BlitAspect aspect = BlitAspect::All);
    void CopyTextureToBuffer(void* texture, u32 level, u32 slice, Origin3 origin, Size3 size,
                             BufferRef dst, u32 bytes_per_row, u32 bytes_per_image,
                             BlitAspect aspect = BlitAspect::All);
    /// Buffers shaders reach by GPU address, declared on every Metal encoder this Encoder opens
    /// from now on (render encoders: read; compute: read/write) so Metal orders them.
    void DeclareIndirect(void* buffer);
    /// True while any slot, pipeline or render state is set (Submit clears all of it).
    [[nodiscard]] bool HasBindings() const noexcept;
    /// Ends encoding and commits with no tick (Device::CommitUntracked); keep-alive objects are
    /// released when it completes. For the scheduler's hoisted-upload command buffer.
    void SubmitUntracked();
};
```

Behaviour changes: `Submit()` and `SubmitUntracked()` clear every binding, pipeline and render
state (resolves the roadmap's "binding lifetime" item: a new command buffer starts unbound, as in
Vulkan); `DeclareIndirect` survives Submit. `~Encoder` ends an open pass before checking for a
command buffer (M1 minor).

- [ ] **Step 1: Write the failing tests** — add to `tests/test_mtl_encoder.mm` and call from `main`:

```objc
void TestFillPattern() {
    Metal::Device device;
    Metal::Encoder encoder{device};
    id<MTLBuffer> b = SharedBuffer(device, 64);
    encoder.FillBuffer(Ref(b), 32, 0x0000000Fu);       // HTile-style pattern: bytes differ
    encoder.FillBuffer(Ref(b, 32), 32, 0x5a5a5a5au);   // uniform bytes: blit
    device.Wait(encoder.Submit());
    const u32* w = static_cast<const u32*>(b.contents);
    bool ok = true;
    for (int i = 0; i < 8; ++i) ok &= w[i] == 0x0000000Fu;
    for (int i = 8; i < 16; ++i) ok &= w[i] == 0x5a5a5a5au;
    Expect(ok, "FillBuffer writes the exact 32-bit pattern");
}

void TestSubmitResetsBindings() {
    Metal::Device device;
    Metal::Encoder encoder{device};
    id<MTLBuffer> b = SharedBuffer(device, 16);
    encoder.SetBuffer(Metal::Stage::Compute, 3, Ref(b));
    Expect(encoder.HasBindings(), "a binding is recorded");
    device.Wait(encoder.Submit());
    Expect(!encoder.HasBindings(), "Submit clears bindings (objects may be freed after it)");
}

void TestUntrackedRunsFirst() {
    Metal::Device device;
    Metal::Encoder upload{device}, main_encoder{device};
    id<MTLBuffer> src = SharedBuffer(device, 64), mid = SharedBuffer(device, 64),
                  dst = SharedBuffer(device, 64);
    std::memset(src.contents, 0x11, 64);
    upload.CopyBuffer(Ref(src), Ref(mid), 64);   // "hoisted" upload
    main_encoder.CopyBuffer(Ref(mid), Ref(dst), 64);
    upload.SubmitUntracked();
    const u64 tick = main_encoder.Submit();
    Expect(tick == 1, "an untracked commit takes no tick");
    device.Wait(tick);
    Expect(static_cast<const u8*>(dst.contents)[63] == 0x11,
           "the untracked command buffer ran before the tracked one");
}

void TestIndirectResidency() {
    // A kernel reads a buffer only through its GPU address (as shaders read the arena).
    Metal::Device device;
    Metal::Encoder encoder{device};
    id<MTLDevice> dev = (__bridge id<MTLDevice>)device.NativeDevice();
    NSError* error = nil;
    id<MTLLibrary> lib = [dev newLibraryWithSource:@R"(
#include <metal_stdlib>
using namespace metal;
kernel void through_address(device uint* out [[buffer(0)]], constant ulong& address [[buffer(1)]],
                            uint i [[thread_position_in_grid]]) {
    out[i] = ((device const uint*)address)[i];
}
)" options:nil error:&error];
    id<MTLComputePipelineState> k =
        [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"through_address"] error:&error];
    id<MTLBuffer> hidden = [dev newBufferWithLength:256 options:MTLResourceStorageModePrivate];
    id<MTLBuffer> source = SharedBuffer(device, 256), out = SharedBuffer(device, 256);
    for (u32 i = 0; i < 64; ++i) static_cast<u32*>(source.contents)[i] = 300 + i;
    device.MakeResident((__bridge void*)hidden);
    encoder.DeclareIndirect((__bridge void*)hidden);
    encoder.CopyBuffer(Ref(source), Ref(hidden), 256);  // written by a blit, read by address
    const u64 address = hidden.gpuAddress;
    encoder.SetComputePipeline((__bridge void*)k);
    encoder.SetBuffer(Metal::Stage::Compute, 0, Ref(out));
    encoder.SetBytes(Metal::Stage::Compute, 1, &address, sizeof(address));
    encoder.Dispatch({1, 1, 1}, {64, 1, 1});
    device.Wait(encoder.Submit());
    Expect(static_cast<const u32*>(out.contents)[63] == 363, "a resident buffer reads by address");
    device.Evict((__bridge void*)hidden);
}

void TestDepthAspectCopy() {
    Metal::Device device;
    Metal::Encoder encoder{device};
    id<MTLDevice> dev = (__bridge id<MTLDevice>)device.NativeDevice();
    MTLTextureDescriptor* d = [MTLTextureDescriptor
        texture2DDescriptorWithPixelFormat:MTLPixelFormatDepth32Float_Stencil8 width:4 height:4 mipmapped:NO];
    d.usage = MTLTextureUsageRenderTarget;
    d.storageMode = MTLStorageModePrivate;
    id<MTLTexture> ds = [dev newTextureWithDescriptor:d];
    Metal::RenderPassDesc pass;
    pass.depth.texture = (__bridge void*)ds;
    pass.depth.has_stencil = true;
    pass.depth.clear_depth = pass.depth.clear_stencil = true;
    pass.depth.depth = 0.5;
    pass.depth.stencil = 7;
    encoder.BeginRenderPass(pass);
    encoder.EndRenderPass();
    id<MTLBuffer> depth = SharedBuffer(device, 64), stencil = SharedBuffer(device, 16);
    encoder.CopyTextureToBuffer((__bridge void*)ds, 0, 0, {}, {4, 4, 1}, Ref(depth), 16, 64,
                                Metal::BlitAspect::Depth);
    encoder.CopyTextureToBuffer((__bridge void*)ds, 0, 0, {}, {4, 4, 1}, Ref(stencil), 4, 16,
                                Metal::BlitAspect::Stencil);
    device.Wait(encoder.Submit());
    Expect(static_cast<const float*>(depth.contents)[5] == 0.5f, "depth aspect copied");
    Expect(static_cast<const u8*>(stencil.contents)[5] == 7, "stencil aspect copied");
}
```

- [ ] **Step 2: Run** the core command. Expected: compile errors (`no member named 'SubmitUntracked'`,
  `'BlitAspect'`, …).

- [ ] **Step 3: Implement.**
  - `mtl_device_impl.h`: add `id<MTLResidencySet> residency;` to `Device::Impl`.
  - `mtl_device.mm` constructor (after the event): create the set and attach it to the queue:

```objc
        MTLResidencySetDescriptor* rd = [MTLResidencySetDescriptor new];
        rd.label = @"bbport arena and address-reached buffers";
        NSError* error = nil;
        impl->residency = [impl->device newResidencySetWithDescriptor:rd error:&error];
        ASSERT_MSG(impl->residency, "No Metal residency set: {}",
                   error ? error.localizedDescription.UTF8String : "unknown");
        [impl->queue addResidencySet:impl->residency];
```

  and the new members:

```objc
void Device::CommitUntracked(void* command_buffer) {
    [(__bridge id<MTLCommandBuffer>)command_buffer commit];
}

void Device::MakeResident(void* buffer) {
    std::scoped_lock lock{residency_mutex};
    [impl->residency addAllocation:(__bridge id<MTLBuffer>)buffer];
    [impl->residency commit];
}

void Device::Evict(void* buffer) {
    std::scoped_lock lock{residency_mutex};
    [impl->residency removeAllocation:(__bridge id<MTLBuffer>)buffer];
    [impl->residency commit];
}
```

  (`std::mutex residency_mutex;` private member in `mtl_device.h`, `#include <mutex>`.)
  - `mtl_encoder.mm`:
    - `Impl` gains `std::vector<void*> indirect;` and `id<MTLComputePipelineState> fill_u32;`.
      When `Render()` opens an encoder, after `pass_opened = true;` add
      `for (void* b : indirect) [render useResource:(__bridge id<MTLBuffer>)b usage:MTLResourceUsageRead stages:MTLRenderStageVertex | MTLRenderStageFragment];`
      and when `Compute()` opens one:
      `for (void* b : indirect) [compute useResource:(__bridge id<MTLBuffer>)b usage:MTLResourceUsageRead | MTLResourceUsageWrite];`
    - `FillBuffer`: keep the blit when `pattern == u32(u8(pattern)) * 0x01010101u`; otherwise
      `ASSERT((dst.offset | size) % 4 == 0)`, build `fill_u32` once from this MSL with
      `newLibraryWithSource`:

```metal
#include <metal_stdlib>
using namespace metal;
kernel void fill_u32(device uint* dst [[buffer(0)]], constant uint2& args [[buffer(1)]],
                     uint i [[thread_position_in_grid]]) {
    if (i < args.y) dst[i] = args.x;
}
```

      then dispatch it through the Encoder's own compute path **without disturbing the caller's
      compute state**: end the current encoders, open a dedicated compute encoder
      (`[Cmd() computeCommandEncoder]`), set `fill_u32`, `setBuffer:dst offset:dst.offset atIndex:0`,
      `setBytes:{pattern, u32(size/4)} length:8 atIndex:1`, `dispatchThreads:MTLSizeMake(size/4,1,1)
      threadsPerThreadgroup:MTLSizeMake(64,1,1)`, `endEncoding`. Remove the M1 warning.
    - `CopyBufferToTexture` / `CopyTextureToBuffer`: add the `aspect` parameter and use the
      `…options:` variants with `MTLBlitOptionNone` / `MTLBlitOptionDepthFromDepthStencil` /
      `MTLBlitOptionStencilFromDepthStencil`.
    - `BeginRenderPass`: color `a.depthPlane = c.depth_plane;`; when `c.resolve_texture`:
      `a.resolveTexture = …; a.resolveLevel; a.resolveSlice; a.storeAction = MTLStoreActionStoreAndMultisampleResolve;`.
      Depth: when `d.stencil_only`, configure only `first.stencilAttachment` (texture, level, slice,
      clear from `clear_stencil`/`stencil`) and skip `depthAttachment`.
    - `DeclareIndirect(void* b)`: append if absent; set the bit so the next encoder declares it
      (an already open encoder: call `useResource` on it now too).
    - `HasBindings()`: any of `render_used`, `compute_pipeline`, or any stage's `*_used` mask set.
    - `Submit()`: after committing, reset: `impl->vertex = {}; impl->fragment = {}; impl->compute_stage = {};
      impl->render_pipeline = impl->depth_stencil = impl->compute_pipeline = nullptr;
      impl->render_used = impl->render_dirty = 0; impl->compute_pipeline_dirty = false;`
      (`viewport/scissor/...` back to defaults). `SubmitUntracked()`: same ending, but
      `impl->device.CommitUntracked(cmd)` instead of `Commit`, with the same keep-alive handler.
    - `~Encoder`: `if (impl->in_pass) EndRenderPass(); if (impl->cmd) Submit();`.

- [ ] **Step 4: Run** the core command. Expected: `PASS` twice (normal and validation layer).
  Existing tests that set state, Submit and then rely on it in a new command buffer must rebind;
  `TestCompute`/`TestLargeBytes` already rebind per command buffer — fix any that do not.

- [ ] **Step 5: Commit**: "Metal-native M2: encoder fills with 32-bit patterns, depth/stencil blit
  aspects, 3D/resolve/stencil-only attachments, residency and indirect declarations, untracked
  commits; Submit clears bindings".

---

### Task 3: Depth-stencil state cache and SPIR-V specialisation helper

**Files:**
- Create: `gpu/shadps4/video_core/renderer_metal/mtl_depth_stencil.{h,mm}`
- Create: `gpu/shadps4/video_core/renderer_metal/mtl_specialize.{h,cpp}` (moved from `mtl_shadow.mm`)
- Modify: `mtl_shadow.mm` (call the shared helper), `gpu/CMakeLists.txt` (`METAL_CORE_SOURCES`)
- Test: `tests/test_mtl_encoder.mm`, `tests/data/spec_const.comp` (new)

**Interfaces (produces):**

```cpp
// mtl_depth_stencil.h
namespace Metal {
struct StencilFace {
    u8 fail{}, pass{}, depth_fail{};   ///< VkStencilOp values
    u8 compare{7};                     ///< VkCompareOp (7 = Always)
    u8 read_mask{0xff}, write_mask{0xff};
    bool operator==(const StencilFace&) const = default;
};
struct DepthStencilKey {
    bool depth_test{}, depth_write{}, stencil_test{};
    u8 depth_compare{7};               ///< VkCompareOp
    StencilFace front{}, back{};
    bool operator==(const DepthStencilKey&) const = default;
};
/// MTLDepthStencilState per key, made once (thread-safe; states live as long as the cache).
class DepthStencilCache {
public:
    explicit DepthStencilCache(Device& device);
    ~DepthStencilCache();
    void* Get(const DepthStencilKey& key); ///< id<MTLDepthStencilState>
private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};
}
// mtl_specialize.h
namespace Metal {
struct SpecConstant { u32 id; u32 value; };
/// Returns `spirv` with every OpSpecConstant/OpSpecConstantTrue/False whose SpecId is in
/// `constants` turned into a plain OpConstant/OpConstantTrue/False with that value (others keep
/// their defaults as plain constants). SPIRV-Cross then emits literals (no function constants).
std::vector<u32> Specialize(std::span<const u32> spirv, std::span<const SpecConstant> constants);
}
```

- [ ] **Step 1: Failing tests.** Create `tests/data/spec_const.comp`:

```glsl
#version 450
layout(local_size_x = 1) in;
layout(constant_id = 0) const uint A = 1;
layout(constant_id = 3) const uint B = 2;
layout(set = 0, binding = 0) buffer Out { uint v[]; };
void main() { v[0] = A; v[1] = B; }
```

  Add to the test (it reads `out/spec_const.spv`, compiled by the test command below):

```objc
std::vector<u32> ReadWords(const char* path) {
    FILE* f = std::fopen(path, "rb");
    std::vector<u32> words;
    if (!f) return words;
    u32 w;
    while (std::fread(&w, 4, 1, f) == 1) words.push_back(w);
    std::fclose(f);
    return words;
}

/// OpConstant (43) result values of the module, by result id.
std::map<u32, u32> Constants(const std::vector<u32>& w) {
    std::map<u32, u32> out;
    for (size_t i = 5; i < w.size();) {
        const u32 count = w[i] >> 16, op = w[i] & 0xffff;
        if (op == 43 && count == 4) out[w[i + 2]] = w[i + 3];
        Expect(op != 50, "no OpSpecConstant left after Specialize");
        i += count ? count : 1;
    }
    return out;
}

void TestSpecialize() {
    const std::vector<u32> spirv = ReadWords("out/spec_const.spv");
    Expect(!spirv.empty(), "out/spec_const.spv exists");
    const Metal::SpecConstant constants[] = {{3, 77}};
    const std::vector<u32> out = Metal::Specialize(spirv, constants);
    bool has77 = false, has1 = false;
    for (const auto& [id, value] : Constants(out)) {
        has77 |= value == 77;
        has1 |= value == 1;
    }
    Expect(has77 && has1, "SpecId 3 takes the given value, SpecId 0 keeps its default");
}

void TestDepthStencilCache() {
    Metal::Device device;
    Metal::DepthStencilCache cache{device};
    Metal::DepthStencilKey a;
    a.depth_test = a.depth_write = true;
    a.depth_compare = 1; // Less
    Metal::DepthStencilKey b = a;
    b.depth_write = false;
    void* sa = cache.Get(a);
    Expect(sa && cache.Get(a) == sa, "same key, same state");
    Expect(cache.Get(b) && cache.Get(b) != sa, "different key, different state");
}
```

  and to the core command, before the clang++ line:
  `glslang -V --target-env vulkan1.2 tests/data/spec_const.comp -o out/spec_const.spv >/dev/null &&`
  (also add `#include <map>` to the test, `#include "video_core/renderer_metal/mtl_depth_stencil.h"`,
  `#include "video_core/renderer_metal/mtl_specialize.h"`).

- [ ] **Step 2: Run** — expected: missing headers.

- [ ] **Step 3: Implement.**
  - `mtl_specialize.cpp`: move `Specialize()` from `mtl_shadow.mm:455-513` and change its input
    from `const VkSpecializationInfo*` to the `SpecConstant` span (look a SpecId's value up in the
    span; keep the existing rewriting of op 48/49/50 → 41/42/43). `mtl_shadow.mm` builds a
    `std::vector<SpecConstant>` from its `VkSpecializationInfo` (map entries: `constantID`,
    value read at `offset` with `size` 4) and calls the helper.
  - `mtl_depth_stencil.mm`: `Impl { id<MTLDevice> device; std::mutex mutex;
    std::unordered_map<DepthStencilKey, Object, Hash> states; }` with a hash over the key's bytes
    (`XXH3_64bits` of a packed copy, or a combination of fields). `Get`: lock; find or build with
    `MTLDepthStencilDescriptor` — `depthCompareFunction = depth_test ? VkToMtl::CompareFunction(compare)
    : MTLCompareFunctionAlways`, `depthWriteEnabled = depth_test && depth_write`; when
    `stencil_test`, front/back `MTLStencilDescriptor` with `stencilCompareFunction`,
    `stencilFailureOperation`, `depthFailureOperation`, `depthStencilPassOperation`,
    `readMask`, `writeMask` via the `VkToMtl` helpers (`renderer_metal/vk_to_mtl.h`; add
    `StencilOperation(VkStencilOp)` if missing). Return `Get()` of the stored object.
  - Add both new sources to `METAL_CORE_SOURCES` (and `vk_to_mtl.mm` if `mtl_depth_stencil.mm`
    needs it — then remove it from bbgpu's glob list too).

- [ ] **Step 4: Run** — expected `PASS` twice.
- [ ] **Step 5: Commit**: "Metal-native M2: depth-stencil state cache; SPIR-V specialisation as a
  shared helper".

---

### Task 4: Scheduler on the Encoder

**Files:**
- Modify: `gpu/shadps4/video_core/renderer_vulkan/vk_scheduler.{h,cpp}`
- Delete: `vk_semaphore.{h,cpp}`, `vk_resource_pool.{h,cpp}` (`CommandPool`, `DescriptorHeap`, generic pool)
- Modify: `gpu/CMakeLists.txt` (drop deleted sources if listed explicitly)

**Interfaces:**
- Consumes: `Metal::Device`, `Metal::Encoder` (incl. Task 2's `SubmitUntracked`, `DeclareIndirect`),
  `Metal::DepthStencilCache`, `Metal::RenderPassDesc`.
- Produces (M3 callers depend on these exact shapes):

```cpp
class Scheduler {
public:
    explicit Scheduler(Metal::Device& device, bool threaded);
    void Flush();                       // submits; no SubmitInfo
    void Finish();                      // t = Submit; device.Wait(t)
    void Wait(u64 tick);                // flushes first if tick >= CurrentTick()
    void PopPendingOperations();
    void BeginRendering(const RenderState& state);
    void EndRendering(std::source_location loc = std::source_location::current(), const char* detail = nullptr);
    void SetSubmitCallback(std::function<void()> callback);
    DynamicState& GetDynamicState();
    template <typename Func> void Record(Func&& func);   // Func(Metal::Encoder&)
    // unchanged: IsRecordingDeferred, ReserveRecordData, RecordData, KickRecording, SyncRecording,
    // WaitHostCopies, RecordHostCopy, SignalAfterHostCopies, WaitDeferredSignals, IsRendering,
    // IsRenderingWith, WillBeginRendering, direct_recordings, DeferOperation, DeferPriorityOperation
    u64 CurrentTick() const noexcept;   // device.CurrentTick()
    bool IsFree(u64 tick) const;        // device.IsFree(tick)
    Metal::Device& GetDevice() noexcept;
    Metal::DepthStencilCache& GetDepthStencilCache() noexcept;
    /// Hoisted copies: recorded now on the calling thread into the upload encoder, which is
    /// committed untracked right before the next submission.
    void RecordHoistedCopy(Metal::BufferRef src, Metal::BufferRef dst,
                           std::span<const BufferCopy> copies, bool ordered);
    static std::mutex submit_mutex;     // one committer at a time (Device::Commit requirement)
};
struct BufferCopy { u64 src_offset; u64 dst_offset; u64 size; };   // VideoCore-wide, same field order as vk::BufferCopy
struct RenderAttachment {
    void* texture{};                    // id<MTLTexture> (the image's own texture or a plain view)
    u32 level{}, slice{}, depth_plane{};
    bool clear{};
    vk::ClearValue clear_value{};
};
struct RenderState {
    std::array<RenderAttachment, 8> colors{};
    u32 num_colors{};
    RenderAttachment depth{};
    bool has_depth{}, has_stencil{}, stencil_only{};
    u32 width{}, height{};
    Metal::RenderPassDesc ToDesc() const;
    bool operator==(const RenderState&) const;   // memberwise (no memcmp: padding)
};
```

`BufferCopy` lives in a new small header `gpu/shadps4/video_core/buffer_cache/buffer_copy.h`
(namespace `VideoCore`), included by the scheduler and both caches.

- [ ] **Step 1: Convert** following the inventory (roadmap-linked agent report, summarised):
  - Members: `Metal::Device& device; Metal::Encoder encoder{device}; Metal::Encoder upload_encoder{device};`
    `bool upload_open{}; std::mutex upload_mutex; Metal::DepthStencilCache depth_stencil{device};`
    Remove `master_semaphore`/`Semaphore`, `CommandPool`s, `current_cmdbuf`, `upload_cmdbuf`,
    Tracy VkCtx, `SubmitInfo` (`vk_scheduler.h:76-98`).
  - `RecordChunk` stores `void(Metal::Encoder&)` callables; `Record` calls `func(encoder)` in direct
    mode and the recorder thread calls `cmd(encoder)`. Rename the recorder thread `bb:MtlRecorder`.
  - `CommandBuffer()` is deleted; add `Metal::Encoder& DirectEncoder()` that does what
    `CommandBuffer()` did before returning (sync the recorder, switch to direct mode) for the few
    callers that need immediate encoding (presenter/upscaler in M4/M5).
  - `SubmitExecution()` (was `.cpp:619-706`): lock `submit_mutex`; call the submit callback;
    `EndRendering()`; `SyncRecording(); WaitHostCopies();` then
    `{ std::scoped_lock l{upload_mutex}; if (upload_open) { upload_encoder.SubmitUntracked(); upload_open = false; } }`
    then `const u64 tick = encoder.Submit();` (assert `tick` equals the `CurrentTick()` read at the
    start), ImGui texture submit stays (it moves to Metal in M5; keep the call compiling or guard
    it with a TODO-free comment "ImGui textures: M5"), `PopPendingOperations()`.
  - `Finish()`: `const u64 t = CurrentTick(); Flush(); device.Wait(t);`. `Wait(tick)`: `if (tick >=
    CurrentTick()) Flush(); device.Wait(tick);`. `PopPendingOperations`: drop the 1-in-32 throttle,
    run ops whose tick `device.IsFree`. The priority-ops thread waits with `device.Wait`.
  - `RecordHoistedCopy`: under `upload_mutex`, `upload_open = true;` and
    `for (c : copies) upload_encoder.CopyBuffer({src.buffer, src.offset + c.src_offset}, {dst.buffer, dst.offset + c.dst_offset}, c.size);`
    (`ordered` needs nothing: one blit encoder executes in order).
  - `BeginRendering(state)`: keep the merge-if-equal logic and `BB_PASS_STATS`; record
    `Record([desc = state.ToDesc()](Metal::Encoder& e) { e.BeginRenderPass(desc); });`.
    `RenderState::ToDesc()` maps colors (texture, level, slice, depth_plane, clear, clear color from
    `clear_value.color.float32`), depth (texture, level, slice, has_stencil, stencil_only, clears from
    `clear_value.depthStencil`), width/height. `EndRendering` records `e.EndRenderPass()`.
  - `DynamicState::CommitWith` emits closures taking `Metal::Encoder&`:
    viewport[0] → `SetViewport`, scissor[0] → `SetScissor`, cull/front → `SetCullMode`/`SetFrontFace`
    (VkCullModeFlags 1=Front, 2=Back, 3 → Front+Back is not in Metal: log once, use Back),
    depth bias → `SetDepthBias(enabled ? constant : 0, enabled ? slope : 0, enabled ? clamp : 0)`,
    stencil references → `SetStencilReference`, blend constants → `SetBlendColor`, and the
    depth/stencil fields → build a `Metal::DepthStencilKey` **on the producer thread**, take
    `void* s = depth_stencil.Get(key)` and emit `e.SetDepthStencil(s)`. Fields Metal has no dynamic
    state for (color write masks, rasterizer discard, primitive restart, line width, depth bounds,
    feedback loop) keep their dirty tracking but emit nothing; add one comment above them: "Metal:
    these belong to the pipeline key (M3) or have no equivalent."
  - `SetSubmitCallback` takes `std::function<void()>`.
  - Delete `Commit(Instance, cmdbuf)` (`.cpp:734`), `BbPassLog` command-buffer naming (or set
    `MTLCommandBuffer.label` through `encoder.CommandBuffer()`), Tracy.
- [ ] **Step 2: Compile** `CMakeFiles/bbgpu.dir/shadps4/video_core/renderer_vulkan/vk_scheduler.cpp.o`.
  Expected: no errors. Run the handle check on `vk_scheduler.{h,cpp}`: no output (except
  `vk::ClearValue`, which is vocabulary).
- [ ] **Step 3: Core command** (unit tests unaffected) — expected PASS twice.
- [ ] **Step 4: Commit**: "Metal-native M2: scheduler records into Metal::Encoder (hoisted uploads as
  an untracked command buffer, dynamic state on the encoder, depth-stencil states cached)".

---

### Task 5: Buffers, stream buffer, staging pool, constant ring

**Files:**
- Modify: `buffer_cache/buffer.{h,cpp}`, `renderer_vulkan/vk_staging_buffer_pool.{h,cpp}`,
  `renderer_vulkan/vk_constant_ring.h`

**Interfaces (produces):**

```cpp
enum class MemoryType { DeviceLocal, HostUncached, HostCached, Stream };  // Sparse removed
class UniqueBuffer {            // buffer.h
public:
    void Create(Metal::Device& device, u64 size, MemoryType type);
    void Destroy();
    void* Handle() const noexcept;          // id<MTLBuffer>
    u64 GpuAddress() const noexcept;
    u8* Contents() const noexcept;          // null for DeviceLocal (private storage)
private:
    Metal::Object buffer;
    u64 gpu_address{};
};
class Buffer {                  // ctor takes Metal::Device& instead of const Vulkan::Instance&
public:
    Buffer(Metal::Device& device, MemoryType type, VAddr cpu_addr, u64 size_bytes);
    void* Handle() const noexcept;
    Metal::BufferRef Ref(u64 offset = 0) const noexcept { return {Handle(), offset}; }
    u64 GpuAddress() const noexcept;
    std::span<u8> mapped_data;              // shared storage only
    // Flush/Invalidate/is_coherent/BufferDeviceAddress removed
};
class StreamBuffer;             // ctor: (Metal::Device&, MemoryType, u64 size); ticks from Device
class StagingBufferPool;        // ctor: (Metal::Device&); StagingBufferRef keeps Buffer* / offset / mapped; Flush/Invalidate removed
class ConstantRing;             // Handle() -> Metal::BufferRef; Stamp/Retire use device.CurrentTick/IsFree
```

- [ ] **Step 1: Convert** per the buffer inventory:
  - `buffer.cpp`: `MemoryType` → `MTLResourceOptions` (`DeviceLocal`: `MTLResourceStorageModePrivate`;
    `HostUncached`: `MTLResourceStorageModeShared | MTLResourceCPUCacheModeWriteCombined`;
    `HostCached`, `Stream`: `MTLResourceStorageModeShared`). `Create`:
    `[device newBufferWithLength:size options:opts]` (keep the "Failed allocating buffer" assert),
    `label` from the debug name, `gpu_address = buf.gpuAddress`. Since `buffer.cpp` becomes
    Objective-C++, rename it `buffer.mm` and list it as `.mm` (the CMake glob picks `.mm` only in
    renderer_metal — add it explicitly with the ARC flags) — **or** keep `buffer.cpp` plain C++ and
    put the three Metal calls (create, label, address) in `renderer_metal/mtl_buffer.mm`
    (`Metal::Object CreateBuffer(Device&, u64 size, MemoryType, std::string_view label, u64* gpu_address, u8** contents)`)
    in `bbmetal_core`. Choose the second: caches stay plain C++.
  - Delete VMA includes and flag mapping (`buffer.cpp:34-64`), the sparse branch (:107-111),
    `ReadFlags`/`AllFlags` (`buffer.h:38-44`), `non_coherent_atom_size`, `StreamBuffer::Commit`'s
    flush (keep `AdvanceAndWatch`), `StagingBufferRef::Flush/Invalidate` and all their callers
    (grep `\.Flush()`/`Invalidate()` on staging refs in buffer_cache, texture_cache, runtime).
  - Staging pool and constant ring: replace `Scheduler&`/`Semaphore` tick reads with
    `Metal::Device&` (`CurrentTick`, `IsFree`, `KnownGpuTick`).
- [ ] **Step 2: Compile** the objects of `buffer.cpp`, `vk_staging_buffer_pool.cpp`; core command
  (if `mtl_buffer.mm` was added to the core, add a quick test: a 64-byte Stream buffer has
  `Contents()`, a DeviceLocal one does not, both have a non-zero GPU address). Handle check on
  the converted files: no output.
- [ ] **Step 3: Commit**: "Metal-native M2: buffers, stream buffer, staging pool and constant ring on MTLBuffer".

---

### Task 6: Buffer cache and fault manager

**Files:**
- Modify: `buffer_cache/buffer_cache.{h,cpp}`, `buffer_cache/fault_manager.{h,cpp}`
- Create: `renderer_metal/mtl_host_kernel.{h,mm}` (in `bbgpu`, not the core: it uses `ShaderCompiler`)

**Interfaces (produces):**

```cpp
// mtl_host_kernel.h
namespace Metal {
struct HostKernel {
    Object pipeline;                         // id<MTLComputePipelineState>
    std::array<u32, 8> buffer_slot{};        // SPIR-V binding (set 0) -> MSL buffer index
    std::array<u32, 8> texture_slot{};
    std::array<u32, 3> local_size{};
};
/// Compiles a host GLSL kernel (SPIR-V from host_shaders) with specialisation constants baked in.
HostKernel BuildHostKernel(Device& device, std::span<const u32> spirv,
                           std::span<const SpecConstant> constants, std::string_view name);
}
// BufferCache ctor: (Metal::Device&, Vulkan::Scheduler&, Vulkan::Runtime&, AmdGpu::Liverpool*, PageManager&)
```

- [ ] **Step 1: `BuildHostKernel`** — `Specialize`, hash the result with `XXH3_64bits`, compile via a
  process-wide `ShaderCompiler` (constructed once with the device and the GPU cache dir used by
  `mtl_shader_check.mm`), `newComputePipelineStateWithFunction`, fill the slot arrays with
  `shader.Find(0, b, ShaderBinding::Kind::Buffer/Texture)->msl_index` for bindings 0-7 that exist.
- [ ] **Step 2: Buffer cache** per the inventory:
  - Delete the block-size probe (`:63-86`, `FindMemoryType`, `ARENA_USAGE`): `block_size = MIN_BLOCK_SIZE`.
  - Page table: `MemoryType::DeviceLocal` buffer, zero-filled with `runtime.FillBuffer`, element `u64`;
    `device.MakeResident(page_table->Handle())`.
  - Arena: drop the `#ifdef __APPLE__` (only path); `max_buffer = device.GetLimits().max_buffer_length`;
    keep `BB_ARENA_GIB`, `BB_MAX_BUFFER_GIB`, assert and printf; `device.MakeResident(arena)` and
    `scheduler` → `encoder.DeclareIndirect(arena->Handle())` via a Record closure at construction
    (`scheduler.Record([h](Metal::Encoder& e){ e.DeclareIndirect(h); })`, and again after every
    Submit is **not** needed: `DeclareIndirect` survives Submit).
  - `EnsureResident`: keep the Apple half (`arena->GpuAddress()`), delete `:535-579`.
  - Delete `GetArena` sparse creation/migration, `SubmitPendingArenaBinds`, `ArenaBinds`,
    `BindsForArena`, `memory_semaphore`, Vulkan fields of `Backing`.
  - `vk::BufferCopy` → `VideoCore::BufferCopy` everywhere (positional `emplace_back` keeps order).
  - Alignments from `device.GetLimits()`; `DownloadMemory`: `scheduler.Finish()` then read
    `mapped_data` (no Invalidate).
  - `SynchronizeMemoryFromImage`: HTile fill uses `runtime.FillBuffer(…, 0xF)` (now exact).
- [ ] **Step 3: Fault manager**: `fault_buffer` DeviceLocal (`MakeResident` + `DeclareIndirect`),
  `download_buffer` HostCached; pipeline = `BuildHostKernel(device, FAULT_BUFFER_PROCESS_COMP,
  {{0, caching_pagebits}, {1, max_page_faults}}, "fault_buffer_process")`. `ProcessFaultBuffer`:
  `EndRendering`, then `Record([...](Metal::Encoder& e){ e.SetComputePipeline(k.pipeline.Get());
  e.SetBuffer(Compute, k.buffer_slot[0], fault->Ref()); e.SetBuffer(Compute, k.buffer_slot[1],
  download->Ref(offset)); e.Dispatch({DivCeil(num_pages / 32, 64u), 1, 1}, {64, 1, 1}); })`. Delete
  descriptor writes and barriers; keep `Wait`, `PopPendingOperations`, `DeferOperation`.
- [ ] **Step 4: Compile** objects of `buffer_cache.cpp`, `fault_manager.cpp`, `mtl_host_kernel.mm`;
  handle check: no output.
- [ ] **Step 5: Commit**: "Metal-native M2: buffer cache (arena and page table resident, no sparse
  code) and fault manager on Metal".

---

### Task 7: Runtime — buffer side; barrier tracker deleted

**Files:**
- Modify: `renderer_vulkan/vk_runtime.{h,cpp}`
- Delete: `renderer_vulkan/vk_barrier_tracker.{h,cpp}`

**Interfaces (produces):** `Runtime(Metal::Device&, Scheduler&)`; `CopyBuffer(Buffer& src,
Buffer& dst, std::span<const VideoCore::BufferCopy>)`, `FillBuffer(Buffer&, u64 offset, u64 size,
u32 pattern)`, `HoistCopy(...)` (logic unchanged, calls `scheduler.RecordHoistedCopy(src.Ref(),
dst.Ref(), copies, ordered)`), `RegisterHoistTarget`, `NoteSubmissionUse` (unchanged), and
`AccessBuffer(...)` kept as a thin function that only calls `NoteSubmissionUse` (external callers in
the rasterizer, shader HLE and tile manager keep compiling in M3 with no barrier semantics).
`FlushBarriers()` kept as an empty inline for the same reason (M3 deletes the calls). `InlineData`
deleted.

- [ ] **Step 1: Convert** (`vk_runtime.cpp:166-245`, `:809-939`): `CopyBuffer` = `EndRendering`,
  `NoteSubmissionUse` for both, `Record` a loop of `e.CopyBuffer`; `FillBuffer` = `Record(e.FillBuffer)`.
  Delete `memory_barrier`, `BarrierTracker`, `access_memo`, `IsBufferAccessed`, the memory half of
  `FlushBarriers`.
- [ ] **Step 2: Compile** `vk_runtime.cpp.o` — it will still fail on the image half (Task 9); confirm
  that the remaining errors are only in image functions (`grep -n "error:"` lines point at
  `UploadImage`/`DownloadImage`/`CopyImage`/… line ranges).
- [ ] **Step 3: Commit**: "Metal-native M2: runtime buffer copies and fills on the encoder; barrier
  tracking removed (Metal hazard-tracks)".

---

### Task 8: Images, views, samplers

**Files:** `texture_cache/image.{h,cpp}`, `texture_cache/image_view.{h,cpp}`,
`texture_cache/sampler.{h,cpp}`; create `renderer_metal/mtl_texture.{h,mm}` (core) for the Metal
calls so the texture cache stays plain C++.

**Interfaces (produces):**

```cpp
// mtl_texture.h (bbmetal_core)
namespace Metal {
enum class TextureType : u8 { Tex2D, Tex2DArray, Tex2DMultisample, Tex2DMultisampleArray, Tex3D };
struct TextureDesc {
    TextureType type{TextureType::Tex2D};
    u32 pixel_format{};            // MTLPixelFormat (from VkToMtl::PixelFormat)
    u32 width{1}, height{1}, depth{1}, levels{1}, layers{1}, samples{1};
    bool render_target{}, shader_write{}, format_views{};
};
Object CreateTexture(Device&, const TextureDesc&, std::string_view label, u64* allocated_size);
Object CreateTextureView(void* texture, u32 pixel_format, TextureType type, u32 base_level,
                         u32 levels, u32 base_layer, u32 layers, std::array<u8, 4> swizzle);
struct SamplerDesc {               // filled from LiverpoolToVK enums through VkToMtl
    u32 min_filter{}, mag_filter{}, mip_filter{};      // MTLSamplerMinMagFilter / MipFilter
    u32 address_u{}, address_v{}, address_w{};         // MTLSamplerAddressMode
    u32 border{};                                       // MTLSamplerBorderColor
    float min_lod{}, max_lod{}; u32 max_anisotropy{1};
    bool compare{}; u32 compare_function{};
    bool normalized{true};
};
Object CreateSampler(Device&, const SamplerDesc&);
bool SupportsSampleCount(Device&, u32 samples);
u32 BytesPerBlock(u32 pixel_format); u32 BlockWidth(u32 pixel_format); u32 BlockHeight(u32 pixel_format);
}
```

- [ ] **Step 1: `mtl_texture.mm`** — straightforward descriptor fill (`storageMode = Private`,
  usage `ShaderRead | (render_target ? RenderTarget : 0) | (shader_write ? ShaderWrite : 0) |
  (format_views ? PixelFormatView : 0)`), view creation with `MTLTextureSwizzleChannels`, sampler
  descriptor (`lodMinClamp/lodMaxClamp`, `maxAnisotropy = clamp(1..16)`,
  `compareFunction` only if `compare`, `supportArgumentBuffers = NO`). Block table: reuse the one
  in `mtl_frame.mm:547-583` (move it here). Add a core unit test: a 4x4 RGBA8 texture, a BC1 8x8
  texture (`BytesPerBlock == 8`), a swizzled view, a sampler — all non-null.
- [ ] **Step 2: Image** per the texture inventory (1): `UniqueImage` → `Metal::Object texture` +
  `TextureDesc desc` + `allocated_size`; type from `ImageInfo` (1D → 2D height 1, 1D array → 2D
  array, 2D with layers → array, MSAA → multisample(array), 3D); `pixel_format =
  VkToMtl::PixelFormat(info.pixel_format)`, a 0 result logs once and falls back to RGBA8Unorm
  (assert in debug); samples clamped to `SupportsSampleCount`; usage: render target when not a
  block format, shader write only when not block, not depth, single-sample (keep the 2026-10-11
  BC rule, `image.cpp:44-48`), `format_views` on all non-block images. Keep `pixel_format`
  (`vk::Format`) in `ImageInfo` as vocabulary. Replace `aspect_mask` with
  `has_depth`/`has_stencil` booleans (from `info.props`). Delete `GetBarriers`, `State`,
  `subresource_states`, `usage_flags`, `format_features`, `supported_samples` (→ a `u32
  supported_samples_mask` from `SupportsSampleCount` for 1/2/4/8), VMA, dedicated allocations.
- [ ] **Step 3: ImageView**: `Metal::Object view`; format and swizzle as the inventory (2) says
  (depth views keep the image's format; stencil aspect of `D32S8` → `X32_Stencil8`); no
  `min_lod` (log once when non-zero: "view min LOD has no Metal equivalent"); expose
  `void* Texture() const` (the view, or the image's texture when the view is identical — format,
  full range, identity swizzle — so render targets can use it).
- [ ] **Step 4: Sampler**: `Metal::Object sampler` from `SamplerDesc`; LOD bias and custom border
  colours have no Metal sampler field — log once each (same text as `mtl_frame.mm`) and use
  OpaqueBlack for unsupported borders (Vulkan's fallback). `GetSampler` returns `void*`.
- [ ] **Step 5: Compile** objects of `image.cpp`, `image_view.cpp`, `sampler.cpp`; core command (with
  the new texture test); handle check: no output.
- [ ] **Step 6: Commit**: "Metal-native M2: images, views and samplers are Metal textures, texture
  views and sampler states".

---

### Task 9: Runtime — image side

**Files:** `renderer_vulkan/vk_runtime.{h,cpp}`

- [ ] **Step 1: Convert** per the texture inventory (5): delete `Transit` and every layout/barrier
  call; `UploadImage`/`DownloadImage` → per-layer `CopyBufferToTexture`/`CopyTextureToBuffer` with
  `bytes_per_row = DivCeil(width, BlockWidth) * BytesPerBlock`, `bytes_per_image = bytes_per_row *
  DivCeil(height, BlockHeight)`, aspect `Depth` for depth downloads (`texture_cache.cpp:92`) and
  uploads with the stencil bit cleared (`:816`), `Stencil` for stencil-only copies;
  `CopyImage`/`CopyMip` → `CopyTexture` per slice (2D↔3D: slice i ↔ z = i; size-compatible
  different formats: copy through a `format_views` view created with `CreateTextureView`);
  `CopyColorAndDepth` → always `CopyImageWithBuffer` (Metal has no colour↔depth blit);
  `CopyDepthStencil` → two buffer bounces (Depth, Stencil aspects); `ResolveImage` → a render pass
  whose colour attachment has `resolve_texture` (no draws: `clear = false`, so add
  `Encoder::BeginRenderPass` support for "resolve only": open the encoder in `EndRenderPass` when
  any attachment has a resolve texture — extend Task 2's code and its test with a 4x MSAA clear +
  resolve check); `ClearImage` → a clear-only pass per level/slice; `SetBackingSamples` → new
  texture from the stored `TextureDesc` with the new sample count, then the BlitHelper (Task 11).
  RGB32 formats (no Metal format): log once and skip the copy (M5 adds a 12→16-byte expand if a
  scene needs it).
- [ ] **Step 2: Compile** `vk_runtime.cpp.o` — expected: no errors. Handle check on `vk_runtime.*`.
- [ ] **Step 3: Commit**: "Metal-native M2: runtime image uploads, downloads, copies, resolves and
  clears on the encoder".

---

### Task 10: Tile manager

**Files:** `texture_cache/tile_manager.{h,cpp}`

- [ ] **Step 1: Convert** per inventory (6): pipelines per `TilingKey` = `BuildHostKernel(device,
  <SPIR-V for the key's micro/macro × bpp variant>, spec constants 0-3 (+4-11 macro) from the
  existing code at `tile_manager.cpp:103-121,154-168`, name)`, cached in a map (mutex). Dispatch
  (`RecordTilingDispatch`, :196-225): `EndRendering`; `Record`: `SetComputePipeline`,
  `SetBuffer(Compute, k.buffer_slot[0], tiled)`, `SetBuffer(Compute, k.buffer_slot[1], linear)`,
  `SetBytes(Compute, k.buffer_slot[2], &params, sizeof(params))` (replaces the stream-buffer copy),
  `Dispatch({dim_x, 1, 1}, {64, 1, 1})`. Delete descriptor/pipeline layouts, push descriptors,
  `FlushBarriers`, `AccessBuffer` hazard calls (keep `NoteSubmissionUse` via `runtime.AccessBuffer`).
- [ ] **Step 2: Compile** `tile_manager.cpp.o`; handle check.
- [ ] **Step 3: Commit**: "Metal-native M2: tile manager kernels through the host-kernel builder".

---

### Task 11: Blit helper

**Files:** `texture_cache/blit_helper.{h,cpp}`; create `renderer_metal/mtl_blit_shaders.h` (MSL
source strings).

- [ ] **Step 1: MSL** for the three fragment programs and the fullscreen triangle, hand-written
  from `host_shaders/{fs_tri.vert,color_to_ms_depth.frag,ms_image_blit.frag,depth_resample.frag}`
  (read each GLSL file; keep inputs at texture 0/1, push constants at buffer 0; `color_to_ms_depth`
  writes `[[depth(any)]]` per `[[sample_id]]`; `ms_image_blit` has MSAA (`texture2d_ms`) and
  non-MSAA variants via two functions).
- [ ] **Step 2: Convert** `ReinterpretColorAsMsDepth` and `CopyBetweenMsImages`: render pipelines
  built once per (format, samples) key with `rasterSampleCount`, depth/stencil attachment formats
  for the depth variant (stencil format for D32S8); a depth-stencil state from the
  `DepthStencilCache` (always, write on); temporary 2D views via `CreateTextureView` kept alive with
  `KeepAlive`; `BeginRenderPass` → `SetRenderPipeline`/`SetDepthStencil`/`SetViewport`/`SetScissor`
  → `SetTexture(Fragment, 0, view)` → `Draw(Triangle, 3)` → `EndRenderPass`, inside one `Record`.
- [ ] **Step 3: Compile** `blit_helper.cpp.o`; handle check.
- [ ] **Step 4: Commit**: "Metal-native M2: blit helper as Metal render pipelines (MSL)".

---

### Task 12: Texture cache; milestone gate

**Files:** `texture_cache/texture_cache.{h,cpp}`, roadmap.

- [ ] **Step 1: Convert** per inventory (8): budget queries → `Metal::Device`
  (`CanReportMemoryUsage` true; usage `currentAllocatedSize` — add `u64 Device::AllocatedSize()`
  to the core with a one-line test; budget and local memory `GetLimits().recommended_working_set`);
  sampler limit `GetLimits().max_samplers`; deferred destruction keeps `DeferOperation` (ticks are
  Metal's). Constructor takes `Metal::Device&`.
- [ ] **Step 2: Gate.** Compile every converted object:

```bash
for f in renderer_vulkan/vk_scheduler.cpp renderer_vulkan/vk_runtime.cpp renderer_vulkan/vk_staging_buffer_pool.cpp \
         buffer_cache/buffer.cpp buffer_cache/buffer_cache.cpp buffer_cache/fault_manager.cpp \
         texture_cache/image.cpp texture_cache/image_view.cpp texture_cache/sampler.cpp \
         texture_cache/texture_cache.cpp texture_cache/tile_manager.cpp texture_cache/blit_helper.cpp \
         renderer_metal/mtl_host_kernel.mm; do
  ninja -C out/gpu-arm64 CMakeFiles/bbgpu.dir/shadps4/video_core/$f.o >/tmp/m2-obj.log 2>&1 \
    && echo "OK  $f" || { echo "ERR $f"; grep -m3 "error:" /tmp/m2-obj.log; }
done
```

  Expected: `OK` for all 13. Then the handle check over `buffer_cache/ texture_cache/
  renderer_vulkan/vk_{scheduler,runtime,staging_buffer_pool}.* vk_constant_ring.h`: no output.
  Core command: PASS twice.
- [ ] **Step 3: Roadmap.** Mark M2 done with the date; add to "Open items" anything deferred
  (RGB32 expand, view min LOD, sampler LOD bias, custom borders, `PixelFormatView` bandwidth cost)
  for M5. List for the M3 plan the external consumers from the inventories (rasterizer:
  `Handle()` ×13, `image_view` ×5, `GetSampler`, `Transit` ×8, `AccessBuffer` ×8; shader HLE;
  scene resolution; upscaler; camera/object motion; presenter).
- [ ] **Step 4: Commit and push**: "Metal-native M2 done: scheduler and caches on Metal".
