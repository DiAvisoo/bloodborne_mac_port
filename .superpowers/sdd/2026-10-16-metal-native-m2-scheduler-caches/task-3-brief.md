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

