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

