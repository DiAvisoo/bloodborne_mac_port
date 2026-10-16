# Task 1 report
- RED: `ninja -C out/gpu-arm64 bbmetal_core` -> `ninja: error: unknown target 'bbmetal_core'`.
- Implemented per brief (gpu/CMakeLists.txt, tests/test_mtl_encoder.mm stubs removed, build_macos.sh encoder block).
- GREEN: core command: `PASS: Metal-native foundation (object, device ticks, encoder)` twice (second with MTL_DEBUG_LAYER=1).
- build_gpu_arm64.sh reconfigured and still links libbbgpu.dylib.
- Concerns: the linker prints a harmless "unable to open object file" warning (stale LTO/debug object path from -g on the static lib); gpu/third_party/fsr-vulkan shows modified in git status (pre-existing, not committed).
