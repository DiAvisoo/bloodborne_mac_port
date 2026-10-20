# Local native Metal ultrawide test

Base: Wooloomooloo2/bloodborne_mac_port `master` at `030346a09c313572e87f52fdf75f27d2b80257b3` (0.5.3 cleanup). Local branch: `metal-ultrawide-test`; app version: `0.5.3-ultrawide.1`. No upstream merge or public release.

- Keep upstream's native Metal renderer, command recording, shader compiler and memory fixes. Do not transplant the older Vulkan/KosmicKrisp engine or driver.
- Add a launcher Ultrawide toggle and native 32:9 / 21:9 presets. Apply Kyo's existing camera, viewport, HUD-coordinate and heap patch at startup. Add our existing 2560×720 variant. Initial M2 test: **3840×1080 native**, no upscaling, 60 FPS cap.
- Keep generic `output_res=1920x1080` neutral in wide mode. The community patch sets the real game buffers and camera; stacking the generic resolution patch would overwrite it. The separate `ultrawide_res` setting survives in-game saves. Disable live display/upscaler controls while the startup wide patch is active. Report the actual patched size in the overlay.
- Allocate the required 9152 MiB of guest direct memory in wide mode (the Metal arena adjusts to 10 GiB). Restore automatic overrides when wide mode is disabled or the game restarts into 16:9; preserve explicit developer overrides and unrelated patches.
- Separate bundle identity, app name, saves, settings and logs: **Bloodborne Metal Ultrawide Test**. The app reads the user's selected game folder; no game assets, UI mod files or saves are included or installed into it. Existing UI mods in that folder are used as-is.
- Retain this port's launcher and controls (`Cmd+,` / L3+R3 for configuration). Our older port's quicksave/quickload and remapping implementation is not transplanted in this first renderer comparison; normal game saving works, and the prior apps/save histories remain available.

Build using `scripts/macos/bootstrap_deps.sh` for x86_64 and arm64, `build_macos.sh`, then `scripts/macos/package_app.sh 0.5.3-ultrawide.1`. This local build uses the macOS 26.5 SDK because the installed linker does not support the default newer SDK. `scripts/macos/build_launcher.sh --test`, Python patch tests, `tests/test_launch_settings.sh`, and `tests/test_settings_keep.cpp` cover persistence, patch sizes and restart/disable behavior. GPU startup, image correctness and FPS must be checked in gameplay; upstream M4 Max measurements are not M2 measurements.
