# Local native Metal ultrawide test

Base: Wooloomooloo2/bloodborne_mac_port `master` at `030346a09c313572e87f52fdf75f27d2b80257b3` (0.5.3 cleanup). Local branch: `metal-ultrawide-test`; app version: `0.5.3-ultrawide.4`. No upstream merge or public release.

- Keep upstream's native Metal renderer, command recording, shader compiler and memory fixes. The GPU process is arm64; the x86-64 game/runtime runs through Rosetta. This does not use Vulkan/KosmicKrisp.
- Add native 32:9 / 21:9 presets to the launcher. Apply Kyo's complete scene-resolution, camera, UI viewport and screen-coordinate patch, omitting only the two graphics-heap expansion writes at `0x04B36DDA` and `0x04B36DE2`. Test setting: **3840×1080**, 60 FPS cap.
- Preserve the resolution patch as a unit: the fallback globals at `0x055289F8/FC` and the scene MOV/NOP overrides at `0x02196A6B/7A` must remain together. Removing those overrides but retaining the globals made `.3` allocate 640×360, causing black menus and an oversized HUD.
- Set `BB_UPSCALER=none`, `live_resolution=0`, upscaler off and native preset. Native game-sized targets replace the unsuccessful live-wide compositor; no additional host resizing/composition or temporal upscaling. The FPS badge takes scene dimensions from the actual G-buffer depth attachment, not the requested setting.
- Keep retail direct memory (5056 MiB; 6 GiB Metal arena). Clear `.1`'s automatic 9152 MiB override on restart, while preserving explicit developer overrides and unrelated patches. Reject conflicting built-in resolution/heap overrides in wide mode.
- Separate app, bundle identity, saves, settings and logs: **Bloodborne Metal Ultrawide Test**. Read the selected game folder without writing game assets. No game assets, UI mod files, saves or icons are bundled. Existing UI mods are used as-is.
- Keep this port's launcher and controls (`Cmd+,` / L3+R3 for configuration). The older port's quicksave/quickload and remapping are not transplanted into this renderer trial. Normal game saving works; prior apps and save histories remain backed up.

Build with `build_macos.sh`, `scripts/macos/build_launcher.sh --test`, then `scripts/macos/package_app.sh 0.5.3-ultrawide.4`; use `scripts/macos/bootstrap_deps.sh` first if dependencies are absent. This machine requires `SDKROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk` because its linker does not support the default newer SDK.

Validation: patch tests check complete scene/UI MOV/NOP instructions and camera aspect for every preset, with no heap expansion. Launcher/settings tests cover native rendering, persistence and disable/restart behavior. The isolated full-native/retail-heap gameplay trial displayed correctly proportioned title menus, gameplay and HUD; the user confirmed movement/menu testing looked correct with no artifacts. Observed clinic FPS varied roughly 33–54 with view; this is not a controlled performance comparison. Longer sessions and the other resolutions remain unverified.

Trial history: `.1` used expanded heaps and showed missing legs/progressive triangles (~55 FPS, not a valid benchmark). `.2` used host scene proxies; no triangle artifacts, but squeezed gameplay and stretched menus (~36 FPS). `.3` combined incomplete resolution patching and live UI composition; it produced black menus and a giant HUD (actual scene 640×360). Removing just the 640×360 globals restored the title but left its placement wrong. The `.4` configuration returns to the complete native resolution patch with retail heaps and passed the initial movement/menu check. Later combat crashed from texture-pool exhaustion; the initial test did not establish the cause of geometry corruption.


## Unpackaged combat/geometry diagnostic

Working-tree changes increase only the second graphics pool to 1536 MiB for ultrawide
presets above 1920x1080 pixel count, with an automatic 6144 MiB DMEM setting. Explicit DMEM
overrides are preserved; switching to 2560x720 or disabling ultrawide clears the automatic
increase. Patch/settings tests pass and the build succeeds. This candidate allows the
previously crashing combat, but the user still sees missing legs and terrain after area
travel. A separate save-copy diagnostic with `BB_READBACKS=2` passed the user's visual,
travel and combat check, but varied from 24-60 FPS, with about 28 FPS in Hunter's Dream.
Profiling shows a sustained GPU bottleneck there, with no shader compiles in the measured
interval. It is **not ready to install or release**; Precise is not a launcher default.
See FINDINGS for the evidence and limits of these short tests.


## Optional shader experiment from PR #20

Commit `ebaf9c7` imports upstream `4be665b` with the original author credited. Set
`BB_SPIRV_PRECISE=invariant` only for the separate diagnostic; leave unset for original
vertex precision. The mode invalidates incompatible shader cache entries, so initial loads
recompile. This is not enabled by the launcher. Build and shader/Metal validation pass;
The user confirmed fairly stable 60 FPS at native 3840x1080; warmed gameplay samples
mostly measured 56-60 FPS. Keep `BB_READBACKS=2` as in that trial. Audio later lost some
music/effects despite continuing menu sounds, so this remains a local diagnostic, not a
release. The confirmed fast build and saves were backed up before audio investigation.
