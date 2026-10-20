# Local native Metal ultrawide test

Base: Wooloomooloo2/bloodborne_mac_port `master` at `030346a09c313572e87f52fdf75f27d2b80257b3`
(0.5.3 cleanup). Local branch: `metal-ultrawide-test`. Current local app: **0.5.3-ultrawide.6**.
No public release or upstream merge.

## Current tested configuration

- Native arm64 Metal GPU host; the x86-64 game/runtime still uses Rosetta. No Vulkan/KosmicKrisp.
- Native **3840×1080, 32:9**, upscaling off, 60 FPS cap. The FPS badge measures the actual
  scene depth attachment. User reports fairly stable 60 FPS; warmed moving-view samples
  mostly measured 56-60 FPS on a Mac Studio M2 Max, 64 GB. These are short playtests, not a
  complete-game benchmark or a guarantee for every area.
- Kyo's complete camera/scene/UI resolution patch, with its dependent MOV/NOP instructions
  and fallback globals kept together. Omit its two very large heap increases. Increase only
  the second texture pool to 1536 MiB above 1920×1080 pixel count; use 6144 MiB DMEM. This
  resolved the observed combat texture allocation failure. Smaller presets retain retail
  memory sizes; explicit developer memory overrides are respected.
- `BB_SPIRV_PRECISE=invariant`: PR #20's optional vertex optimization, imported from
  Mojahed Abushawish's `4be665b5bcae16b4f132bb73cdf52a702297904e` with attribution retained.
  Invariant position output and Metal preserveInvariance replace blanket vertex NoContraction;
  tessellation, geometry and export keep their precision rules. Mode-specific shader caches.
- `BB_READBACKS=2`: retained from the test where travel, combat, legs and terrain looked
  correct. The launcher supplies these two defaults; explicit environment overrides win.
  The renderer's standalone default remains unchanged. Further long-session testing is needed.
- Launcher stays available behind the game. Play becomes **Stop**, with a Stopping state.
  It terminates only its retained child process; the native renderer's existing parent-watch
  path closes the renderer. Intentional stops do not open a crash report. Reopening the
  launcher from its Dock icon exposes Stop; quitting the launcher also stops its game.
- Selecting 90/120 FPS automatically disables display sync to avoid an observed 60 FPS
  CAMetalLayer ceiling even on a 120 Hz monitor. Lower caps keep their existing behavior.
  Tearing is possible; explicit `BB_PRESENT_MODE` overrides win. The cap is a maximum:
  observed gameplay can remain GPU-limited around 67-86 FPS across the tested views.
- Separate app, identity, saves, settings and logs: **Bloodborne Metal Ultrawide Test**.
  Stable older apps remain untouched. Game files are read-only. No game assets, saves,
  UI mod files or game icons are bundled; existing UI mods are used as-is.
- This renderer trial uses the newer port's launcher/controls (`Cmd+,` / L3+R3 for settings).
  The older port's quicksave/quickload and remapping are not transplanted. Normal game saves
  work; the original app and save histories remain backed up.

The first invariant-mode launch recompiles affected shaders. An initial sound dropout was
reported; a subsequent session with the same audio implementation worked correctly according
to the user. No audio fix is claimed. Optional audio diagnostics now report signal levels,
invalid samples and decoder/output failures if it recurs; they are off in the launcher.

## Build and validation

Build with `build_macos.sh`, `scripts/macos/build_launcher.sh --test`, then
`scripts/macos/package_app.sh 0.5.3-ultrawide.6`. Bootstrap dependencies first when absent.
This machine needs `SDKROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk` because
its linker does not support the newer default SDK.

Patch/settings tests cover full resolution patch dependencies, pool sizing, restart and
disable transitions. Both architecture builds pass. Shader tests pass in original and
invariant modes, including Metal validation; emitted SPIR-V validates and position invariant
decorations differ as intended. Launcher model tests cover defaults and explicit overrides.
See FINDINGS for measurements and remaining limitations.

## Trial history

`.1`: large heaps, disappearing geometry/triangles. `.2`: host scene proxies, squeezed scene
and stretched UI. `.3`: incomplete resolution patch/live UI composition, black menus and
oversized HUD, actual scene 640×360. `.4`: complete native patch with retail heaps; correct
initial visuals, but first-hit combat texture exhaustion. The bounded pool resolved that
crash; Precise readbacks passed the subsequent visual test but still gave about 28 FPS in
Hunter's Dream. The invariant shader trial produced the user-confirmed performance gain.

Sources: [PR #20](https://github.com/Wooloomooloo2/bloodborne_mac_port/pull/20),
[geometry issue #18](https://github.com/Wooloomooloo2/bloodborne_mac_port/issues/18).
