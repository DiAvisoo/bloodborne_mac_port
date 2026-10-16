# Metal-native M4 — decision record (rulings and deferred minors)

Copied from the execution ledger when M4 finished (2026-10-16). Each ruling: what was decided — why — cost if wrong.

## Rulings
- Ruling: T4 run by an opus subagent with explicit save-safety rules — runs the game end to end — cost if wrong: a test save touched (restored from backups).
- Task 1: Ruling: carry to Task 2 — Present always inside WithAutoreleasePool (Begin/Present autorelease); every acquired drawable is presented (also blank/clear-only), unpresented drawables return only per refresh; shutdown never waits on an uncommitted tick (bounded dtor wait + LOG_ERROR) — cost if wrong: present stalls/hang at exit
- Task 2: Ruling: accept deviations — PrepareLastFrame null when the last frame is in flight (only caller DrawLastFrame null-checks; no black flash); FindView keeps format swizzle (alpha One via mapping); resized frame texture replaced directly (KeepAlive + cmdbuf refs) — cost if wrong: none seen
- Task 3: Ruling: root cause = Metal archiver fails serializing an already-serialized/loaded archive after adding reused functions; fix A (fresh archive rebuilt from session descriptors at each Flush) + unique entry names (MSL cache v5) — WarmUp preloads every stored pipeline so the session set covers the cache — cost if wrong: unused stale pipelines recompile from MSL cache
- Final: Ruling: M1 (split-mode _exit skips shutdown archive write) re-graded Important by effect — default mode loses the last ~10 s of compiled pipelines on every quit (stutter next launch) — fix wave = I1 + M1 + backlog entries for deferred items (docs) — cost: small

## Deferred minors
- Task 1: minor (deferred): post-process pass loads the output (no DontCare option); TestPresentLetterbox readback duplicated, hdr=1 passthrough untested; linker "(arm64) unable to open object file" warning on test links
- Task 2: minor (deferred): PrepareBlankFrame(true) may block the swap thread when both frames queued (pre-existing, Stability backlog); last_presented set on skipped presents; dtor resets present queue before draw Finish; TestBoundedWaits prints an intentional error line; unused blit_helper.h include and liverpool member
- Task 3: minor (deferred): writer's due time computed before the timed wait (Flush during wait loosens throttle); periodic 10 s path untested (checked with temporary 300 ms build); Sync does not drain the build pool before writing
- Task 4: minor (deferred): 0.4.0 draws range narrowed in FINDINGS (886-887 vs 884-888); compiled counter counts try_emplace race losers; display-on worst frames 17-40 ms on both builds (measure M5 gate with display on)
- Final: minor (deferred): ~Presenter resets present_queue while swap thread may present (single-process only, M7 removes); archive write re-serializes everything every 10 s (log duration); ready_tick read before Flush (Flush should return tick); msl-check labels archive_adds as compiled and misreports "could not be written" with no compiles; TestAutoreleaseLoop has no presentDrawable path; last_presented/flip counter on skipped presents; roadmap M4 row lists bbgpu.cpp; record array outside @autoreleasepool
- Final: re-review — all addressed, no new breakage; minor (deferred): duplicated doc comment in mtl_present.h; bbgpu_flush_caches call_once flushes once per process; stop_host unbounded waitpid
