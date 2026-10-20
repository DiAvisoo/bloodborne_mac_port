#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-2.0-or-later
# bbport.ini fps_cap / fullscreen -> environment (scripts/launch_settings.sh).
set -euo pipefail
cd -- "$(dirname -- "$0")/.."
ini=$(mktemp)
trap 'rm -f "$ini"' EXIT
check() { # ini text, expected "LIMIT FULLSCREEN" (- = unset), extra environment...
    printf '%s' "$1" > "$ini"
    local got
    got=$(env -u BB_FPS_LIMIT -u BB_FULLSCREEN "${@:3}" /bin/bash -c \
        'source scripts/launch_settings.sh; bb_launch_settings "$1"; printf "%s %s" "${BB_FPS_LIMIT:--}" "${BB_FULLSCREEN:--}"' _ "$ini")
    if [[ $got != "$2" ]]; then
        echo "FAIL: launch settings for $(printf %q "$1") ${*:3}: got '$got', want '$2'"
        exit 1
    fi
}
check $'fps_cap=40\nfullscreen=1\n' "40 1"
check $'fps_cap=120\n' "120 -"
check $'fps_cap=30\nfullscreen=0\n' "30 -"
check $'fps_cap=60' "60 -"                      # no final newline
check $'fps_cap=40\r\nfullscreen=1\r\n' "40 1"  # CRLF
check $'fps_cap=45\n' "- -"                     # not a cap the launcher offers: ignored
check $'upscaler=off\n' "- -"                   # no keys: today's behaviour
check $'fps_cap=40\nfullscreen=1\n' "90 1" BB_FPS_LIMIT=90
check $'fullscreen=1\n' "- 0" BB_FULLSCREEN=0
# High caps remove the observed 60 Hz Metal drawable wait; standard caps and external
# choices retain their original presentation behavior. Re-evaluate after an inherited restart.
(
    source scripts/launch_settings.sh
    unset BB_PRESENT_MODE BB_AUTO_PRESENT_MODE BB_FPS_LIMIT BB_FULLSCREEN
    printf 'fps_cap=120\n' > "$ini"
    bb_launch_settings "$ini"
    [[ $BB_FPS_LIMIT == 120 && $BB_PRESENT_MODE == Immediate && $BB_AUTO_PRESENT_MODE == 1 ]]
    export BB_FPS_LIMIT=60
    bb_launch_settings "$ini"
    [[ -z ${BB_PRESENT_MODE:-} && -z ${BB_AUTO_PRESENT_MODE:-} ]]
    export BB_PRESENT_MODE=Mailbox BB_FPS_LIMIT=90
    bb_launch_settings "$ini"
    [[ $BB_PRESENT_MODE == Mailbox && -z ${BB_AUTO_PRESENT_MODE:-} ]]
    unset BB_PRESENT_MODE
    bb_launch_settings "$ini"
    [[ $BB_PRESENT_MODE == Immediate && $BB_AUTO_PRESENT_MODE == 1 ]]
    export BB_FPS_LIMIT=40
    bb_launch_settings "$ini"
    [[ -z ${BB_PRESENT_MODE:-} && -z ${BB_AUTO_PRESENT_MODE:-} ]]
) || { echo 'FAIL: presentation mode for high FPS caps'; exit 1; }
echo 'PASS: presentation mode defaults, overrides and restart'
rm -f "$ini"
/bin/bash -c 'source scripts/launch_settings.sh; bb_launch_settings /nonexistent/bbport.ini' ||
    { echo "FAIL: launch settings with a missing file"; exit 1; }
echo "PASS: launch settings (fps_cap, fullscreen)"

# In-game restart inherits the old environment: disabling wide mode must clear only our
# overrides and keep the user's external patch list and explicit memory limit.
(
    source scripts/launch_settings.sh
    unset BB_DMEM_MB BB_ULTRAWIDE_AUTO_DMEM BB_ULTRAWIDE_SESSION BB_UPSCALER
    export BB_PATCHES='Skip Intro'
    bb_native_ultrawide 'Resolution Patch 3840x1080 (32:9)'
    [[ $launch_patches == 'Skip Intro' ]]
    [[ $BB_DMEM_MB == 6144 && $BB_ULTRAWIDE_AUTO_DMEM == 1 && $BB_UPSCALER == none && $BB_ULTRAWIDE_RES == 3840x1080 ]]
    [[ -z ${BB_RENDER_RES:-} && -z ${BB_OUTPUT_RES:-} ]]
    # An inherited .1 test environment must shed its unsafe automatic heap override.
    export BB_DMEM_MB=9152 BB_ULTRAWIDE_AUTO_DMEM=1
    bb_native_ultrawide 'Resolution Patch 3840x1080 (32:9)'
    [[ $BB_DMEM_MB == 6144 && $BB_ULTRAWIDE_AUTO_DMEM == 1 ]]
    bb_native_ultrawide 'Resolution Patch 2560x720 (32:9)'
    [[ -z ${BB_DMEM_MB:-} && -z ${BB_ULTRAWIDE_AUTO_DMEM:-} ]]
    bb_native_ultrawide ''
    [[ $launch_patches == 'Skip Intro' && -z ${BB_DMEM_MB:-} && -z ${BB_UPSCALER:-} && -z ${BB_ULTRAWIDE_RES:-} ]]
    export BB_DMEM_MB=10240
    bb_native_ultrawide 'Resolution Patch 2560x1080 (21:9)'
    bb_native_ultrawide ''
    [[ $BB_DMEM_MB == 10240 ]]
) || { echo 'FAIL: ultrawide restart/disable'; exit 1; }
echo 'PASS: native ultrawide restart and disable'
