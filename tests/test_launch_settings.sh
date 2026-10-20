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
    [[ $launch_patches == 'Skip Intro;Resolution Patch 3840x1080 (32:9)' ]]
    [[ $BB_DMEM_MB == 9152 && $BB_UPSCALER == none && $BB_ULTRAWIDE_RES == 3840x1080 ]]
    bb_native_ultrawide ''
    [[ $launch_patches == 'Skip Intro' && -z ${BB_DMEM_MB:-} && -z ${BB_UPSCALER:-} && -z ${BB_ULTRAWIDE_RES:-} ]]
    export BB_DMEM_MB=10240
    bb_native_ultrawide 'Resolution Patch 2560x1080 (21:9)'
    bb_native_ultrawide ''
    [[ $BB_DMEM_MB == 10240 ]]
) || { echo 'FAIL: ultrawide restart/disable'; exit 1; }
echo 'PASS: native ultrawide restart and disable'
