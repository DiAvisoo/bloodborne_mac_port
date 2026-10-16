// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: what the shim calls of the settings overlay, whose source stays out of the build until
// it is converted to Metal (M5; BB_M5_PENDING in gpu/CMakeLists.txt).

#include "bbport_overlay.h"
#include "stubs_m3.h"

namespace BbOverlay {

bool HandleEvent(const SDL_Event&) {
    return false;
}

void UpdateTextInput(SDL_Window*) {}

bool Visible() {
    return false;
}

bool CapturesInput() {
    return false;
}

} // namespace BbOverlay
