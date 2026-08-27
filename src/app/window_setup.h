#pragma once

// ====================================================================
// OpenGL / CUDA window and buffer initialisation.
//
// These functions are called once at startup and are independent of
// the rendering pipeline.  The implementation receives the application
// state explicitly instead of reading globals from main.cpp.
// ====================================================================

#include "app/app_state.h"

bool init(AppState& app);
