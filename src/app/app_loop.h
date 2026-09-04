#pragma once

#include "app/app_state.h"

bool initializeSaveOutput(AppState& app);
void saveImage(AppState& app);
bool saveImage(AppState& app, bool manual);
void mainLoop(AppState& app);
