#pragma once

// ====================================================================
// Runtime Configuration File Loading
//
// Loads config.local.json (or a user-specified path) to override
// code defaults without recompiling.
//
// Priority:  CLI flags  >  config.local.json  >  code defaults
//
// The config file is loaded BEFORE CLI parsing in main(), so any
// --compact=N / --sort=N / --rng=N flag naturally overrides it.
// ====================================================================

#include <string>

/**
 * Load runtime configuration from a JSON file and apply via setters.
 *
 * Loading strategy:
 *   1. If `explicitPath` is non-empty, load from that path.
 *   2. If `explicitPath` is empty, look for "config.local.json" in CWD.
 *   3. If neither exists, do nothing (pure code-default path).
 *
 * Missing JSON keys leave the corresponding runtime value unchanged
 * (code default or prior override).
 */
void loadConfigFile(const std::string& explicitPath);
