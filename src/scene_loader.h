#pragma once

// ====================================================================
// Scene Loader
//
// Parses JSON scene files and OBJ mesh files into a Scene data
// container.  This module is the only place where scene-file formats
// are handled — Scene itself is a pure data struct with no parsing
// logic.
//
// Supported scene formats: .json (custom CIS 565 format)
// Supported mesh formats: .obj (via tinyobjloader)
// ====================================================================

#include "scene.h"

#include <string>

namespace SceneLoader {

    /**
     * Load a complete scene from a JSON file.
     *
     * Parses materials, objects (cube/sphere/mesh referencing OBJ files),
     * and camera settings from the JSON.  OBJ file paths are resolved
     * relative to the JSON file's directory.
     */
    Scene loadFromJSON(const std::string& jsonName);

} // namespace SceneLoader
