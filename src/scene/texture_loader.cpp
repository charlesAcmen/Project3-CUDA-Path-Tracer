// ====================================================================
// Texture loading: decode PNG/JPG image files, or in-memory bytes (a .glb
// bufferView / embedded glTF buffer), into Scene::textures.
//
// This file does NOT define STB_IMAGE_IMPLEMENTATION — the main app gets
// stb's implementation from src/stb.cpp, and loader_test.cu defines it in
// its own TU (see tests/loader_test/loader_test.cu).  It only includes the
// header for declarations.
// ====================================================================

#include "scene/loader_internal.h"

#include "utils/logger.h"

#include <stb_image.h>   // stbi_load / stbi_load_from_memory

#include <cmath>

using namespace std;

namespace SceneLoader {

// Upload decoded RGB texels into Scene::textures.
//
// The shading pipeline works in linear space, and color PNG/JPG texels are
// sRGB, so linearization happens here once, at load time — the GPU sampler
// then returns linear colors that feed the accumulation buffer directly.
// DATA maps (normal / metallic-roughness / occlusion) whose bytes are already
// linear pass through untouched (srgb=false).
//
// @return Index into Scene::textures (>= 0), or -1 on failure
static int uploadTexturePixels(Scene& scene, const stbi_uc* data,
                               int w, int h, bool srgb)
{
    TextureData td;
    td.width  = w;
    td.height = h;
    td.pixels.resize((size_t)w * (size_t)h);

    // sRGB → linear: c <= 0.04045 ? c/12.92 : ((c+0.055)/1.055)^2.4
    const auto lin = [](float c) {
        return (c <= 0.04045f) ? c / 12.92f
                               : std::pow((c + 0.055f) / 1.055f, 2.4f);
    };
    for (int i = 0; i < w * h; i++)
    {
        const float r = data[3 * i + 0] / 255.0f;
        const float g = data[3 * i + 1] / 255.0f;
        const float b = data[3 * i + 2] / 255.0f;
        td.pixels[i] = srgb
            ? glm::vec3(lin(r), lin(g), lin(b))
            : glm::vec3(r, g, b);
    }

    const int id = (int)scene.textures.size();
    scene.textures.push_back(std::move(td));
    return id;
}

/**
 * Load an image file (PNG/JPG via stb_image) into Scene::textures.
 *
 * @param scene  Scene to append the texture to
 * @param path   Absolute path to the image file
 * @param srgb   True (default): linearize sRGB texels on load (color maps).
 *               False: keep raw byte values (normal/ORM/occlusion maps).
 * @return       Index into Scene::textures (>= 0), or -1 on failure
 */
int loadTextureFile(Scene& scene, const string& path, bool srgb)
{
    int w = 0, h = 0, comp = 0;
    // req_comp = 3 forces RGB (3 channels), so texels are always vec3.
    stbi_uc* data = stbi_load(path.c_str(), &w, &h, &comp, 3);
    if (data == nullptr)
    {
        Log::error("Scene", "Failed to load texture image: %s", path.c_str());
        return -1;
    }

    const int id = uploadTexturePixels(scene, data, w, h, srgb);
    stbi_image_free(data);
    return id;
}

/**
 * Load an image decoded from in-memory bytes (a .glb bufferView / embedded
 * glTF buffer) into Scene::textures.  stbi_load_from_memory auto-detects the
 * format (PNG/JPG/…) from the magic bytes.
 *
 * @param scene  Scene to append the texture to
 * @param bytes  Decoded image payload (already loaded by cgltf_load_buffers)
 * @param len    Byte length of `bytes`
 * @param srgb   True: linearize sRGB texels (color maps); false: keep raw
 *               byte values (normal/ORM/occlusion maps)
 * @return       Index into Scene::textures (>= 0), or -1 on failure
 */
int loadTextureMemory(Scene& scene, const unsigned char* bytes,
                      int len, bool srgb)
{
    int w = 0, h = 0, comp = 0;
    stbi_uc* data = stbi_load_from_memory(bytes, len, &w, &h, &comp, 3);
    if (data == nullptr)
    {
        Log::error("Scene",
                   "Failed to decode embedded texture image (%d bytes)", len);
        return -1;
    }

    const int id = uploadTexturePixels(scene, data, w, h, srgb);
    stbi_image_free(data);
    return id;
}

} // namespace SceneLoader
