#include "image.h"

#include "utils/logger.h"

#include <stb_image_write.h>

#include <string>

Image::Image(int x, int y)
    : xSize(x), ySize(y), pixels(new glm::vec3[x * y]) 
{}

Image::~Image()
{
    delete[] pixels;  // pixels was allocated with new glm::vec3[x * y]
}

void Image::setPixel(int x, int y, const glm::vec3 &pixel)
{
    assert(x >= 0 && y >= 0 && x < xSize && y < ySize);
    pixels[(y * xSize) + x] = pixel;
}

bool Image::savePNG(const std::string &filename)
{
    unsigned char *bytes = new unsigned char[3 * xSize * ySize];
    for (int y = 0; y < ySize; y++)
    {
        for (int x = 0; x < xSize; x++)
        {
            int i = y * xSize + x;
            glm::vec3 pix = glm::clamp(pixels[i], glm::vec3(), glm::vec3(1)) * 255.f;
            bytes[3 * i + 0] = (unsigned char) pix.x;
            bytes[3 * i + 1] = (unsigned char) pix.y;
            bytes[3 * i + 2] = (unsigned char) pix.z;
        }
    }

    const bool written = stbi_write_png(filename.c_str(), xSize, ySize, 3,
                                        bytes, xSize * 3) != 0;
    if (written)
        Log::info("Image", "Saved %s", filename.c_str());
    else
        Log::error("Image", "Could not save %s", filename.c_str());

    delete[] bytes;
    return written;
}

void Image::saveHDR(const std::string &baseFilename)
{
    std::string filename = baseFilename + ".hdr";
    stbi_write_hdr(filename.c_str(), xSize, ySize, 3, (const float *) pixels);
    Log::info("Image", "Saved %s", filename.c_str());
}
