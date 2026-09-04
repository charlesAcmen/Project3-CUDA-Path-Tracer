#pragma once

#include <filesystem>
#include <string>

namespace SaveOutput
{
std::filesystem::path createUniqueRunDirectory(
    const std::filesystem::path& outputRoot,
    const std::string& imageName,
    const std::string& runId,
    std::string& error);

std::filesystem::path imagePath(
    const std::filesystem::path& runDirectory,
    unsigned int pass,
    int iteration,
    bool manual);
} // namespace SaveOutput
