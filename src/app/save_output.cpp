#include "app/save_output.h"

#include <iomanip>
#include <sstream>
#include <system_error>

namespace fs = std::filesystem;

namespace SaveOutput
{
namespace
{
std::string outputStem(const std::string& imageName)
{
    const fs::path imagePath(imageName);
    const std::string stem = imagePath.filename().stem().string();
    return stem.empty() ? "render" : stem;
}
} // namespace

fs::path createUniqueRunDirectory(
    const fs::path& outputRoot,
    const std::string& imageName,
    const std::string& runId,
    std::string& error)
{
    const fs::path sceneDirectory = outputRoot / outputStem(imageName);
    for (unsigned int suffix = 1; ; ++suffix)
    {
        const std::string directoryName = suffix == 1
            ? runId
            : runId + "-" + (suffix < 10 ? "0" : "") + std::to_string(suffix);
        const fs::path candidate = sceneDirectory / directoryName;

        std::error_code ec;
        const bool candidateExists = fs::exists(candidate, ec);
        if (ec)
        {
            error = "could not inspect output directory '" + candidate.string()
                + "': " + ec.message();
            return {};
        }
        if (candidateExists)
        {
            continue;
        }

        if (fs::create_directories(candidate, ec))
        {
            return candidate;
        }

        if (ec)
        {
            error = "could not create output directory '" + candidate.string()
                + "': " + ec.message();
            return {};
        }
    }
}

fs::path imagePath(
    const fs::path& runDirectory,
    unsigned int pass,
    int iteration,
    bool manual)
{
    std::ostringstream filename;
    filename << "pass-" << std::setw(2) << std::setfill('0') << pass
             << "." << std::setw(6) << std::setfill('0') << iteration << "spp";
    if (manual)
    {
        filename << ".manual";
    }
    filename << ".png";
    return runDirectory / filename.str();
}
} // namespace SaveOutput
