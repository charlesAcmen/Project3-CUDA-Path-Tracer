#include "config.h"

#include "logger.h"
#include "pathtrace.h"
#include "json.hpp"

#include <filesystem>
#include <fstream>

void loadConfigFile(const std::string& explicitPath)
{
    // ---- Resolve path ----
    std::string path = explicitPath;
    if (path.empty())
    {
        if (std::filesystem::exists("config.local.json"))
            path = "config.local.json";
        else
            return;    // no config file, use code defaults
    }

    // ---- Open ----
    std::ifstream f(path);
    if (!f.is_open())
    {
        Log::warn("Config", "Could not open '%s'", path.c_str());
        return;
    }

    Log::info("Config", "Loading: %s", path.c_str());
    auto data = nlohmann::json::parse(f);

    // ---- Apply — each key is optional (missing == keep default) ----

    if (data.contains("compactMethod"))
        setCompactMethod(
            static_cast<CompactMethod>(data["compactMethod"].get<int>()));

    if (data.contains("sortByMaterial"))
        setSortByMaterial(data["sortByMaterial"].get<bool>());

    if (data.contains("rngMode"))
        setRngMode(static_cast<RngMode>(data["rngMode"].get<int>()));

    // Bloom
    if (data.contains("bloom"))
    {
        const auto& b = data["bloom"];
        if (b.contains("enabled"))   setBloomEnabled(b["enabled"].get<bool>());
        if (b.contains("threshold")) setBloomThreshold(b["threshold"].get<float>());
        if (b.contains("intensity")) setBloomIntensity(b["intensity"].get<float>());
        if (b.contains("radius"))    setBloomRadius(b["radius"].get<int>());
        if (b.contains("sigma"))     setBloomSigma(b["sigma"].get<float>());
    }

    // Chromatic aberration
    if (data.contains("chromaticAberration"))
    {
        const auto& ca = data["chromaticAberration"];
        if (ca.contains("enabled"))
            setChromaticAberrationEnabled(ca["enabled"].get<bool>());
        if (ca.contains("intensity"))
            setChromaticAberrationIntensity(ca["intensity"].get<float>());
    }

    // Vignette
    if (data.contains("vignette"))
    {
        const auto& v = data["vignette"];
        if (v.contains("enabled"))
            setVignetteEnabled(v["enabled"].get<bool>());
        if (v.contains("intensity"))
            setVignetteIntensity(v["intensity"].get<float>());
        if (v.contains("exponent"))
            setVignetteExponent(v["exponent"].get<float>());
    }

    Log::info("Config", "Applied: compact=%d  sort=%d  rng=%d",
           static_cast<int>(getCompactMethod()),
           getSortByMaterial() ? 1 : 0,
           static_cast<int>(getRngMode()));
}
