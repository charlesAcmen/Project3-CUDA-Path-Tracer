#include "config/config.h"

#include "utils/logger.h"

#include <algorithm>
#include <cctype>     // std::tolower — case-insensitive enum-name parsing
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using json = nlohmann::json;

// ---- Singleton: the one true runtime config --------------------------------
// Function-local static = constructed on first call, no init-order fiasco.
// All runtime code (setters, pipeline .cuh files) reads from this single instance.

AppConfig& appConfig() {
    static AppConfig s_config;
    return s_config;
}

void initAppConfig(int argc, char** argv)
{
    AppConfig& cfg = appConfig();
    mergeConfigJson(cfg, loadConfigFile(""));
    parseCliFlags(cfg, argc, argv);
}

// ====================================================================
// Config file loading
// ====================================================================

json loadConfigFile(const std::string& path)
{
    if (path.empty())
    {
        if (std::filesystem::exists("config.local.json"))
            return loadConfigFile("config.local.json");
        return json::object();
    }

    std::ifstream f(path);
    if (!f.is_open())
    {
        Log::warn("Config", "Could not open '%s'", path.c_str());
        return json::object();
    }

    Log::info("Config", "Loading: %s", path.c_str());
    return json::parse(f);
}

// ====================================================================
// Enum string parsing (config JSON accepts names OR legacy numbers)
//
// mergeConfigJson accepts either the enum's string name (case-insensitive,
// e.g. "SharedMem" / "sharedmem") or the legacy integer value (e.g. 3), so
// existing numeric config files keep working.  An unknown name is NOT a
// silent no-op — it logs a warning and keeps the current value.
// ====================================================================

static std::string toLower(std::string s)
{
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return (char)std::tolower(c); });
    return s;
}

static CompactMethod parseCompactMethod(const json& v, CompactMethod fallback)
{
    if (v.is_number_integer())
        return static_cast<CompactMethod>(v.get<int>());
    if (!v.is_string())
    {
        Log::warn("Config", "compactMethod must be a name or int — keeping %s",
                  toString(fallback));
        return fallback;
    }
    const std::string s = toLower(v.get<std::string>());
    if (s == "off")        return CompactMethod::Off;
    if (s == "globalscan") return CompactMethod::GlobalScan;
    if (s == "thrust")     return CompactMethod::Thrust;
    if (s == "sharedmem")  return CompactMethod::SharedMem;
    Log::warn("Config", "unknown compactMethod '%s' — keeping %s",
              v.get<std::string>().c_str(), toString(fallback));
    return fallback;
}

static RngMode parseRngMode(const json& v, RngMode fallback)
{
    if (v.is_number_integer())
        return static_cast<RngMode>(v.get<int>());
    if (!v.is_string())
    {
        Log::warn("Config", "rngMode must be a name or int — keeping %s",
                  toString(fallback));
        return fallback;
    }
    const std::string s = toLower(v.get<std::string>());
    if (s == "lcg")    return RngMode::LCG;
    if (s == "halton") return RngMode::HALTON;
    Log::warn("Config", "unknown rngMode '%s' — keeping %s",
              v.get<std::string>().c_str(), toString(fallback));
    return fallback;
}

// ====================================================================
// JSON → AppConfig merge (lowest priority)
// ====================================================================

void mergeConfigJson(AppConfig& cfg, const json& data)
{
    if (data.is_null() || data.empty()) return;

    if (data.contains("compactMethod"))
        cfg.compactMethod = parseCompactMethod(data["compactMethod"], cfg.compactMethod);

    if (data.contains("sortByMaterial"))
        cfg.sortByMaterial = data["sortByMaterial"].get<bool>();

    if (data.contains("rngMode"))
        cfg.rngMode = parseRngMode(data["rngMode"], cfg.rngMode);

    // Bloom
    if (data.contains("bloom"))
    {
        const auto& b = data["bloom"];
        if (b.contains("enabled"))   cfg.bloom.enabled   = b["enabled"].get<bool>();
        if (b.contains("threshold")) cfg.bloom.threshold = b["threshold"].get<float>();
        if (b.contains("intensity")) cfg.bloom.intensity = b["intensity"].get<float>();
        if (b.contains("radius"))    cfg.bloom.radius    = b["radius"].get<int>();
        if (b.contains("sigma"))     cfg.bloom.sigma     = b["sigma"].get<float>();
    }

    // Chromatic aberration
    if (data.contains("chromaticAberration"))
    {
        const auto& ca = data["chromaticAberration"];
        if (ca.contains("enabled"))   cfg.chromaticAberration.enabled   = ca["enabled"].get<bool>();
        if (ca.contains("intensity")) cfg.chromaticAberration.intensity = ca["intensity"].get<float>();
    }

    // Vignette
    if (data.contains("vignette"))
    {
        const auto& v = data["vignette"];
        if (v.contains("enabled"))   cfg.vignette.enabled   = v["enabled"].get<bool>();
        if (v.contains("intensity")) cfg.vignette.intensity = v["intensity"].get<float>();
        if (v.contains("exponent"))  cfg.vignette.exponent  = v["exponent"].get<float>();
    }



    // Profiler
    if (data.contains("profiler"))
    {
        const auto& p = data["profiler"];
        if (p.contains("enabled"))  cfg.profCfg.enabled     = p["enabled"].get<bool>();
        if (p.contains("warmup"))   cfg.profCfg.warmupIters = p["warmup"].get<int>();
    }

    // Clamp merged values to their legal ranges (the constexpr bounds on each
    // config struct).  This choke point guarantees a config file can never
    // overflow the device bloomWeights buffer (radius > MAX_BLOOM_RADIUS) or
    // push a visual parameter outside the range the ImGui sliders expose.
    cfg.bloom.clamp();
    cfg.chromaticAberration.clamp();
    cfg.vignette.clamp();
}

// ====================================================================
// CLI flags → AppConfig (highest priority)
// ====================================================================

void parseCliFlags(AppConfig& cfg, int argc, char** argv)
{
    // ---- Handle --config=PATH: reload config with explicit path ----
    for (int i = 1; i < argc; ++i)
    {
        std::string a = argv[i];
        if (a.rfind("--config=", 0) == 0)
        {
            json alt = loadConfigFile(a.substr(9));
            mergeConfigJson(cfg, alt);
            break;
        }
    }

    // ---- Parse flags ----
    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];

        if (arg == "-h" || arg == "--help")
        {
            cfg.showHelp = true;
        }
        else if (arg == "--benchmark")
        {
            cfg.profCfg.enabled = true;
        }
        else if (arg == "--save")
        {
            cfg.autoSave = true;
        }
        else if (arg.rfind("--save-at=", 0) == 0)
        {
            cfg.autoSave = true;
            std::string list = arg.substr(10);
            std::stringstream ss(list);
            std::string token;
            while (std::getline(ss, token, ','))
            {
                if (!token.empty())
                    cfg.saveAtIterations.push_back(std::stoi(token));
            }
            std::sort(cfg.saveAtIterations.begin(),
                      cfg.saveAtIterations.end());
        }
        else if (arg.rfind("--config=", 0) == 0)
        {
            continue;   // already handled above
        }
        else if (arg.rfind("--compact=", 0) == 0)
        {
            int v = std::stoi(arg.substr(10));
            cfg.compactMethod = static_cast<CompactMethod>(v);
        }
        else if (arg.rfind("--sort=", 0) == 0)
        {
            cfg.sortByMaterial = (std::stoi(arg.substr(7)) != 0);
        }
        else if (arg.rfind("--rng=", 0) == 0)
        {
            int v = std::stoi(arg.substr(6));
            cfg.rngMode = static_cast<RngMode>(v);
        }
        else if (arg.rfind("--warmup=", 0) == 0)
        {
            cfg.profCfg.warmupIters = std::stoi(arg.substr(9));
        }
        else if (arg[0] != '-')
        {
            cfg.sceneFile = arg;
        }
    }

    // ---- Seed profCfg metadata ----
    cfg.profCfg.compactMethod  = cfg.compactMethod;
    cfg.profCfg.sortByMaterial = cfg.sortByMaterial;

    // ---- Derive scene name for CSV ----
    if (!cfg.sceneFile.empty())
    {
        std::string s = cfg.sceneFile;
        size_t slash = s.find_last_of("/\\");
        if (slash != std::string::npos) s = s.substr(slash + 1);
        size_t dot = s.find_last_of('.');
        if (dot != std::string::npos) s = s.substr(0, dot);
        cfg.profCfg.sceneName = s;
    }

    Log::info("Config", "compactMethod=%s  sortByMaterial=%s  rngMode=%s",
           toString(cfg.compactMethod),
           cfg.sortByMaterial ? "yes" : "no",
           toString(cfg.rngMode));
}

// ====================================================================
// Display: startup help text
// ====================================================================

void printStartupHelp(const char* exeName)
{
    Log::raw("\n");
    Log::raw("======================================================================\n");
    Log::raw("  CIS 565 Path Tracer - Command Line Help\n");
    Log::raw("======================================================================\n");
    Log::raw("  Usage:\n");
    Log::raw("    %s SCENEFILE.json [options]\n", exeName);
    Log::raw("\n");
    Log::raw("  Examples:\n");
    Log::raw("    %s ../scenes/cornell.json\n", exeName);
    Log::raw("    %s ../scenes/cornell.json --benchmark --compact=2 --warmup=1\n", exeName);
    Log::raw("    %s ../scenes/cornell.json --benchmark --sort=0 --save\n", exeName);
    Log::raw("\n");
    Log::raw("  Options:\n");
    Log::raw("    --benchmark    Enable profiler CSV output.\n");
    Log::raw("    --compact=N    Compaction mode: 0=off, 1=global scan, 2=Thrust copy_if,\n");
    Log::raw("                   3=shared-memory scan (default).\n");
    Log::raw("    --sort=N       Material sorting: 0=off, nonzero=on (default on).\n");
    Log::raw("    --rng=N        RNG mode: 0=LCG (default), 1=scrambled Halton.\n");
    Log::raw("    --warmup=N     Warmup iterations excluded from profiler stats.\n");
    Log::raw("    --save         Save the final rendered image on exit.\n");
    Log::raw("                   (default: yes)\n");
    Log::raw("    --save-at=N1,N2,...  Auto-save at specific iteration counts\n");
    Log::raw("                   (e.g., --save-at=50,200,1000).  Implies --save.\n");
    Log::raw("    --config=PATH  Load runtime config from a JSON file.\n");
    Log::raw("                   Default: config.local.json in CWD.\n");
    Log::raw("    -h, --help     Show this help text.\n");
    Log::raw("\n");
    Log::raw("  Notes:\n");
    Log::raw("    - Flags and scene file are order-independent.\n");
    Log::raw("    - Profiler CSVs are written to profiler_output/<scene>_<timestamp>/\n");
    Log::raw("      when --benchmark is enabled.\n");
    Log::raw("    - Nonzero values for --sort are treated as enabled.\n");
    Log::raw("    - Only compact values 0..3 have defined behavior.\n");
    Log::raw("======================================================================\n");
    Log::raw("\n");
}
