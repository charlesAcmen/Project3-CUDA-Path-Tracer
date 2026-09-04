#include "config/config.h"

#include "utils/json_utils.h"
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

static std::string_view trim(std::string_view value)
{
    while (!value.empty() && std::isspace(static_cast<unsigned char>(value.front())))
        value.remove_prefix(1);
    while (!value.empty() && std::isspace(static_cast<unsigned char>(value.back())))
        value.remove_suffix(1);
    return value;
}

static bool normalizeSaveAtIterations(std::vector<int>& iterations,
                                      std::string& error)
{
    for (const int iteration : iterations)
    {
        if (iteration <= 0)
        {
            error = "saveAt values must be positive integers";
            return false;
        }
    }

    std::sort(iterations.begin(), iterations.end());
    const auto duplicate = std::adjacent_find(iterations.begin(), iterations.end());
    if (duplicate != iterations.end())
    {
        error = "saveAt must not contain duplicate iteration "
            + std::to_string(*duplicate);
        return false;
    }
    return true;
}

static bool parseSaveAtJson(const json& value,
                            std::vector<int>& iterations,
                            std::string& error)
{
    if (!value.is_array())
    {
        error = "saveAt must be an array of positive integers";
        return false;
    }

    std::vector<int> parsed;
    parsed.reserve(value.size());
    for (std::size_t index = 0; index < value.size(); ++index)
    {
        const json& item = value[index];
        if (!item.is_number_integer() && !item.is_number_unsigned())
        {
            error = "saveAt[" + std::to_string(index)
                + "] must be a positive integer";
            return false;
        }

        std::uint64_t parsedValue = 0;
        try
        {
            parsedValue = item.get<std::uint64_t>();
        }
        catch (const json::exception&)
        {
            error = "saveAt[" + std::to_string(index)
                + "] is outside the supported iteration range";
            return false;
        }

        if (parsedValue == 0 || parsedValue > static_cast<std::uint64_t>(INT_MAX))
        {
            error = "saveAt[" + std::to_string(index)
                + "] is outside the supported positive integer range";
            return false;
        }
        parsed.push_back(static_cast<int>(parsedValue));
    }

    if (!normalizeSaveAtIterations(parsed, error)) return false;
    iterations = std::move(parsed);
    return true;
}

static bool parseSaveAtCli(std::string_view value,
                           std::vector<int>& iterations,
                           std::string& error)
{
    if (value.empty())
    {
        error = "--save-at requires a comma-separated list of positive integers";
        return false;
    }

    std::vector<int> parsed;
    std::size_t tokenStart = 0;
    while (tokenStart <= value.size())
    {
        const std::size_t comma = value.find(',', tokenStart);
        const std::size_t tokenEnd = comma == std::string_view::npos
            ? value.size() : comma;
        const std::string_view token = trim(value.substr(tokenStart, tokenEnd - tokenStart));
        if (token.empty())
        {
            error = "--save-at contains an empty item";
            return false;
        }

        int parsedValue = 0;
        const auto result = std::from_chars(token.data(), token.data() + token.size(),
                                            parsedValue);
        if (result.ec != std::errc() || result.ptr != token.data() + token.size()
            || parsedValue <= 0)
        {
            error = "--save-at item '" + std::string(token)
                + "' must be a positive integer";
            return false;
        }
        parsed.push_back(parsedValue);

        if (comma == std::string_view::npos) break;
        tokenStart = comma + 1;
    }

    if (!normalizeSaveAtIterations(parsed, error)) return false;
    iterations = std::move(parsed);
    return true;
}

// ====================================================================
// JSON → AppConfig merge (lowest priority)
// ====================================================================

void mergeConfigJson(AppConfig& cfg, const json& data)
{
    if (data.is_null() || data.empty()) return;

    if (const auto* value = JsonUtil::findKey(data, "compactMethod"))
        cfg.compactMethod = parseCompactMethod(*value, cfg.compactMethod);

    if (const auto* value = JsonUtil::findKey(data, "sortByMaterial"))
        cfg.sortByMaterial = value->get<bool>();

    if (const auto* value = JsonUtil::findKey(data, "rngMode"))
        cfg.rngMode = parseRngMode(*value, cfg.rngMode);

    if (const auto* value = JsonUtil::findKey(data, "directLighting"))
        cfg.directLighting = value->get<bool>();

    // Bloom
    if (const auto* b = JsonUtil::findKey(data, "bloom"))
    {
        if (const auto* value = JsonUtil::findKey(*b, "enabled"))   cfg.bloom.enabled   = value->get<bool>();
        if (const auto* value = JsonUtil::findKey(*b, "threshold")) cfg.bloom.threshold = value->get<float>();
        if (const auto* value = JsonUtil::findKey(*b, "intensity")) cfg.bloom.intensity = value->get<float>();
        if (const auto* value = JsonUtil::findKey(*b, "radius"))    cfg.bloom.radius    = value->get<int>();
        if (const auto* value = JsonUtil::findKey(*b, "sigma"))     cfg.bloom.sigma     = value->get<float>();
    }

    // Chromatic aberration
    if (const auto* ca = JsonUtil::findKey(data, "chromaticAberration"))
    {
        if (const auto* value = JsonUtil::findKey(*ca, "enabled"))   cfg.chromaticAberration.enabled   = value->get<bool>();
        if (const auto* value = JsonUtil::findKey(*ca, "intensity")) cfg.chromaticAberration.intensity = value->get<float>();
    }

    // Vignette
    if (const auto* v = JsonUtil::findKey(data, "vignette"))
    {
        if (const auto* value = JsonUtil::findKey(*v, "enabled"))   cfg.vignette.enabled   = value->get<bool>();
        if (const auto* value = JsonUtil::findKey(*v, "intensity")) cfg.vignette.intensity = value->get<float>();
        if (const auto* value = JsonUtil::findKey(*v, "exponent"))  cfg.vignette.exponent  = value->get<float>();
    }



    // Profiler
    if (const auto* p = JsonUtil::findKey(data, "profiler"))
    {
        if (const auto* value = JsonUtil::findKey(*p, "enabled")) cfg.profCfg.enabled     = value->get<bool>();
        if (const auto* value = JsonUtil::findKey(*p, "warmup"))  cfg.profCfg.warmupIters = value->get<int>();
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
        else if (arg == "--save-at")
        {
            cfg.errors.push_back("--save-at requires '=N1,N2,...'");
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
        else if (arg.rfind("--direct-lighting=", 0) == 0)
        {
            cfg.directLighting = (std::stoi(arg.substr(18)) != 0);
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

    Log::info("Config", "compactMethod=%s  sortByMaterial=%s  rngMode=%s  directLighting=%s",
           toString(cfg.compactMethod),
           cfg.sortByMaterial ? "yes" : "no",
           toString(cfg.rngMode),
           cfg.directLighting ? "yes" : "no");
}

bool validateSaveAtIterations(const std::vector<int>& iterations,
                              unsigned int maximum,
                              std::string& error)
{
    for (const int iteration : iterations)
    {
        if (static_cast<std::uint64_t>(iteration) > maximum)
        {
            error = "saveAt iteration " + std::to_string(iteration)
                + " exceeds this scene's ITERATIONS=" + std::to_string(maximum);
            return false;
        }
    }
    return true;
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
    Log::raw("    %s ../scenes/cornell.json --save-at=50,200,1000\n", exeName);
    Log::raw("\n");
    Log::raw("  Options:\n");
    Log::raw("    --benchmark    Enable profiler CSV output.\n");
    Log::raw("    --compact=N    Compaction mode: 0=off, 1=global scan, 2=Thrust copy_if,\n");
    Log::raw("                   3=shared-memory scan (default).\n");
    Log::raw("    --sort=N       Material sorting: 0=off, nonzero=on (default on).\n");
    Log::raw("    --rng=N        RNG mode: 0=LCG (default), 1=scrambled Halton.\n");
    Log::raw("    --direct-lighting=N  Next-event estimation: 0=off, nonzero=on (default).\n");
    Log::raw("    --warmup=N     Warmup iterations excluded from profiler stats.\n");
    Log::raw("    --save-at=N1,N2,...  Override config saveAt with checkpoints\n");
    Log::raw("                   (e.g., --save-at=50,200,1000).\n");
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
