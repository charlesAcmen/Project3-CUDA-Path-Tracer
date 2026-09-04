/**
 * @file config_test.cu
 * @brief Unit tests for the three-layer config priority chain.
 *
 * Compiles as host-only CUDA (like rng_test) to satisfy the
 * profiler/profiler.h include path via nvcc without needing a GPU.
 *
 * Tests:
 *   1. Code defaults are correct
 *   2. JSON merge overrides defaults
 *   3. CLI flags override JSON
 *   4. JSON + CLI combined priority (CLI wins)
 *   5. Real config file → loadConfigFile → mergeConfigJson overrides defaults
 *      (the exact chain config.local.json takes in the app)
 */

#include "config.h"
#include "app/save_output.h"
#include "app/save_schedule.h"
#include <json.hpp>

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <cassert>
#include <fstream>    // std::ofstream — write a temp config file
#include <string>
#include <system_error>
#include <vector>

using json = nlohmann::json;

// ---- helpers -----------------------------------------------------------

static int  s_tests  = 0;
static int  s_passed = 0;

#define TEST(name)  do { ++s_tests; printf("  %s ... ", name); } while(0)
#define PASS()      do { ++s_passed; printf("OK\n"); } while(0)
#define FAIL(msg)   do { printf("FAIL: %s\n", msg); return 1; } while(0)

static int checkEq(const char* field, int got, int expected) {
    if (got != expected) {
        printf("FAIL: %s = %d, expected %d\n", field, got, expected);
        return 1;
    }
    return 0;
}

static int checkBool(const char* field, bool got, bool expected) {
    if (got != expected) {
        printf("FAIL: %s = %s, expected %s\n", field,
               got ? "true" : "false", expected ? "true" : "false");
        return 1;
    }
    return 0;
}

static int checkStr(const char* field, const std::string& got,
                    const std::string& expected) {
    if (got != expected) {
        printf("FAIL: %s = '%s', expected '%s'\n", field,
               got.c_str(), expected.c_str());
        return 1;
    }
    return 0;
}

static int checkIterations(const char* field, const std::vector<int>& got,
                           const std::vector<int>& expected) {
    if (got != expected) {
        printf("FAIL: %s differs from expected checkpoint list\n", field);
        return 1;
    }
    return 0;
}

// ---- Test: code defaults -----------------------------------------------

static int testDefaults()
{
    TEST("code defaults");
    AppConfig cfg;
    if (checkEq("compactMethod", (int)cfg.compactMethod, (int)CompactMethod::SharedMem)) return 1;
    if (checkBool("sortByMaterial", cfg.sortByMaterial, false)) return 1;
    if (checkEq("rngMode", (int)cfg.rngMode, (int)RngMode::LCG)) return 1;
    if (checkBool("directLighting", cfg.directLighting, true)) return 1;
    if (checkBool("bloom.enabled", cfg.bloom.enabled, false)) return 1;
    if (checkBool("profCfg.enabled", cfg.profCfg.enabled, false)) return 1;
    if (checkIterations("saveAtIterations", cfg.saveAtIterations, {})) return 1;
    PASS();
    return 0;
}

// ---- Test: JSON merge --------------------------------------------------

static int testJsonMerge()
{
    TEST("JSON merge overrides defaults");
    AppConfig cfg;
    json j = json::parse(R"({
        "compactMethod": "Off",
        "sortByMaterial": true,
        "rngMode": "Halton",
        "directLighting": false,
        "saveAt": [50, 1, 10],
        "bloom": { "enabled": true, "threshold": 0.5, "intensity": 0.3, "radius": 5 },
        "chromaticAberration": { "enabled": true },
        "vignette": { "enabled": true, "intensity": 0.8, "exponent": 4.0 },
        "profiler": { "enabled": true, "warmup": 10 }
    })");
    mergeConfigJson(cfg, j);

    if (checkEq("compactMethod", (int)cfg.compactMethod, (int)CompactMethod::Off)) return 1;
    if (checkBool("sortByMaterial", cfg.sortByMaterial, true)) return 1;
    if (checkEq("rngMode", (int)cfg.rngMode, (int)RngMode::HALTON)) return 1;
    if (checkBool("directLighting", cfg.directLighting, false)) return 1;
    if (checkIterations("saveAt (normalized)", cfg.saveAtIterations, {1, 10, 50})) return 1;
    if (checkBool("bloom.enabled", cfg.bloom.enabled, true)) return 1;
    if (checkBool("profCfg.enabled", cfg.profCfg.enabled, true)) return 1;
    if (checkEq("profCfg.warmupIters", cfg.profCfg.warmupIters, 10)) return 1;
    PASS();
    return 0;
}

// ---- Test: CLI override -------------------------------------------------

static int testCliOverride()
{
    TEST("CLI flags override defaults");
    AppConfig cfg;
    const char* argv[] = {
        "prog", "--direct-lighting=0", "--compact=1", "--sort=1", "--rng=1",
        "--save-at=50,10,100", "--warmup=5", "--benchmark", "test.json"
    };
    int argc = sizeof(argv) / sizeof(argv[0]);
    parseCliFlags(cfg, argc, (char**)argv);

    if (checkEq("compactMethod", (int)cfg.compactMethod, (int)CompactMethod::GlobalScan)) return 1;
    if (checkBool("sortByMaterial", cfg.sortByMaterial, true)) return 1;
    if (checkEq("rngMode", (int)cfg.rngMode, (int)RngMode::HALTON)) return 1;
    if (checkBool("directLighting", cfg.directLighting, false)) return 1;
    if (checkIterations("CLI saveAt (normalized)", cfg.saveAtIterations, {10, 50, 100})) return 1;
    if (checkEq("profCfg.warmupIters", cfg.profCfg.warmupIters, 5)) return 1;
    if (checkBool("profCfg.enabled", cfg.profCfg.enabled, true)) return 1;
    if (checkStr("sceneFile", cfg.sceneFile, "test.json")) return 1;
    PASS();
    return 0;
}

// ---- Test: JSON + CLI priority (CLI wins) ------------------------------

static int testPriority()
{
    TEST("CLI overrides JSON (compactMethod=Off vs --compact=2)");
    AppConfig base;
    json j = json::parse(R"({ "compactMethod": 0, "sortByMaterial": false, "directLighting": false, "saveAt": [10, 20] })");
    mergeConfigJson(base, j);                    // JSON sets compact=Off, sort=no
    if (checkEq("after JSON compactMethod", (int)base.compactMethod, (int)CompactMethod::Off)) return 1;

    const char* argv[] = { "prog", "--compact=2", "--sort=1", "--direct-lighting=1", "--save-at=100,50" };
    int argc = 5;
    parseCliFlags(base, argc, (char**)argv);

    if (checkEq("final compactMethod", (int)base.compactMethod, (int)CompactMethod::Thrust)) return 1;
    if (checkBool("final sortByMaterial", base.sortByMaterial, true)) return 1;
    if (checkBool("final directLighting", base.directLighting, true)) return 1;
    if (checkIterations("CLI replaces JSON saveAt", base.saveAtIterations, {50, 100})) return 1;
    PASS();
    return 0;
}

// ---- Test: JSON partial merge (missing keys leave defaults) ------------

static int testPartialJson()
{
    TEST("partial JSON leaves unchanged values at defaults");
    AppConfig cfg;
    json j = json::parse(R"({ "compactMethod": 2 })");
    mergeConfigJson(cfg, j);

    if (checkEq("compactMethod", (int)cfg.compactMethod, (int)CompactMethod::Thrust)) return 1;
    if (checkBool("sortByMaterial (default)", cfg.sortByMaterial, false)) return 1;
    if (checkEq("rngMode (default)", (int)cfg.rngMode, (int)RngMode::LCG)) return 1;
    if (checkBool("directLighting (default)", cfg.directLighting, true)) return 1;
    if (checkBool("bloom.enabled (default)", cfg.bloom.enabled, false)) return 1;
    PASS();
    return 0;
}

// ---- Test: empty JSON does nothing -------------------------------------

static int testEmptyJson()
{
    TEST("empty JSON leaves all defaults unchanged");
    AppConfig cfg;
    cfg.compactMethod = CompactMethod::Thrust;  // change from default
    json j = json::object();
    mergeConfigJson(cfg, j);
    if (checkEq("compactMethod preserved", (int)cfg.compactMethod, (int)CompactMethod::Thrust)) return 1;
    PASS();
    return 0;
}

// ---- Test: real config file → loadConfigFile → mergeConfigJson ---------
// Exercises the exact chain the app runs for config.local.json: the file is
// read off disk by loadConfigFile (nlohmann parse), then merged into
// AppConfig by mergeConfigJson.  A temp file is used so the test never
// depends on the developer's local config.local.json (which is gitignored).

static int testConfigFileLoad()
{
    TEST("config file (loadConfigFile + merge) overrides code defaults");
    const char* path = "_test_config.json";
    {
        std::ofstream f(path);
        f << R"({ "compactMethod": "Thrust", "sortByMaterial": true, "rngMode": "Halton", "directLighting": false,
                  "bloom": { "enabled": true, "threshold": 0.7 },
                  "profiler": { "enabled": true, "warmup": 7 },
                  "saveAt": [100, 5] })";
    }

    AppConfig cfg;
    json j = loadConfigFile(path);    // real file → parsed JSON
    mergeConfigJson(cfg, j);          // JSON → AppConfig (same path config.local.json takes)

    std::remove(path);

    if (checkEq("compactMethod", (int)cfg.compactMethod, (int)CompactMethod::Thrust)) return 1;
    if (checkBool("sortByMaterial", cfg.sortByMaterial, true)) return 1;
    if (checkEq("rngMode", (int)cfg.rngMode, (int)RngMode::HALTON)) return 1;
    if (checkBool("directLighting", cfg.directLighting, false)) return 1;
    if (checkBool("bloom.enabled", cfg.bloom.enabled, true)) return 1;
    if (checkBool("bloom.threshold", cfg.bloom.threshold == 0.7f, true)) return 1;
    if (checkBool("profCfg.enabled", cfg.profCfg.enabled, true)) return 1;
    if (checkEq("profCfg.warmupIters", cfg.profCfg.warmupIters, 7)) return 1;
    if (checkIterations("saveAt", cfg.saveAtIterations, {5, 100})) return 1;
    // untouched keys stay at code defaults
    if (checkBool("vignette.enabled (default)", cfg.vignette.enabled, false)) return 1;
    PASS();
    return 0;
}

// ---- Test: enum values accept names (case-insensitive) OR legacy numbers --

static int testEnumStringParsing()
{
    TEST("enum values accept string names, legacy numbers, reject unknowns");
    AppConfig cfg;

    // String name, mixed case
    mergeConfigJson(cfg, json::parse(R"({ "compactMethod": "Thrust" })"));
    if (checkEq("compactMethod 'Thrust'", (int)cfg.compactMethod, (int)CompactMethod::Thrust)) return 1;

    // Lowercase name
    mergeConfigJson(cfg, json::parse(R"({ "compactMethod": "sharedmem" })"));
    if (checkEq("compactMethod 'sharedmem'", (int)cfg.compactMethod, (int)CompactMethod::SharedMem)) return 1;

    // Legacy number still accepted (backward compat)
    mergeConfigJson(cfg, json::parse(R"({ "compactMethod": 1 })"));
    if (checkEq("compactMethod 1 (legacy)", (int)cfg.compactMethod, (int)CompactMethod::GlobalScan)) return 1;

    // rngMode names + case
    mergeConfigJson(cfg, json::parse(R"({ "rngMode": "Halton" })"));
    if (checkEq("rngMode 'Halton'", (int)cfg.rngMode, (int)RngMode::HALTON)) return 1;
    mergeConfigJson(cfg, json::parse(R"({ "rngMode": "lcg" })"));
    if (checkEq("rngMode 'lcg'", (int)cfg.rngMode, (int)RngMode::LCG)) return 1;

    // Unknown name keeps the current value (falls back, does not throw)
    cfg.compactMethod = CompactMethod::Thrust;
    mergeConfigJson(cfg, json::parse(R"({ "compactMethod": "nope" })"));
    if (checkEq("unknown compactMethod keeps value", (int)cfg.compactMethod, (int)CompactMethod::Thrust)) return 1;

    PASS();
    return 0;
}

// ---- Test: JSON object keys are case-insensitive at every config level --

static int testCaseInsensitiveJsonKeys()
{
    TEST("JSON object keys are case-insensitive, including nested blocks");
    AppConfig cfg;
    mergeConfigJson(cfg, json::parse(R"({
        "COMPACTMETHOD": "Thrust",
        "SORTBYMATERIAL": true,
        "RNGMODE": "Halton",
        "DIRECTLIGHTING": false,
        "SAVEAT": [10, 1],
        "BLOOM": { "ENABLED": true, "THRESHOLD": 0.7, "RADIUS": 4 },
        "CHROMATICABERRATION": { "ENABLED": true, "INTENSITY": 0.006 },
        "VIGNETTE": { "ENABLED": true, "EXPONENT": 3.0 },
        "PROFILER": { "ENABLED": true, "WARMUP": 6 }
    })"));

    if (checkEq("compactMethod", (int)cfg.compactMethod, (int)CompactMethod::Thrust)) return 1;
    if (checkBool("sortByMaterial", cfg.sortByMaterial, true)) return 1;
    if (checkEq("rngMode", (int)cfg.rngMode, (int)RngMode::HALTON)) return 1;
    if (checkBool("directLighting", cfg.directLighting, false)) return 1;
    if (checkIterations("saveAt", cfg.saveAtIterations, {1, 10})) return 1;
    if (checkBool("bloom.enabled", cfg.bloom.enabled, true)) return 1;
    if (checkBool("chromaticAberration.enabled", cfg.chromaticAberration.enabled, true)) return 1;
    if (checkBool("vignette.enabled", cfg.vignette.enabled, true)) return 1;
    if (checkBool("profCfg.enabled", cfg.profCfg.enabled, true)) return 1;
    if (checkEq("profCfg.warmupIters", cfg.profCfg.warmupIters, 6)) return 1;
    PASS();
    return 0;
}

// ---- Test: sceneFile empty check ---------------------------------------

static int testMissingSceneFile()
{
    TEST("no positional arg → sceneFile empty");
    AppConfig base;
    const char* argv[] = { "prog", "--benchmark" };
    int argc = 2;
    parseCliFlags(base, argc, (char**)argv);
    if (checkBool("sceneFile.empty", base.sceneFile.empty(), true)) return 1;
    PASS();
    return 0;
}

// ---- main ---------------------------------------------------------------

// ---- Test: JSON only, no CLI overrides ----------------------------------

static int testJsonOnly()
{
    TEST("JSON values survive when no CLI flags given");
    AppConfig cfg;
    json j = json::parse(R"({
        "compactMethod": 0,
        "sortByMaterial": true,
        "rngMode": 1,
        "directLighting": false,
        "bloom": { "enabled": true, "threshold": 0.8 }
    })");
    mergeConfigJson(cfg, j);

    // CLI with only a scene file — no override flags
    const char* argv[] = { "prog", "scene.json" };
    int argc = 2;
    parseCliFlags(cfg, argc, (char**)argv);

    if (checkEq("compactMethod", (int)cfg.compactMethod, (int)CompactMethod::Off)) return 1;
    if (checkBool("sortByMaterial", cfg.sortByMaterial, true)) return 1;
    if (checkEq("rngMode", (int)cfg.rngMode, (int)RngMode::HALTON)) return 1;
    if (checkBool("directLighting", cfg.directLighting, false)) return 1;
    if (checkBool("bloom.enabled", cfg.bloom.enabled, true)) return 1;
    if (checkStr("sceneFile", cfg.sceneFile, "scene.json")) return 1;
    PASS();
    return 0;
}

// ---- Test: checkpoint validation ---------------------------------------

static int testInvalidSaveAt()
{
    TEST("saveAt rejects malformed, duplicate, and out-of-range values");

    AppConfig duplicate;
    mergeConfigJson(duplicate, json::parse(R"({ "saveAt": [1, 1] })"));
    if (duplicate.valid()) FAIL("duplicate JSON values accepted");

    AppConfig negative;
    mergeConfigJson(negative, json::parse(R"({ "saveAt": [-1] })"));
    if (negative.valid()) FAIL("negative JSON value accepted");

    AppConfig nonInteger;
    mergeConfigJson(nonInteger, json::parse(R"({ "saveAt": [1, 2.5] })"));
    if (nonInteger.valid()) FAIL("non-integer JSON value accepted");

    AppConfig cliEmptyItem;
    const char* emptyItemArgv[] = { "prog", "--save-at=1,,10" };
    parseCliFlags(cliEmptyItem, 2, (char**)emptyItemArgv);
    if (cliEmptyItem.valid()) FAIL("empty CLI item accepted");

    AppConfig cliZero;
    const char* zeroArgv[] = { "prog", "--save-at=0" };
    parseCliFlags(cliZero, 2, (char**)zeroArgv);
    if (cliZero.valid()) FAIL("zero CLI value accepted");

    AppConfig cliRepeated;
    const char* repeatedArgv[] = { "prog", "--save-at=1", "--save-at=10" };
    parseCliFlags(cliRepeated, 3, (char**)repeatedArgv);
    if (cliRepeated.valid()) FAIL("repeated CLI flag accepted");

    std::string error;
    if (validateSaveAtIterations({1, 101}, 100, error))
        FAIL("checkpoint above scene ITERATIONS accepted");

    PASS();
    return 0;
}

// ---- Test: checkpoint pass state ---------------------------------------

static int testSaveSchedule()
{
    TEST("save schedule consumes exact checkpoints and resets per pass");
    SaveSchedule schedule({1, 10});
    if (schedule.pass() != 1) FAIL("initial pass is not one");
    if (schedule.consumeCheckpointAt(0)) FAIL("checkpoint consumed before target");
    if (!schedule.consumeCheckpointAt(1)) FAIL("first checkpoint not consumed");
    if (schedule.consumeCheckpointAt(1)) FAIL("checkpoint consumed twice");
    if (!schedule.consumeCheckpointAt(10)) FAIL("second checkpoint not consumed");

    schedule.beginNextPass();
    if (schedule.pass() != 2) FAIL("reset did not advance pass");
    if (!schedule.consumeCheckpointAt(1)) FAIL("reset did not replay checkpoints");

    SaveSchedule finalCheckpoint({10});
    if (!finalCheckpoint.shouldSaveAt(10, 10)) FAIL("final checkpoint was not saved");
    if (finalCheckpoint.shouldSaveAt(10, 10)) FAIL("final checkpoint was saved twice");
    PASS();
    return 0;
}

// ---- Test: output directory and filename construction -----------------

static int testSaveOutput()
{
    TEST("save output directories and names are unique and collision-free");
    namespace fs = std::filesystem;
    const fs::path root = fs::temp_directory_path() / "cis565_save_output_test";
    std::error_code ec;
    fs::remove_all(root, ec);

    std::string error;
    const fs::path first = SaveOutput::createUniqueRunDirectory(
        root, "nee_small_light.png", "2026-09-04_00-00-00Z", error);
    const fs::path second = SaveOutput::createUniqueRunDirectory(
        root, "nee_small_light.png", "2026-09-04_00-00-00Z", error);
    if (first.empty() || second.empty()) FAIL("run directory was not created");
    if (first == second) FAIL("run directories collided");
    if (first.filename() != "2026-09-04_00-00-00Z") FAIL("first run name changed");
    if (second.filename() != "2026-09-04_00-00-00Z-02") FAIL("second run suffix incorrect");

    const fs::path automatic = SaveOutput::imagePath(first, 1, 50, false);
    const fs::path manual = SaveOutput::imagePath(first, 1, 50, true);
    if (automatic.filename() != "pass-01.000050spp.png") FAIL("automatic name incorrect");
    if (manual.filename() != "pass-01.000050spp.manual.png") FAIL("manual name incorrect");

    fs::remove_all(root, ec);
    PASS();
    return 0;
}

// Stub globals needed by config.cpp's printStartupSummary.
// The test never calls printStartupSummary, but the linker needs them.
std::string  startTimeString;
int          width    = 0;
int          height   = 0;
RenderState* renderState = nullptr;

int main()
{
    printf("Config priority chain tests\n");
    printf("==========================\n\n");

    int failures = 0;
    failures += testDefaults();
    failures += testJsonMerge();
    failures += testCliOverride();
    failures += testPriority();
    failures += testPartialJson();
    failures += testEmptyJson();
    failures += testConfigFileLoad();
    failures += testEnumStringParsing();
    failures += testCaseInsensitiveJsonKeys();
    failures += testMissingSceneFile();
    failures += testJsonOnly();
    failures += testInvalidSaveAt();
    failures += testSaveSchedule();
    failures += testSaveOutput();

    printf("\n%d / %d tests passed", s_passed, s_tests);
    if (failures) printf("  (%d FAILED)", failures);
    printf("\n");
    return failures ? 1 : 0;
}
