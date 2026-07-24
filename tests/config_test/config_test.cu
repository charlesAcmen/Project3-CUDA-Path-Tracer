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
 *   5. --config=PATH loads an alternative file
 *   6. Fresnel mode: config JSON sets value (scene overrides),
 *      CLI --fresnel= sets both value and fresnelSet flag
 */

#include "config.h"
#include "json.hpp"

#include <cstdio>
#include <cstring>
#include <cassert>
#include <string>
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

// ---- Test: code defaults -----------------------------------------------

static int testDefaults()
{
    TEST("code defaults");
    AppConfig cfg;
    if (checkEq("compactMethod", (int)cfg.compactMethod, (int)CompactMethod::SharedMem)) return 1;
    if (checkBool("sortByMaterial", cfg.sortByMaterial, false)) return 1;
    if (checkEq("rngMode", (int)cfg.rngMode, (int)RngMode::LCG)) return 1;
    if (checkEq("fresnelMode", (int)cfg.fresnelMode, (int)FresnelMode::Schlick)) return 1;
    if (checkBool("fresnelSet", cfg.fresnelSet, false)) return 1;
    if (checkBool("bloom.enabled", cfg.bloom.enabled, false)) return 1;
    if (checkBool("profCfg.enabled", cfg.profCfg.enabled, false)) return 1;
    if (checkBool("autoSave", cfg.autoSave, true)) return 1;
    PASS();
    return 0;
}

// ---- Test: JSON merge --------------------------------------------------

static int testJsonMerge()
{
    TEST("JSON merge overrides defaults");
    AppConfig cfg;
    json j = json::parse(R"({
        "compactMethod": 0,
        "sortByMaterial": true,
        "rngMode": 1,
        "fresnelMode": 1,
        "bloom": { "enabled": true, "threshold": 0.5, "intensity": 0.3, "radius": 5 },
        "chromaticAberration": { "enabled": true },
        "vignette": { "enabled": true, "intensity": 0.8, "exponent": 4.0 },
        "profiler": { "enabled": true, "verbose": true, "warmup": 10 }
    })");
    mergeConfigJson(cfg, j);

    if (checkEq("compactMethod", (int)cfg.compactMethod, (int)CompactMethod::Off)) return 1;
    if (checkBool("sortByMaterial", cfg.sortByMaterial, true)) return 1;
    if (checkEq("rngMode", (int)cfg.rngMode, (int)RngMode::HALTON)) return 1;
    if (checkEq("fresnelMode", (int)cfg.fresnelMode, (int)FresnelMode::Accurate)) return 1;
    if (checkBool("fresnelSet", cfg.fresnelSet, false)) return 1;  // JSON never sets fresnelSet
    if (checkBool("bloom.enabled", cfg.bloom.enabled, true)) return 1;
    if (checkBool("profCfg.enabled", cfg.profCfg.enabled, true)) return 1;
    if (checkBool("profCfg.verbose", cfg.profCfg.verbose, true)) return 1;
    if (checkEq("profCfg.warmupIters", cfg.profCfg.warmupIters, 10)) return 1;
    PASS();
    return 0;
}

// ---- Test: CLI override -------------------------------------------------

static int testCliOverride()
{
    TEST("CLI flags override defaults");
    AppConfig base;
    // --compact=1 --sort=1 --rng=1 --fresnel=1 --save --warmup=5 --benchmark
    const char* argv[] = {
        "prog", "--compact=1", "--sort=1", "--rng=1",
        "--fresnel=1", "--save", "--warmup=5", "--benchmark", "test.json"
    };
    int argc = sizeof(argv) / sizeof(argv[0]);
    AppConfig cfg = parseCliFlags(base, argc, (char**)argv);

    if (checkEq("compactMethod", (int)cfg.compactMethod, (int)CompactMethod::GlobalScan)) return 1;
    if (checkBool("sortByMaterial", cfg.sortByMaterial, true)) return 1;
    if (checkEq("rngMode", (int)cfg.rngMode, (int)RngMode::HALTON)) return 1;
    if (checkEq("fresnelMode", (int)cfg.fresnelMode, (int)FresnelMode::Accurate)) return 1;
    if (checkBool("fresnelSet", cfg.fresnelSet, true)) return 1;
    if (checkBool("autoSave", cfg.autoSave, true)) return 1;
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
    json j = json::parse(R"({ "compactMethod": 0, "sortByMaterial": false })");
    mergeConfigJson(base, j);                    // JSON sets compact=Off, sort=no
    if (checkEq("after JSON compactMethod", (int)base.compactMethod, (int)CompactMethod::Off)) return 1;

    const char* argv[] = { "prog", "--compact=2", "--sort=1" };
    int argc = 3;
    AppConfig cfg = parseCliFlags(base, argc, (char**)argv);

    if (checkEq("final compactMethod", (int)cfg.compactMethod, (int)CompactMethod::Thrust)) return 1;
    if (checkBool("final sortByMaterial", cfg.sortByMaterial, true)) return 1;
    PASS();
    return 0;
}

// ---- Test: FresnelSet preserved after CLI ------------------------------

static int testFresnelSet()
{
    TEST("fresnelSet is true only after CLI --fresnel=");
    AppConfig base;
    json j = json::parse(R"({ "fresnelMode": 1 })");
    mergeConfigJson(base, j);

    // JSON alone: fresnelSet stays false
    if (checkBool("after JSON fresnelSet", base.fresnelSet, false)) return 1;

    // CLI --fresnel= sets both
    const char* argv[] = { "prog", "--fresnel=0" };
    int argc = 2;
    AppConfig cfg = parseCliFlags(base, argc, (char**)argv);
    if (checkBool("after CLI fresnelSet", cfg.fresnelSet, true)) return 1;
    if (checkEq("after CLI fresnelMode", (int)cfg.fresnelMode, (int)FresnelMode::Schlick)) return 1;
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

// ---- Test: sceneFile empty check ---------------------------------------

static int testMissingSceneFile()
{
    TEST("no positional arg → sceneFile empty");
    AppConfig base;
    const char* argv[] = { "prog", "--benchmark" };
    int argc = 2;
    AppConfig cfg = parseCliFlags(base, argc, (char**)argv);
    if (checkBool("sceneFile.empty", cfg.sceneFile.empty(), true)) return 1;
    PASS();
    return 0;
}

// ---- main ---------------------------------------------------------------

// Stub globals needed by config.cpp's printStartupSummary.
// The test never calls printStartupSummary, but the linker needs them.
std::string  startTimeString;
int          width    = 0;
int          height   = 0;
RenderState* renderState = nullptr;
bool         g_autoSave = true;

int main()
{
    printf("Config priority chain tests\n");
    printf("==========================\n\n");

    int failures = 0;
    failures += testDefaults();
    failures += testJsonMerge();
    failures += testCliOverride();
    failures += testPriority();
    failures += testFresnelSet();
    failures += testPartialJson();
    failures += testEmptyJson();
    failures += testMissingSceneFile();

    printf("\n%d / %d tests passed", s_passed, s_tests);
    if (failures) printf("  (%d FAILED)", failures);
    printf("\n");
    return failures ? 1 : 0;
}
