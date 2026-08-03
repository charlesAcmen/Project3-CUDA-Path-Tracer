/**
 * @file rng_compare.cu
 * @brief Standalone RNG comparison: LCG vs Halton(CP-legacy) vs Halton(Owen).
 *
 * Compiles with nvcc as a pure host program (no kernel launches).
 * Includes the real rng.h from the path tracer — tests the actual
 * implementation, not a re-implementation.
 *
 * Usage:
 *   rng_compare --samples 4096 --pixels 16 --bounces 3 --out data.csv
 *
 * Output (CSV):
 *   pixel,iter,bounce,dim,lcg,halton_cp,halton_owen
 *
 * Three columns for the same (pixel, iter, bounce, dim) tuple:
 *   lcg         — thrust LCG (baseline)
 *   halton_cp   — legacy Cranley-Patterson Halton (BUGGY, full-uint32 offset)
 *   halton_owen — new Owen-scrambled Halton (current implementation)
 *
 * The halton_cp column is computed inline here (not via RngMode) so that
 * the legacy path is available for comparison even after rng.h was updated.
 */

#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cerrno>
#include <vector>
#include <algorithm>

#ifdef _WIN32
#include <direct.h>   // _mkdir
#else
#include <sys/stat.h> // mkdir
#endif

// Helper: create directory (recursive single-level for our use case)
static bool ensureDir(const char* path) {
#ifdef _WIN32
    return _mkdir(path) == 0 || errno == EEXIST;
#else
    return mkdir(path, 0755) == 0 || errno == EEXIST;
#endif
}

// Include the project's actual RNG header.
// Compiled with -I../../src so this resolves to src/rng/rng.h.
// This gives us the NEW Owen-Scrambled Halton implementation.
#include "rng/rng.h"

// ============================================================================
// Legacy CP-Halton — inlined here for comparison
// ============================================================================
// This re-implements the OLD (buggy) CP-rotation Halton as it existed before
// Owen Scrambling was introduced.  We inline it rather than using RngMode so
// we can compare both in a single test run without modifying sceneStructs.h.

/** Reproduce the old (LEGACY) CP rotation seed — linear sum, collision-prone. */
static float haltonCP_next(unsigned int haltonIndex,
                           unsigned int pixelIndex,
                           unsigned int bounceIndex,
                           int dim)
{
    int base = getHaltonPrime(dim);
    float raw = radicalInverse(base, haltonIndex);

    // BUG: linear sum seed — same as the original buggy code
    unsigned int h = utilhash(
        pixelIndex  * 131u
        + bounceIndex * 17u
        + (unsigned int)dim * 11u);
    float offset = (float)(h & 0xFFFFFFu) * (1.0f / 16777216.0f);
    return cpRotate(raw, offset);
}

/** Reproduce the old (LEGACY) makeRngState haltonIndex — full uint32 offset. */
static unsigned int haltonCP_index(int iter, int pixelIndex, int depth)
{
    // This is the BUGGY formula: unmasked full-uint32 hash + iter
    // means the walk starts at a random position ~2.7B away from 0.
    return mixHaltonBaseOffset((unsigned int)pixelIndex, (unsigned int)depth)
           + (unsigned int)iter;
}

// ============================================================================
// CLI argument parsing (minimal — no external dependencies)
// ============================================================================

struct Args {
    int    numSamples = 4096;   // iterations per pixel
    int    numPixels  = 16;     // number of pixels to simulate
    int    numBounces = 3;      // bounce depths
    int    numDims    = 10;     // dimensions 0..9
    const char* outFile = "profiler_output/rng_test/rng_data.csv";
};

static void printUsage(const char* prog) {
    fprintf(stderr,
        "Usage: %s [options]\n"
        "  --samples N    iterations per pixel  (default: 4096)\n"
        "  --pixels  N    number of pixels      (default: 16)\n"
        "  --bounces N    bounce depths         (default: 3)\n"
        "  --dims    N    dimensions            (default: 10)\n"
        "  --out     FILE output CSV path       (default: rng_data.csv)\n"
        "  --help         show this message\n",
        prog);
}

static Args parseArgs(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            printUsage(argv[0]);
            exit(0);
        } else if (strcmp(argv[i], "--samples") == 0 && i + 1 < argc) {
            args.numSamples = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--pixels") == 0 && i + 1 < argc) {
            args.numPixels = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--bounces") == 0 && i + 1 < argc) {
            args.numBounces = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--dims") == 0 && i + 1 < argc) {
            args.numDims = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--out") == 0 && i + 1 < argc) {
            args.outFile = argv[++i];
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            exit(1);
        }
    }
    return args;
}

// ============================================================================
// CSV output
// ============================================================================

static void writeHeader(FILE* f) {
    fprintf(f, "pixel,iter,bounce,dim,lcg,halton_cp,halton_owen\n");
}

static void writeRow(FILE* f,
                     int pixel, int iter, int bounce, int dim,
                     float lcgVal, float cpVal, float owenVal)
{
    fprintf(f, "%d,%d,%d,%d,%.9g,%.9g,%.9g\n",
            pixel, iter, bounce, dim, lcgVal, cpVal, owenVal);
}

// ============================================================================
// Main: generate all sequences
// ============================================================================

int main(int argc, char** argv) {
    Args args = parseArgs(argc, argv);

    long long total = (long long)args.numPixels * args.numSamples
                    * args.numBounces * args.numDims;
    fprintf(stdout,
        "RNG Compare: %d pixels × %d iters × %d bounces × %d dims = %lld rows\n"
        "  LCG | Halton-CP (legacy/buggy) | Halton-Owen (new)\n"
        "  Output: %s\n",
        args.numPixels, args.numSamples, args.numBounces, args.numDims,
        total, args.outFile);

    FILE* f = fopen(args.outFile, "w");
    if (!f) {
        ensureDir("profiler_output");
        ensureDir("profiler_output/rng_test");
        f = fopen(args.outFile, "w");
    }
    if (!f) {
        fprintf(stderr, "Error: cannot open %s for writing\n", args.outFile);
        return 1;
    }

    writeHeader(f);

    int totalCombos = args.numPixels * args.numSamples * args.numBounces;
    int progressInterval = std::max(1, totalCombos / 20);

    int combo = 0;
    for (int pixel = 0; pixel < args.numPixels; ++pixel) {
        for (int iter = 0; iter < args.numSamples; ++iter) {
            for (int bounce = 0; bounce < args.numBounces; ++bounce) {
                int bounceIndex = bounce * MAX_DRAWS_PER_BOUNCE;

                // --- LCG state ---
                RngState rngLcg = makeRngState(iter, pixel, bounceIndex, RngMode::LCG);

                // --- Owen Halton state (current implementation via rng.h) ---
                RngState rngOwen = makeRngState(iter, pixel, bounceIndex, RngMode::HALTON);

                // --- Legacy CP halton index (old buggy formula) ---
                unsigned int cpIdx = haltonCP_index(iter, pixel, bounceIndex);

                for (int dim = 0; dim < args.numDims; ++dim) {
                    float lcgVal  = rngLcg.next(dim);
                    float owenVal = rngOwen.next(dim);
                    float cpVal   = haltonCP_next(cpIdx,
                                                  (unsigned int)pixel,
                                                  (unsigned int)bounceIndex,
                                                  dim);

                    writeRow(f, pixel, iter, bounce, dim, lcgVal, cpVal, owenVal);
                }

                ++combo;
                if (combo % progressInterval == 0) {
                    fprintf(stdout, "\r  Progress: %3d%% (%d/%d)",
                            (combo * 100) / totalCombos, combo, totalCombos);
                    fflush(stdout);
                }
            }
        }
    }

    fprintf(stdout, "\r  Progress: 100%% (%d/%d)\n", totalCombos, totalCombos);
    fclose(f);
    fprintf(stdout, "Done. Generated %s\n", args.outFile);
    return 0;
}
