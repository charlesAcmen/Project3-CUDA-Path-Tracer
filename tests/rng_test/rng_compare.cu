/**
 * @file rng_compare.cu
 * @brief Standalone RNG comparison: LCG vs Halton — sequence generator.
 *
 * Compiles with nvcc as a pure host program (no kernel launches).
 * Includes the real rng.h from the path tracer — tests the actual
 * implementation, not a re-implementation.
 *
 * Usage:
 *   rng_compare --samples 4096 --pixels 4 --bounces 2 --out data.csv
 *
 * Output (CSV):
 *   pixel,iter,bounce,dim,lcg,halton
 *   0,0,0,0,0.123456,0.500000
 *   ...
 *
 * Each row is one sample from both RNG modes for the same
 * (pixel, iter, bounce, dim) combination, so the analysis script
 * can compare them directly.
 */

// Define __host__ __device__ as empty macros for non-CUDA compilation
// (nvcc defines __CUDACC__ automatically — this path is taken on NVCC)
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

// Include the project's actual RNG header
// Compiled with -I../../src so this resolves to src/rng/rng.h
#include "rng/rng.h"

// ============================================================================
// CLI argument parsing (minimal — no external dependencies)
// ============================================================================

struct Args {
    int    numSamples = 4096;    // iterations per pixel
    int    numPixels  = 4;       // number of pixels to simulate
    int    numBounces = 2;       // bounce depths (0 = primary ray, 1 = first scatter)
    int    numDims    = 10;      // dimensions 0..9 (matches current HALTON_NUM_DIMS allocation)
    const char* outFile = "profiler_output/rng_test/rng_data.csv";
};

static void printUsage(const char* prog) {
    fprintf(stderr,
        "Usage: %s [options]\n"
        "  --samples N    iterations per pixel  (default: 4096)\n"
        "  --pixels  N    number of pixels      (default: 4)\n"
        "  --bounces N    bounce depths         (default: 2)\n"
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
    fprintf(f, "pixel,iter,bounce,dim,lcg,halton\n");
}

static void writeRow(FILE* f, int pixel, int iter, int bounce, int dim,
                     float lcgVal, float haltonVal)
{
    // Write with enough precision to distinguish float values
    fprintf(f, "%d,%d,%d,%d,%.9g,%.9g\n",
            pixel, iter, bounce, dim, lcgVal, haltonVal);
}

// ============================================================================
// Main: generate all sequences
// ============================================================================

int main(int argc, char** argv) {
    Args args = parseArgs(argc, argv);

    fprintf(stdout,
        "RNG Compare: %d pixels × %d iters × %d bounces × %d dims = %d samples\n"
        "  Output: %s\n",
        args.numPixels, args.numSamples, args.numBounces, args.numDims,
        args.numPixels * args.numSamples * args.numBounces * args.numDims,
        args.outFile);

    FILE* f = fopen(args.outFile, "w");
    if (!f) {
        // Try creating the output directory and retry
        ensureDir("profiler_output");
        ensureDir("profiler_output/rng_test");
        f = fopen(args.outFile, "w");
    }
    if (!f) {
        fprintf(stderr, "Error: cannot open %s for writing\n", args.outFile);
        return 1;
    }

    writeHeader(f);

    const int totalCombos = args.numPixels * args.numSamples * args.numBounces;
    int progressInterval = std::max(1, totalCombos / 20);  // ~5% granularity

    int combo = 0;
    for (int pixel = 0; pixel < args.numPixels; ++pixel) {
        for (int iter = 0; iter < args.numSamples; ++iter) {
            for (int bounce = 0; bounce < args.numBounces; ++bounce) {
                int bounceIndex = bounce * MAX_DRAWS_PER_BOUNCE;

                // Create both RNG states with identical seeds.
                // LCG: randomness from seeded engine.
                // Halton: deterministic low-discrepancy + CP rotation.
                RngState rngLcg    = makeRngState(iter, pixel, bounceIndex, RngMode::LCG);
                RngState rngHalton = makeRngState(iter, pixel, bounceIndex, RngMode::HALTON);

                for (int dim = 0; dim < args.numDims; ++dim) {
                    float lcgVal    = rngLcg.next(dim);
                    float haltonVal = rngHalton.next(dim);

                    writeRow(f, pixel, iter, bounce, dim, lcgVal, haltonVal);
                }

                ++combo;
                if (combo % progressInterval == 0) {
                    int pct = (combo * 100) / totalCombos;
                    fprintf(stdout, "\r  Progress: %3d%% (%d/%d)", pct, combo, totalCombos);
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
