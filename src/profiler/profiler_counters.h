#pragma once

// Optional benchmark-only counters written by shadeMaterial. The pointer
// passed to the kernel is null during normal rendering and throughput runs,
// so atomics, counter-buffer clears, and readbacks are skipped unless an
// explicit counter run is selected.
struct DeviceBounceCounters
{
    unsigned int alreadyTerminated = 0;
    unsigned int surfaceHits = 0;
    unsigned int misses = 0;
    unsigned int emissiveTerminations = 0;
    unsigned int invalidSurfaceTerminations = 0;
    unsigned int russianRouletteTerminations = 0;
    unsigned int maxDepthTerminations = 0;
    unsigned int debugTerminations = 0;
    unsigned int survivors = 0;

    // Counter-mode-only work diagnostics. Traversal functions accumulate
    // per-thread locals and issue one atomic add per completed query.
    unsigned long long closestBvhNodeTests = 0;
    unsigned long long closestBvhTriangleTests = 0;
    unsigned long long shadowBvhNodeTests = 0;
    unsigned long long shadowBvhTriangleTests = 0;
    unsigned int lightSelections = 0;
    unsigned int validLightSamples = 0;
    unsigned int shadowRays = 0;
    unsigned int visibleLightSamples = 0;
    unsigned int occludedLightSamples = 0;
};
