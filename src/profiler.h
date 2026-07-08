#pragma once

#include <chrono>
#include <string>
#include <vector>
#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// Enumeration of every measurable operation.
// Starter-code kernels previously excluded are now measured when they are
// needed to quantify cross-cutting optimisations such as stream compaction.
// ---------------------------------------------------------------------------
enum class ProfilerOp : int {
    ShadeMaterial = 0,         // GPU kernel -- GPU timer (user-modified)
    GatherTerminatedPaths,     // GPU kernel -- GPU timer (user-written)
    SortByMaterial,            // async Thrust -- GPU timer (user-written)
                               //   Thrust transform/sequence/sort_by_key/gather are
                               //   all asynchronous; CPU timer would miss GPU work.
    CompactPaths,              // Thrust copy_if/scan -- CPU timer (user-written)
                               //   copy_if returns a host-visible iterator, forcing
                               //   an internal sync; CPU timer naturally captures it.
    ComputeIntersections,      // GPU kernel -- GPU timer (starter code)
    COUNT
};

const char* profilerOpName(ProfilerOp op);

// ---------------------------------------------------------------------------
// Configuration (populated from command-line flags)
// ---------------------------------------------------------------------------
struct ProfilerConfig {
    bool        enabled        = false;
    bool        verbose        = false;    // Control debug printf output
    int         warmupIters    = 3;
    std::string sceneName      = "unknown";
    
    // IMPORTANT: These fields are for CSV metadata tagging ONLY, not for controlling runtime behavior.
    // They are written to every CSV row so plotting scripts can generate correct labels.
    // 
    // Runtime behavior is controlled by g_compactMethod and g_sortByMaterial in pathtrace.cu.
    // 
    // Synchronization happens in main.cpp:
    //   1. profCfg is initialized by calling getCompactMethod() and getSortByMaterial()
    //   2. If command-line flags (--compact=N --sort=0/1) are provided, they override both
    //      profCfg fields AND pathtrace.cu runtime variables (via setters)
    // 
    // This ensures CSV metadata always matches actual runtime configuration, and you only
    // need to change defaults in ONE place (pathtrace.cu).
    int         compactMethod  = 1;       // 0=none, 1=global-mem scan, 2=Thrust, 3=shared-mem scan
    bool        sortByMaterial = true;    // placeholder, auto-synced from pathtrace.cu
};

// ---------------------------------------------------------------------------
// One timing measurement = one row in the per-iteration timing CSV
// ---------------------------------------------------------------------------
struct TimingRecord {
    int         iteration;
    int         bounce;
    ProfilerOp  op;
    float       time_ms;
    int         num_active_paths;
};

// ---------------------------------------------------------------------------
// Per-bounce path survival data point
// ---------------------------------------------------------------------------
struct PathCountRecord {
    int iteration;
    int bounce;
    int num_paths;
};

// ---------------------------------------------------------------------------
// Profiler singleton
// ---------------------------------------------------------------------------
class Profiler {
public:
    Profiler();
    ~Profiler();

    // Non-copyable, non-movable
    Profiler(const Profiler&)            = delete;
    Profiler& operator=(const Profiler&) = delete;

    // ---- Lifecycle ----
    void init(const ProfilerConfig& cfg);
    void shutdown();   // writes CSVs, frees cudaEvents

    // ---- Per-frame context ----
    void beginIteration(int iter);
    void endIteration();

    // ---- GPU timing (cudaEvent) ----
    void gpuStart(ProfilerOp op);
    void gpuStop(ProfilerOp op);

    // ---- CPU timing (std::chrono) ----
    void cpuStart(ProfilerOp op);
    void cpuStop(ProfilerOp op);

    // ---- Per-bounce data ----
    void recordBounce(int bounce, int num_paths);

    // ---- Accessors ----
    const ProfilerConfig& config() const { return m_cfg; }
    bool enabled() const { return m_cfg.enabled; }
    bool verbose() const { return m_cfg.verbose; }

    // ---- GUI data update ----
    void updateGuiData(class GuiDataContainer* guiData);

private:
    ProfilerConfig m_cfg;

    cudaEvent_t m_eventStart = nullptr;
    cudaEvent_t m_eventStop  = nullptr;

    // CPU timing state (non-nesting; one op at a time)
    std::chrono::high_resolution_clock::time_point m_cpuStartTime;
    ProfilerOp m_pendingCpuOp;
    bool       m_cpuTiming = false;

    // Accumulated records
    std::vector<TimingRecord>    m_timingRecords;
    std::vector<PathCountRecord> m_pathCounts;

    // Current iteration context
    int m_currentIteration = 0;

    // Timestamp for output filename deduplication
    std::string m_timestamp;

    // Internal helpers
    void writeTimingCSV(const std::string& filepath);
    void writePathSurvivalCSV(const std::string& filepath);
    void writeSummaryCSV(const std::string& filepath);
};

// Global accessor (defined in profiler.cu)
Profiler& g_profiler();
