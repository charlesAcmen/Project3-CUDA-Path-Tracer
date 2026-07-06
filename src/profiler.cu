#include "profiler.h"

#include <cstdio>
#include <ctime>
#include <cmath>
#include <fstream>

#ifdef _WIN32
#include <direct.h>   // _mkdir
#else
#include <sys/stat.h> // mkdir
#include <sys/types.h>
#endif

// ---------------------------------------------------------------------------
// Operation name lookup (for CSV headers)
// ---------------------------------------------------------------------------
const char* profilerOpName(ProfilerOp op)
{
    switch (op) {
        case ProfilerOp::ShadeMaterial:         return "ShadeMaterial";
        case ProfilerOp::GatherTerminatedPaths:  return "GatherTerminatedPaths";
        case ProfilerOp::SortByMaterial:         return "SortByMaterial";
        case ProfilerOp::CompactPaths:           return "CompactPaths";
        default:                                 return "Unknown";
    }
}

// ---------------------------------------------------------------------------
// Timestamp generation ("YYYYMMDD_HHMMSS")
// ---------------------------------------------------------------------------
static std::string generateTimestamp()
{
    time_t now = time(nullptr);
    struct tm tstruct;
#ifdef _WIN32
    localtime_s(&tstruct, &now);
#else
    localtime_r(&now, &tstruct);
#endif
    char buf[32];
    strftime(buf, sizeof(buf), "%Y%m%d_%H%M%S", &tstruct);
    return std::string(buf);
}

// ---------------------------------------------------------------------------
// Singleton
// ---------------------------------------------------------------------------
static Profiler s_profiler;

Profiler& g_profiler()
{
    return s_profiler;
}

// ---------------------------------------------------------------------------
// File-scope state shared by recordBounce / gpuStop / cpuStop
// ---------------------------------------------------------------------------
namespace {
    int  s_lastBounce     = -1;
    int  s_lastPathCount  = 0;
}

// ---------------------------------------------------------------------------
// Construction / Destruction
// ---------------------------------------------------------------------------
Profiler::Profiler() {}

Profiler::~Profiler()
{
    if (m_eventStart) cudaEventDestroy(m_eventStart);
    if (m_eventStop)  cudaEventDestroy(m_eventStop);
}

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------
void Profiler::init(const ProfilerConfig& cfg)
{
    m_cfg = cfg;
    if (!m_cfg.enabled) return;

    m_timestamp = generateTimestamp();

    cudaEventCreate(&m_eventStart);
    cudaEventCreate(&m_eventStop);

    m_timingRecords.clear();
    m_timingRecords.reserve(8192);
    m_pathCounts.clear();
    m_pathCounts.reserve(1024);

    printf("[Profiler] Enabled. Output: scripts/%s_%s_*.csv\n",
           m_cfg.sceneName.c_str(), m_timestamp.c_str());
}

void Profiler::shutdown()
{
    if (!m_cfg.enabled) return;

    std::string prefix = "scripts/" + m_cfg.sceneName + "_" + m_timestamp;

    if (!m_timingRecords.empty())
    {
        writeTimingCSV(prefix + "_timing.csv");
        writeSummaryCSV(prefix + "_summary.csv");
        printf("[Profiler] Wrote %zu timing records.\n", m_timingRecords.size());
    }

    if (!m_pathCounts.empty())
    {
        writePathSurvivalCSV(prefix + "_path_survival.csv");
        printf("[Profiler] Wrote %zu path survival records.\n", m_pathCounts.size());
    }

    m_timingRecords.clear();
    m_pathCounts.clear();
}

// ---------------------------------------------------------------------------
// Per-frame context
// ---------------------------------------------------------------------------
void Profiler::beginIteration(int iter)
{
    m_currentIteration = iter;
    s_lastBounce    = -1;
    s_lastPathCount = 0;
}

void Profiler::endIteration()
{
    // per-iteration flush not needed; everything stays in vectors
}

// ---------------------------------------------------------------------------
// GPU timing
// ---------------------------------------------------------------------------
void Profiler::gpuStart(ProfilerOp op)
{
    if (!m_cfg.enabled) return;
    (void)op;
    cudaEventRecord(m_eventStart, 0);
}

void Profiler::gpuStop(ProfilerOp op)
{
    if (!m_cfg.enabled) return;

    cudaEventRecord(m_eventStop, 0);
    cudaEventSynchronize(m_eventStop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, m_eventStart, m_eventStop);

    m_timingRecords.push_back({
        m_currentIteration,
        s_lastBounce,
        op,
        ms,
        s_lastPathCount
    });
}

// ---------------------------------------------------------------------------
// CPU timing
// ---------------------------------------------------------------------------
void Profiler::cpuStart(ProfilerOp op)
{
    if (!m_cfg.enabled) return;
    m_pendingCpuOp = op;
    m_cpuStartTime = std::chrono::high_resolution_clock::now();
    m_cpuTiming = true;
}

void Profiler::cpuStop(ProfilerOp op)
{
    if (!m_cfg.enabled || !m_cpuTiming) return;

    auto endTime = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(
        endTime - m_cpuStartTime).count();

    m_timingRecords.push_back({
        m_currentIteration,
        s_lastBounce,
        op,
        static_cast<float>(ms),
        s_lastPathCount
    });

    m_cpuTiming = false;
}

// ---------------------------------------------------------------------------
// Per-bounce data
// ---------------------------------------------------------------------------
void Profiler::recordBounce(int bounce, int num_paths)
{
    if (!m_cfg.enabled) return;

    s_lastBounce    = bounce;
    s_lastPathCount = num_paths;

    m_pathCounts.push_back({
        m_currentIteration,
        bounce,
        num_paths
    });
}

// ---------------------------------------------------------------------------
// CSV writers (internal)
// ---------------------------------------------------------------------------

static void ensureScriptsDir()
{
    // Create scripts/ in the current working directory if it doesn't exist.
#ifdef _WIN32
    _mkdir("scripts");
#else
    mkdir("scripts", 0755);
#endif
}

void Profiler::writeTimingCSV(const std::string& filepath)
{
    ensureScriptsDir();
    std::ofstream f(filepath);
    if (!f.is_open()) {
        printf("[Profiler] ERROR: cannot write %s\n", filepath.c_str());
        return;
    }

    f << "iteration,bounce_depth,operation,time_ms,num_active_paths,compact_method,sort_by_material\n";

    for (const auto& r : m_timingRecords)
    {
        f << r.iteration << ","
          << r.bounce << ","
          << profilerOpName(r.op) << ","
          << r.time_ms << ","
          << r.num_active_paths << ","
          << m_cfg.compactMethod << ","
          << (m_cfg.sortByMaterial ? 1 : 0) << "\n";
    }
}

void Profiler::writePathSurvivalCSV(const std::string& filepath)
{
    ensureScriptsDir();
    std::ofstream f(filepath);
    if (!f.is_open()) {
        printf("[Profiler] ERROR: cannot write %s\n", filepath.c_str());
        return;
    }

    f << "iteration,bounce_depth,num_active_paths,compact_method,sort_by_material\n";

    for (const auto& p : m_pathCounts)
    {
        f << p.iteration << ","
          << p.bounce << ","
          << p.num_paths << ","
          << m_cfg.compactMethod << ","
          << (m_cfg.sortByMaterial ? 1 : 0) << "\n";
    }
}

void Profiler::writeSummaryCSV(const std::string& filepath)
{
    ensureScriptsDir();
    if (m_timingRecords.empty()) return;

    // Compute per-operation statistics (excluding warmup iterations)
    struct OpStats {
        double sum   = 0.0;
        double sumSq = 0.0;
        double minVal = 1e30;
        double maxVal = 0.0;
        int    count = 0;
    };
    OpStats stats[static_cast<int>(ProfilerOp::COUNT)];

    for (const auto& r : m_timingRecords)
    {
        if (r.iteration < m_cfg.warmupIters) continue;
        int idx = static_cast<int>(r.op);
        if (idx < 0 || idx >= static_cast<int>(ProfilerOp::COUNT)) continue;

        OpStats& s = stats[idx];
        s.sum   += r.time_ms;
        s.sumSq += r.time_ms * r.time_ms;
        if (r.time_ms < s.minVal) s.minVal = r.time_ms;
        if (r.time_ms > s.maxVal) s.maxVal = r.time_ms;
        s.count++;
    }

    std::ofstream f(filepath);
    if (!f.is_open()) {
        printf("[Profiler] ERROR: cannot write %s\n", filepath.c_str());
        return;
    }

    f << "operation,mean_ms,std_ms,min_ms,max_ms,num_samples\n";

    for (int i = 0; i < static_cast<int>(ProfilerOp::COUNT); ++i)
    {
        const OpStats& s = stats[i];
        if (s.count == 0) continue;

        double mean = s.sum / s.count;
        double variance = (s.sumSq / s.count) - (mean * mean);
        if (variance < 0.0) variance = 0.0;
        double stddev = std::sqrt(variance);

        f << profilerOpName(static_cast<ProfilerOp>(i)) << ","
          << mean << ","
          << stddev << ","
          << s.minVal << ","
          << s.maxVal << ","
          << s.count << "\n";
    }
}
