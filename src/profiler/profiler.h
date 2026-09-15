#pragma once

#include <chrono>
#include <cstddef>
#include <limits>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "profiler/profiler_counters.h"
#include "sceneStructs.h"

inline constexpr int kProfilerSchemaVersion = 2;
inline constexpr std::size_t kInvalidProfilerToken =
    std::numeric_limits<std::size_t>::max();

enum class ProfilerMode : int {
    Throughput = 0,
    Detail = 1
};

inline const char* toString(ProfilerMode mode)
{
    return mode == ProfilerMode::Detail ? "detail" : "throughput";
}

enum class ProfilerScope : int {
    Setup = 0,
    Frame,
    Bounce
};

enum class ProfilerTimerDomain : int {
    CPU = 0,
    GPU
};

// Python discovers operations from timing.csv; plot scripts never carry a
// second hard-coded list that can drift after a renderer refactor.
enum class ProfilerOp : int {
    FrameGpu = 0,
    PathtraceInit,
    AllocateCoreBuffers,
    BuildSceneBvh,
    BuildLightSampling,
    UploadSceneData,
    UploadTextures,
    AllocatePipelineBuffers,
    GenerateCameraRays,
    ComputeIntersections,
    SortByMaterial,
    ShadeMaterial,
    GatherTerminatedPaths,
    CompactPaths,
    FinalGather,
    BloomWeights,
    BloomThreshold,
    BloomBlurHorizontal,
    BloomBlurVertical,
    PrepareDisplay,
    Tonemap,
    ChromaticAberration,
    Vignette,
    PostProcessCopy,
    SendImageToPbo,
    COUNT
};

inline constexpr int kProfilerOpCount = static_cast<int>(ProfilerOp::COUNT);

const char* profilerOpName(ProfilerOp op);
const char* profilerScopeName(ProfilerScope scope);
const char* profilerTimerDomainName(ProfilerTimerDomain domain);

struct ProfilerRuntimeConfig
{
    CompactMethod compactMethod = CompactMethod::SharedMem;
    bool sortByMaterial = false;
    RngMode rngMode = RngMode::LCG;
    bool directLighting = true;

    bool bloomEnabled = false;
    float bloomThreshold = 1.0f;
    float bloomIntensity = 0.5f;
    int bloomRadius = 10;
    float bloomSigma = 5.0f;

    bool chromaticAberrationEnabled = false;
    float chromaticAberrationIntensity = 0.003f;
    bool vignetteEnabled = false;
    float vignetteIntensity = 0.5f;
    float vignetteExponent = 2.0f;
};

struct ProfilerConfig
{
    bool enabled = false;
    int warmupIters = 3;
    ProfilerMode mode = ProfilerMode::Detail;
    bool collectCounters = false;
    std::string outputDir = "profiler_output";
    std::string runTag;

    std::string sceneName = "unknown";
    std::string sceneFile;
    int width = 0;
    int height = 0;
    int iterations = 0;
    int traceDepth = 0;
    int rrMinBounces = 0;
    int numObjects = 0;
    int numMeshes = 0;
    int numMaterials = 0;
    int numTriangles = 0;
    int numTextures = 0;
    std::size_t texturePixels = 0;

    ProfilerRuntimeConfig runtime;
};

struct ProfilerMemoryStats
{
    std::size_t pathBuffersBytes = 0;
    std::size_t sceneBuffersBytes = 0;
    std::size_t bvhBytes = 0;
    std::size_t lightSamplingBytes = 0;
    std::size_t textureBytes = 0;
    std::size_t postProcessBytes = 0;
    std::size_t compactionWorkspaceBytes = 0;
    std::size_t materialSortWorkspaceBytes = 0;
    std::size_t counterBytes = 0;
    std::size_t displayInteropBytes = 0;
};

struct TimingRecord
{
    int epoch = 0;
    int iteration = 0;
    ProfilerScope scope = ProfilerScope::Frame;
    int bounce = -1;
    ProfilerOp op = ProfilerOp::FrameGpu;
    ProfilerTimerDomain timerDomain = ProfilerTimerDomain::GPU;
    float timeMs = 0.0f;
    int workItems = 0;
};

struct BounceCounterRecord
{
    int epoch = 0;
    int iteration = 0;
    int bounce = 0;
    int processedPaths = 0;
    DeviceBounceCounters counters;
};

struct MaterialHitRecord
{
    int epoch = 0;
    int iteration = 0;
    int bounce = 0;
    int materialId = 0;
    unsigned int hitCount = 0;
};

struct FrameTimeRecord
{
    int epoch = 0;
    int iteration = 0;
    float endToEndMs = 0.0f;
    float gpuPipelineMs = 0.0f;
};

struct EpochRecord
{
    int epoch = 0;
    ProfilerRuntimeConfig runtime;
};

struct GuiDataContainer
{
    int TracedDepth = 0;
    float perKernelMs[kProfilerOpCount] = {};
    int perKernelCalls[kProfilerOpCount] = {};
    int lastBounceCount = 0;
};

class Profiler
{
public:
    Profiler();
    ~Profiler();

    Profiler(const Profiler&) = delete;
    Profiler& operator=(const Profiler&) = delete;

    void init(const ProfilerConfig& cfg);
    void shutdown();
    void resetForNewAccumulation(const ProfilerRuntimeConfig& runtime);

    void beginIteration(int iteration);
    void endIteration();
    void beginFrame();
    void endFrame();

    std::size_t gpuStart(
        ProfilerOp op,
        ProfilerScope scope,
        int bounce = -1,
        int workItems = 0);
    void gpuStop(std::size_t token);

    std::size_t cpuStart(
        ProfilerOp op,
        ProfilerScope scope,
        int bounce = -1,
        int workItems = 0);
    void cpuStop(std::size_t token);

    void recordBounceCounters(
        int bounce,
        int processedPaths,
        const DeviceBounceCounters& counters);
    void recordMaterialHits(
        int bounce,
        const std::vector<unsigned int>& hitCounts);
    void recordMemoryStats(const ProfilerMemoryStats& stats) { m_memoryStats = stats; }

    void updateGuiData();

    const ProfilerConfig& config() const { return m_cfg; }
    bool enabled() const { return m_cfg.enabled; }
    bool detailed() const { return m_cfg.enabled && m_cfg.mode == ProfilerMode::Detail; }
    bool collectCounters() const { return m_cfg.enabled && m_cfg.collectCounters; }
    const std::string& outputDirectory() const { return m_experimentDir; }
    GuiDataContainer& guiData() { return m_guiData; }
    const GuiDataContainer& guiData() const { return m_guiData; }

private:
    struct GpuRangeState
    {
        cudaEvent_t start = nullptr;
        cudaEvent_t stop = nullptr;
        TimingRecord record;
        bool stopped = false;
    };

    struct CpuRangeState
    {
        std::chrono::steady_clock::time_point start;
        TimingRecord record;
        bool stopped = false;
    };

    ProfilerConfig m_cfg;
    std::string m_runId;
    std::string m_experimentDir;

    std::vector<GpuRangeState> m_gpuRanges;
    std::vector<std::size_t> m_gpuStopOrder;
    std::size_t m_gpuRangesUsed = 0;
    std::vector<CpuRangeState> m_cpuRanges;

    std::vector<TimingRecord> m_timingRecords;
    std::vector<BounceCounterRecord> m_bounceCounters;
    std::vector<MaterialHitRecord> m_materialHits;
    std::vector<FrameTimeRecord> m_frameTimes;
    std::vector<EpochRecord> m_epochs;
    ProfilerMemoryStats m_memoryStats;

    int m_currentEpoch = 0;
    int m_currentIteration = 0;
    float m_currentFrameGpuMs = 0.0f;
    std::chrono::steady_clock::time_point m_frameStartTime;
    bool m_frameTiming = false;
    bool m_shutdown = false;

    GuiDataContainer m_guiData;

    bool isWarmup(int iteration) const;
    void resolveGpuRanges();
    void destroyGpuEvents();
    void writeTimingCSV(const std::string& filepath) const;
    void writeBounceCountersCSV(const std::string& filepath) const;
    void writeMaterialHitsCSV(const std::string& filepath) const;
    void writeFrameTimesCSV(const std::string& filepath) const;
    void writeEpochsCSV(const std::string& filepath) const;
    void writeRunJson(const std::string& filepath) const;
};

Profiler& g_profiler();
