#include "profiler/profiler.h"

#include "utils/logger.h"

#include <algorithm>
#include <cctype>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>

#include <json.hpp>

namespace {

Profiler s_profiler;

std::string generateTimestampUtc()
{
    const std::time_t now = std::time(nullptr);
    std::tm value{};
#ifdef _WIN32
    gmtime_s(&value, &now);
#else
    gmtime_r(&now, &value);
#endif
    char buffer[32];
    std::strftime(buffer, sizeof(buffer), "%Y%m%d_%H%M%SZ", &value);
    return buffer;
}

std::string sanitizeComponent(std::string value)
{
    for (char& c : value)
    {
        const unsigned char uc = static_cast<unsigned char>(c);
        if (!std::isalnum(uc) && c != '-' && c != '_') c = '_';
    }
    return value.empty() ? "run" : value;
}

std::filesystem::path uniqueExperimentPath(
    const std::filesystem::path& root,
    const std::string& baseName)
{
    std::filesystem::path candidate = root / baseName;
    int suffix = 1;
    while (std::filesystem::exists(candidate))
    {
        candidate = root / (baseName + "_" + std::to_string(suffix++));
    }
    return candidate;
}

nlohmann::json runtimeJson(const ProfilerRuntimeConfig& runtime)
{
    return {
        {"compact_method", static_cast<int>(runtime.compactMethod)},
        {"compact_method_name", toString(runtime.compactMethod)},
        {"sort_by_material", runtime.sortByMaterial},
        {"rng_mode", static_cast<int>(runtime.rngMode)},
        {"rng_mode_name", toString(runtime.rngMode)},
        {"direct_lighting", runtime.directLighting},
        {"bloom", {
            {"enabled", runtime.bloomEnabled},
            {"threshold", runtime.bloomThreshold},
            {"intensity", runtime.bloomIntensity},
            {"radius", runtime.bloomRadius},
            {"sigma", runtime.bloomSigma}
        }},
        {"chromatic_aberration", {
            {"enabled", runtime.chromaticAberrationEnabled},
            {"intensity", runtime.chromaticAberrationIntensity}
        }},
        {"vignette", {
            {"enabled", runtime.vignetteEnabled},
            {"intensity", runtime.vignetteIntensity},
            {"exponent", runtime.vignetteExponent}
        }}
    };
}

} // namespace

const char* profilerOpName(ProfilerOp op)
{
    switch (op)
    {
        case ProfilerOp::FrameGpu: return "FrameGpu";
        case ProfilerOp::PathtraceInit: return "PathtraceInit";
        case ProfilerOp::AllocateCoreBuffers: return "AllocateCoreBuffers";
        case ProfilerOp::BuildSceneBvh: return "BuildSceneBvh";
        case ProfilerOp::BuildLightSampling: return "BuildLightSampling";
        case ProfilerOp::UploadSceneData: return "UploadSceneData";
        case ProfilerOp::UploadTextures: return "UploadTextures";
        case ProfilerOp::AllocatePipelineBuffers: return "AllocatePipelineBuffers";
        case ProfilerOp::GenerateCameraRays: return "GenerateCameraRays";
        case ProfilerOp::ComputeIntersections: return "ComputeIntersections";
        case ProfilerOp::SortByMaterial: return "SortByMaterial";
        case ProfilerOp::ShadeMaterial: return "ShadeMaterial";
        case ProfilerOp::GatherTerminatedPaths: return "GatherTerminatedPaths";
        case ProfilerOp::CompactPaths: return "CompactPaths";
        case ProfilerOp::FinalGather: return "FinalGather";
        case ProfilerOp::BloomWeights: return "BloomWeights";
        case ProfilerOp::BloomThreshold: return "BloomThreshold";
        case ProfilerOp::BloomBlurHorizontal: return "BloomBlurHorizontal";
        case ProfilerOp::BloomBlurVertical: return "BloomBlurVertical";
        case ProfilerOp::PrepareDisplay: return "PrepareDisplay";
        case ProfilerOp::Tonemap: return "Tonemap";
        case ProfilerOp::ChromaticAberration: return "ChromaticAberration";
        case ProfilerOp::Vignette: return "Vignette";
        case ProfilerOp::PostProcessCopy: return "PostProcessCopy";
        case ProfilerOp::SendImageToPbo: return "SendImageToPbo";
        default: return "Unknown";
    }
}

const char* profilerScopeName(ProfilerScope scope)
{
    switch (scope)
    {
        case ProfilerScope::Setup: return "setup";
        case ProfilerScope::Frame: return "frame";
        case ProfilerScope::Bounce: return "bounce";
        default: return "unknown";
    }
}

const char* profilerTimerDomainName(ProfilerTimerDomain domain)
{
    return domain == ProfilerTimerDomain::GPU ? "gpu" : "cpu";
}

Profiler& g_profiler()
{
    return s_profiler;
}

Profiler::Profiler() = default;

Profiler::~Profiler()
{
    destroyGpuEvents();
}

void Profiler::init(const ProfilerConfig& cfg)
{
    m_cfg = cfg;
    m_shutdown = false;
    if (!m_cfg.enabled) return;

    m_cfg.warmupIters = std::max(0, m_cfg.warmupIters);
    const std::string timestamp = generateTimestampUtc();
    const std::string tag = m_cfg.runTag.empty()
        ? std::string()
        : "_" + sanitizeComponent(m_cfg.runTag);
    m_runId = sanitizeComponent(m_cfg.sceneName) + "_" + timestamp + tag;

    const std::filesystem::path root = m_cfg.outputDir.empty()
        ? std::filesystem::path("profiler_output")
        : std::filesystem::path(m_cfg.outputDir);
    std::error_code error;
    std::filesystem::create_directories(root, error);
    const std::filesystem::path experiment = uniqueExperimentPath(root, m_runId);
    std::filesystem::create_directories(experiment, error);
    if (error)
    {
        Log::error("Profiler", "Cannot create output directory %s: %s",
                   experiment.string().c_str(), error.message().c_str());
        m_cfg.enabled = false;
        return;
    }
    m_experimentDir = experiment.string();
    m_runId = experiment.filename().string();

    m_timingRecords.clear();
    m_timingRecords.reserve(16384);
    m_bounceCounters.clear();
    m_bounceCounters.reserve(2048);
    m_materialHits.clear();
    m_materialHits.reserve(4096);
    m_cpuRanges.clear();
    m_cpuRanges.reserve(4096);
    m_frameTimes.clear();
    m_frameTimes.reserve(512);
    m_epochs.clear();
    m_epochs.push_back({0, m_cfg.runtime});
    m_currentEpoch = 0;
    m_memoryStats = {};
    m_guiData = {};

    Log::info("Profiler", "Enabled (%s%s). Output directory: %s",
              toString(m_cfg.mode),
              m_cfg.collectCounters ? ", counters" : "",
              m_experimentDir.c_str());
    // Machine-readable marker consumed by benchmark_runner.py.
    Log::raw("PROFILER_OUTPUT_DIR=%s\n", m_experimentDir.c_str());
}

void Profiler::shutdown()
{
    if (!m_cfg.enabled || m_shutdown) return;
    m_shutdown = true;

    resolveGpuRanges();
    const std::filesystem::path root(m_experimentDir);
    writeRunJson((root / "run.json").string());
    writeEpochsCSV((root / "epochs.csv").string());
    if (!m_timingRecords.empty())
        writeTimingCSV((root / "timing.csv").string());
    if (!m_frameTimes.empty())
        writeFrameTimesCSV((root / "frame_times.csv").string());
    if (!m_bounceCounters.empty())
        writeBounceCountersCSV((root / "bounce_counters.csv").string());
    if (!m_materialHits.empty())
        writeMaterialHitsCSV((root / "material_hits.csv").string());

    Log::info("Profiler", "Wrote schema-v%d run with %zu timing records",
              kProfilerSchemaVersion, m_timingRecords.size());
    destroyGpuEvents();
}

void Profiler::resetForNewAccumulation(const ProfilerRuntimeConfig& runtime)
{
    if (!m_cfg.enabled) return;
    ++m_currentEpoch;
    m_currentIteration = 0;
    m_epochs.push_back({m_currentEpoch, runtime});
    m_guiData = {};
}

void Profiler::beginIteration(int iteration)
{
    if (!m_cfg.enabled) return;
    m_currentIteration = iteration;
    m_currentFrameGpuMs = 0.0f;
    m_gpuStopOrder.clear();
    m_gpuRangesUsed = 0;
    m_cpuRanges.clear();
}

void Profiler::endIteration()
{
    // GPU ranges are resolved by endFrame(), after the application's required
    // CUDA completion point. This avoids adding a synchronize per operation.
}

void Profiler::beginFrame()
{
    if (!m_cfg.enabled) return;
    m_frameStartTime = std::chrono::steady_clock::now();
    m_frameTiming = true;
}

void Profiler::endFrame()
{
    if (!m_cfg.enabled || !m_frameTiming) return;
    const auto end = std::chrono::steady_clock::now();
    resolveGpuRanges();
    const double milliseconds =
        std::chrono::duration<double, std::milli>(end - m_frameStartTime).count();
    m_frameTimes.push_back({
        m_currentEpoch,
        m_currentIteration,
        static_cast<float>(milliseconds),
        m_currentFrameGpuMs
    });
    m_frameTiming = false;
    updateGuiData();
}

std::size_t Profiler::gpuStart(
    ProfilerOp op,
    ProfilerScope scope,
    int bounce,
    int workItems)
{
    if (!m_cfg.enabled) return kInvalidProfilerToken;
    if (m_cfg.mode == ProfilerMode::Throughput && op != ProfilerOp::FrameGpu)
        return kInvalidProfilerToken;

    const std::size_t token = m_gpuRangesUsed++;
    if (token >= m_gpuRanges.size())
    {
        GpuRangeState state;
        cudaEventCreate(&state.start);
        cudaEventCreate(&state.stop);
        m_gpuRanges.push_back(state);
    }

    GpuRangeState& state = m_gpuRanges[token];
    state.record = {
        m_currentEpoch, m_currentIteration, scope, bounce, op,
        ProfilerTimerDomain::GPU, 0.0f, workItems
    };
    state.stopped = false;
    cudaEventRecord(state.start, 0);
    return token;
}

void Profiler::gpuStop(std::size_t token)
{
    if (!m_cfg.enabled || token == kInvalidProfilerToken ||
        token >= m_gpuRangesUsed) return;
    GpuRangeState& state = m_gpuRanges[token];
    cudaEventRecord(state.stop, 0);
    state.stopped = true;
    m_gpuStopOrder.push_back(token);
}

std::size_t Profiler::cpuStart(
    ProfilerOp op,
    ProfilerScope scope,
    int bounce,
    int workItems)
{
    if (!detailed()) return kInvalidProfilerToken;
    CpuRangeState state;
    state.start = std::chrono::steady_clock::now();
    state.record = {
        m_currentEpoch, m_currentIteration, scope, bounce, op,
        ProfilerTimerDomain::CPU, 0.0f, workItems
    };
    m_cpuRanges.push_back(state);
    return m_cpuRanges.size() - 1;
}

void Profiler::cpuStop(std::size_t token)
{
    if (!m_cfg.enabled || token == kInvalidProfilerToken ||
        token >= m_cpuRanges.size()) return;
    CpuRangeState& state = m_cpuRanges[token];
    if (state.stopped) return;
    const auto end = std::chrono::steady_clock::now();
    state.record.timeMs = static_cast<float>(
        std::chrono::duration<double, std::milli>(end - state.start).count());
    state.stopped = true;
    m_timingRecords.push_back(state.record);
}

void Profiler::recordBounceCounters(
    int bounce,
    int processedPaths,
    const DeviceBounceCounters& counters)
{
    if (!collectCounters()) return;
    m_bounceCounters.push_back({
        m_currentEpoch, m_currentIteration, bounce, processedPaths, counters
    });
}

void Profiler::recordMaterialHits(
    int bounce,
    const std::vector<unsigned int>& hitCounts)
{
    if (!collectCounters()) return;
    for (std::size_t materialId = 0; materialId < hitCounts.size(); ++materialId)
    {
        m_materialHits.push_back({
            m_currentEpoch, m_currentIteration, bounce,
            static_cast<int>(materialId), hitCounts[materialId]
        });
    }
}

bool Profiler::isWarmup(int iteration) const
{
    return iteration > 0 && iteration <= m_cfg.warmupIters;
}

void Profiler::resolveGpuRanges()
{
    if (!m_cfg.enabled || m_gpuStopOrder.empty()) return;

    // endFrame() is called after the renderer's required cudaDeviceSynchronize.
    // Keep this fallback for abnormal/manual call paths without introducing a
    // synchronize at every phase boundary.
    const cudaEvent_t finalEvent = m_gpuRanges[m_gpuStopOrder.back()].stop;
    if (cudaEventQuery(finalEvent) == cudaErrorNotReady)
        cudaEventSynchronize(finalEvent);

    for (std::size_t i = 0; i < m_gpuRangesUsed; ++i)
    {
        GpuRangeState& state = m_gpuRanges[i];
        if (!state.stopped) continue;
        cudaEventElapsedTime(&state.record.timeMs, state.start, state.stop);
        if (state.record.op == ProfilerOp::FrameGpu)
            m_currentFrameGpuMs = state.record.timeMs;
        m_timingRecords.push_back(state.record);
        state.stopped = false;
    }
    m_gpuStopOrder.clear();
    m_gpuRangesUsed = 0;
}

void Profiler::destroyGpuEvents()
{
    for (GpuRangeState& state : m_gpuRanges)
    {
        if (state.start) cudaEventDestroy(state.start);
        if (state.stop) cudaEventDestroy(state.stop);
        state.start = nullptr;
        state.stop = nullptr;
    }
    m_gpuRanges.clear();
    m_gpuStopOrder.clear();
    m_gpuRangesUsed = 0;
}

void Profiler::updateGuiData()
{
    if (!m_cfg.enabled) return;
    for (int i = 0; i < kProfilerOpCount; ++i)
    {
        m_guiData.perKernelMs[i] = 0.0f;
        m_guiData.perKernelCalls[i] = 0;
    }

    int maxBounce = -1;
    // Records for one iteration are appended contiguously. Walking backward
    // avoids an O(total-history) GUI scan every frame during long benchmarks.
    for (auto it = m_timingRecords.rbegin(); it != m_timingRecords.rend(); ++it)
    {
        const TimingRecord& record = *it;
        if (record.epoch != m_currentEpoch ||
            record.iteration != m_currentIteration) break;
        const int index = static_cast<int>(record.op);
        if (index >= 0 && index < kProfilerOpCount)
        {
            m_guiData.perKernelMs[index] += record.timeMs;
            ++m_guiData.perKernelCalls[index];
        }
        if (record.scope == ProfilerScope::Bounce)
            maxBounce = std::max(maxBounce, record.bounce);
    }
    m_guiData.lastBounceCount = maxBounce + 1;
}

void Profiler::writeTimingCSV(const std::string& filepath) const
{
    std::ofstream file(filepath);
    if (!file) return;
    file << "schema_version,run_id,epoch,iteration,is_warmup,scope,bounce_depth,"
            "operation,timer_domain,time_ms,work_items\n";
    file << std::setprecision(9);
    for (const TimingRecord& record : m_timingRecords)
    {
        file << kProfilerSchemaVersion << ',' << m_runId << ','
             << record.epoch << ',' << record.iteration << ','
             << (isWarmup(record.iteration) ? 1 : 0) << ','
             << profilerScopeName(record.scope) << ',';
        if (record.bounce >= 0) file << record.bounce;
        file << ',' << profilerOpName(record.op) << ','
             << profilerTimerDomainName(record.timerDomain) << ','
             << record.timeMs << ',' << record.workItems << '\n';
    }
}

void Profiler::writeBounceCountersCSV(const std::string& filepath) const
{
    std::ofstream file(filepath);
    if (!file) return;
    file << "schema_version,run_id,epoch,iteration,is_warmup,bounce_depth,"
            "processed_paths,active_after_bounce,already_terminated,surface_hits,"
            "misses,emissive_terminations,invalid_surface_terminations,"
            "russian_roulette_terminations,max_depth_terminations,debug_terminations,"
            "closest_bvh_node_tests,closest_bvh_triangle_tests,"
            "shadow_bvh_node_tests,shadow_bvh_triangle_tests,light_selections,"
            "valid_light_samples,shadow_rays,visible_light_samples,"
            "occluded_light_samples\n";
    for (const BounceCounterRecord& record : m_bounceCounters)
    {
        const DeviceBounceCounters& c = record.counters;
        file << kProfilerSchemaVersion << ',' << m_runId << ','
             << record.epoch << ',' << record.iteration << ','
             << (isWarmup(record.iteration) ? 1 : 0) << ','
             << record.bounce << ',' << record.processedPaths << ','
             << c.survivors << ',' << c.alreadyTerminated << ','
             << c.surfaceHits << ',' << c.misses << ','
             << c.emissiveTerminations << ',' << c.invalidSurfaceTerminations << ','
             << c.russianRouletteTerminations << ','
             << c.maxDepthTerminations << ',' << c.debugTerminations << ','
             << c.closestBvhNodeTests << ',' << c.closestBvhTriangleTests << ','
             << c.shadowBvhNodeTests << ',' << c.shadowBvhTriangleTests << ','
             << c.lightSelections << ',' << c.validLightSamples << ','
             << c.shadowRays << ',' << c.visibleLightSamples << ','
             << c.occludedLightSamples << '\n';
    }
}

void Profiler::writeFrameTimesCSV(const std::string& filepath) const
{
    std::ofstream file(filepath);
    if (!file) return;
    file << "schema_version,run_id,epoch,iteration,is_warmup,end_to_end_ms,gpu_pipeline_ms\n";
    file << std::setprecision(9);
    for (const FrameTimeRecord& record : m_frameTimes)
    {
        file << kProfilerSchemaVersion << ',' << m_runId << ','
             << record.epoch << ',' << record.iteration << ','
             << (isWarmup(record.iteration) ? 1 : 0) << ','
             << record.endToEndMs << ',' << record.gpuPipelineMs << '\n';
    }
}

void Profiler::writeMaterialHitsCSV(const std::string& filepath) const
{
    std::ofstream file(filepath);
    if (!file) return;
    file << "schema_version,run_id,epoch,iteration,is_warmup,bounce_depth,"
            "material_id,hit_count\n";
    for (const MaterialHitRecord& record : m_materialHits)
    {
        file << kProfilerSchemaVersion << ',' << m_runId << ','
             << record.epoch << ',' << record.iteration << ','
             << (isWarmup(record.iteration) ? 1 : 0) << ','
             << record.bounce << ',' << record.materialId << ','
             << record.hitCount << '\n';
    }
}

void Profiler::writeEpochsCSV(const std::string& filepath) const
{
    std::ofstream file(filepath);
    if (!file) return;
    file << "schema_version,run_id,epoch,compact_method,compact_method_name,"
            "sort_by_material,rng_mode,rng_mode_name,direct_lighting,bloom_enabled,"
            "bloom_threshold,bloom_intensity,bloom_radius,bloom_sigma,"
            "chromatic_aberration_enabled,chromatic_aberration_intensity,"
            "vignette_enabled,vignette_intensity,vignette_exponent\n";
    for (const EpochRecord& record : m_epochs)
    {
        const ProfilerRuntimeConfig& r = record.runtime;
        file << kProfilerSchemaVersion << ',' << m_runId << ',' << record.epoch << ','
             << static_cast<int>(r.compactMethod) << ',' << toString(r.compactMethod) << ','
             << (r.sortByMaterial ? 1 : 0) << ','
             << static_cast<int>(r.rngMode) << ',' << toString(r.rngMode) << ','
             << (r.directLighting ? 1 : 0) << ','
             << (r.bloomEnabled ? 1 : 0) << ',' << r.bloomThreshold << ','
             << r.bloomIntensity << ',' << r.bloomRadius << ',' << r.bloomSigma << ','
             << (r.chromaticAberrationEnabled ? 1 : 0) << ','
             << r.chromaticAberrationIntensity << ','
             << (r.vignetteEnabled ? 1 : 0) << ','
             << r.vignetteIntensity << ',' << r.vignetteExponent << '\n';
    }
}

void Profiler::writeRunJson(const std::string& filepath) const
{
    nlohmann::json root;
    root["schema_version"] = kProfilerSchemaVersion;
    root["run_id"] = m_runId;
    root["profiler"] = {
        {"mode", toString(m_cfg.mode)},
        {"warmup_iterations", m_cfg.warmupIters},
        {"collect_counters", m_cfg.collectCounters},
        {"counter_overhead_in_frame_times", m_cfg.collectCounters},
        {"timing_policy", "CUDA events resolved after the frame completion point"}
    };
    root["scene"] = {
        {"name", m_cfg.sceneName},
        {"file", m_cfg.sceneFile},
        {"width", m_cfg.width},
        {"height", m_cfg.height},
        {"iterations", m_cfg.iterations},
        {"trace_depth", m_cfg.traceDepth},
        {"rr_min_bounces", m_cfg.rrMinBounces},
        {"objects", m_cfg.numObjects},
        {"meshes", m_cfg.numMeshes},
        {"materials", m_cfg.numMaterials},
        {"triangles", m_cfg.numTriangles},
        {"textures", m_cfg.numTextures},
        {"texture_pixels", m_cfg.texturePixels}
    };
    root["initial_runtime"] = runtimeJson(m_cfg.runtime);
    const std::size_t knownMemoryTotal =
        m_memoryStats.pathBuffersBytes + m_memoryStats.sceneBuffersBytes +
        m_memoryStats.bvhBytes + m_memoryStats.lightSamplingBytes +
        m_memoryStats.textureBytes + m_memoryStats.postProcessBytes +
        m_memoryStats.compactionWorkspaceBytes +
        m_memoryStats.materialSortWorkspaceBytes + m_memoryStats.counterBytes +
        m_memoryStats.displayInteropBytes;
    root["estimated_device_memory_bytes"] = {
        {"path_buffers", m_memoryStats.pathBuffersBytes},
        {"scene_buffers", m_memoryStats.sceneBuffersBytes},
        {"bvh", m_memoryStats.bvhBytes},
        {"light_sampling", m_memoryStats.lightSamplingBytes},
        {"textures", m_memoryStats.textureBytes},
        {"post_process", m_memoryStats.postProcessBytes},
        {"compaction_workspace", m_memoryStats.compactionWorkspaceBytes},
        {"material_sort_workspace", m_memoryStats.materialSortWorkspaceBytes},
        {"counter_only", m_memoryStats.counterBytes},
        {"display_interop_nominal", m_memoryStats.displayInteropBytes},
        {"known_total", knownMemoryTotal},
        {"note", "Known renderer allocations; display interop is nominal RGBA8; CUDA/Thrust/OpenGL driver internals are excluded"}
    };

    int device = 0;
    cudaDeviceProp properties{};
    if (cudaGetDevice(&device) == cudaSuccess &&
        cudaGetDeviceProperties(&properties, device) == cudaSuccess)
    {
        int driverVersion = 0;
        int runtimeVersion = 0;
        cudaDriverGetVersion(&driverVersion);
        cudaRuntimeGetVersion(&runtimeVersion);
        root["device"] = {
            {"id", device},
            {"name", properties.name},
            {"compute_capability", std::to_string(properties.major) + "." +
                                   std::to_string(properties.minor)},
            {"total_global_memory_bytes", properties.totalGlobalMem},
            {"multiprocessors", properties.multiProcessorCount},
            {"driver_version", driverVersion},
            {"runtime_version", runtimeVersion}
        };
    }

    std::ofstream file(filepath);
    if (file) file << std::setw(2) << root << '\n';
}
