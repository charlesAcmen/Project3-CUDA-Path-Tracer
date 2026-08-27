#include "app/render_ui.h"

#include "app/camera_controller.h"
#include "config/config.h"
#include "pathtrace.h"
#include "profiler/profiler.h"

#include "ImGui/imgui_impl_glfw.h"
#include "ImGui/imgui_impl_opengl3.h"

#include <cfloat>
#include <cstdio>

void renderImGui(AppState& app)
{
    app.mouseOverImGuiWindow = app.io->WantCaptureMouse;

    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    ImGui::Begin("Path Tracer Analytics");

    ImGui::Text("Traced Depth %d", g_profiler().guiData().TracedDepth);
    ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);

    if (g_profiler().enabled()) {
        ImGui::Separator();
        ImGui::Text("Per-frame phase timing (sum across all calls):");
        const GuiDataContainer& profGui = g_profiler().guiData();
        const auto phaseTiming = [&](const char* name, int op) {
            const int calls = profGui.perKernelCalls[op];
            const float total = profGui.perKernelMs[op];
            const float perCall = (calls > 0) ? total / calls : 0.0f;
            ImGui::Text("  %-23s %.3f ms total | %.3f ms/call | %d calls",
                        name, total, perCall, calls);
        };
        phaseTiming("ComputeIntersections",  static_cast<int>(ProfilerOp::ComputeIntersections));
        phaseTiming("ShadeMaterial",         static_cast<int>(ProfilerOp::ShadeMaterial));
        phaseTiming("GatherTerminatedPaths", static_cast<int>(ProfilerOp::GatherTerminatedPaths));
        phaseTiming("SortByMaterial",        static_cast<int>(ProfilerOp::SortByMaterial));
        phaseTiming("CompactPaths",          static_cast<int>(ProfilerOp::CompactPaths));
        phaseTiming("BloomPass",             static_cast<int>(ProfilerOp::BloomPass));
        phaseTiming("PostProcessTail",       static_cast<int>(ProfilerOp::PostProcessTail));
        ImGui::Text("Bounces Last Frame: %d", g_profiler().guiData().lastBounceCount);
    }

    if (app.renderState != nullptr) {
        ImGui::Separator();
        ImGui::Text("Camera Settings (JSON format):");
        Camera& cam = app.renderState->camera;
        char jsonBuf[384];
        sprintf(jsonBuf,
            "\"EYE\": [%.4f, %.4f, %.4f],\n"
            "\"LOOKAT\": [%.4f, %.4f, %.4f],\n"
            "\"UP\": [%.4f, %.4f, %.4f],\n"
            "\"FOVY\": %.2f",
            cam.position.x, cam.position.y, cam.position.z,
            cam.lookAt.x, cam.lookAt.y, cam.lookAt.z,
            cam.up.x, cam.up.y, cam.up.z,
            cam.fov.y
        );
        ImGui::InputTextMultiline("##json_cam", jsonBuf, sizeof(jsonBuf), ImVec2(-FLT_MIN, ImGui::GetTextLineHeight() * 4.5f), ImGuiInputTextFlags_ReadOnly);

        ImGui::Separator();
        ImGui::Text("Movement:");
        ImGui::SliderFloat("Fly Speed", &app.cameraMoveSpeed, 0.05f, 5.0f, "%.2f");

        ImGui::Separator();
        ImGui::Text("DOF Debug:");
        DebugConfig& dbg = app.renderState->debug;
        if (ImGui::SliderFloat("Focal Distance", &cam.focalDistance, 0.5f, 30.0f))
            markCameraChanged(app);
        if (ImGui::SliderFloat("Lens Radius", &cam.lensRadius, 0.0f, 1.0f))
            markCameraChanged(app);
        if (ImGui::Checkbox("Focal Plane Overlay", &dbg.showDOFOverlay))
            markCameraChanged(app);
        if (dbg.showDOFOverlay) {
            if (ImGui::SliderFloat("Focal Tolerance", &dbg.focalTolerance, 0.05f, 5.0f))
                markCameraChanged(app);
        }

        ImGui::Separator();
        ImGui::Separator();
        ImGui::Text("RNG Mode:");
        int currentRng = static_cast<int>(getRngMode());
        if (ImGui::RadioButton("LCG", &currentRng, 0))  { setRngMode(RngMode::LCG); markCameraChanged(app); }
        ImGui::SameLine();
        if (ImGui::RadioButton("Halton", &currentRng, 1)) { setRngMode(RngMode::HALTON); markCameraChanged(app); }

        ImGui::Separator();
        ImGui::Text("Compaction:");
        int curCompact = static_cast<int>(getCompactMethod());
        if (ImGui::RadioButton("Off",        &curCompact, 0)) { setCompactMethod(CompactMethod::Off);        markCameraChanged(app); }
        if (ImGui::RadioButton("Global",     &curCompact, 1)) { setCompactMethod(CompactMethod::GlobalScan);  markCameraChanged(app); }
        if (ImGui::RadioButton("Thrust",     &curCompact, 2)) { setCompactMethod(CompactMethod::Thrust);      markCameraChanged(app); }
        if (ImGui::RadioButton("SharedMem",  &curCompact, 3)) { setCompactMethod(CompactMethod::SharedMem);   markCameraChanged(app); }

        bool sortEnabled = getSortByMaterial();
        if (ImGui::Checkbox("Sort by material", &sortEnabled)) {
            setSortByMaterial(sortEnabled);
            markCameraChanged(app);
        }

        ImGui::Separator();
        ImGui::Text("Bloom:");
        bool bloomEnabled = getBloomEnabled();
        if (ImGui::Checkbox("Enable Bloom", &bloomEnabled))
            setBloomEnabled(bloomEnabled);

        if (bloomEnabled) {
            // Slider bounds come from the same constexpr ranges that clamp
            // config.json ingestion (config.h) — one source of truth.
            float threshold = getBloomThreshold();
            if (ImGui::SliderFloat("Threshold", &threshold,
                    BloomConfig::kThresholdMin, BloomConfig::kThresholdMax, "%.2f"))
                setBloomThreshold(threshold);

            float intensity = getBloomIntensity();
            if (ImGui::SliderFloat("Intensity", &intensity,
                    BloomConfig::kIntensityMin, BloomConfig::kIntensityMax, "%.2f"))
                setBloomIntensity(intensity);

            int radius = getBloomRadius();
            if (ImGui::SliderInt("Radius", &radius,
                    BloomConfig::kRadiusMin, BloomConfig::kRadiusMax))
                setBloomRadius(radius);
        }

        ImGui::Separator();
        ImGui::Text("Chromatic Aberration:");
        bool caEnabled = getChromaticAberrationEnabled();
        if (ImGui::Checkbox("Enable Chromatic Aberration", &caEnabled))
            setChromaticAberrationEnabled(caEnabled);
        if (caEnabled) {
            float caIntensity = getChromaticAberrationIntensity();
            if (ImGui::SliderFloat("CA Intensity", &caIntensity,
                    ChromaticAberrationConfig::kIntensityMin,
                    ChromaticAberrationConfig::kIntensityMax, "%.5f"))
                setChromaticAberrationIntensity(caIntensity);
        }

        ImGui::Separator();
        ImGui::Text("Vignette:");
        bool vigEnabled = getVignetteEnabled();
        if (ImGui::Checkbox("Enable Vignette", &vigEnabled))
            setVignetteEnabled(vigEnabled);
        if (vigEnabled) {
            float vigIntensity = getVignetteIntensity();
            if (ImGui::SliderFloat("Vignette Intensity", &vigIntensity,
                    VignetteConfig::kIntensityMin, VignetteConfig::kIntensityMax, "%.2f"))
                setVignetteIntensity(vigIntensity);
            float vigExponent = getVignetteExponent();
            if (ImGui::SliderFloat("Vignette Exponent", &vigExponent,
                    VignetteConfig::kExponentMin, VignetteConfig::kExponentMax, "%.1f"))
                setVignetteExponent(vigExponent);
        }

        ImGui::Separator();
        {
            SceneStats stats = computeSceneStats(*app.scene);
            ImGui::Text("Scene: %d objects  (%d meshes)",
                        stats.numObjects, stats.numMeshes);
            ImGui::Text("  %d triangles, %d materials",
                        stats.numTriangles, stats.numMaterials);
        }
    }
    ImGui::End();

    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}
