#include "app/app_loop.h"

#include "app/camera_controller.h"
#include "app/save_output.h"
#include "image.h"
#include "pathtrace.h"
#include "profiler/profiler.h"
#include "app/render_ui.h"
#include "utils/logger.h"

#include "ImGui/imgui_impl_glfw.h"
#include "ImGui/imgui_impl_opengl3.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <string>

namespace {

constexpr float CAMERA_MAX_FRAME_DT = 0.1f; // clamp against first-frame / lag jumps

// ====================================================================
// Rendering Pipeline
// ====================================================================

void runCuda(AppState& app)
{
    updateCameraOrientation(app);

    // Map OpenGL buffer object for writing from CUDA on a single GPU
    // No data is moved (Win & Linux). When mapped to CUDA, OpenGL should not use this buffer

    if (!app.pathtraceInitialized)
    {
        // First frame: full one-time init — build the BVH, upload materials /
        // triangles / textures, allocate all device buffers.
        pathtraceInit(app.scene);
        app.pathtraceInitialized = true;
    }
    else if (app.iteration == 0)
    {
        // Camera or a runtime setting changed → restart MC accumulation.
        // The scene never changed, so only the HDR accumulation buffer is
        // zeroed — NOT the old pathtraceFree + pathtraceInit cycle, which
        // rebuilt the BVH and re-uploaded the whole scene every interactive
        // frame (the cause of the FPS drop while dragging the camera).
        pathtraceResetAccumulation();
        app.saveSchedule.beginNextPass();
    }

    if (app.iteration < app.renderState->iterations)
    {
        uchar4* pbo_dptr = NULL;
        app.iteration++;

        // ---- Map PBO for CUDA write access (modern stream-level interop) ----
        // Modern API: cudaGraphicsMapResources avoids the full-context
        // synchronisation that the deprecated cudaGLMapBufferObject performed,
        // which on WDDM could accumulate the entire frame's GPU work into a
        // single KMD submission window, intermittently triggering TDR.
        cudaGraphicsMapResources(1, &app.cudaPboResource, 0);
        cudaGraphicsResourceGetMappedPointer((void**)&pbo_dptr,
                                             NULL, app.cudaPboResource);

        // execute the kernel
        g_profiler().beginFrame();
        pathtrace(pbo_dptr, app.iteration);
        g_profiler().endFrame();

        // Ensure CUDA work completes before GL touches the PBO.
        cudaDeviceSynchronize();

        // unmap buffer object
        cudaGraphicsUnmapResources(1, &app.cudaPboResource, 0);

        // A checkpoint can only be consumed at its exact iteration. Completion
        // is also an automatic save; SaveSchedule coalesces a final checkpoint
        // into that single request.
        if (app.saveSchedule.shouldSaveAt(app.iteration,
                                          app.renderState->iterations))
        {
            saveImage(app, false);
        }
    }
    else
    {
        // Write CSVs and destroy CUDA events BEFORE tearing down the context.
        // The atexit handler will fire again during exit() but is a no-op
        // (vectors already cleared, events already null).
        g_profiler().shutdown();
        pathtraceFree();
        // Null out the PBO so the atexit(cleanupCuda) handler doesn't try
        // cudaGLUnregisterBufferObject after the context was destroyed.
        app.pbo = 0;
        cudaDeviceReset();
        exit(EXIT_SUCCESS);
    }
}

} // namespace

bool initializeSaveOutput(AppState& app)
{
    std::string error;
    app.saveOutputDirectory = SaveOutput::createUniqueRunDirectory(
        "outputs", app.renderState->imageName, app.startTimeString, error);
    if (!app.saveOutputDirectory.empty())
    {
        Log::info("Image", "Save output: %s", app.saveOutputDirectory.string().c_str());
        return true;
    }

    Log::error("Image", "%s", error.c_str());
    return false;
}

void saveImage(AppState& app)
{
    saveImage(app, true);
}

bool saveImage(AppState& app, bool manual)
{
    // Fetch the latest tonemapped display buffer on demand.
    pathtraceCopyDisplayToHost();
    // No /samples (already averaged in prepareDisplayKernel)
    // and no tonemapping here (already applied by tonemapKernel).
    Image img(app.width, app.height);

    for (int x = 0; x < app.width; x++)
    {
        for (int y = 0; y < app.height; y++)
        {
            int index = x + (y * app.width);
            glm::vec3 pix = app.renderState->image[index];
            img.setPixel(x, y, pix);
        }
    }

    const std::filesystem::path filename = SaveOutput::imagePath(
        app.saveOutputDirectory, app.saveSchedule.pass(), app.iteration, manual);
    return img.savePNG(filename.string());
}

void mainLoop(AppState& app)
{
    double lastTime = glfwGetTime();
    while (!glfwWindowShouldClose(app.window))
    {
        glfwPollEvents();

        // Frame-rate independent WASD / Space / Shift fly movement.
        double now = glfwGetTime();
        float dt = (float)std::fmin(now - lastTime, CAMERA_MAX_FRAME_DT);
        lastTime = now;
        updateCameraMovement(app, dt);

        runCuda(app);

        std::string title = "CIS565 Path Tracer | " + std::to_string(app.iteration) + " Iterations";
        glfwSetWindowTitle(app.window, title.c_str());
        // Centre the rendered scene inside the larger window so ImGui has room.
        int fbW, fbH;
        glfwGetFramebufferSize(app.window, &fbW, &fbH);
        int vpX = (fbW - app.width) / 2;
        int vpY = (fbH - app.height) / 2;
        glViewport(vpX, vpY, app.width, app.height);

        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, app.pbo);
        glBindTexture(GL_TEXTURE_2D, app.displayImage);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, app.width, app.height, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
        glClear(GL_COLOR_BUFFER_BIT);

        // Binding GL_PIXEL_UNPACK_BUFFER back to default
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

        // VAO, shader program, and texture already bound
        glDrawElements(GL_TRIANGLES, 6,  GL_UNSIGNED_SHORT, 0);

        // Render ImGui Stuff
        renderImGui(app);

        glfwSwapBuffers(app.window);
    }

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(app.window);
    glfwTerminate();
}
