#include "camera_controller.h"

#include "app_loop.h"
#include "constants.h"

#include <glm/glm.hpp>

#include <cmath>

namespace {

// Camera control feel parameters (orbit + WASD fly).
constexpr float CAMERA_MIN_THETA      = 0.001f; // orbit latitude pole guard
constexpr float CAMERA_MIN_ZOOM       = 0.1f;   // min camera–target distance
constexpr float CAMERA_SCROLL_ZOOM_IN = 0.97f;  // scroll-up zoom multiplier
constexpr float CAMERA_SCROLL_ZOOM_OUT = 1.03f; // scroll-down zoom multiplier
constexpr float CAMERA_PAN_SPEED      = 0.01f;  // middle-drag pan sensitivity

AppState* appFromWindow(GLFWwindow* window)
{
    return static_cast<AppState*>(glfwGetWindowUserPointer(window));
}

} // namespace

void initializeCameraController(AppState& app)
{
    Camera& cam = app.renderState->camera;

    glm::vec3 view = cam.view;
    glm::vec3 up = cam.up;
    glm::vec3 right = glm::cross(view, up);
    up = glm::cross(right, view);

    // compute phi (horizontal) and theta (vertical) relative 3D axis
    // so, (0 0 1) is forward, (0 1 0) is up
    glm::vec3 v = cam.position - cam.lookAt;
    app.zoom = glm::length(v);
    app.theta = (app.zoom > 0.0f) ? glm::acos(v.y / app.zoom) : 0.0f;
    app.phi = atan2(v.x, v.z);

    // Remember the loaded view so R (recenter) can restore it exactly.
    app.originalCameraPosition = cam.position;
    app.originalZoom = app.zoom;
    app.originalTheta = app.theta;
    app.originalPhi = app.phi;
}

void markCameraChanged(AppState& app)
{
    app.cameraChanged = true;
}

// WASD / Space / Shift fly-translation.
//
// In the free-fly model cam.position is independent state, so translating it
// along the camera's own axes moves the camera through the scene while the
// orientation (theta, phi) — and thus where it points — stays fixed.
//   W/S forward/backward (cam.view), A/D left/right (cam.right),
//   Space/Shift up/down (cam.up).
void updateCameraMovement(AppState& app, float dt)
{
    if (app.io && app.io->WantCaptureKeyboard)
    {
        return; // ImGui text input active
    }

    Camera& cam = app.renderState->camera;

    glm::vec3 translate(0.0f);
    if (glfwGetKey(app.window, GLFW_KEY_W)             == GLFW_PRESS) translate += cam.view;
    if (glfwGetKey(app.window, GLFW_KEY_S)             == GLFW_PRESS) translate -= cam.view;
    if (glfwGetKey(app.window, GLFW_KEY_D)             == GLFW_PRESS) translate += cam.right;
    if (glfwGetKey(app.window, GLFW_KEY_A)             == GLFW_PRESS) translate -= cam.right;
    if (glfwGetKey(app.window, GLFW_KEY_SPACE)         == GLFW_PRESS) translate += cam.up;
    if (glfwGetKey(app.window, GLFW_KEY_LEFT_SHIFT)    == GLFW_PRESS) translate -= cam.up;

    if (translate == glm::vec3(0.0f))
    {
        return;
    }

    // Speed scales with the camera-target distance so the same key feel
    // works at both macro and micro scale (roughly zoom distance per 0.66s).
    float speed = app.zoom * app.cameraMoveSpeed * dt;
    cam.position += glm::normalize(translate) * speed;
    app.cameraChanged = true; // resets accumulation & recomputes view/right/up/lookAt
}

void updateCameraOrientation(AppState& app)
{
    if (!app.cameraChanged) return;

    app.iteration = 0;
    Camera& cam = app.renderState->camera;

    // Free-fly camera: cam.position is independent state (translated by
    // WASD / middle-pan / scroll-dolly) and is NOT touched here.  (theta,
    // phi) describe the view direction, changed only by left-drag, so
    // rotating turns the camera in place.
    //   D        = (sin(phi)sin(theta), cos(theta), cos(phi)sin(theta))
    //   cam.view = -D  →  the direction the camera faces (camera → lookAt)
    glm::vec3 D(sin(app.phi) * sin(app.theta), cos(app.theta), cos(app.phi) * sin(app.theta));
    cam.view = -glm::normalize(D);

    // right/up MUST stay unit length: ray_generation.cu builds the pixel
    // grid as view − right·pixelLength·offset, and pixelLength is calibrated
    // for unit vectors — a short right/up would silently narrow the FOV
    // whenever the camera is pitched.  (cross(view, (0,1,0)) is unit only
    // when looking horizontally.)
    glm::vec3 u = glm::vec3(0, 1, 0);
    glm::vec3 r = glm::normalize(glm::cross(cam.view, u));
    cam.up = glm::cross(r, cam.view); // r ⟂ view, both unit → up is unit too
    cam.right = r;

    // lookAt is a derived reference point zoom units ahead along the view
    // axis (purely bookkeeping — rendering uses position/view/up/right).
    cam.lookAt = cam.position + cam.view * app.zoom;
    app.cameraChanged = false;
}

void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    AppState* app = appFromWindow(window);
    if (!app || action != GLFW_PRESS) return;

    switch (key)
    {
        case GLFW_KEY_ESCAPE:
            saveImage(*app);
            glfwSetWindowShouldClose(window, GL_TRUE);
            break;
        case GLFW_KEY_P: // save image (S is now walk-backward)
            saveImage(*app);
            break;
        case GLFW_KEY_R: // recenter to original position + orientation (was SPACE)
            app->cameraChanged = true;
            app->renderState = &app->scene->state;
            Camera& cam = app->renderState->camera;
            cam.position = app->originalCameraPosition;
            app->zoom = app->originalZoom;
            app->theta = app->originalTheta;
            app->phi = app->originalPhi;
            break;
    }
}

void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods)
{
    AppState* app = appFromWindow(window);
    if (!app || app->mouseOverImGuiWindow)
    {
        return;
    }

    app->leftMousePressed = (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS);
    app->rightMousePressed = (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS);
    app->middleMousePressed = (button == GLFW_MOUSE_BUTTON_MIDDLE && action == GLFW_PRESS);
}

void scrollCallback(GLFWwindow* window, double xoffset, double yoffset)
{
    AppState* app = appFromWindow(window);
    if (!app || (app->io && app->io->WantCaptureMouse)) return;
    // Dolly along the view axis: scrolling changes the reference distance
    // (zoom) and moves the camera to match, so the focused point stays put —
    // "zoom in" physically flies the camera toward what it is looking at.
    Camera& cam = app->renderState->camera;
    const float oldZoom = app->zoom;
    app->zoom *= (yoffset > 0.0) ? CAMERA_SCROLL_ZOOM_IN : CAMERA_SCROLL_ZOOM_OUT;
    app->zoom = std::fmax(CAMERA_MIN_ZOOM, app->zoom);
    cam.position += cam.view * (oldZoom - app->zoom);
    app->cameraChanged = true;
}

void mousePositionCallback(GLFWwindow* window, double xpos, double ypos)
{
    AppState* app = appFromWindow(window);
    if (!app) return;

    if (xpos == app->lastX || ypos == app->lastY)
    {
        return; // otherwise, clicking back into window causes re-start
    }

    if (app->leftMousePressed)
    {
        // compute new camera parameters
        app->phi -= (xpos - app->lastX) / app->width;
        app->theta -= (ypos - app->lastY) / app->height;
        app->theta = std::fmax(CAMERA_MIN_THETA, std::fmin(app->theta, PI));
        app->cameraChanged = true;
    }
    else if (app->rightMousePressed)
    {
        // Dolly along the view axis (same as scroll), driven by drag delta.
        Camera& cam = app->renderState->camera;
        const float oldZoom = app->zoom;
        app->zoom += (ypos - app->lastY) / app->height;
        app->zoom = std::fmax(CAMERA_MIN_ZOOM, app->zoom);
        cam.position += cam.view * (oldZoom - app->zoom);
        app->cameraChanged = true;
    }
    else if (app->middleMousePressed)
    {
        app->renderState = &app->scene->state;
        Camera& cam = app->renderState->camera;

        // Pan the camera in its image plane (screen space).
        //   Horizontal: along the camera's right vector
        //   Vertical:   along the camera's up vector (Y unlocked)
        cam.position -= (float)(xpos - app->lastX) * cam.right * CAMERA_PAN_SPEED;
        cam.position += (float)(ypos - app->lastY) * cam.up    * CAMERA_PAN_SPEED;
        app->cameraChanged = true;
    }

    app->lastX = xpos;
    app->lastY = ypos;
}
