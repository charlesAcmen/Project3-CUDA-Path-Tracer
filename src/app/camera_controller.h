#pragma once

#include "app/app_state.h"

void initializeCameraController(AppState& app);
void updateCameraMovement(AppState& app, float dt);
void updateCameraOrientation(AppState& app);
void markCameraChanged(AppState& app);

// GLFW requires plain function pointers for these callbacks.  Each callback
// obtains AppState from glfwGetWindowUserPointer(window).
void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);
void mousePositionCallback(GLFWwindow* window, double xpos, double ypos);
void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods);
void scrollCallback(GLFWwindow* window, double xoffset, double yoffset);
