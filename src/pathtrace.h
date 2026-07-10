#pragma once

#include "scene.h"
#include "utilities.h"

void InitDataContainer(GuiDataContainer* guiData);
void pathtraceInit(Scene *scene);
void pathtraceFree();
void pathtrace(uchar4 *pbo, int frame, int iteration);

// Runtime configuration overrides (call before pathtraceInit)
void setCompactMethod(int method);
void setSortByMaterial(bool enable);
int  getCompactMethod();
bool getSortByMaterial();
void setAutoSave(bool enable);
bool getAutoSave();
void setFresnelMode(int mode);
int  getFresnelMode();
