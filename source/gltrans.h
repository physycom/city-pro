#pragma once

#include <FL/gl.h>

void save_model(void);
int project(GLdouble objx,GLdouble objy,GLdouble objz,GLdouble *winx,GLdouble *winy,GLdouble *winz);
int unproject(GLdouble winx,GLdouble winy,GLdouble winz,GLdouble *objx,GLdouble *objy,GLdouble *objz);
