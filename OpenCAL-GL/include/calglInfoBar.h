#ifndef calglInfoBar_h
#define calglInfoBar_h

#include <calgl2D.h>
#include <calgl3D.h>
#include <calglCommon.h>
#include <calglGlobalSettings.h>

struct CALGLInfoBar{
	const char* substateName;
	enum CALGL_TYPE_INFO_USE infoUse;
	GLdouble* min;
	GLdouble* max;

	GLfloat xPosition;
	GLfloat yPosition;
	GLint width;
	GLint height;

	CALbyte barInitialization;
	GLint constWidth;
	GLint constHeight;

	enum CALGL_INFO_BAR_ORIENTATION orientation;
};

/*! Constructor
*/
struct CALGLInfoBar* calglCreateInfoBar2Db(const char* substateName, enum CALGL_TYPE_INFO_USE infoUse, struct CALDrawModel2D* calDrawModel, struct CALSubstate2Db* substate, enum CALGL_INFO_BAR_ORIENTATION orientation);
struct CALGLInfoBar* calglCreateInfoBar2Di(const char* substateName, enum CALGL_TYPE_INFO_USE infoUse, struct CALDrawModel2D* calDrawModel, struct CALSubstate2Di* substate, enum CALGL_INFO_BAR_ORIENTATION orientation);
struct CALGLInfoBar* calglCreateInfoBar2Dr(const char* substateName, enum CALGL_TYPE_INFO_USE infoUse, struct CALDrawModel2D* calDrawModel, struct CALSubstate2Dr* substate, enum CALGL_INFO_BAR_ORIENTATION orientation);
struct CALGLInfoBar* calglCreateInfoBar3Db(const char* substateName, enum CALGL_TYPE_INFO_USE infoUse, struct CALDrawModel3D* calDrawModel, struct CALSubstate3Db* substate, enum CALGL_INFO_BAR_ORIENTATION orientation);
struct CALGLInfoBar* calglCreateInfoBar3Di(const char* substateName, enum CALGL_TYPE_INFO_USE infoUse, struct CALDrawModel3D* calDrawModel, struct CALSubstate3Di* substate, enum CALGL_INFO_BAR_ORIENTATION orientation);
struct CALGLInfoBar* calglCreateInfoBar3Dr(const char* substateName, enum CALGL_TYPE_INFO_USE infoUse, struct CALDrawModel3D* calDrawModel, struct CALSubstate3Dr* substate, enum CALGL_INFO_BAR_ORIENTATION orientation);

/*! Destructor
*/
void calglDestroyInfoBar(struct CALGLInfoBar* infoBar);

void calglSetInfoBarConstDimension(struct CALGLInfoBar* infoBar, GLfloat width, GLfloat height);

#endif
