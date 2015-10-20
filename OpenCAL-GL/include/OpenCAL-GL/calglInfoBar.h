// (C) Copyright University of Calabria and others.
// All rights reserved. This program and the accompanying materials
// are made available under the terms of the GNU Lesser General Public License
// (LGPL) version 2.1 which accompanies this distribution, and is available at
// http://www.gnu.org/licenses/lgpl-2.1.html
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
// Lesser General Public License for more details.

#ifndef calglInfoBar_h
#define calglInfoBar_h

#include <OpenCAL-GL/calgl2D.h>
#include <OpenCAL-GL/calgl3D.h>
#include <OpenCAL-GL/calglCommon.h>
#include <OpenCAL-GL/calglGlobalSettings.h>
//#include <calCommon.h>
struct CALDrawModel2D;
struct CALDrawModel3D;
struct CALGLInfoBar{
	const char* substateName;
	enum CALGL_TYPE_INFO_USE infoUse;
	GLdouble* min;
	GLdouble* max;

	GLfloat xPosition;
	GLfloat yPosition;
	GLint width;
	GLint height;

	enum CALGL_INFO_BAR_DIMENSION dimension;
	enum CALGL_INFO_BAR_ORIENTATION orientation;

	CALbyte barInitialization;
	GLint constWidth;
	GLint constHeight;
};

/*! Constructor
*/
struct CALGLInfoBar* calglCreateRelativeInfoBar2Db(const char* substateName, enum CALGL_TYPE_INFO_USE infoUse, struct CALDrawModel2D* calDrawModel, struct CALSubstate2Db* substate, enum CALGL_INFO_BAR_ORIENTATION orientation);
struct CALGLInfoBar* calglCreateRelativeInfoBar2Di(const char* substateName, enum CALGL_TYPE_INFO_USE infoUse, struct CALDrawModel2D* calDrawModel, struct CALSubstate2Di* substate, enum CALGL_INFO_BAR_ORIENTATION orientation);
struct CALGLInfoBar* calglCreateRelativeInfoBar2Dr(const char* substateName, enum CALGL_TYPE_INFO_USE infoUse, struct CALDrawModel2D* calDrawModel, struct CALSubstate2Dr* substate, enum CALGL_INFO_BAR_ORIENTATION orientation);
struct CALGLInfoBar* calglCreateRelativeInfoBar3Db(const char* substateName, enum CALGL_TYPE_INFO_USE infoUse, struct CALDrawModel3D* calDrawModel, struct CALSubstate3Db* substate, enum CALGL_INFO_BAR_ORIENTATION orientation);
struct CALGLInfoBar* calglCreateRelativeInfoBar3Di(const char* substateName, enum CALGL_TYPE_INFO_USE infoUse, struct CALDrawModel3D* calDrawModel, struct CALSubstate3Di* substate, enum CALGL_INFO_BAR_ORIENTATION orientation);
struct CALGLInfoBar* calglCreateRelativeInfoBar3Dr(const char* substateName, enum CALGL_TYPE_INFO_USE infoUse, struct CALDrawModel3D* calDrawModel, struct CALSubstate3Dr* substate, enum CALGL_INFO_BAR_ORIENTATION orientation);
struct CALGLInfoBar* calglCreateAbsoluteInfoBar2Db(const char* substateName, enum CALGL_TYPE_INFO_USE infoUse, struct CALDrawModel2D* calDrawModel, struct CALSubstate2Db* substate, GLfloat xPosition, GLfloat yPosition, GLint width, GLint height);
struct CALGLInfoBar* calglCreateAbsoluteInfoBar2Di(const char* substateName, enum CALGL_TYPE_INFO_USE infoUse, struct CALDrawModel2D* calDrawModel, struct CALSubstate2Di* substate, GLfloat xPosition, GLfloat yPosition, GLint width, GLint height);
struct CALGLInfoBar* calglCreateAbsoluteInfoBar2Dr(const char* substateName, enum CALGL_TYPE_INFO_USE infoUse, struct CALDrawModel2D* calDrawModel, struct CALSubstate2Dr* substate, GLfloat xPosition, GLfloat yPosition, GLint width, GLint height);
struct CALGLInfoBar* calglCreateAbsoluteInfoBar3Db(const char* substateName, enum CALGL_TYPE_INFO_USE infoUse, struct CALDrawModel3D* calDrawModel, struct CALSubstate3Db* substate, GLfloat xPosition, GLfloat yPosition, GLint width, GLint height);
struct CALGLInfoBar* calglCreateAbsoluteInfoBar3Di(const char* substateName, enum CALGL_TYPE_INFO_USE infoUse, struct CALDrawModel3D* calDrawModel, struct CALSubstate3Di* substate, GLfloat xPosition, GLfloat yPosition, GLint width, GLint height);
struct CALGLInfoBar* calglCreateAbsoluteInfoBar3Dr(const char* substateName, enum CALGL_TYPE_INFO_USE infoUse, struct CALDrawModel3D* calDrawModel, struct CALSubstate3Dr* substate, GLfloat xPosition, GLfloat yPosition, GLint width, GLint height);

/*! Destructor
*/
void calglDestroyInfoBar(struct CALGLInfoBar* infoBar);

void calglSetInfoBarConstDimension(struct CALGLInfoBar* infoBar, GLfloat width, GLfloat height);

#endif
