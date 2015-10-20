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

#ifndef calgl3D_h
#define calgl3D_h

#ifdef __APPLE__
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif

#include <OpenCAL/cal3D.h>
#include <OpenCAL-GL/calgl3DNodeData.h>
#include <OpenCAL-GL/calglModelViewParameter.h>
#include <OpenCAL-GL/calglLightParameter.h>
#include <OpenCAL-GL/calgl3DUpdater.h>
#include <OpenCAL-GL/calglInfoBar.h>
#include <OpenCAL-GL/calglGlobalSettings.h>

/*! Structure that contains informations regarding drawing type
*/
struct CALDrawModel3D{
	const char* name;
	enum CALGL_DRAW_MODE drawMode;
	struct CALNode3Db* byteModel;
	struct CALNode3Di* intModel;
	struct CALNode3Dr* realModel;
	struct CALModel3D* calModel;

	struct CALGLModelViewParameter* modelView;
	struct CALGLLightParameter* modelLight;
	struct CALUpdater3D* calUpdater;
	struct CALGLInfoBar* infoBar;

	GLshort* drawKCells;
	GLshort* drawICells;
	GLshort* drawJCells;

	// Private data for const color
	GLfloat redComponent;
	GLfloat greenComponent;
	GLfloat blueComponent;
	GLfloat alphaComponent;
};

/*! Constructor
*/
struct CALDrawModel3D* calglDefDrawModel3D(enum CALGL_DRAW_MODE mode, const char* name, struct CALModel3D* calModel, struct CALRun3D* calRun);

/*! Destructor
*/
void calglDestoyDrawModel3D(struct CALDrawModel3D* drawModel);

#pragma region AddData
void calglAddToDrawModel3Db(struct CALDrawModel3D* drawModel, struct CALSubstate3Db* substateFather, struct CALSubstate3Db** substateToAdd, enum CALGL_TYPE_INFO typeInfo, enum CALGL_TYPE_INFO_USE typeInfoUseSubstate, enum CALGL_DATA_TYPE dataType);
void calglAddToDrawModel3Di(struct CALDrawModel3D* drawModel, struct CALSubstate3Di* substateFather, struct CALSubstate3Di** substateToAdd, enum CALGL_TYPE_INFO typeInfo, enum CALGL_TYPE_INFO_USE typeInfoUseSubstate, enum CALGL_DATA_TYPE dataType);
void calglAddToDrawModel3Dr(struct CALDrawModel3D* drawModel, struct CALSubstate3Dr* substateFather, struct CALSubstate3Dr** substateToAdd, enum CALGL_TYPE_INFO typeInfo, enum CALGL_TYPE_INFO_USE typeInfoUseSubstate, enum CALGL_DATA_TYPE dataType);
#pragma endregion

#pragma region SearchSubstate
void calglSearchSubstateDrawModel3Db(struct CALNode3Db* currentNode, struct CALSubstate3Db* substateToSearch, struct CALNode3Db** nodeSearched);
void calglSearchSubstateDrawModel3Di(struct CALNode3Di* currentNode, struct CALSubstate3Di* substateToSearch, struct CALNode3Di** nodeSearched);
void calglSearchSubstateDrawModel3Dr(struct CALNode3Dr* currentNode, struct CALSubstate3Dr* substateToSearch, struct CALNode3Dr** nodeSearched);
#pragma endregion

void calglDisplayModel3D(struct CALDrawModel3D* calDrawModel);

#pragma region DrawDiscreetModel3D
void calglDrawDiscreetModel3D(struct CALDrawModel3D* calDrawModel);

void calglDrawDiscreetModelDisplayNode3Db(struct CALDrawModel3D* calDrawModel, struct CALNode3Db* calNode);
void calglDrawDiscreetModelDisplayNode3Di(struct CALDrawModel3D* calDrawModel, struct CALNode3Di* calNode);
void calglDrawDiscreetModelDisplayNode3Dr(struct CALDrawModel3D* calDrawModel, struct CALNode3Dr* calNode);

void calglDrawDiscreetModelDisplayCurrentNode3Db(struct CALDrawModel3D* calDrawModel, struct CALNode3Db* calNode);
void calglDrawDiscreetModelDisplayCurrentNode3Di(struct CALDrawModel3D* calDrawModel, struct CALNode3Di* calNode);
void calglDrawDiscreetModelDisplayCurrentNode3Dr(struct CALDrawModel3D* calDrawModel, struct CALNode3Dr* calNode);
#pragma endregion

#pragma region DrawRealModel3D
void calglDrawRealModel3D(struct CALDrawModel3D* calDrawModel);
#pragma endregion

#pragma region ComputeExtremes
void calglComputeExtremesDrawModel3Db(struct CALDrawModel3D* calDrawModel, struct CALNode3Db* calNode, GLdouble* m, GLdouble* M);
void calglComputeExtremesDrawModel3Di(struct CALDrawModel3D* calDrawModel, struct CALNode3Di* calNode, GLdouble* m, GLdouble* M);
void calglComputeExtremesDrawModel3Dr(struct CALDrawModel3D* calDrawModel, struct CALNode3Dr* calNode, GLdouble* m, GLdouble* M);
#pragma endregion

#pragma region ComputeExtremesToAll
void calglComputeExtremesToAll3Db(struct CALDrawModel3D* calDrawModel, struct CALNode3Db* calNode);
void calglComputeExtremesToAll3Di(struct CALDrawModel3D* calDrawModel, struct CALNode3Di* calNode);
void calglComputeExtremesToAll3Dr(struct CALDrawModel3D* calDrawModel, struct CALNode3Dr* calNode);
#pragma endregion

#pragma region SetNormalData
void calglSetNormalData3Db(struct CALDrawModel3D* calDrawModel, struct CALNode3Db* calNode, GLint i, GLint j, GLint k);
void calglSetNormalData3Di(struct CALDrawModel3D* calDrawModel, struct CALNode3Di* calNode, GLint i, GLint j, GLint k);
void calglSetNormalData3Dr(struct CALDrawModel3D* calDrawModel, struct CALNode3Dr* calNode, GLint i, GLint j, GLint k);
#pragma endregion

#pragma region SetColorData
GLboolean calglSetColorData3Db(struct CALDrawModel3D* calDrawModel, struct CALNode3Db* calNode, GLint i, GLint j, GLint k);
GLboolean calglSetColorData3Di(struct CALDrawModel3D* calDrawModel, struct CALNode3Di* calNode, GLint i, GLint j, GLint k);
GLboolean calglSetColorData3Dr(struct CALDrawModel3D* calDrawModel, struct CALNode3Dr* calNode, GLint i, GLint j, GLint k);
#pragma endregion

void calglColor3D(struct CALDrawModel3D* calDrawModel, GLfloat redComponent, GLfloat greenComponent, GLfloat blueComponent, GLfloat alphaComponent);

void calglSetModelViewParameter3D(struct CALDrawModel3D* calDrawModel, struct CALGLModelViewParameter* modelView);

void calglSetLightParameter3D(struct CALDrawModel3D* calDrawModel, struct CALGLLightParameter* modelLight);

#pragma region BoundingBox
void calglDrawBoundingBox3D(struct CALDrawModel3D* calDrawModel);
#pragma endregion

#pragma region InfoBar
void calglRelativeInfoBar3Db(struct CALDrawModel3D* calDrawModel, struct CALSubstate3Db* substate, const char* substateName, enum CALGL_TYPE_INFO_USE infoUse, enum CALGL_INFO_BAR_ORIENTATION orientation);
void calglRelativeInfoBar3Di(struct CALDrawModel3D* calDrawModel, struct CALSubstate3Di* substate, const char* substateName, enum CALGL_TYPE_INFO_USE infoUse, enum CALGL_INFO_BAR_ORIENTATION orientation);
void calglRelativeInfoBar3Dr(struct CALDrawModel3D* calDrawModel, struct CALSubstate3Dr* substate, const char* substateName, enum CALGL_TYPE_INFO_USE infoUse, enum CALGL_INFO_BAR_ORIENTATION orientation);
void calglAbsoluteInfoBar3Db(struct CALDrawModel3D* calDrawModel, struct CALSubstate3Db* substate, const char* substateName, enum CALGL_TYPE_INFO_USE infoUse, GLfloat xPosition, GLfloat yPosition, GLint width, GLint height);
void calglAbsoluteInfoBar3Di(struct CALDrawModel3D* calDrawModel, struct CALSubstate3Di* substate, const char* substateName, enum CALGL_TYPE_INFO_USE infoUse, GLfloat xPosition, GLfloat yPosition, GLint width, GLint height);
void calglAbsoluteInfoBar3Dr(struct CALDrawModel3D* calDrawModel, struct CALSubstate3Dr* substate, const char* substateName, enum CALGL_TYPE_INFO_USE infoUse, GLfloat xPosition, GLfloat yPosition, GLint width, GLint height);
#pragma endregion

#pragma region DrawIntervals
void calglDisplayDrawKBound3D(struct CALDrawModel3D* calDrawModel, GLint min, GLint max);
void calglDisplayDrawIBound3D(struct CALDrawModel3D* calDrawModel, GLint min, GLint max);
void calglDisplayDrawJBound3D(struct CALDrawModel3D* calDrawModel, GLint min, GLint max);
void calglHideDrawKBound3D(struct CALDrawModel3D* calDrawModel, GLint min, GLint max);
void calglHideDrawIBound3D(struct CALDrawModel3D* calDrawModel, GLint min, GLint max);
void calglHideDrawJBound3D(struct CALDrawModel3D* calDrawModel, GLint min, GLint max);
#pragma endregion

#endif
