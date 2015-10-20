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

#ifndef calgl2D_h
#define calgl2D_h

#ifdef __APPLE__
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif

#include <OpenCAL/calCommon.h>

#include <OpenCAL-GL/calgl2DNodeData.h>
#include <OpenCAL-GL/calglModelViewParameter.h>
#include <OpenCAL-GL/calglLightParameter.h>
#include <OpenCAL-GL/calgl2DUpdater.h>
#include <OpenCAL-GL/calglInfoBar.h>
#include <OpenCAL-GL/calglGlobalSettings.h>



/*! Structure that contains informations regarding drawing type
*/
struct CALDrawModel2D{
	const char* name;
	enum CALGL_DRAW_MODE drawMode;
	struct CALNode2Db* byteModel;
	struct CALNode2Di* intModel;
	struct CALNode2Dr* realModel;
	struct CALModel2D* calModel;

	struct CALGLModelViewParameter* modelView;
	struct CALGLLightParameter* modelLight;
	struct CALUpdater2D* calUpdater;
	struct CALGLInfoBar* infoBar;

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
struct CALDrawModel2D* calglDefDrawModel2D(enum CALGL_DRAW_MODE mode, const char* name, struct CALModel2D* calModel, struct CALRun2D* calRun);

/*! Destructor
*/
void calglDestoyDrawModel2D(struct CALDrawModel2D* drawModel);

#pragma region AddData
void calglAddToDrawModel2Db(struct CALDrawModel2D* drawModel, struct CALSubstate2Db* substateFather, struct CALSubstate2Db** substateToAdd, enum CALGL_TYPE_INFO typeInfo, enum CALGL_TYPE_INFO_USE typeInfoUseSubstate, enum CALGL_DATA_TYPE dataType);
void calglAddToDrawModel2Di(struct CALDrawModel2D* drawModel, struct CALSubstate2Di* substateFather, struct CALSubstate2Di** substateToAdd, enum CALGL_TYPE_INFO typeInfo, enum CALGL_TYPE_INFO_USE typeInfoUseSubstate, enum CALGL_DATA_TYPE dataType);
void calglAddToDrawModel2Dr(struct CALDrawModel2D* drawModel, struct CALSubstate2Dr* substateFather, struct CALSubstate2Dr** substateToAdd, enum CALGL_TYPE_INFO typeInfo, enum CALGL_TYPE_INFO_USE typeInfoUseSubstate, enum CALGL_DATA_TYPE dataType);
#pragma endregion

#pragma region SearchSubstate
void calglSearchSubstateDrawModel2Db(struct CALNode2Db* currentNode, struct CALSubstate2Db* substateToSearch, struct CALNode2Db** nodeSearched);
void calglSearchSubstateDrawModel2Di(struct CALNode2Di* currentNode, struct CALSubstate2Di* substateToSearch, struct CALNode2Di** nodeSearched);
void calglSearchSubstateDrawModel2Dr(struct CALNode2Dr* currentNode, struct CALSubstate2Dr* substateToSearch, struct CALNode2Dr** nodeSearched);
#pragma endregion

void calglDisplayModel2D(struct CALDrawModel2D* calDrawModel);

#pragma region DrawDiscreetModel2D
void calglDrawDiscreetModel2D(struct CALDrawModel2D* calDrawModel);

void calglDrawDiscreetModelDisplayNode2Db(struct CALDrawModel2D* calDrawModel, struct CALNode2Db* calNode);
void calglDrawDiscreetModelDisplayNode2Di(struct CALDrawModel2D* calDrawModel, struct CALNode2Di* calNode);
void calglDrawDiscreetModelDisplayNode2Dr(struct CALDrawModel2D* calDrawModel, struct CALNode2Dr* calNode);

void calglDrawDiscreetModelDisplayCurrentNode2Db(struct CALDrawModel2D* calDrawModel, struct CALNode2Db* calNode);
void calglDrawDiscreetModelDisplayCurrentNode2Di(struct CALDrawModel2D* calDrawModel, struct CALNode2Di* calNode);
void calglDrawDiscreetModelDisplayCurrentNode2Dr(struct CALDrawModel2D* calDrawModel, struct CALNode2Dr* calNode);
#pragma endregion

#pragma region DrawRealModel2D
void calglDrawRealModel2D(struct CALDrawModel2D* calDrawModel);

void calglDrawRealModelDisplayNode2Db(struct CALDrawModel2D* calDrawModel, struct CALNode2Db* calNode);
void calglDrawRealModelDisplayNode2Di(struct CALDrawModel2D* calDrawModel, struct CALNode2Di* calNode);
void calglDrawRealModelDisplayNode2Dr(struct CALDrawModel2D* calDrawModel, struct CALNode2Dr* calNode);

void calglDrawRealModelDisplayCurrentNode2Db(struct CALDrawModel2D* calDrawModel, struct CALNode2Db* calNode);
void calglDrawRealModelDisplayCurrentNode2Di(struct CALDrawModel2D* calDrawModel, struct CALNode2Di* calNode);
void calglDrawRealModelDisplayCurrentNode2Dr(struct CALDrawModel2D* calDrawModel, struct CALNode2Dr* calNode);

void calglDrawRealModelDisplayCurrentNodeSetVertexData2Db(struct CALDrawModel2D* calDrawModel, struct CALNode2Db*calNode, GLint i, GLint j);
void calglDrawRealModelDisplayCurrentNodeSetVertexData2Di(struct CALDrawModel2D* calDrawModel, struct CALNode2Di*calNode, GLint i, GLint j);
void calglDrawRealModelDisplayCurrentNodeSetVertexData2Dr(struct CALDrawModel2D* calDrawModel, struct CALNode2Dr*calNode, GLint i, GLint j);
#pragma endregion

#pragma region ComputeExtremes
void calglComputeExtremesDrawModel2Db(struct CALDrawModel2D* calDrawModel, struct CALNode2Db* calNode, GLdouble* m, GLdouble* M);
void calglComputeExtremesDrawModel2Di(struct CALDrawModel2D* calDrawModel, struct CALNode2Di* calNode, GLdouble* m, GLdouble* M);
void calglComputeExtremesDrawModel2Dr(struct CALDrawModel2D* calDrawModel, struct CALNode2Dr* calNode, GLdouble* m, GLdouble* M);
#pragma endregion

#pragma region ComputeExtremesToAll
void calglComputeExtremesToAll2Db(struct CALDrawModel2D* calDrawModel, struct CALNode2Db* calNode);
void calglComputeExtremesToAll2Di(struct CALDrawModel2D* calDrawModel, struct CALNode2Di* calNode);
void calglComputeExtremesToAll2Dr(struct CALDrawModel2D* calDrawModel, struct CALNode2Dr* calNode);
#pragma endregion

#pragma region SetNormalData
void calglSetNormalData2Db(struct CALDrawModel2D* calDrawModel, struct CALNode2Db* calNode, GLint i, GLint j);
void calglSetNormalData2Di(struct CALDrawModel2D* calDrawModel, struct CALNode2Di* calNode, GLint i, GLint j);
void calglSetNormalData2Dr(struct CALDrawModel2D* calDrawModel, struct CALNode2Dr* calNode, GLint i, GLint j);
#pragma endregion

#pragma region SetColorData
GLboolean calglSetColorData2Db(struct CALDrawModel2D* calDrawModel, struct CALNode2Db* calNode, GLint i, GLint j);
GLboolean calglSetColorData2Di(struct CALDrawModel2D* calDrawModel, struct CALNode2Di* calNode, GLint i, GLint j);
GLboolean calglSetColorData2Dr(struct CALDrawModel2D* calDrawModel, struct CALNode2Dr* calNode, GLint i, GLint j);
#pragma endregion

void calglColor2D(struct CALDrawModel2D* calDrawModel, GLfloat redComponent, GLfloat greenComponent, GLfloat blueComponent, GLfloat alphaComponent);

void calglSetModelViewParameter2D(struct CALDrawModel2D* calDrawModel, struct CALGLModelViewParameter* modelView);

void calglSetLightParameter2D(struct CALDrawModel2D* calDrawModel, struct CALGLLightParameter* modelLight);

#pragma region BoundingBox
void calglDrawBoundingSquare2D(struct CALDrawModel2D* calDrawModel);
void calglDrawBoundingBox2D(struct CALDrawModel2D* calDrawModel, GLfloat height, GLfloat low);
#pragma endregion

#pragma region InfoBar
void calglRelativeInfoBar2Db(struct CALDrawModel2D* calDrawModel, struct CALSubstate2Db* substate, const char* substateName, enum CALGL_TYPE_INFO_USE infoUse, enum CALGL_INFO_BAR_ORIENTATION orientation);
void calglRelativeInfoBar2Di(struct CALDrawModel2D* calDrawModel, struct CALSubstate2Di* substate, const char* substateName, enum CALGL_TYPE_INFO_USE infoUse, enum CALGL_INFO_BAR_ORIENTATION orientation);
void calglRelativeInfoBar2Dr(struct CALDrawModel2D* calDrawModel, struct CALSubstate2Dr* substate, const char* substateName, enum CALGL_TYPE_INFO_USE infoUse, enum CALGL_INFO_BAR_ORIENTATION orientation);
void calglAbsoluteInfoBar2Db(struct CALDrawModel2D* calDrawModel, struct CALSubstate2Db* substate, const char* substateName, enum CALGL_TYPE_INFO_USE infoUse, GLfloat xPosition, GLfloat yPosition, GLint width, GLint height);
void calglAbsoluteInfoBar2Di(struct CALDrawModel2D* calDrawModel, struct CALSubstate2Di* substate, const char* substateName, enum CALGL_TYPE_INFO_USE infoUse, GLfloat xPosition, GLfloat yPosition, GLint width, GLint height);
void calglAbsoluteInfoBar2Dr(struct CALDrawModel2D* calDrawModel, struct CALSubstate2Dr* substate, const char* substateName, enum CALGL_TYPE_INFO_USE infoUse, GLfloat xPosition, GLfloat yPosition, GLint width, GLint height);
#pragma endregion

#pragma region DrawIntervals
void calglDisplayDrawIBound2D(struct CALDrawModel2D* calDrawModel, GLint min, GLint max);
void calglDisplayDrawJBound2D(struct CALDrawModel2D* calDrawModel, GLint min, GLint max);
void calglHideDrawIBound2D(struct CALDrawModel2D* calDrawModel, GLint min, GLint max);
void calglHideDrawJBound2D(struct CALDrawModel2D* calDrawModel, GLint min, GLint max);
#pragma endregion

#endif
