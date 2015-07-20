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

#ifndef calgl2DNodeData_h
#define calgl2DNodeData_h

#include <OpenCAL-GL/calgl2D.h>
#include <OpenCAL-GL/calglCommon.h>

#pragma region DataDefinition
struct CALNode2Db {
	enum CALGL_DATA_TYPE dataType;
	GLuint* callList;

	enum CALGL_TYPE_INFO typeInfoSubstate;
	enum CALGL_TYPE_INFO_USE typeInfoUseSubstate;	
	struct CALSubstate2Db* substate;

	GLdouble min;
	GLdouble max;
	CALbyte noData;

	// ColorValue
	GLfloat redComponent;
	GLfloat greenComponent;
	GLfloat blueComponent;
	GLfloat alphaComponent;

	int capacityNode;
	int insertedNode;
	struct CALNode2Db** nodes; // the first is the father
};
struct CALNode2Di {
	enum CALGL_DATA_TYPE dataType;
	GLuint* callList;

	enum CALGL_TYPE_INFO typeInfoSubstate;
	enum CALGL_TYPE_INFO_USE typeInfoUseSubstate;	
	struct CALSubstate2Di* substate;

	GLdouble min;
	GLdouble max;
	CALint noData;

	// ColorValue
	GLfloat redComponent;
	GLfloat greenComponent;
	GLfloat blueComponent;
	GLfloat alphaComponent;

	int capacityNode;
	int insertedNode;
	struct CALNode2Di** nodes; // the first is the father
};
struct CALNode2Dr {
	enum CALGL_DATA_TYPE dataType;
	GLuint* callList;

	enum CALGL_TYPE_INFO typeInfoSubstate;
	enum CALGL_TYPE_INFO_USE typeInfoUseSubstate;	
	struct CALSubstate2Dr* substate;

	GLdouble min;
	GLdouble max;
	CALreal noData;

	// ColorValue
	GLfloat redComponent;
	GLfloat greenComponent;
	GLfloat blueComponent;
	GLfloat alphaComponent;

	int capacityNode;
	int insertedNode;
	struct CALNode2Dr** nodes; // the first is the father
};
#pragma endregion

#pragma region Create
struct CALNode2Db* calglCreateNode2Db(struct CALNode2Db* father);
struct CALNode2Di* calglCreateNode2Di(struct CALNode2Di* father);
struct CALNode2Dr* calglCreateNode2Dr(struct CALNode2Dr* father);
#pragma endregion

#pragma region Destroy
void calglDestroyNode2Db(struct CALNode2Db* node);
void calglDestroyNode2Di(struct CALNode2Di* node);
void calglDestroyNode2Dr(struct CALNode2Dr* node);
#pragma endregion

#pragma region IncreaseData
void calglIncreaseDataNode2Db(struct CALNode2Db* node);
void calglIncreaseDataNode2Di(struct CALNode2Di* node);
void calglIncreaseDataNode2Dr(struct CALNode2Dr* node);
#pragma endregion

#pragma region DecreaseData
void calglDecreaseDataNode2Db(struct CALNode2Db* node);
void calglDecreaseDataNode2Di(struct CALNode2Di* node);
void calglDecreaseDataNode2Dr(struct CALNode2Dr* node);
#pragma endregion

#pragma region AddData
struct CALNode2Db* calglAddDataNode2Db(struct CALNode2Db* node, struct CALSubstate2Db* substate, enum CALGL_TYPE_INFO typeInfoSubstate, enum CALGL_TYPE_INFO_USE typeInfoUseSubstate, enum CALGL_DATA_TYPE dataType);
struct CALNode2Di* calglAddDataNode2Di(struct CALNode2Di* node, struct CALSubstate2Di* substate, enum CALGL_TYPE_INFO typeInfoSubstate, enum CALGL_TYPE_INFO_USE typeInfoUseSubstate, enum CALGL_DATA_TYPE dataType);
struct CALNode2Dr* calglAddDataNode2Dr(struct CALNode2Dr* node, struct CALSubstate2Dr* substate, enum CALGL_TYPE_INFO typeInfoSubstate, enum CALGL_TYPE_INFO_USE typeInfoUseSubstate, enum CALGL_DATA_TYPE dataType);
#pragma endregion

#pragma region RemoveData
void calglRemoveDataNode2Db(struct CALNode2Db* node, struct CALSubstate2Db* substate);
void calglRemoveDataNode2Di(struct CALNode2Di* node, struct CALSubstate2Di* substate);
void calglRemoveDataNode2Dr(struct CALNode2Dr* node, struct CALSubstate2Dr* substate);
#pragma endregion

#pragma region ShiftLeftFromIndex
void calglShiftLeftFromIndexNode2Db(struct CALNode2Db* node, int index);
void calglShiftLeftFromIndexNode2Di(struct CALNode2Di* node, int index);
void calglShiftLeftFromIndexNode2Dr(struct CALNode2Dr* node, int index);
#pragma endregion

#pragma region GetFather
struct CALNode2Db* calglGetFatherNode2Db(struct CALNode2Db* node);
struct CALNode2Di* calglGetFatherNode2Di(struct CALNode2Di* node);
struct CALNode2Dr* calglGetFatherNode2Dr(struct CALNode2Dr* node);
#pragma endregion

#pragma region SetNoData
void calglSetNoDataToNode2Db(struct CALNode2Db* node, CALbyte noData);
void calglSetNoDataToNode2Di(struct CALNode2Di* node, CALint noData);
void calglSetNoDataToNode2Dr(struct CALNode2Dr* node, CALreal noData);
#pragma endregion

#endif

