#ifndef calgl3DNodeData_h
#define calgl3DNodeData_h

#include <calglCommon.h>
#include <calgl3D.h>

#pragma region DataDefinition
struct CALNode3Db {
	enum CALGL_DATA_TYPE dataType;
	GLuint* callList;

	enum CALGL_TYPE_INFO typeInfoSubstate;
	enum CALGL_TYPE_INFO_USE typeInfoUseSubstate;	
	struct CALSubstate3Db* substate;

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
	struct CALNode3Db** nodes; // the first is the father
};
struct CALNode3Di {
	enum CALGL_DATA_TYPE dataType;
	GLuint* callList;

	enum CALGL_TYPE_INFO typeInfoSubstate;
	enum CALGL_TYPE_INFO_USE typeInfoUseSubstate;	
	struct CALSubstate3Di* substate;

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
	struct CALNode3Di** nodes; // the first is the father
};
struct CALNode3Dr {
	enum CALGL_DATA_TYPE dataType;
	GLuint* callList;

	enum CALGL_TYPE_INFO typeInfoSubstate;
	enum CALGL_TYPE_INFO_USE typeInfoUseSubstate;	
	struct CALSubstate3Dr* substate;

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
	struct CALNode3Dr** nodes; // the first is the father
};
#pragma endregion

#pragma region Create
struct CALNode3Db* calglCreateNode3Db(struct CALNode3Db* father);
struct CALNode3Di* calglCreateNode3Di(struct CALNode3Di* father);
struct CALNode3Dr* calglCreateNode3Dr(struct CALNode3Dr* father);
#pragma endregion

#pragma region Destroy
void calglDestroyNode3Db(struct CALNode3Db* node);
void calglDestroyNode3Di(struct CALNode3Di* node);
void calglDestroyNode3Dr(struct CALNode3Dr* node);
#pragma endregion

#pragma region IncreaseData
void calglIncreaseDataNode3Db(struct CALNode3Db* node);
void calglIncreaseDataNode3Di(struct CALNode3Di* node);
void calglIncreaseDataNode3Dr(struct CALNode3Dr* node);
#pragma endregion

#pragma region DecreaseData
void calglDecreaseDataNode3Db(struct CALNode3Db* node);
void calglDecreaseDataNode3Di(struct CALNode3Di* node);
void calglDecreaseDataNode3Dr(struct CALNode3Dr* node);
#pragma endregion

#pragma region AddData
struct CALNode3Db* calglAddDataNode3Db(struct CALNode3Db* node, struct CALSubstate3Db* substate, enum CALGL_TYPE_INFO typeInfoSubstate, enum CALGL_TYPE_INFO_USE typeInfoUseSubstate, enum CALGL_DATA_TYPE dataType);
struct CALNode3Di* calglAddDataNode3Di(struct CALNode3Di* node, struct CALSubstate3Di* substate, enum CALGL_TYPE_INFO typeInfoSubstate, enum CALGL_TYPE_INFO_USE typeInfoUseSubstate, enum CALGL_DATA_TYPE dataType);
struct CALNode3Dr* calglAddDataNode3Dr(struct CALNode3Dr* node, struct CALSubstate3Dr* substate, enum CALGL_TYPE_INFO typeInfoSubstate, enum CALGL_TYPE_INFO_USE typeInfoUseSubstate, enum CALGL_DATA_TYPE dataType);
#pragma endregion

#pragma region RemoveData
void calglRemoveDataNode3Db(struct CALNode3Db* node, struct CALSubstate3Db* substate);
void calglRemoveDataNode3Di(struct CALNode3Di* node, struct CALSubstate3Di* substate);
void calglRemoveDataNode3Dr(struct CALNode3Dr* node, struct CALSubstate3Dr* substate);
#pragma endregion

#pragma region ShiftLeftFromIndex
void calglShiftLeftFromIndexNode3Db(struct CALNode3Db* node, int index);
void calglShiftLeftFromIndexNode3Di(struct CALNode3Di* node, int index);
void calglShiftLeftFromIndexNode3Dr(struct CALNode3Dr* node, int index);
#pragma endregion

#pragma region GetFather
struct CALNode3Db* calglGetFatherNode3Db(struct CALNode3Db* node);
struct CALNode3Di* calglGetFatherNode3Di(struct CALNode3Di* node);
struct CALNode3Dr* calglGetFatherNode3Dr(struct CALNode3Dr* node);
#pragma endregion

#pragma region SetNoData
void calglSetNoDataToNode3Db(struct CALNode3Db* node, CALbyte noData);
void calglSetNoDataToNode3Di(struct CALNode3Di* node, CALint noData);
void calglSetNoDataToNode3Dr(struct CALNode3Dr* node, CALreal noData);
#pragma endregion

#endif

