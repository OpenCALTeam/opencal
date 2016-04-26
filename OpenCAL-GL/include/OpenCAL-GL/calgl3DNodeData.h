/*
 * Copyright (c) 2016 OpenCALTeam (https://github.com/OpenCALTeam),
 * Telesio Research Group,
 * Department of Mathematics and Computer Science,
 * University of Calabria, Italy.
 *
 * This file is part of OpenCAL (Open Computing Abstraction Layer).
 *
 * OpenCAL is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3 of
 * the License, or (at your option) any later version.
 *
 * OpenCAL is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with OpenCAL. If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef calgl3DNodeData_h
#define calgl3DNodeData_h

#include <OpenCAL-GL/calgl3D.h>
#include <OpenCAL-GL/calglCommon.h>

#pragma region DataDefinition
/*! \brief Recursive structure that is used to model the concept of hierarchy tree.
	It is used for storing the data that will be used for draw the cellular automata.
	The more relevant details about this structure is that it has a list of pointer to other CALNode structure, where in this list the first pointer is referred to the father and the rest of the pointers are referred to the children of the current node.
	This structure is for 3D byte data.
*/
struct CALNode3Db {
	enum CALGL_DATA_TYPE dataType;					//!< Specify if this node contains static or dynamic data.
	GLuint* callList;								//!< Index used for the display list, where possible (Static data).
	enum CALGL_TYPE_INFO typeInfoSubstate;			//!< Type of data that the node contains (Vertex, color, normal etc.).
	enum CALGL_TYPE_INFO_USE typeInfoUseSubstate;	//!< Specify how to use the color information.
	struct CALSubstate3Db* substate;				//!< The substatewhich the node contains.
	GLdouble min;									//!< Min value of the substate.
	GLdouble max;									//!< Max value of the substate.
	CALbyte noData;									//!< Value that specify a value to ignore if contained in the substate.
	GLfloat redComponent;							//!< Red color component used for constant color.
	GLfloat greenComponent;							//!< Green color component used for constant color.
	GLfloat blueComponent;							//!< Blue color component used for constant color.
	GLfloat alphaComponent;							//!< Alpha component used for constant color.
	int capacityNode;								//!< Max number of children that the node can contains before to increase its dimension.
	int insertedNode;								//!< Number of children that the node contains.
	struct CALNode3Db** nodes; 						//!< List of nodes connected to this, the first is the father the other the children.
};
/*! \brief Recursive structure that is used to model the concept of hierarchy tree.
	It is used for storing the data that will be used for draw the cellular automata.
	The more relevant details about this structure is that it has a list of pointer to other CALNode structure, where in this list the first pointer is referred to the father and the rest of the pointers are referred to the children of the current node.
	This structure is for 3D int data.
*/
struct CALNode3Di {
	enum CALGL_DATA_TYPE dataType;					//!< Specify if this node contains static or dynamic data.
	GLuint* callList;								//!< Index used for the display list, where possible (Static data).
	enum CALGL_TYPE_INFO typeInfoSubstate;			//!< Type of data that the node contains (Vertex, color, normal etc.).
	enum CALGL_TYPE_INFO_USE typeInfoUseSubstate;	//!< Specify how to use the color information.
	struct CALSubstate3Di* substate;				//!< The substatewhich the node contains.
	GLdouble min;									//!< Min value of the substate.
	GLdouble max;									//!< Max value of the substate.
	CALint noData;									//!< Value that specify a value to ignore if contained in the substate.
	GLfloat redComponent;							//!< Red color component used for constant color.
	GLfloat greenComponent;							//!< Green color component used for constant color.
	GLfloat blueComponent;							//!< Blue color component used for constant color.
	GLfloat alphaComponent;							//!< Alpha component used for constant color.
	int capacityNode;								//!< Max number of children that the node can contains before to increase its dimension.
	int insertedNode;								//!< Number of children that the node contains.
	struct CALNode3Di** nodes; 						//!< List of nodes connected to this, the first is the father the other the children.
};
/*! \brief Recursive structure that is used to model the concept of hierarchy tree.
	It is used for storing the data that will be used for draw the cellular automata.
	The more relevant details about this structure is that it has a list of pointer to other CALNode structure, where in this list the first pointer is referred to the father and the rest of the pointers are referred to the children of the current node.
	This structure is for 3D real data.
*/
struct CALNode3Dr {
	enum CALGL_DATA_TYPE dataType;					//!< Specify if this node contains static or dynamic data.
	GLuint* callList;								//!< Index used for the display list, where possible (Static data).
	enum CALGL_TYPE_INFO typeInfoSubstate;			//!< Type of data that the node contains (Vertex, color, normal etc.).
	enum CALGL_TYPE_INFO_USE typeInfoUseSubstate;	//!< Specify how to use the color information.
	struct CALSubstate3Dr* substate;				//!< The substatewhich the node contains.
	GLdouble min;									//!< Min value of the substate.
	GLdouble max;									//!< Max value of the substate.
	CALreal noData;									//!< Value that specify a value to ignore if contained in the substate.
	GLfloat redComponent;							//!< Red color component used for constant color.
	GLfloat greenComponent;							//!< Green color component used for constant color.
	GLfloat blueComponent;							//!< Blue color component used for constant color.
	GLfloat alphaComponent;							//!< Alpha component used for constant color.
	int capacityNode;								//!< Max number of children that the node can contains before to increase its dimension.
	int insertedNode;								//!< Number of children that the node contains.
	struct CALNode3Dr** nodes; 						//!< List of nodes connected to this, the first is the father the other the children.
};
#pragma endregion


#pragma region Create
/*! \brief Function for create a node in which tha father must be specified.
This function is for 3D byte data.
*/
DllExport
struct CALNode3Db* calglCreateNode3Db(
struct CALNode3Db* father	//!< Father node.
	);
/*! \brief Function for create a node in which tha father must be specified.
This function is for 3D int data.
*/
DllExport
struct CALNode3Di* calglCreateNode3Di(
struct CALNode3Di* father	//!< Father node.
	);
/*! \brief Function for create a node in which tha father must be specified.
This function is for 3D real data.
*/
DllExport
struct CALNode3Dr* calglCreateNode3Dr(
	struct CALNode3Dr* father	//!< Father node.
	);
#pragma endregion

#pragma region Destroy
/*! \brief Function for de-allocate memory.
This function is for 3D byte data.
*/
DllExport
void calglDestroyNode3Db(
	struct CALNode3Db* node	//!< Node to destroy.
	);
/*! \brief Function for de-allocate memory.
This function is for 3D int data.
*/
DllExport
void calglDestroyNode3Di(
	struct CALNode3Di* node	//!< Node to destroy.
	);
/*! \brief Function for de-allocate memory.
This function is for 3D real data.
*/
DllExport
void calglDestroyNode3Dr(
	struct CALNode3Dr* node	//!< Node to destroy.
	);
#pragma endregion

#pragma region IncreaseData
/*! \brief Function for increase the capacity of the node.
It is increased of three units.
This function is for 3D byte data.
*/
DllExport
void calglIncreaseDataNode3Db(
	struct CALNode3Db* node	//!< Node to which increase capacity.
	);
/*! \brief Function for increase the capacity of the node.
It is increased of three units.
This function is for 3D int data.
*/
DllExport
void calglIncreaseDataNode3Di(
	struct CALNode3Di* node	//!< Node to which increase capacity.
	);
/*! \brief Function for increase the capacity of the node.
It is increased of three units.
This function is for 3D real data.
*/
DllExport
void calglIncreaseDataNode3Dr(
	struct CALNode3Dr* node	//!< Node to which increase capacity.
	);
#pragma endregion

#pragma region DecreaseData
/*! \brief Function for decrease the capacity of the node.
It is Decreased of three units.
This function is for 3D byte data.
*/
DllExport
void calglDecreaseDataNode3Db(
	struct CALNode3Db* node	//!< Node to which Decrease capacity.
	);
/*! \brief Function for decrease the capacity of the node.
It is Decreased of three units.
This function is for 3D Det data.
*/
DllExport
void calglDecreaseDataNode3Di(
	struct CALNode3Di* node	//!< Node to which Decrease capacity.
	);
/*! \brief Function for decrease the capacity of the node.
It is Decreased of three units.
This function is for 3D real data.
*/
DllExport
void calglDecreaseDataNode3Dr(
	struct CALNode3Dr* node	//!< Node to which Decrease capacity.
	);
#pragma endregion

#pragma region AddData
/*! \brief Function for insert a substate in the hierarchy tree.
This function is for 3D byte data.
*/
DllExport
struct CALNode3Db* calglAddDataNode3Db(
	struct CALNode3Db* node,						//!< Node father.
	struct CALSubstate3Db* substate,				//!< Substate to add.
	enum CALGL_TYPE_INFO typeInfoSubstate,			//!< Type of the information.
	enum CALGL_TYPE_INFO_USE typeInfoUseSubstate,	//!< Color gradient to use.
	enum CALGL_DATA_TYPE dataType					//!< Static or dynamic data.
	);
/*! \brief Function for insert a substate in the hierarchy tree.
This function is for 3D int data.
*/
DllExport
struct CALNode3Di* calglAddDataNode3Di(
	struct CALNode3Di* node, 						//!< Node father.
	struct CALSubstate3Di* substate, 				//!< Substate to add.
	enum CALGL_TYPE_INFO typeInfoSubstate, 			//!< Type of the information.
	enum CALGL_TYPE_INFO_USE typeInfoUseSubstate, 	//!< Color gradient to use.
	enum CALGL_DATA_TYPE dataType                   //!< Static or dynamic data.
	);
/*! \brief Function for insert a substate in the hierarchy tree.
This function is for 3D real data.
*/
DllExport
struct CALNode3Dr* calglAddDataNode3Dr(
	struct CALNode3Dr* node, 						//!< Node father.
	struct CALSubstate3Dr* substate, 				//!< Substate to add.
	enum CALGL_TYPE_INFO typeInfoSubstate, 			//!< Type of the information.
	enum CALGL_TYPE_INFO_USE typeInfoUseSubstate, 	//!< Color gradient to use.
	enum CALGL_DATA_TYPE dataType					//!< Static or dynamic data.
	);
#pragma endregion

#pragma region RemoveData
/*! \brief Function for remove a substate from the children of a node.
This function is for 3D byte data.
*/
DllExport
void calglRemoveDataNode3Db(
	struct CALNode3Db* node,		//!< Node from which remove data.
	struct CALSubstate3Db* substate	//!< Substate relative to the node to remove.
	);
/*! \brief Function for remove a substate from the children of a node.
This function is for 3D int data.
*/
DllExport
void calglRemoveDataNode3Di(
	struct CALNode3Di* node, 		//!< Node from which remove data.
	struct CALSubstate3Di* substate	//!< Substate relative to the node to remove.
	);
/*! \brief Function for remove a substate from the children of a node.
This function is for 3D real data.
*/
DllExport
void calglRemoveDataNode3Dr(
	struct CALNode3Dr* node, 		//!< Node from which remove data.
	struct CALSubstate3Dr* substate	//!< Substate relative to the node to remove.
	);
#pragma endregion

#pragma region ShiftLeftFromIndex
/*! \brief Function that execute a left shift of the children of a node starting from index.
This function is for 3D byte data.
*/
DllExport
void calglShiftLeftFromIndexNode3Db(
	struct CALNode3Db* node,	//!< Node from which shift children.
	int index	 				//!< Index to begin shifting.
	);
/*! \brief Function that execute a left shift of the children of a node starting from index.
This function is for 3D int data.
*/
DllExport
void calglShiftLeftFromIndexNode3Di(
	struct CALNode3Di* node,	//!< Node from which shift children.
	int index	 				//!< Index to begin shifting.
	);
/*! \brief Function that execute a left shift of the children of a node starting from index.
This function is for 3D real data.
*/
DllExport
void calglShiftLeftFromIndexNode3Dr(
	struct CALNode3Dr* node,	//!< Node from which shift children.
	int index	 				//!< Index to begin shifting.
	);
#pragma endregion

#pragma region GetFather
/*! \brief Function that return the father of a node.
This function is for 3D byte data.
*/
DllExport
struct CALNode3Db* calglGetFatherNode3Db(
	struct CALNode3Db* node	//!< Node from which return the father.
	);
/*! \brief Function that return the father of a node.
This function is for 3D int data.
*/
DllExport
struct CALNode3Di* calglGetFatherNode3Di(
	struct CALNode3Di* node	//!< Node from which shift children.
	);
/*! \brief Function that return the father of a node.
This function is for 3D real data.
*/
DllExport
struct CALNode3Dr* calglGetFatherNode3Dr(
	struct CALNode3Dr* node	//!< Node from which shift children.
	);
#pragma endregion

#pragma region SetNoData
/*! \brief Function for set the no data that will be discarded in drawing fase.
This function is for 3D byte data.
*/
DllExport
void calglSetNoDataToNode3Db(
	struct CALNode3Db* node,	//!< Node to which set no data value.
	CALbyte noData				//!< No data value.
	);
/*! \brief Function for set the no data that will be discarded in drawing fase.
This function is for 3D int data.
*/
DllExport
void calglSetNoDataToNode3Di(
	struct CALNode3Di* node, 	//!< Node to which set no data value.
	CALint noData				//!< No data value.
	);
/*! \brief Function for set the no data that will be discarded in drawing fase.
This function is for 3D real data.
*/
DllExport
void calglSetNoDataToNode3Dr(
	struct CALNode3Dr* node, 	//!< Node to which set no data value.
	CALreal noData				//!< No data value.
	);
#pragma endregion

#endif
