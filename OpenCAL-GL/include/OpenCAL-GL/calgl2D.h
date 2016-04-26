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
#include <OpenCAL-GL/calgl2DRun.h>
#include <OpenCAL-GL/calglInfoBar.h>
#include <OpenCAL-GL/calglGlobalSettings.h>

/*! \brief Structure that contains informations regarding the drawing type.
	Internally it contains a hierarchy tree of substates.
	In which each substate has several pointer to the other substates.
	This is for 2D type of cellular automata.
	*/
struct CALGLDrawModel2D {
	const char* name;							//!< Name of the drawing model.
	enum CALGL_DRAW_MODE drawMode;				//!< Type of the drawing mode.
	struct CALNode2Db* byteModel;				//!< Pointer to the byte type node tree that contains informations of the cellular automata's substate.
	struct CALNode2Di* intModel;				//!< Pointer to the int type node tree that contains informations of the cellular automata's substate.
	struct CALNode2Dr* realModel;				//!< Pointer to the real type node tree that contains informations of the cellular automata's substate.
	struct CALModel2D* calModel;				//!< Pointer to the cellular automata.
	struct CALGLModelViewParameter* modelView;	//!< Pointer to the model view matrix transformation.
	struct CALGLLightParameter* modelLight;		//!< Pointer to the lights parameters.
	struct CALGLRun2D* calglRun;			//!< Pointer to the object that update the cellular automata
	struct CALGLInfoBar* infoBar;				//!< Pointer to the information bar that contains informations releated to the reference sub-state.
	GLshort* drawICells;						//!< Array of value for the design of the automaton slices, in this case the rows.
	GLshort* drawJCells;						//!< Array of value for the design of the automaton slices, in this case the columns.
	GLfloat redComponent;						//!< Component for the channel of the red color, this is used for a const color.
	GLfloat greenComponent;						//!< Component for the channel of the blue color, this is used for a const color.
	GLfloat blueComponent;						//!< Component for the channel of the green color, this is used for a const color.
	GLfloat alphaComponent;						//!< Component for the alpha channel, this is used for a const color.
	int moving;									//!< Component for control the drawing of the model when the user is moving the automata.
	GLfloat heightOffset;						//!< Component that specifies the offset between two substates.
};

/*! \brief Constructor for creating the drawing model
*/
DllExport
struct CALGLDrawModel2D* calglDefDrawModel2D(
	enum CALGL_DRAW_MODE mode,		//!< Type of drawing (es. FLAT, SURFACE).
	const char* name,				//!< Name of the drawing model.
	struct CALModel2D* calModel,	//!< Pointer to the cellular automata.
	struct CALRun2D* calRun		//!< Pointer to the struct CALRun2D.
	);


/*! \brief Constructor for creating the drawing model
*/
DllExport
struct CALGLDrawModel2D* calglDefDrawModelCL2D(
	enum CALGL_DRAW_MODE mode,		//!< Type of drawing (es. FLAT, SURFACE).
	const char* name,				//!< Name of the drawing model.
	struct CALModel2D* calModel,	//!< Pointer to the cellular automata.
	struct CALGLRun2D* calglRun  //!< Pointer to the struct CALGLRun2D.
	);

/*! \brief Destructor for destroying the drawing model
*/
DllExport
void calglDestoyDrawModel2D(
	struct CALGLDrawModel2D* drawModel	//!< Pointer to the CALDrawModel to destroy.
	);

#pragma region AddData
/*! \brief Add data to the byte drawing model.
*/
DllExport
void calglAdd2Db(
	struct CALGLDrawModel2D* drawModel,				//!< The CALDrawModel to which adding data.
	struct CALSubstate2Db* substateFather,			//!< The substate father to which add the new data.
	struct CALSubstate2Db** substateToAdd,			//!< The new data to add.
	enum CALGL_TYPE_INFO typeInfo,					//!< The information that the new substate contains.
	enum CALGL_TYPE_INFO_USE typeInfoUseSubstate,	//!< Information about the mode in which the data had to be used.
	enum CALGL_DATA_TYPE dataType					//!< Specify if the data will change (Dynamic data) or will not (Static data).
	);
/*! \brief Add data to the int drawing model.
*/
DllExport
void calglAdd2Di(
	struct CALGLDrawModel2D* drawModel,				//!< The CALDrawModel to which adding data.
	struct CALSubstate2Di* substateFather,			//!< The substate father to which add the new data.
	struct CALSubstate2Di** substateToAdd,			//!< The new data to add.
	enum CALGL_TYPE_INFO typeInfo,					//!< The information that the new substate contains.
	enum CALGL_TYPE_INFO_USE typeInfoUseSubstate,	//!< Information about the mode in which the data had to be used.
	enum CALGL_DATA_TYPE dataType					//!< Specify if the data will change (Dynamic data) or will not (Static data).
	);
/*! \brief Add data to the real drawing model.
*/
DllExport
void calglAdd2Dr(
	struct CALGLDrawModel2D* drawModel,				//!< The CALDrawModel to which adding data.
	struct CALSubstate2Dr* substateFather,			//!< The substate father to which add the new data.
	struct CALSubstate2Dr** substateToAdd,			//!< The new data to add.
	enum CALGL_TYPE_INFO typeInfo,					//!< The information that the new substate contains.
	enum CALGL_TYPE_INFO_USE typeInfoUseSubstate,	//!< Information about the mode in which the data had to be used.
	enum CALGL_DATA_TYPE dataType					//!< Specify if the data will change (Dynamic data) or will not (Static data).
	);
#pragma endregion

#pragma region SearchSubstate
/*! \brief Functions that search a node in the byte hierarchy tree.
*/
DllExport
void calglSearchSubstateDrawModel2Db(
	struct CALNode2Db* currentNode,				//!< The hierarchy tree that contains the node to search.
	struct CALSubstate2Db* substateToSearch,	//!< The substate to search.
	struct CALNode2Db** nodeSearched			//!< The node searched. NULL if it doesn't exists.
	);
/*! \brief Functions that search a node in the int hierarchy tree.
*/
DllExport
void calglSearchSubstateDrawModel2Di(
	struct CALNode2Di* currentNode,				//!< The hierarchy tree that contains the node to search.
	struct CALSubstate2Di* substateToSearch,	//!< The substate to search.
	struct CALNode2Di** nodeSearched			//!< The node searched. NULL if it doesn't exists.
	);
/*! \brief Functions that search a node in the real hierarchy tree.
*/
DllExport
void calglSearchSubstateDrawModel2Dr(
	struct CALNode2Dr* currentNode,				//!< The hierarchy tree that contains the node to search.
	struct CALSubstate2Dr* substateToSearch,	//!< The substate to search.
	struct CALNode2Dr** nodeSearched			//!< The node searched. NULL if it doesn't exists.
	);
#pragma endregion

/*! \brief Functions that display the model specified.
	It must be called into the OpenGL display function callback.
	*/
DllExport
void calglDisplayModel2D(
	struct CALGLDrawModel2D* calDrawModel	//!< The CALDrawModel to display.
	);

#pragma region DrawDiscreetModel2D
/*! \brief Functions for the discreet displaying.
	It is called by the calglDisplayModel.
	*/
DllExport
void calglDrawDiscreetModel2D(
	struct CALGLDrawModel2D* calDrawModel	//!< The CALDrawModel to display.
	);

/*! \brief Functions for the byte discreet displaying.
	It recursively display all node of the hierarchy tree.
	*/
DllExport
void calglDrawDiscreetModelDisplayNode2Db(
	struct CALGLDrawModel2D* calDrawModel,	//!< The CALDrawModel to display.
	struct CALNode2Db* calNode				//!< The node from which begin the drawing.
	);
/*! \brief Functions for the int discreet displaying.
	It recursively display all node of the hierarchy tree.
	*/
DllExport
void calglDrawDiscreetModelDisplayNode2Di(
	struct CALGLDrawModel2D* calDrawModel,	//!< The CALDrawModel to display.
	struct CALNode2Di* calNode				//!< The node from which begin the drawing.
	);
/*! \brief Functions for the real discreet displaying.
	It recursively display all node of the hierarchy tree.
	*/
DllExport
void calglDrawDiscreetModelDisplayNode2Dr(
	struct CALGLDrawModel2D* calDrawModel,	//!< The CALDrawModel to display.
	struct CALNode2Dr* calNode				//!< The node from which begin the drawing.
	);

/*! \brief Functions for the byte discreet displaying of the current node.
	This function is recursively called for all node in the hierarchy tree by the calglDrawDiscreetModelDisplayNode2Db
	*/
DllExport
void calglDrawDiscreetModelDisplayCurrentNode2Db(
	struct CALGLDrawModel2D* calDrawModel,	//!< The CALDrawModel to display.
	struct CALNode2Db* calNode				//!< The current node to drawing.
	);
/*! \brief Functions for the byte discreet displaying of the current node.
	This function is recursively called for all node in the hierarchy tree by the calglDrawDiscreetModelDisplayNode2Di
	*/
DllExport
void calglDrawDiscreetModelDisplayCurrentNode2Di(
	struct CALGLDrawModel2D* calDrawModel,	//!< The CALDrawModel to display.
	struct CALNode2Di* calNode				//!< The current node to drawing.
	);
/*! \brief Functions for the byte discreet displaying of the current node.
	This function is recursively called for all node in the hierarchy tree by the calglDrawDiscreetModelDisplayNode2Dr
	*/
DllExport
void calglDrawDiscreetModelDisplayCurrentNode2Dr(
	struct CALGLDrawModel2D* calDrawModel,	//!< The CALDrawModel to display.
	struct CALNode2Dr* calNode				//!< The current node to drawing.
	);
#pragma endregion

#pragma region DrawRealModel2D
/*! \brief Functions for the real displaying.
	It is called by the calglDisplayModel.
	*/
DllExport
void calglDrawRealModel2D(
	struct CALGLDrawModel2D* calDrawModel	//!< The CALDrawModel to display.
	);

/*! \brief Functions for the byte real displaying.
	It recursively display all node of the hierarchy tree.
	*/
DllExport
void calglDrawRealModelDisplayNode2Db(
	struct CALGLDrawModel2D* calDrawModel,	//!< The CALDrawModel to display.
	struct CALNode2Db* calNode				//!< The node from which begin the drawing.
	);
/*! \brief Functions for the int real displaying.
	It recursively display all node of the hierarchy tree.
	*/
DllExport
void calglDrawRealModelDisplayNode2Di(
	struct CALGLDrawModel2D* calDrawModel,	//!< The CALDrawModel to display.
	struct CALNode2Di* calNode				//!< The node from which begin the drawing.
	);
/*! \brief Functions for the real real displaying.
	It recursively display all node of the hierarchy tree.
	*/
DllExport
void calglDrawRealModelDisplayNode2Dr(
	struct CALGLDrawModel2D* calDrawModel,	//!< The CALDrawModel to display.
	struct CALNode2Dr* calNode				//!< The node from which begin the drawing.
	);

/*! \brief Functions for the byte real displaying of the current node.
	This function is recursively called for all node in the hierarchy tree by the calglDrawRealModelDisplayNode2Db
	*/
DllExport
void calglDrawRealModelDisplayCurrentNode2Db(
	struct CALGLDrawModel2D* calDrawModel,	//!< The CALDrawModel to display.
	struct CALNode2Db* calNode				//!< The current node to drawing.
	);
/*! \brief Functions for the int real displaying of the current node.
	This function is recursively called for all node in the hierarchy tree by the calglDrawRealModelDisplayNode2Di
	*/
DllExport
void calglDrawRealModelDisplayCurrentNode2Di(
	struct CALGLDrawModel2D* calDrawModel,	//!< The CALDrawModel to display.
	struct CALNode2Di* calNode				//!< The current node to drawing.
	);
/*! \brief Functions for the real real displaying of the current node.
	This function is recursively called for all node in the hierarchy tree by the calglDrawRealModelDisplayNode2Dr
	*/
DllExport
void calglDrawRealModelDisplayCurrentNode2Dr(
	struct CALGLDrawModel2D* calDrawModel,	//!< The CALDrawModel to display.
	struct CALNode2Dr* calNode				//!< The current node to drawing.
	);

/*! \brief Functions that specify, for the current row and column (i, j) in the current substate, the vertex.
	This is for byte substate.
	*/
DllExport
void calglDrawRealModelDisplayCurrentNodeSetVertexData2Db(
	struct CALGLDrawModel2D* calDrawModel,	//!< The CALDrawModel to display.
	struct CALNode2Db*calNode,				//!< The current node to drawing.
	GLint i,								//!< The current row.
	GLint j									//!< The current column.
	);
/*! \brief Functions that specify, for the current row and column (i, j) in the current substate, the vertex.
	This is for int substate.
	*/
DllExport
void calglDrawRealModelDisplayCurrentNodeSetVertexData2Di(
	struct CALGLDrawModel2D* calDrawModel,	//!< The CALDrawModel to display.
	struct CALNode2Di*calNode,				//!< The current node to drawing.
	GLint i,								//!< The current row.
	GLint j									//!< The current column.
	);
/*! \brief Functions that specify, for the current row and column (i, j) in the current substate, the vertex.
	This is for real substate.
	*/
DllExport
void calglDrawRealModelDisplayCurrentNodeSetVertexData2Dr(
	struct CALGLDrawModel2D* calDrawModel,	//!< The CALDrawModel to display.
	struct CALNode2Dr*calNode,				//!< The current node to drawing.
	GLint i,								//!< The current row.
	GLint j									//!< The current column.
	);
#pragma endregion

#pragma region ComputeExtremes
/*! \brief Functions that calculate the minimum and maximum value for the specified substate.
	This is for byte substate.
	*/
DllExport
void calglComputeExtremesDrawModel2Db(
	struct CALGLDrawModel2D* calDrawModel,	//!< The pointer to CALDrawModel.
	struct CALNode2Db* calNode,				//!< The pointer to the substate for calculating the minimum and maximum.
	GLdouble* m,							//!< The minimum value.
	GLdouble* M								//!< The maximum value.
	);
/*! \brief Functions that calculate the minimum and maximum value for the specified substate.
	This is for int substate.
	*/
DllExport
void calglComputeExtremesDrawModel2Di(
	struct CALGLDrawModel2D* calDrawModel, 	//!< The pointer to CALDrawModel.
	struct CALNode2Di* calNode,				//!< The pointer to the substate for calculating the minimum and maximum.
	GLdouble* m,							//!< The minimum value.
	GLdouble* M								//!< The maximum value.
	);
/*! \brief Functions that calculate the minimum and maximum value for the specified substate.
	This is for real substate.
	*/
DllExport
void calglComputeExtremesDrawModel2Dr(
	struct CALGLDrawModel2D* calDrawModel, 	//!< The pointer to CALDrawModel.
	struct CALNode2Dr* calNode,				//!< The pointer to the substate for calculating the minimum and maximum.
	GLdouble* m,							//!< The minimum value.
	GLdouble* M								//!< The maximum value.
	);
#pragma endregion

#pragma region ComputeExtremesToAll
/*! \brief Functions that calculate the minimum and maximum for all substate in the tree.
	This is for byte substates.
	*/
DllExport
void calglComputeExtremesToAll2Db(
	struct CALGLDrawModel2D* calDrawModel,	//!< The pointer to CALDrawModel.
	struct CALNode2Db* calNode				//!< The pointer to CALNode tree.
	);
/*! \brief Functions that calculate the minimum and maximum for all substate in the tree.
	This is for int substates.
	*/
DllExport
void calglComputeExtremesToAll2Di(
	struct CALGLDrawModel2D* calDrawModel,	//!< The pointer to CALDrawModel.
	struct CALNode2Di* calNode				//!< The pointer to CALNode tree.
	);
/*! \brief Functions that calculate the minimum and maximum for all substate in the tree.
	This is for real substates.
	*/
DllExport
void calglComputeExtremesToAll2Dr(
	struct CALGLDrawModel2D* calDrawModel,	//!< The pointer to CALDrawModel.
	struct CALNode2Dr* calNode				//!< The pointer to CALNode tree.
	);
#pragma endregion

#pragma region SetNormalData
/*! \brief Functions that specify, for the current row and column (i, j) in the current substate, the normal data.
	This is a necessary information for the lights.
	This is for byte substate.
	*/
DllExport
void calglSetNormalData2Db(
	struct CALGLDrawModel2D* calDrawModel,	//!< The CALDrawModel to display.
	struct CALNode2Db* calNode,				//!< The current node to drawing.
	GLint i, 								//!< The current row.
	GLint j									//!< The current column.
	);
/*! \brief Functions that specify, for the current row and column (i, j) in the current substate, the normal data.
	This is a necessary information for the lights.
	This is for int substate.
	*/
DllExport
void calglSetNormalData2Di(
	struct CALGLDrawModel2D* calDrawModel,	//!< The CALDrawModel to display.
	struct CALNode2Di* calNode,				//!< The current node to drawing.
	GLint i, 								//!< The current row.
	GLint j									//!< The current column.
	);
/*! \brief Functions that specify, for the current row and column (i, j) in the current substate, the normal data.
	This is a necessary information for the lights.
	This is for real substate.
	*/
DllExport
void calglSetNormalData2Dr(
	struct CALGLDrawModel2D* calDrawModel,	//!< The CALDrawModel to display.
	struct CALNode2Dr* calNode,				//!< The current node to drawing.
	GLint i, 								//!< The current row.
	GLint j									//!< The current column.
	);
#pragma endregion

#pragma region SetColorData
/*! \brief Functions that specify, for the current row and column (i, j) in the current substate, the color data.
	This is for byte substate.
	*/
DllExport
GLboolean calglSetColorData2Db(
	struct CALGLDrawModel2D* calDrawModel,	//!< The CALDrawModel to display.
	struct CALNode2Db* calNode,				//!< The current node to drawing.
	GLint i, 								//!< The current row.
	GLint j									//!< The current column.
	);
/*! \brief Functions that specify, for the current row and column (i, j) in the current substate, the color data.
	This is for int substate.
	*/
DllExport
GLboolean calglSetColorData2Di(
	struct CALGLDrawModel2D* calDrawModel,	//!< The CALDrawModel to display.
	struct CALNode2Di* calNode,				//!< The current node to drawing.
	GLint i, 								//!< The current row.
	GLint j									//!< The current column.
	);
/*! \brief Functions that specify, for the current row and column (i, j) in the current substate, the color data.
	This is for real substate.
	*/
DllExport
GLboolean calglSetColorData2Dr(
	struct CALGLDrawModel2D* calDrawModel,	//!< The CALDrawModel to display.
	struct CALNode2Dr* calNode,				//!< The current node to drawing.
	GLint i, 								//!< The current row.
	GLint j									//!< The current column.
	);
#pragma endregion

/*! \brief Functions that set a constant color to use in the drawing.
*/
DllExport
void calglColor2D(
struct CALGLDrawModel2D* calDrawModel,	//!< The pointer to CALDrawModel.
	GLfloat redComponent,					//!< The red component
	GLfloat greenComponent, 				//!< The blue component
	GLfloat blueComponent, 					//!< The green component
	GLfloat alphaComponent					//!< The alpha component
	);

/*! \brief Functions that set the model view parameters.
	It contains informations about the translation, rotation and scaling of the model.
	*/
DllExport
void calglSetModelViewParameter2D(
	struct CALGLDrawModel2D* calDrawModel,			//!< The pointer to CALDrawModel.
	struct CALGLModelViewParameter* modelView		//!< The pointer to CALGLModelViewParameter.
	);

/*! \brief Functions that set the light parameters.
	It contains informations about the ambient, diffuse and specular component of the light.
	*/
DllExport
void calglSetLightParameter2D(
	struct CALGLDrawModel2D* calDrawModel,		//!< The pointer to CALDrawModel.
	struct CALGLLightParameter* modelLight		//!< The pointer to CALGLLightParameter.
	);

#pragma region BoundingBox
/*! \brief Functions that in the drawing fase display a bounding box which contains the model.
	This function is called in the discreet drawing.
	*/
DllExport
void calglDrawBoundingSquare2D(
	struct CALGLDrawModel2D* calDrawModel		//!< The pointer to CALDrawModel.
	);
/*! \brief Functions that in the drawing fase display a bounding box which contains the model.
	This function is called in the real drawing.
	*/
DllExport
void calglDrawBoundingBox2D(
	struct CALGLDrawModel2D* calDrawModel,	//!< The pointer to CALDrawModel.
	GLfloat height,							//!< The maximum height of the model.
	GLfloat low								//!< The minimum height of the model.
	);
#pragma endregion

#pragma region InfoBar
/*! \brief Functions that display an information bar which contains informations about a specified substate.
	This is the relative version, in which the display bar is positioned automatically according to the type of orientation.
	This is for byte substate.
	*/
DllExport
void calglRelativeInfoBar2Db(
	struct CALGLDrawModel2D* calDrawModel,		//!< The pointer to CALDrawModel.
	struct CALSubstate2Db* substate, 			//!< The pointer to the substate to which display informations.
	const char* substateName,  					//!< The substate name.
	enum CALGL_TYPE_INFO_USE infoUse,			//!< Enum used to specify the gradient color.
	enum CALGL_INFO_BAR_ORIENTATION orientation	//!< The type of orientation (Vertical, Oriziontal).
	);
/*! \brief Functions that display an information bar which contains informations about a specified substate.
	This is the relative version, in which the display bar is positioned automatically according to the type of orientation.
	This is for byte substate.
	*/
DllExport
void calglRelativeInfoBar2Di(
	struct CALGLDrawModel2D* calDrawModel,		//!< The pointer to CALDrawModel.
	struct CALSubstate2Di* substate,  			//!< The pointer to the substate to which display informations.
	const char* substateName,   				//!< The substate name.
	enum CALGL_TYPE_INFO_USE infoUse, 			//!< Enum used to specify the gradient color.
	enum CALGL_INFO_BAR_ORIENTATION orientation	//!< The type of orientation (Vertical, Oriziontal).
	);
/*! \brief Functions that display an information bar which contains informations about a specified substate.
	This is the relative version, in which the display bar is positioned automatically according to the type of orientation.
	This is for byte substate.
	*/
DllExport
void calglRelativeInfoBar2Dr(
	struct CALGLDrawModel2D* calDrawModel,		//!< The pointer to CALDrawModel.
	struct CALSubstate2Dr* substate, 			//!< The pointer to the substate to which display informations.
	const char* substateName,   				//!< The substate name.
	enum CALGL_TYPE_INFO_USE infoUse, 			//!< Enum used to specify the gradient color.
	enum CALGL_INFO_BAR_ORIENTATION orientation	//!< The type of orientation (Vertical, Oriziontal).
	);
/*! \brief Functions that display an information bar which contains informations about a specified substate.
	This is the absolute version, in which the display bar is positioned according to the position specified by the user.
	This is for byte substate.
	*/
DllExport
void calglInfoBar2Db(
	struct CALGLDrawModel2D* calDrawModel,		//!< The pointer to CALDrawModel.
	struct CALSubstate2Db* substate,  			//!< The pointer to the substate to which display informations.
	const char* substateName,   				//!< The substate name.
	enum CALGL_TYPE_INFO_USE infoUse, 			//!< Enum used to specify the gradient color.
	GLfloat xPosition,							//!< The window x position.
	GLfloat yPosition, 							//!< The window y position.
	GLint width, 								//!< The width dimension.
	GLint height								//!< The height dimension.
	);
/*! \brief Functions that display an information bar which contains informations about a specified substate.
	This is the absolute version, in which the display bar is positioned according to the position specified by the user.
	This is for byte substate.
	*/
DllExport
void calglInfoBar2Di(
	struct CALGLDrawModel2D* calDrawModel,		//!< The pointer to CALDrawModel.
	struct CALSubstate2Di* substate,  			//!< The pointer to the substate to which display informations.
	const char* substateName,   				//!< The substate name.
	enum CALGL_TYPE_INFO_USE infoUse, 			//!< Enum used to specify the gradient color.
	GLfloat xPosition, 							//!< The window x position.
	GLfloat yPosition, 							//!< The window x position.
	GLint width, 								//!< The width dimension.
	GLint height								//!< The height dimension.
	);
/*! \brief Functions that display an information bar which contains informations about a specified substate.
	This is the absolute version, in which the display bar is positioned according to the position specified by the user.
	This is for byte substate.
	*/
DllExport
void calglInfoBar2Dr(
	struct CALGLDrawModel2D* calDrawModel,		//!< The pointer to CALDrawModel.
	struct CALSubstate2Dr* substate,  			//!< The pointer to the substate to which display informations.
	const char* substateName,   				//!< The substate name.
	enum CALGL_TYPE_INFO_USE infoUse, 			//!< Enum used to specify the gradient color.
	GLfloat xPosition, 							//!< The window x position.
	GLfloat yPosition, 							//!< The window x position.
	GLint width, 								//!< The width dimension.
	GLint height								//!< The height dimension.
	);
#pragma endregion

#pragma region DrawIntervals
/*! \brief Functions that set the slices to show of the cellular automata.
	This is for the rows.
	It shows the rows in the interval [min, max].
	*/
DllExport
void calglDisplayDrawIBound2D(
	struct CALGLDrawModel2D* calDrawModel,		//!< The pointer to CALDrawModel.
	GLint min, 									//!< The minimum of the interval.
	GLint max 									//!< The maximum of the interval.
	);
/*! \brief Functions that set the slices to show of the cellular automata.
	This is for the columns.
	It shows the columns in the interval [min, max].
	*/
DllExport
void calglDisplayDrawJBound2D(
	struct CALGLDrawModel2D* calDrawModel,		//!< The pointer to CALDrawModel.
	GLint min, 									//!< The minimum of the interval.
	GLint max 									//!< The maximum of the interval.
	);
/*! \brief Functions that set the slices to hide of the cellular automata.
	This is for the rows.
	It hides the rows in the interval [min, max].
	*/
DllExport
void calglHideDrawIBound2D(
	struct CALGLDrawModel2D* calDrawModel, 		//!< The pointer to CALDrawModel.
	GLint min,									//!< The minimum of the interval.
	GLint max 									//!< The maximum of the interval.
	);
/*! \brief Functions that set the slices to hide of the cellular automata.
	This is for the columns.
	It hides the columns in the interval [min, max].
	*/
DllExport
void calglHideDrawJBound2D(
	struct CALGLDrawModel2D* calDrawModel,		//!< The pointer to CALDrawModel.
	GLint min, 									//!< The minimum of the interval.
	GLint max 									//!< The maximum of the interval.
	);
#pragma endregion

DllExport
void calglSetHeightOffset2D(
	struct CALGLDrawModel2D* calDrawModel,		//!< The pointer to CALDrawModel.
	float heightOffset							//!< The offset between two substates.
	);

#endif
