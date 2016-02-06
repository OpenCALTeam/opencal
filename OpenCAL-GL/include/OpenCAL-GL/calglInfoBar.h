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
struct CALGLDrawModel2D;
struct CALGLDrawModel3D;

/*! \brief Structure that model the concept of the information bar.
	It is used to display information about the substate of reference.
*/
struct CALGLInfoBar{
	const char* substateName;						//!< Infomation bar's name.
	enum CALGL_TYPE_INFO_USE infoUse;				//!< Color gradient that must be used.
	GLdouble* min;									//!< Minimum value of the bar.
	GLdouble* max;									//!< Maximum value of the bar.
	GLfloat xPosition;								//!< X position value of the bar.
	GLfloat yPosition;								//!< Y position value of the bar.
	GLint width;									//!< Width value of the bar.
	GLint height;									//!< Heigth value of the bar.
	enum CALGL_INFO_BAR_DIMENSION dimension;		//!< Type of dimension (absolute or relative).
	enum CALGL_INFO_BAR_ORIENTATION orientation;	//!< Type of orientation (vertical or horizontal).
	CALbyte barInitialization;						//!< Boolean value which tell if a bar has been instantiated.
	GLint constWidth;								//!< Const width for relative bar.
	GLint constHeight;								//!< Const height for relative bar.
};

/*! \brief Constructor for a relative information bar.
	This is for a 2D byte substate.
*/
struct CALGLInfoBar* calglCreateRelativeInfoBar2Db(
	const char* substateName,						//!< Infomation bar's name.
	enum CALGL_TYPE_INFO_USE infoUse,				//!< Color gradient that must be used.
	struct CALGLDrawModel2D* calDrawModel,			//!< Pointer to the structure for which retrive the substate. 
	struct CALSubstate2Db* substate,				//!< Pointer to the substate for which display information.
	enum CALGL_INFO_BAR_ORIENTATION orientation		//!< Type of orientation (vertical or horizontal).
	);
/*! \brief Constructor for a relative information bar.
	This is for a 2D int substate.
*/
struct CALGLInfoBar* calglCreateRelativeInfoBar2Di(
	const char* substateName, 						//!< Infomation bar's name.
	enum CALGL_TYPE_INFO_USE infoUse, 				//!< Color gradient that must be used.
	struct CALGLDrawModel2D* calDrawModel, 			//!< Pointer to the structure for which retrive the substate. 
	struct CALSubstate2Di* substate, 				//!< Pointer to the substate for which display information.
	enum CALGL_INFO_BAR_ORIENTATION orientation		//!< Type of orientation (vertical or horizontal).
	);
/*! \brief Constructor for a relative information bar.
	This is for a 2D real substate.
*/
struct CALGLInfoBar* calglCreateRelativeInfoBar2Dr(
	const char* substateName, 						//!< Infomation bar's name.
	enum CALGL_TYPE_INFO_USE infoUse, 				//!< Color gradient that must be used.
	struct CALGLDrawModel2D* calDrawModel, 			//!< Pointer to the structure for which retrive the substate. 
	struct CALSubstate2Dr* substate, 				//!< Pointer to the substate for which display information.
	enum CALGL_INFO_BAR_ORIENTATION orientation		//!< Type of orientation (vertical or horizontal).
	);
/*! \brief Constructor for a relative information bar.
	This is for a 3D byte substate.
*/
struct CALGLInfoBar* calglCreateRelativeInfoBar3Db(
	const char* substateName, 						//!< Infomation bar's name.
	enum CALGL_TYPE_INFO_USE infoUse, 				//!< Color gradient that must be used.
	struct CALGLDrawModel3D* calDrawModel, 			//!< Pointer to the structure for which retrive the substate. 
	struct CALSubstate3Db* substate, 				//!< Pointer to the substate for which display information.
	enum CALGL_INFO_BAR_ORIENTATION orientation		//!< Type of orientation (vertical or horizontal).
	);
/*! \brief Constructor for a relative information bar.
	This is for a 3D int substate.
*/
struct CALGLInfoBar* calglCreateRelativeInfoBar3Di(
	const char* substateName, 						//!< Infomation bar's name.
	enum CALGL_TYPE_INFO_USE infoUse, 				//!< Color gradient that must be used.
	struct CALGLDrawModel3D* calDrawModel, 			//!< Pointer to the structure for which retrive the substate. 
	struct CALSubstate3Di* substate, 				//!< Pointer to the substate for which display information.
	enum CALGL_INFO_BAR_ORIENTATION orientation		//!< Type of orientation (vertical or horizontal).
	);
/*! \brief Constructor for a relative information bar.
	This is for a 3D real substate.
*/
struct CALGLInfoBar* calglCreateRelativeInfoBar3Dr(
	const char* substateName, 						//!< Infomation bar's name.
	enum CALGL_TYPE_INFO_USE infoUse, 				//!< Color gradient that must be used.
	struct CALGLDrawModel3D* calDrawModel, 			//!< Pointer to the structure for which retrive the substate. 
	struct CALSubstate3Dr* substate, 				//!< Pointer to the substate for which display information.
	enum CALGL_INFO_BAR_ORIENTATION orientation		//!< Type of orientation (vertical or horizontal).
	);
/*! \brief Constructor for an absolute information bar.
	This is for a 2D byte substate.
*/
struct CALGLInfoBar* calglCreateInfoBar2Db(
	const char* substateName, 						//!< Infomation bar's name.
	enum CALGL_TYPE_INFO_USE infoUse, 				//!< Color gradient that must be used.
	struct CALGLDrawModel2D* calDrawModel, 			//!< Pointer to the structure for which retrive the substate. 
	struct CALSubstate2Db* substate, 				//!< Pointer to the substate for which display information.
	GLfloat xPosition,								//!< X position value of the bar.
	GLfloat yPosition, 								//!< Y position value of the bar.
	GLint width,									//!< Width value of the bar.
	GLint height									//!< Heigth value of the bar.
	);
/*! \brief Constructor for an absolute information bar.
	This is for a 2D int substate.
*/
struct CALGLInfoBar* calglCreateInfoBar2Di(
	const char* substateName, 						//!< Infomation bar's name.
	enum CALGL_TYPE_INFO_USE infoUse, 				//!< Color gradient that must be used.
	struct CALGLDrawModel2D* calDrawModel, 			//!< Pointer to the structure for which retrive the substate. 
	struct CALSubstate2Di* substate,				//!< Pointer to the substate for which display information.
	GLfloat xPosition, 								//!< X position value of the bar.
	GLfloat yPosition, 								//!< Y position value of the bar. 
	GLint width, 									//!< Width value of the bar.
	GLint height									//!< Heigth value of the bar.
	);
/*! \brief Constructor for an absolute information bar.
	This is for a 2D real substate.
*/
struct CALGLInfoBar* calglCreateInfoBar2Dr(
	const char* substateName, 						//!< Infomation bar's name.
	enum CALGL_TYPE_INFO_USE infoUse, 				//!< Color gradient that must be used.
	struct CALGLDrawModel2D* calDrawModel, 			//!< Pointer to the structure for which retrive the substate. 
	struct CALSubstate2Dr* substate, 				//!< Pointer to the substate for which display information.
	GLfloat xPosition, 								//!< X position value of the bar.
	GLfloat yPosition, 								//!< Y position value of the bar.
	GLint width, 									//!< Width value of the bar.
	GLint height									//!< Heigth value of the bar.
	);
/*! \brief Constructor for an absolute information bar.
	This is for a 3D byte substate.
*/
struct CALGLInfoBar* calglCreateInfoBar3Db(
	const char* substateName, 						//!< Infomation bar's name.
	enum CALGL_TYPE_INFO_USE infoUse, 				//!< Color gradient that must be used.
	struct CALGLDrawModel3D* calDrawModel, 			//!< Pointer to the structure for which retrive the substate. 
	struct CALSubstate3Db* substate, 				//!< Pointer to the substate for which display information.
	GLfloat xPosition, 								//!< X position value of the bar.
	GLfloat yPosition, 								//!< Y position value of the bar.
	GLint width, 									//!< Width value of the bar.
	GLint height									//!< Heigth value of the bar.
	);
/*! \brief Constructor for an absolute information bar.
	This is for a 3D int substate.
*/
struct CALGLInfoBar* calglCreateInfoBar3Di(
	const char* substateName, 						//!< Infomation bar's name.
	enum CALGL_TYPE_INFO_USE infoUse, 				//!< Color gradient that must be used.
	struct CALGLDrawModel3D* calDrawModel, 			//!< Pointer to the structure for which retrive the substate. 
	struct CALSubstate3Di* substate, 				//!< Pointer to the substate for which display information.
	GLfloat xPosition, 								//!< X position value of the bar.
	GLfloat yPosition, 								//!< Y position value of the bar.
	GLint width, 									//!< Width value of the bar.
	GLint height									//!< Heigth value of the bar.
	);
/*! \brief Constructor for an absolute information bar.
	This is for a 3D real substate.
*/
struct CALGLInfoBar* calglCreateInfoBar3Dr(
	const char* substateName, 						//!< Infomation bar's name.
	enum CALGL_TYPE_INFO_USE infoUse, 				//!< Color gradient that must be used.
	struct CALGLDrawModel3D* calDrawModel, 			//!< Pointer to the structure for which retrive the substate. 
	struct CALSubstate3Dr* substate, 				//!< Pointer to the substate for which display information.
	GLfloat xPosition, 								//!< X position value of the bar.
	GLfloat yPosition, 								//!< Y position value of the bar.
	GLint width, 									//!< Width value of the bar.
	GLint height									//!< Heigth value of the bar.
	);

/*! \brief Destructor for de-allocate memory allocated before.
*/
void calglDestroyInfoBar(
	struct CALGLInfoBar* infoBar //!< Pointer to object to destroy.
	);

/*! \brief Function for setting constant width and height for relative bar.
*/
void calglSetInfoBarConstDimension(
	struct CALGLInfoBar* infoBar,	//!< Infomation bar's pointer.
	GLfloat width,					//!< Infomation bar's const width.
	GLfloat height					//!< Infomation bar's const height.
	);

#endif
