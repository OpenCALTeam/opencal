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

#ifndef calgl3DWindow_h
#define calgl3DWindow_h

#include <OpenCAL-GL/calgl3D.h>
#include <string.h>

#ifdef __APPLE__
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif

/*! \brief Structure for creating a simple window system to which draw the cellular automata.
This is for 3D type of cellular automata.
*/
struct CALGLWindow3D {
	GLuint id;										//!< Id of the main window.
	struct CALGLGlobalSettings* globalSettings;		//!< Reference to global settings.
	GLint noModels;									//!< No of models for which a no of sub window must be created.
	struct CALGLDrawModel3D** models;					//!< List of models.
	GLuint* subWindowID;							//!< List of sub windows ids.
	GLint* positionsX;								//!< List of sub windows x positions.
	GLint* positionsY;								//!< List of sub windows y positions.
	GLint sub_width;								//!< Width dimension of the sub windows.
	GLint sub_height;								//!< Height dimension of the sub windows.
	GLvoid* font_style;								//!< Font style.
};

/*! \brief Constructor for create a window.
*/
DllExport
struct CALGLWindow3D* calglCreateWindow3D(
	int argc,									//!< argc value passed from Main func.
	char** argv,								//!< argv value passed from Main func.
	struct CALGLGlobalSettings* globalSettings, //!< Reference to global settings.
	struct CALGLDrawModel3D** models,				//!< List of models to draw.
	int size									//!< List size of models to draw.
	);

/*! \brief Destructor for de-allocate memory.
*/
DllExport
void calglDestroyWindow3D(
	struct CALGLWindow3D* window	//!< Window to destroy.
	);

/*! \brief Function for redisplay the main window and all sub windows.
*/
DllExport
void calglRedisplayAllWindow3D(void);

/*! \brief Display main window callback.
*/
DllExport
void calglDisplayWindow3D(void);

/*! \brief Reshape main window callback.
*/
DllExport
void calglReshapeWindow3D(int w, int h);

/*! \brief Display sub window callback.
*/
DllExport
void calglSubDisplayWindow3D(void);

/*! \brief Reshape sub window callback.
*/
DllExport
void calglSubReshapeWindow3D(int w, int h);

/*! \brief Function for auto create positions and dimensions of all sub windows.
*/
DllExport
void calglCalculatePositionAndDimensionWindow3D(
	struct CALGLWindow3D* window	//!< Pointer to window.
	);

/*! \brief Function that start the window system.
*/
DllExport
void calglMainLoop3D(
	int argc,	//!< argc value passed from Main func.
	char** argv	//!< argv value passed from Main func.
	);

/*! \brief Function that set the font.
*/
DllExport
void calglSetfontWindow3D(
	struct CALGLWindow3D* window,	//!< Pointer to window.
	char* name, 				//!< Font name.
	int size					//!< Font size.
	);

/*! \brief Function that print a string on the screen of the relative sub window.
*/
DllExport
void calglDrawStringWindow3D(
	struct CALGLWindow3D* window,	//!< Pointer to window.
	GLuint x, 					//!< X position to print.
	GLuint y, 					//!< Y position to print.
	char* format, 				//!< String to print.
	...
	);

/*! \brief Special keyboard callback.
*/
DllExport
void calglSpecialKeyboardEventWindow3D(int key, int x, int y);

/*! \brief Keyboard callback.
*/
DllExport
void calglKeyboardEventWindow3D(unsigned char key, int x, int y);

/*! \brief Keyboard up callback.
*/
DllExport
void calglKeyboardUpEventWindow3D(unsigned char key, int x, int y);

/*! \brief Mouse callback.
*/
DllExport
void calglMouseWindow3D(int button, int state, int x, int y);

/*! \brief Motion mouse callback.
*/
DllExport
void calglMotionMouseWindow3D(int x, int y);

/*! \brief Idle function callback.
*/
DllExport
void calglIdleFuncWindow3D(void);

/*! \brief Timer function callback.
*/
DllExport
void calglTimeFunc3D(int value);

/*! \brief Destroy all models from the list of models.
*/
DllExport
void calglCleanDrawModelList3D();

/*! \brief Insert a model in the list of models.
*/
DllExport
void calglShowModel3D(
	struct CALGLDrawModel3D* model	//!< Pointer to CALGLDrawModel3D.
	);

/*! \brief Increase the capacity of the list of models.
*/
DllExport
void calglIncreaseDrawModel3D();

/*! \brief Print on the console some utility informations.
*/
DllExport
void calglPrintfInfoCommand3D();

/*! \brief Function for render the information bar.
*/
DllExport
void calglDisplayBar3D(
	struct CALGLInfoBar* infoBar	//!< Pointer to CALGLInfoBar.
	);

/*! \brief Utility function for getting a string from a number.
*/
DllExport
char* calglGetString3D(
	GLdouble number	//!< Number to convert to string.
	);

/*! \brief Function that print a string on the screen of the relative sub window.
*/
DllExport
void calglPrintString3D(
	GLfloat x,		//!< X position to print.
	GLfloat y,		//!< Y position to print.
	char *string	//!< String to print.
	);

/*! \brief Function that print a const string on the screen of the relative sub window.
*/
DllExport
void calglPrintConstString3D(
	GLfloat x,			//!< X position to print.
	GLfloat y,			//!< Y position to print.
	const char *string	//!< String to print.
	);

/*! \brief Function that set horizontal/vertical layout when there are only two models to display.
*/
DllExport
void calglSetLayoutOrientation3D(
	enum CALGL_LAYOUT_ORIENTATION newOrientation	//!< Type of orientation.
	);

#endif
