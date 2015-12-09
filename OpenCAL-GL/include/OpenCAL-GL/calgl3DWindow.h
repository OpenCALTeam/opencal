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
struct CALWindow3D {
	GLuint id;										//!< Id of the main window.
	struct CALGLGlobalSettings* globalSettings;		//!< Reference to global settings.
	GLint noModels;									//!< No of models for which a no of sub window must be created.
	struct CALDrawModel3D** models;					//!< List of models.
	GLuint* subWindowID;							//!< List of sub windows ids.
	GLint* positionsX;								//!< List of sub windows x positions.
	GLint* positionsY;								//!< List of sub windows y positions.
	GLint sub_width;								//!< Width dimension of the sub windows.
	GLint sub_height;								//!< Height dimension of the sub windows.
	GLvoid* font_style;								//!< Font style.
};

/*! \brief Constructor for create a window.
*/
struct CALWindow3D* calglCreateWindow3D(
	int argc,									//!< argc value passed from Main func.
	char** argv,								//!< argv value passed from Main func.
struct CALGLGlobalSettings* globalSettings, //!< Reference to global settings.
struct CALDrawModel3D** models,				//!< List of models to draw.
	int size									//!< List size of models to draw.
	);

/*! \brief Destructor for de-allocate memory.
*/
void calglDestroyWindow3D(
struct CALWindow3D* window	//!< Window to destroy.
	);

/*! \brief Function for redisplay the main window and all sub windows.
*/
void calglRedisplayAllWindow3D(void);

/*! \brief Display main window callback.
*/
void calglDisplayWindow3D(void);

/*! \brief Reshape main window callback.
*/
void calglReshapeWindow3D(int w, int h);

/*! \brief Display sub window callback.
*/
void calglSubDisplayWindow3D(void);

/*! \brief Reshape sub window callback.
*/
void calglSubReshapeWindow3D(int w, int h);

/*! \brief Function for auto create positions and dimensions of all sub windows.
*/
void calglCalculatePositionAndDimensionWindow3D(
struct CALWindow3D* window	//!< Pointer to window.
	);

/*! \brief Function that start the window system.
*/
void calglMainLoop3D(
	int argc,	//!< argc value passed from Main func.
	char** argv	//!< argv value passed from Main func.
	);

/*! \brief Function that set the font.
*/
void calglSetfontWindow3D(
struct CALWindow3D* window,	//!< Pointer to window.
	char* name, 				//!< Font name.
	int size					//!< Font size.
	);

/*! \brief Function that print a string on the screen of the relative sub window.
*/
void calglDrawStringWindow3D(
struct CALWindow3D* window,	//!< Pointer to window. 
	GLuint x, 					//!< X position to print.
	GLuint y, 					//!< Y position to print.
	char* format, 				//!< String to print.
	...
	);

/*! \brief Special keyboard callback.
*/
void calglSpecialKeyboardEventWindow3D(int key, int x, int y);

/*! \brief Keyboard callback.
*/
void calglKeyboardEventWindow3D(unsigned char key, int x, int y);

/*! \brief Keyboard up callback.
*/
void calglKeyboardUpEventWindow3D(unsigned char key, int x, int y);

/*! \brief Mouse callback.
*/
void calglMouseWindow3D(int button, int state, int x, int y);

/*! \brief Motion mouse callback.
*/
void calglMotionMouseWindow3D(int x, int y);

/*! \brief Idle function callback.
*/
void calglIdleFuncWindow3D(void);

/*! \brief Timer function callback.
*/
void calglTimeFunc3D(int value);

/*! \brief Destroy all models from the list of models.
*/
void calglCleanDrawModelList3D();

/*! \brief Insert a model in the list of models.
*/
void calglShowModel3D(
struct CALDrawModel3D* model	//!< Pointer to CALDrawModel3D.
	);

/*! \brief Increase the capacity of the list of models.
*/
void calglIncreaseDrawModel3D();

/*! \brief Print on the console some utility informations.
*/
void calglPrintfInfoCommand3D();

/*! \brief Function for render the information bar.
*/
void calglDisplayBar3D(
struct CALGLInfoBar* infoBar	//!< Pointer to CALGLInfoBar.
	);

/*! \brief Utility function for getting a string from a number.
*/
char* calglGetString3D(
	GLdouble number	//!< Number to convert to string.
	);

/*! \brief Function that print a string on the screen of the relative sub window.
*/
void calglPrintString3D(
	GLfloat x,		//!< X position to print.
	GLfloat y,		//!< Y position to print.
	char *string	//!< String to print.
	);

/*! \brief Function that print a const string on the screen of the relative sub window.
*/
void calglPrintConstString3D(
	GLfloat x,			//!< X position to print.
	GLfloat y,			//!< Y position to print.
	const char *string	//!< String to print.
	);

/*! \brief Function that set horizontal/vertical layout when there are only two models to display.
*/
void calglSetLayoutOrientation3D(
enum CALGL_LAYOUT_ORIENTATION newOrientation	//!< Type of orientation.
	);

#endif
