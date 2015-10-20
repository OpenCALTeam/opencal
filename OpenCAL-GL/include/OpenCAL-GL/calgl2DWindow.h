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

#ifndef calgl2DWindow_h
#define calgl2DWindow_h

#include <OpenCAL-GL/calgl2D.h>
#include <string.h>

#ifdef __APPLE__
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif

struct CALWindow2D {
	GLuint id;
	struct CALGLGlobalSettings* globalSettings;
	// Sub Window
	GLint noModels;
	struct CALDrawModel2D** models;
	GLuint* subWindowID;
	GLint* positionsX;
	GLint* positionsY;
	GLint sub_width;
	GLint sub_height;

	GLvoid* font_style;
};

/*! Constructor
*/
struct CALWindow2D* calglCreateWindow2D(int argc, char** argv, struct CALGLGlobalSettings* globalSettings, struct CALDrawModel2D** models, int size);

/*! Destructor
*/
void calglDestroyWindow2D(struct CALWindow2D* window);

/*! 
*/
void calglRedisplayAllWindow2D(void);

/*! 
*/
void calglDisplayWindow2D(void);

/*! 
*/
void calglReshapeWindow2D(int w, int h);

/*! 
*/
void calglSubDisplayWindow2D(void);

/*! 
*/
void calglSubReshapeWindow2D(int w, int h);

/*! 
*/
void calglCalculatePositionAndDimensionWindow2D(struct CALWindow2D* window);

/*! 
*/
void calglStartProcessWindow2D(int argc, char** argv);

void calglSetfontWindow2D(struct CALWindow2D* window, char* name, int size);

void calglDrawStringWindow2D(struct CALWindow2D* window, GLuint x, GLuint y, char* format, ...);

void calglSpecialKeyboardEventWindow2D(int key, int x, int y);

void calglKeyboardEventWindow2D(unsigned char key, int x, int y);

void calglKeyboardUpEventWindow2D(unsigned char key, int x, int y);

void calglMouseWindow2D(int button, int state, int x, int y);

void calglMotionMouseWindow2D(int x, int y);

void calglIdleFuncWindow2D(void);

void calglTimeFunc2D(int value);

void calglCleanDrawModelList2D();

void calglShowModel2D(struct CALDrawModel2D* model);

void calglIncreaseDrawModel2D();

void calglPrintfInfoCommand2D();

void calglDisplayBar2D(struct CALGLInfoBar* infoBar);

char* calglGetString2D(GLdouble number);

void calglPrintString2D(GLfloat x, GLfloat y, char *string);

void calglPrintConstString2D(GLfloat x, GLfloat y, const char *string);

#endif
