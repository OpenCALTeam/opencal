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

struct CALWindow3D {
	GLuint id;
	struct CALGLGlobalSettings* globalSettings;
	// Sub Window
	GLint noModels;
	struct CALDrawModel3D** models;
	GLuint* subWindowID;
	GLint* positionsX;
	GLint* positionsY;
	GLint sub_width;
	GLint sub_height;

	GLfloat backgroundColor[4];
	GLfloat backgroundSubColor[4];
	GLvoid* font_style;
};

/*! Constructor
*/
struct CALWindow3D* calglCreateWindow3D(int argc, char** argv, struct CALGLGlobalSettings* globalSettings, struct CALDrawModel3D** models, int size);

/*! Destructor
*/
void calglDestroyWindow3D(struct CALWindow3D* window);

/*! 
*/
void calglRedisplayAllWindow3D(void);

/*! 
*/
void calglDisplayWindow3D(void);

/*! 
*/
void calglReshapeWindow3D(int w, int h);

/*! 
*/
void calglSubDisplayWindow3D(void);

/*! 
*/
void calglSubReshapeWindow3D(int w, int h);

/*! 
*/
void calglCalculatePositionAndDimensionWindow3D(struct CALWindow3D* window);

/*! 
*/
void calglStartProcessWindow3D(int argc, char** argv);

void calglSetfontWindow3D(struct CALWindow3D* window, char* name, int size);

void calglDrawStringWindow3D(struct CALWindow3D* window, GLuint x, GLuint y, char* format, ...);

void calglSpecialKeyboardEventWindow3D(int key, int x, int y);

void calglKeyboardEventWindow3D(unsigned char key, int x, int y);

void calglKeyboardUpEventWindow3D(unsigned char key, int x, int y);

void calglMouseWindow3D(int button, int state, int x, int y);

void calglMotionMouseWindow3D(int x, int y);

void calglIdleFuncWindow3D(void);

void calglTimeFunc3D(int value);

void calglCleanDrawModelList3D();

void calglShowModel3D(struct CALDrawModel3D* model);

void calglIncreaseDrawModel3D();

void calglPrintfInfoCommand3D();

void calglDisplayBar3D(struct CALGLInfoBar* infoBar);

char* calglGetString3D(GLdouble number);

void calglPrintString3D(GLfloat x, GLfloat y, char *string);

void calglPrintConstString3D(GLfloat x, GLfloat y, const char *string);

#endif
