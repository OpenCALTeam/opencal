/*
 * Copyright (c) 2016 OpenCALTeam (https://github.com/OpenCALTeam),
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

#include <math.h>
#include <OpenCAL-GL/calgl2DWindow.h>
#include <stdio.h>
#include <stdarg.h>

#ifdef __APPLE__
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif

#define SPACE 20
#define SEPARATOR_SPACE 5
#define TRANSLATION_CONSTANT 20

static struct CALGLWindow2D* window2D = NULL;
// Models data
static GLint noModels2D = 0;
static GLint currentModel2D = 0;
static GLint capacityModels2D = 1;
static struct CALGLDrawModel2D** models2D = NULL;
// Worlds Properties
static GLfloat mouseSensitivity = 0.2f;
static GLfloat keyboardSensitivity = 0.02f;
static GLfloat defaultPositionAndRotation = 0.0f;
static GLfloat *xPos = NULL, *yPos = NULL, *zPos = NULL;
static GLfloat *xRot = NULL, *yRot = NULL, *zRot = NULL;
// Mouse And Keyboard data
static GLint oldestY = 0;
static GLint old_x, old_y;
static CALbyte leftPressed = CAL_FALSE;
static CALbyte rightPressed = CAL_FALSE;
static GLint key_old_x, key_old_y;
static CALbyte translationOn = CAL_FALSE;
static GLint activeSubWindow = -1;

static GLfloat fovy = 45.0;

static enum CALGL_LAYOUT_ORIENTATION orientation = CALGL_LAYOUT_ORIENTATION_UNKNOW;

struct CALGLWindow2D* calglCreateWindow2D(int argc, char** argv, struct CALGLGlobalSettings* globalSettings, struct CALGLDrawModel2D** models, int size) {
	struct CALGLWindow2D* window = (struct CALGLWindow2D*) malloc(sizeof(struct CALGLWindow2D));
	GLint i = 0;

	window->globalSettings = globalSettings;

	window->noModels = size;
	window->models = models;
	window->subWindowID = (GLuint*) malloc(sizeof(GLuint)*window->noModels);
	window->positionsX = (GLint*) malloc(sizeof(GLint)*window->noModels);
	window->positionsY = (GLint*) malloc(sizeof(GLint)*window->noModels);

	// Setting parameter for translate
	xPos = (GLfloat*) malloc(sizeof(GLfloat)*size);
	yPos = (GLfloat*) malloc(sizeof(GLfloat)*size);
	zPos = (GLfloat*) malloc(sizeof(GLfloat)*size);
	xRot = (GLfloat*) malloc(sizeof(GLfloat)*size);
	yRot = (GLfloat*) malloc(sizeof(GLfloat)*size);
	zRot = (GLfloat*) malloc(sizeof(GLfloat)*size);

	for(i = 0; i<window->noModels; i++) {
		xPos[i] = 0.0f;	yPos[i] = 0.0f;	zPos[i] = 0.0f;
		xRot[i] = 0.0f; yRot[i] = 0.0f; zRot[i] = 0.0f;
		xRot[i] = 90.0f;
	}

	glutInitWindowSize(window->globalSettings->width, window->globalSettings->height);
	glutInitWindowPosition(window->globalSettings->positionX, window->globalSettings->positionY);
	window->id = glutCreateWindow(window->globalSettings->applicationName);
	glutReshapeFunc(calglReshapeWindow2D);
	glutDisplayFunc(calglDisplayWindow2D);
	glutIdleFunc(calglIdleFuncWindow2D);
	glutTimerFunc(calglGetGlobalSettings()->refreshTime, calglTimeFunc2D, calglGetGlobalSettings()->refreshTime);

	calglCalculatePositionAndDimensionWindow2D(window);

	for(i = 0; i<window->noModels; i++) {
		window->subWindowID[i] = glutCreateSubWindow(window->id, window->positionsX[i], window->positionsY[i], window->sub_width, window->sub_height);
		glutReshapeFunc(calglSubReshapeWindow2D);
		glutDisplayFunc(calglSubDisplayWindow2D);

		glutMouseFunc(calglMouseWindow2D);
		glutMotionFunc(calglMotionMouseWindow2D);
		glutKeyboardFunc(calglKeyboardEventWindow2D);
		glutKeyboardUpFunc(calglKeyboardUpEventWindow2D);
		glutSpecialFunc(calglSpecialKeyboardEventWindow2D);
	}

	window->font_style = GLUT_BITMAP_8_BY_13;

	if(orientation==CALGL_LAYOUT_ORIENTATION_UNKNOW)
		orientation = CALGL_LAYOUT_ORIENTATION_HORIZONTAL;

	window2D = window;
	return window;
}

void calglDestroyWindow2D(struct CALGLWindow2D* window) {
	GLint i = 0;

	if(window) {
		free(window->subWindowID);
		free(window->positionsX);
		free(window->positionsY);
		for(i = 0; i<window->noModels; i++) {
			calglDestoyDrawModel2D(window->models[i]);
		}
		free(window);

		free(xPos);
		free(yPos);
		free(zPos);
		free(xRot);
		free(yRot);
		free(zRot);
	}

	calglCleanDrawModelList2D();
	calglDestroyGlobalSettings();
}

void calglRedisplayAllWindow2D(void) {
	GLint i = 0;
	for(i = 0; i<window2D->noModels; i++) {
		glutSetWindow(window2D->subWindowID[i]);
		calglSubReshapeWindow2D(window2D->sub_width, window2D->sub_height);
		glutPostRedisplay();
	}
}

void calglDisplayWindow2D(void) {
	GLint i = 0;
	glClearColor(0.8f, 0.8f, 0.8f, 0.0f);
	glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);

	glColor3ub(0, 0, 0);
	for(i = 0; i<window2D->noModels; i++) {
		calglDrawStringWindow2D(window2D, window2D->positionsX[i], window2D->positionsY[i]-SEPARATOR_SPACE, (char*) window2D->models[i]->name);
	}

	glutSwapBuffers();
}

void calglReshapeWindow2D(int w, int h) {
	GLint i = 0;

	if(w<200) {
		w = 200;
	}
	if(h<200) {
		h = 200;
	}

	calglSetWindowDimension(w, h);

	glViewport(0, 0, window2D->globalSettings->width, window2D->globalSettings->height);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluOrtho2D(0, window2D->globalSettings->width, window2D->globalSettings->height, 0);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	calglCalculatePositionAndDimensionWindow2D(window2D);

	for(i = 0; i<window2D->noModels; i++) {
		glutSetWindow(window2D->subWindowID[i]);
		glutPositionWindow(window2D->positionsX[i], window2D->positionsY[i]);
		glutReshapeWindow(window2D->sub_width, window2D->sub_height);
	}
}

void calglSubDisplayWindow2D(void) {
	glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);

	currentModel2D = glutGetWindow()-2;

	//if(window2D->models[currentModel2D]->modelLight)
	//	calglApplyLightParameter(window2D->models[currentModel2D]->modelLight);

	glPushMatrix();	{
		glTranslatef(xPos[currentModel2D], yPos[currentModel2D], zPos[currentModel2D]);

		glRotatef(xRot[currentModel2D], 1.0f, 0.0f, 0.0f);
		glRotatef(yRot[currentModel2D], 0.0f, 1.0f, 0.0f);
		glRotatef(zRot[currentModel2D], 0.0f, 0.0f, 1.0f);

		if(window2D->models[currentModel2D]->modelLight) {
			glEnable(GL_LIGHTING);
			glEnable(window2D->models[currentModel2D]->modelLight->currentLight);
			glEnable(GL_COLOR_MATERIAL);
			glColorMaterial(GL_FRONT, GL_AMBIENT_AND_DIFFUSE);
			glEnable(GL_NORMALIZE);

			// Light Cone Drawing - Enable this for debugging
			//glPushMatrix();	{
			//	glTranslatef(window2D->models[currentModel2D]->modelLight->lightPosition[0],
			//		window2D->models[currentModel2D]->modelLight->lightPosition[1],
			//		window2D->models[currentModel2D]->modelLight->lightPosition[2]);
			//	glRotatef(-90, 1.0f, 0.0f, 0.0f);
			//	glColor3f(1.0f, 0.0f, 0.0f);
			//	glutSolidCone(0.5, 1.5, 10, 12);
			//	glColor3f(1.0f, 1.0f, 0.0f);
			//	glutSolidSphere(0.4, 10, 12);
			//}	glPopMatrix();
		}

		calglDisplayModel2D(window2D->models[currentModel2D]);

		if(window2D->models[currentModel2D]->infoBar) {
			calglDisplayBar2D(window2D->models[currentModel2D]->infoBar);
		}

	}	glPopMatrix();

	glutSwapBuffers();
}

void calglSubReshapeWindow2D(int w, int h) {
	glViewport(0, 0, w, h);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(fovy, (GLdouble) w/(GLdouble) h, window2D->globalSettings->zNear, window2D->globalSettings->zFar);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	gluLookAt(0.0f, 0.0f, 10.0f,
		0.0f, 0.0f, -1.0f,
		0.0f, 1.0f, 0.0f);

	glClearColor(0.2f, 0.2f, 1.0f, 1.0f);
	glEnable(GL_DEPTH_TEST);
}

void calglCalculatePositionAndDimensionWindow2D(struct CALGLWindow2D* window) {
	GLint noSubWindowX = 0;
	GLint noSubWindowY = 0;
	GLint i = 0, j = 0, k = 0;

	if(window->noModels==1) {
		noSubWindowX = noSubWindowY = 1;
	} else if(window->noModels==2) {
		if(orientation==CALGL_LAYOUT_ORIENTATION_HORIZONTAL) {
			noSubWindowX = 2;
			noSubWindowY = 1;
		} else {
			noSubWindowX = 1;
			noSubWindowY = 2;
		}
	} else {
		noSubWindowX = (GLint) ceil(sqrt((GLdouble) window->noModels));
		noSubWindowY = noSubWindowX;
	}

	window->sub_width = (window->globalSettings->width-SPACE*(noSubWindowY+1))/noSubWindowY;
	window->sub_height = (window->globalSettings->height-SPACE*(noSubWindowX+1))/noSubWindowX;

	for(i = 0; i<noSubWindowY && k<window->noModels; i++) {
		for(j = 0; j<noSubWindowX && k<window->noModels; j++) {
			window->positionsX[k] = SPACE+(window->sub_width+SPACE)*i;
			window->positionsY[k] = SPACE+(window->sub_height+SPACE)*j;
			k++;
		}
	}
}

void calglMainLoop2D(int argc, char** argv) {
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE|GLUT_RGB|GLUT_DEPTH);

	glClearColor(0.5f, 0.5f, 1.0f, 1.0f);
	glShadeModel(GL_SMOOTH);
	glEnable(GL_DEPTH_TEST);

	calglCreateWindow2D(argc, argv, calglGetGlobalSettings(), models2D, noModels2D);

	calglPrintfInfoCommand2D();

	calglRedisplayAllWindow2D();

	glutMainLoop();

	calglDestroyWindow2D(window2D);
}

void calglSetfontWindow2D(struct CALGLWindow2D* window, char* name, int size) {
	window->font_style = GLUT_BITMAP_HELVETICA_10;
	if(strcmp(name, "helvetica")==0) {
		if(size==12)
			window->font_style = GLUT_BITMAP_HELVETICA_12;
		else if(size==18)
			window->font_style = GLUT_BITMAP_HELVETICA_18;
	} else if(strcmp(name, "times roman")==0) {
		window->font_style = GLUT_BITMAP_TIMES_ROMAN_10;
		if(size==24)
			window->font_style = GLUT_BITMAP_TIMES_ROMAN_24;
	} else if(strcmp(name, "8x13")==0) {
		window->font_style = GLUT_BITMAP_8_BY_13;
	} else if(strcmp(name, "9x15")==0) {
		window->font_style = GLUT_BITMAP_9_BY_15;
	}
}

void calglDrawStringWindow2D(struct CALGLWindow2D* window, GLuint x, GLuint y, char* format, ...) {
	va_list args;
	char buffer[255], *s;

	va_start(args, format);
	vsprintf(buffer, format, args);
	va_end(args);

	glRasterPos2i(x, y);
	for(s = buffer; *s; s++)
		glutBitmapCharacter(window->font_style, *s);
}

void calglSpecialKeyboardEventWindow2D(int key, int x, int y) {
	activeSubWindow = glutGetWindow()-2;
	switch(key) {
	case GLUT_KEY_UP: {
		yPos[activeSubWindow] += TRANSLATION_CONSTANT*keyboardSensitivity;
	}; break;
	case GLUT_KEY_RIGHT: {
		xPos[activeSubWindow] += TRANSLATION_CONSTANT*keyboardSensitivity;
	}; break;
	case GLUT_KEY_DOWN: {
		yPos[activeSubWindow] -= TRANSLATION_CONSTANT*keyboardSensitivity;
	}; break;
	case GLUT_KEY_LEFT: {
		xPos[activeSubWindow] -= TRANSLATION_CONSTANT*keyboardSensitivity;
	}; break;
	default: break;
	}

	calglRedisplayAllWindow2D();
}

void calglKeyboardEventWindow2D(unsigned char key, int x, int y) {
	int i = 0;

	if(key=='t') {
		x -= window2D->sub_width/2;
		y -= window2D->sub_height/2;

		if(translationOn) {
			yPos[activeSubWindow] += -(y-key_old_y)*keyboardSensitivity;
			xPos[activeSubWindow] += (x-key_old_x)*keyboardSensitivity;

			key_old_x = x;
			key_old_y = y;
		} else {
			activeSubWindow = glutGetWindow()-2;
			key_old_x = x;
			key_old_y = y;
			translationOn = CAL_TRUE;
		}
	}

	if(key=='h') {
		calglGetGlobalSettings()->onlyModel = !calglGetGlobalSettings()->onlyModel;
	}

	if(key=='x'||key=='s') {
		if(window2D->models[0]->calglRun) {
			window2D->models[0]->calglRun->active = !window2D->models[0]->calglRun->active;
		}
	}

	if(key=='r') {
		for(i = 0; i<window2D->noModels; i++) {
			xPos[i] = 0.0f;	yPos[i] = 0.0f;	zPos[i] = 0.0f;
			xRot[i] = 90.0f; yRot[i] = 0.0f; zRot[i] = 0.0f;
		}
	}

	calglRedisplayAllWindow2D();
}

void calglKeyboardUpEventWindow2D(unsigned char key, int x, int y) {
	if(key=='t') {
		x -= window2D->sub_width/2;
		y -= window2D->sub_height/2;

		key_old_x = x;
		key_old_y = y;
		translationOn = CAL_FALSE;
	}
}

void calglMouseWindow2D(int button, int state, int x, int y) {
	if(button==0) {  // Left click
		leftPressed = CAL_TRUE;
		old_x = x-window2D->sub_width/2;
		old_y = y-window2D->sub_height/2;

		//models2D[glutGetWindow()-2]->moving = CAL_TRUE;
	} else if(button==2) { // Right click
		rightPressed = CAL_TRUE;
		oldestY = y-window2D->sub_height/2;

		//models2D[glutGetWindow()-2]->moving = CAL_TRUE;
	}

	if(state==GLUT_UP) {
		leftPressed = CAL_FALSE;
		rightPressed = CAL_FALSE;

		//models2D[glutGetWindow()-2]->moving = CAL_FALSE;
	} else {
		calglRedisplayAllWindow2D();

		//models2D[glutGetWindow()-2]->moving = CAL_FALSE;
	}
}

void calglMotionMouseWindow2D(int x, int y) {
	GLfloat rot_x = 0.0f, rot_y = 0.0f;
	GLfloat transY = 0;

	if(leftPressed) {
		x -= window2D->sub_width/2;
		y -= window2D->sub_height/2;

		rot_x = (float) (x-old_x) * mouseSensitivity;
		rot_y = -(float) (y-old_y) * mouseSensitivity;
		old_x = x;
		old_y = y;

		if(models2D[glutGetWindow()-2]->drawMode==CALGL_DRAW_MODE_FLAT||translationOn) {
			return;
		}

		xRot[glutGetWindow()-2] -= rot_y;
		yRot[glutGetWindow()-2] += rot_x;

		/*window2D->models[glutGetWindow()-2]->modelView->xRotation -= rot_y;
		window2D->models[glutGetWindow()-2]->modelView->yRotation += rot_x;*/

		calglRedisplayAllWindow2D();

		//models2D[glutGetWindow()-2]->moving = CAL_TRUE;
	} else if(rightPressed) {
		y -= window2D->sub_height/2;
		transY = -(y-oldestY) * mouseSensitivity;
		oldestY = y;

		zPos[glutGetWindow()-2] += transY;

		calglRedisplayAllWindow2D();

		//models2D[glutGetWindow()-2]->moving = CAL_TRUE;
	} else {

		//models2D[glutGetWindow()-2]->moving = CAL_FALSE;
	}
}

void calglIdleFuncWindow2D(void) {
	if(calglGetGlobalSettings()->fixedDisplay && window2D->models[0]->calglRun->step%calglGetGlobalSettings()->fixedStep==0) {
		calglRedisplayAllWindow2D();
	}
}

void calglTimeFunc2D(int value) {
	glutTimerFunc(calglGetGlobalSettings()->refreshTime, calglTimeFunc2D, value);

	if(!calglGetGlobalSettings()->fixedDisplay) {
		calglRedisplayAllWindow2D();
	}
}

void calglCleanDrawModelList2D() {
	if(models2D) {
		free(models2D);
	}
}

void calglShowModel2D(struct CALGLDrawModel2D* model) {
	if(!models2D) {
		models2D = (struct CALGLDrawModel2D**) malloc(sizeof(struct CALGLDrawModel2D));
	}

	if(noModels2D>=capacityModels2D) {
		calglIncreaseDrawModel2D();
	}

	models2D[noModels2D++] = model;
}

void calglIncreaseDrawModel2D() {
	int i = 0;
	struct CALGLDrawModel2D** models = NULL;

	capacityModels2D += 3;
	models = (struct CALGLDrawModel2D**) malloc(sizeof(struct CALGLDrawModel2D) * (capacityModels2D));

	for(i = 0; i<(capacityModels2D-3); i++) {
		models[i] = models2D[i];
	}

	free(models2D);
	models2D = models;
}

void calglPrintfInfoCommand2D() {
#ifdef WIN32
	system("cls");
#else
	system("clear");
#endif
	printf("*---------------------  Command  ---------------------*\n");
	printf(" Point mouse over a sub window                       \n");
	printf(" Keep T key down and move mouse -> translate model   \n");
	printf(" Use arrows -> translate model						  \n");
	printf(" Left click and move mouse -> rotate model           \n");
	printf(" Right click and move mouse -> zoom in/out model     \n");
	printf(" Press X/S key -> Start/Stop Simulation              \n");
	printf(" Press H key -> Toggle on/off Information Bar Draw   \n");
	printf(" Press R key -> Reset models positions			      \n");
	printf("*-----------------------------------------------------*\n");
}

void calglDisplayBar2D(struct CALGLInfoBar* infoBar) {
	static GLfloat minimumDistanceX = 0;
	static GLfloat minimumDistanceY = 0;
	static GLint sub_width;
	static GLint sub_height;
	static GLfloat smallDimension = 5.0f;
	static GLfloat normalDimension = 10.0f;
	GLint i = 0;
	GLfloat internalDistance = 0.0f;

	if(calglGetGlobalSettings()->onlyModel) {
		return;
	}

	if(infoBar->dimension==CALGL_INFO_BAR_DIMENSION_RELATIVE) {
#pragma region InfoBarRelativeDrawing
		sub_width = window2D->sub_width;
		sub_height = window2D->sub_height;

		minimumDistanceX = (0.1f)*sub_width;
		minimumDistanceY = (0.1f)*sub_height;

		glPushAttrib(GL_LIGHTING_BIT); {
			glDisable(GL_LIGHTING);

			glPushMatrix(); {
				glDepthMask(GL_FALSE);
				glDisable(GL_DEPTH_TEST);

				glMatrixMode(GL_PROJECTION);
				glLoadIdentity();

				gluOrtho2D(0, sub_width, 0, sub_height); //left,right,bottom,top
				glMatrixMode(GL_MODELVIEW);
				glLoadIdentity();

				if(infoBar->orientation==CALGL_INFO_BAR_ORIENTATION_VERTICAL) { // Vertical bar
					minimumDistanceX = (0.2f)*sub_width;
					minimumDistanceY = (0.1f)*sub_height;

					if(infoBar->barInitialization) {
						calglSetInfoBarConstDimension(infoBar, sub_width * 0.1f, -1);
						infoBar->width = infoBar->constWidth;
					}
					infoBar->height = (GLint) (sub_height-minimumDistanceY*2);

					infoBar->xPosition = sub_width-minimumDistanceX-infoBar->width;
					infoBar->yPosition = sub_height-minimumDistanceY;

					glBegin(GL_QUADS); {
						switch(infoBar->infoUse) {
						case CALGL_TYPE_INFO_USE_GRAY_SCALE:
							glColor3f(1.0f, 1.0f, 1.0f);
							break;
						case CALGL_TYPE_INFO_USE_RED_SCALE:
							glColor3f(1.0f, 1.0f, 0.0f);
							break;
						case CALGL_TYPE_INFO_USE_GREEN_SCALE:
							glColor3f(0.0f, 1.0f, 0.0f);
							break;
						case CALGL_TYPE_INFO_USE_BLUE_SCALE:
							glColor3f(0.0f, 0.0f, 1.0f);
							break;
						default:
							glColor3f(1.0f, 1.0f, 1.0f);
							break;
						}
						glVertex2f(infoBar->xPosition+infoBar->width, infoBar->yPosition);
						glVertex2f(infoBar->xPosition, infoBar->yPosition);

						switch(infoBar->infoUse) {
						case CALGL_TYPE_INFO_USE_RED_SCALE:
							glColor3f(1.0f, 0.0f, 0.0f);
							break;
						default:
							glColor3f(0.0f, 0.0f, 0.0f);
							break;
						}
						glVertex2f(infoBar->xPosition, infoBar->yPosition-infoBar->height);
						glVertex2f(infoBar->xPosition+infoBar->width, infoBar->yPosition-infoBar->height);
					}glEnd();

					glColor3f(1.0f, 1.0f, 1.0f);
					calglPrintString2D(infoBar->xPosition+infoBar->width+normalDimension*3, infoBar->yPosition, calglGetString2D(*infoBar->max));
					calglPrintString2D(infoBar->xPosition+infoBar->width+normalDimension*3, infoBar->yPosition-infoBar->height, calglGetString2D(*infoBar->min));

					internalDistance = infoBar->height/10.0f;

					for(i = 0; i<11; i++) {
						glBegin(GL_LINES); {
							if(i%5==0) {
								glVertex2f(infoBar->xPosition+infoBar->width, infoBar->yPosition-i*internalDistance);
								glVertex2f(infoBar->xPosition+infoBar->width+normalDimension*2, infoBar->yPosition-i*internalDistance);
							} else {
								glVertex2f(infoBar->xPosition+infoBar->width, infoBar->yPosition-i*internalDistance);
								glVertex2f(infoBar->xPosition+infoBar->width+smallDimension*2, infoBar->yPosition-i*internalDistance);
							}
						} glEnd();
					}
				} else { // Horizontal bar
					minimumDistanceX = (0.1f)*sub_width;
					minimumDistanceY = (0.1f)*sub_height;

					if(infoBar->barInitialization) {
						calglSetInfoBarConstDimension(infoBar, -1, sub_height * 0.1f);
						infoBar->height = infoBar->constHeight;
					}
					infoBar->width = (GLint) (sub_width-minimumDistanceX*2);

					infoBar->xPosition = minimumDistanceX;
					infoBar->yPosition = minimumDistanceY+infoBar->height;

					glBegin(GL_QUADS); {
						switch(infoBar->infoUse) {
						case CALGL_TYPE_INFO_USE_RED_SCALE:
							glColor3f(1.0f, 0.0f, 0.0f);
							break;
						default:
							glColor3f(0.0f, 0.0f, 0.0f);
							break;
						}
						glVertex2f(infoBar->xPosition, infoBar->yPosition);
						glVertex2f(infoBar->xPosition, infoBar->yPosition-infoBar->height);

						switch(infoBar->infoUse) {
						case CALGL_TYPE_INFO_USE_GRAY_SCALE:
							glColor3f(1.0f, 1.0f, 1.0f);
							break;
						case CALGL_TYPE_INFO_USE_RED_SCALE:
							glColor3f(1.0f, 1.0f, 0.0f);
							break;
						case CALGL_TYPE_INFO_USE_GREEN_SCALE:
							glColor3f(0.0f, 1.0f, 0.0f);
							break;
						case CALGL_TYPE_INFO_USE_BLUE_SCALE:
							glColor3f(0.0f, 0.0f, 1.0f);
							break;
						default:
							glColor3f(1.0f, 1.0f, 1.0f);
							break;
						}
						glVertex2f(infoBar->xPosition+infoBar->width, infoBar->yPosition-infoBar->height);
						glVertex2f(infoBar->xPosition+infoBar->width, infoBar->yPosition);
					}glEnd();

					glColor3f(1.0f, 1.0f, 1.0f);
					calglPrintString2D(infoBar->xPosition+infoBar->width, infoBar->yPosition-infoBar->height-normalDimension*3, calglGetString2D(*infoBar->max));
					calglPrintString2D(infoBar->xPosition, infoBar->yPosition-infoBar->height-normalDimension*3, calglGetString2D(*infoBar->min));

					internalDistance = infoBar->width/10.0f;

					for(i = 0; i<11; i++) {
						glBegin(GL_LINES); {
							if(i%5==0) {
								glVertex2f(infoBar->xPosition+i*internalDistance, infoBar->yPosition-infoBar->height);
								glVertex2f(infoBar->xPosition+i*internalDistance, infoBar->yPosition-infoBar->height-normalDimension*2);
							} else {
								glVertex2f(infoBar->xPosition+i*internalDistance, infoBar->yPosition-infoBar->height);
								glVertex2f(infoBar->xPosition+i*internalDistance, infoBar->yPosition-infoBar->height-smallDimension*2);
							}
						} glEnd();
					}
				}

				// Print name
				glColor3f(1.0f, 1.0f, 1.0f);
				calglPrintConstString2D(infoBar->xPosition, infoBar->yPosition+21, infoBar->substateName);

				glDepthMask(GL_TRUE);
			}glPopMatrix();
		}glPopAttrib();
#pragma endregion
	} else if(infoBar->dimension==CALGL_INFO_BAR_DIMENSION_ABSOLUTE) {
#pragma region InfoBarAbsoluteDrawing
		sub_width = window2D->sub_width;
		sub_height = window2D->sub_height;

		minimumDistanceX = (0.1f)*sub_width;
		minimumDistanceY = (0.1f)*sub_height;

		glPushAttrib(GL_LIGHTING_BIT); {
			glDisable(GL_LIGHTING);

			glPushMatrix(); {
				glDepthMask(GL_FALSE);
				glDisable(GL_DEPTH_TEST);

				glMatrixMode(GL_PROJECTION);
				glLoadIdentity();

				gluOrtho2D(0, sub_width, 0, sub_height); //left,right,bottom,top
				glMatrixMode(GL_MODELVIEW);
				glLoadIdentity();

				if(infoBar->orientation==CALGL_INFO_BAR_ORIENTATION_VERTICAL) { // Vertical bar
					minimumDistanceX = (0.2f)*sub_width;
					minimumDistanceY = (0.1f)*sub_height;

					glBegin(GL_QUADS); {
						switch(infoBar->infoUse) {
						case CALGL_TYPE_INFO_USE_GRAY_SCALE:
							glColor3f(1.0f, 1.0f, 1.0f);
							break;
						case CALGL_TYPE_INFO_USE_RED_SCALE:
							glColor3f(1.0f, 1.0f, 0.0f);
							break;
						case CALGL_TYPE_INFO_USE_GREEN_SCALE:
							glColor3f(0.0f, 1.0f, 0.0f);
							break;
						case CALGL_TYPE_INFO_USE_BLUE_SCALE:
							glColor3f(0.0f, 0.0f, 1.0f);
							break;
						default:
							glColor3f(1.0f, 1.0f, 1.0f);
							break;
						}
						glVertex2f(infoBar->xPosition+infoBar->width, infoBar->yPosition);
						glVertex2f(infoBar->xPosition, infoBar->yPosition);

						switch(infoBar->infoUse) {
						case CALGL_TYPE_INFO_USE_RED_SCALE:
							glColor3f(1.0f, 0.0f, 0.0f);
							break;
						default:
							glColor3f(0.0f, 0.0f, 0.0f);
							break;
						}
						glVertex2f(infoBar->xPosition, infoBar->yPosition-infoBar->height);
						glVertex2f(infoBar->xPosition+infoBar->width, infoBar->yPosition-infoBar->height);
					}glEnd();

					glColor3f(1.0f, 1.0f, 1.0f);
					calglPrintString2D(infoBar->xPosition+infoBar->width+normalDimension*3, infoBar->yPosition, calglGetString2D(*infoBar->max));
					calglPrintString2D(infoBar->xPosition+infoBar->width+normalDimension*3, infoBar->yPosition-infoBar->height, calglGetString2D(*infoBar->min));

					internalDistance = infoBar->height/10.0f;

					for(i = 0; i<11; i++) {
						glBegin(GL_LINES); {
							if(i%5==0) {
								glVertex2f(infoBar->xPosition+infoBar->width, infoBar->yPosition-i*internalDistance);
								glVertex2f(infoBar->xPosition+infoBar->width+normalDimension*2, infoBar->yPosition-i*internalDistance);
							} else {
								glVertex2f(infoBar->xPosition+infoBar->width, infoBar->yPosition-i*internalDistance);
								glVertex2f(infoBar->xPosition+infoBar->width+smallDimension*2, infoBar->yPosition-i*internalDistance);
							}
						} glEnd();
					}
				} else { // Horizontal bar
					minimumDistanceX = (0.1f)*sub_width;
					minimumDistanceY = (0.1f)*sub_height;

					glBegin(GL_QUADS); {
						switch(infoBar->infoUse) {
						case CALGL_TYPE_INFO_USE_RED_SCALE:
							glColor3f(1.0f, 0.0f, 0.0f);
							break;
						default:
							glColor3f(0.0f, 0.0f, 0.0f);
							break;
						}
						glVertex2f(infoBar->xPosition, infoBar->yPosition);
						glVertex2f(infoBar->xPosition, infoBar->yPosition-infoBar->height);

						switch(infoBar->infoUse) {
						case CALGL_TYPE_INFO_USE_GRAY_SCALE:
							glColor3f(1.0f, 1.0f, 1.0f);
							break;
						case CALGL_TYPE_INFO_USE_RED_SCALE:
							glColor3f(1.0f, 1.0f, 0.0f);
							break;
						case CALGL_TYPE_INFO_USE_GREEN_SCALE:
							glColor3f(0.0f, 1.0f, 0.0f);
							break;
						case CALGL_TYPE_INFO_USE_BLUE_SCALE:
							glColor3f(0.0f, 0.0f, 1.0f);
							break;
						default:
							glColor3f(1.0f, 1.0f, 1.0f);
							break;
						}
						glVertex2f(infoBar->xPosition+infoBar->width, infoBar->yPosition-infoBar->height);
						glVertex2f(infoBar->xPosition+infoBar->width, infoBar->yPosition);
					}glEnd();

					glColor3f(1.0f, 1.0f, 1.0f);
					calglPrintString2D(infoBar->xPosition+infoBar->width, infoBar->yPosition-infoBar->height-normalDimension*3, calglGetString2D(*infoBar->max));
					calglPrintString2D(infoBar->xPosition, infoBar->yPosition-infoBar->height-normalDimension*3, calglGetString2D(*infoBar->min));

					internalDistance = infoBar->width/10.0f;

					for(i = 0; i<11; i++) {
						glBegin(GL_LINES); {
							if(i%5==0) {
								glVertex2f(infoBar->xPosition+i*internalDistance, infoBar->yPosition-infoBar->height);
								glVertex2f(infoBar->xPosition+i*internalDistance, infoBar->yPosition-infoBar->height-normalDimension*2);
							} else {
								glVertex2f(infoBar->xPosition+i*internalDistance, infoBar->yPosition-infoBar->height);
								glVertex2f(infoBar->xPosition+i*internalDistance, infoBar->yPosition-infoBar->height-smallDimension*2);
							}
						} glEnd();
					}
				}

				// Print name
				glColor3f(1.0f, 1.0f, 1.0f);
				calglPrintConstString2D(infoBar->xPosition, infoBar->yPosition+21, infoBar->substateName);

				glDepthMask(GL_TRUE);
			}glPopMatrix();
		}glPopAttrib();
#pragma endregion
	}

	calglSubReshapeWindow2D(sub_width, sub_height);
}

char* calglGetString2D(GLdouble number) {
	char* toReturn = NULL;
	GLint tmp = (GLint) (number*100);
	GLint tmpSave = tmp;
	GLint dimension = 0;
	GLint i = 0;

	while(tmp>0) {
		tmp /= 10;
		dimension++;
	}
	dimension += 2;
	tmp = tmpSave;

	toReturn = (char*) malloc(sizeof(char)*dimension);

	toReturn[dimension-1] = '\0';
	for(i = dimension-2; i>=0; i--) {
		if(i==dimension-4) {
			toReturn[i] = ',';
		} else {
			switch(tmp%10) {
			case 0: toReturn[i] = '0'; break;
			case 1: toReturn[i] = '1'; break;
			case 2: toReturn[i] = '2'; break;
			case 3: toReturn[i] = '3'; break;
			case 4: toReturn[i] = '4'; break;
			case 5: toReturn[i] = '5'; break;
			case 6: toReturn[i] = '6'; break;
			case 7: toReturn[i] = '7'; break;
			case 8: toReturn[i] = '8'; break;
			case 9: toReturn[i] = '9'; break;
			default: toReturn[i] = ' '; break;
			}
			tmp /= 10;
		}
	}

	return toReturn;
}

void calglPrintString2D(GLfloat x, GLfloat y, char *string) {
	int i = 0;
	//get the length of the string to display
	int len = (int) strlen(string);
	//set the position of the text in the window using the x and y coordinates
	glRasterPos2f(x, y);
	//loop to display character by character
	for(i = 0; i<len; i++) {
		glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, string[i]);
	}
	free(string);
}

void calglPrintConstString2D(GLfloat x, GLfloat y, const char *string) {
	int i = 0;
	//get the length of the string to display
	int len = (int) strlen(string);
	//set the position of the text in the window using the x and y coordinates
	glRasterPos2f(x, y);
	//loop to display character by character
	for(i = 0; i<len; i++) {
		glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, string[i]);
	}
}

void calglSetLayoutOrientation2D(enum CALGL_LAYOUT_ORIENTATION newOrientation) {
	orientation = newOrientation;
}
