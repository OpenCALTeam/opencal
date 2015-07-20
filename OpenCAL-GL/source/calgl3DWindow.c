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

#include <math.h>
#include <OpenCAL-GL/calgl3DWindow.h>
#include <stdio.h>
#include <stdarg.h>

#define SPACE 20
#define SEPARATOR_SPACE 5

static struct CALWindow3D* window3D = NULL;
// Models data
static GLint noModels3D = 0;
static GLint currentModel3D = 0;
static GLint capacityModels3D = 1;
static struct CALDrawModel3D** models3D = NULL;
// Worlds Properties
static GLfloat keyboardSensitivity = 0.02f;
static GLfloat mouseSensitivity = 0.2f;
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

struct CALWindow3D* calglCreateWindow3D(int argc, char** argv, struct CALGLGlobalSettings* globalSettings, struct CALDrawModel3D** models, int size){
	struct CALWindow3D* window = (struct CALWindow3D*) malloc(sizeof(struct CALWindow3D));
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

	for(i = 0; i<window->noModels; i++){
		xPos[i] = 0.0f;	yPos[i] = 0.0f;	zPos[i] = 0.0f;
		xRot[i] = 0.0f; yRot[i] = 0.0f; zRot[i] = 0.0f;
	}

	glutInitWindowSize(window->globalSettings->width, window->globalSettings->height); 
	glutInitWindowPosition(window->globalSettings->positionX, window->globalSettings->positionY); 
	window->id = glutCreateWindow(window->globalSettings->applicationName);
	glutReshapeFunc(calglReshapeWindow3D);
	glutDisplayFunc(calglDisplayWindow3D);
	glutIdleFunc(calglIdleFuncWindow3D);
	glutTimerFunc(calglGetGlobalSettings()->refreshTime, calglTimeFunc3D, calglGetGlobalSettings()->refreshTime);

	calglCalculatePositionAndDimensionWindow3D(window);

	for(i = 0; i<window->noModels; i++){
		window->subWindowID[i] = glutCreateSubWindow(window->id, window->positionsX[i], window->positionsY[i], window->sub_width, window->sub_height);
		glutReshapeFunc(calglSubReshapeWindow3D);
		glutDisplayFunc(calglSubDisplayWindow3D);

		glutMouseFunc(calglMouseWindow3D);
		glutMotionFunc(calglMotionMouseWindow3D);
		glutKeyboardFunc(calglKeyboardEventWindow3D);
		glutKeyboardUpFunc(calglKeyboardUpEventWindow3D);
	}

	window->font_style = GLUT_BITMAP_8_BY_13;

	window3D = window;
	return window;
}

void calglDestroyWindow3D(struct CALWindow3D* window){
	GLint i = 0;

	if(window){
		free(window->subWindowID);
		free(window->positionsX);
		free(window->positionsY);
		for(i = 0; i<window->noModels; i++){
			calglDestoyDrawModel3D(window->models[i]);
		}
		free(window);

		free(xPos);
		free(yPos);
		free(zPos);
		free(xRot);
		free(yRot);
		free(zRot);
	}

	calglCleanDrawModelList3D();
}

void calglRedisplayAllWindow3D(void){
	GLint i = 0;
	for(i = 0; i<window3D->noModels; i++){
		glutSetWindow(window3D->subWindowID[i]);
		calglSubReshapeWindow3D(window3D->sub_width, window3D->sub_height);
		glutPostRedisplay();
	}
}

void calglDisplayWindow3D(void){
	GLint i = 0;
	glClearColor(0.8f, 0.8f, 0.8f, 0.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glColor3ub(0,0,0);
	for(i = 0; i<window3D->noModels; i++){
		calglDrawStringWindow3D(window3D, window3D->positionsX[i], window3D->positionsY[i]-SEPARATOR_SPACE, (char*) window3D->models[i]->name);
	}

	glutSwapBuffers();
}

void calglReshapeWindow3D(int w, int h){
	GLint i = 0;

	if(w < 200){
		w = 200;
	}
	if(h < 200){
		h = 200;
	}

	calglSetWindowDimensionGlobalSettings(w, h);

	glViewport(0, 0, window3D->globalSettings->width, window3D->globalSettings->height);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluOrtho2D(0, window3D->globalSettings->width, window3D->globalSettings->height, 0);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	calglCalculatePositionAndDimensionWindow3D(window3D);

	for(i = 0; i<window3D->noModels; i++){
		glutSetWindow(window3D->subWindowID[i]);
		glutPositionWindow(window3D->positionsX[i], window3D->positionsY[i]);
		glutReshapeWindow(window3D->sub_width, window3D->sub_height);
	}
}

void calglSubDisplayWindow3D(void){
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	currentModel3D = glutGetWindow()-2;

	glPushMatrix();	{
		glTranslatef(xPos[currentModel3D], yPos[currentModel3D], zPos[currentModel3D]);

		glRotatef((GLfloat) xRot[currentModel3D], 1.0f, 0.0f, 0.0f);
		glRotatef((GLfloat) yRot[currentModel3D], 0.0f, 1.0f, 0.0f);	
		glRotatef((GLfloat) zRot[currentModel3D], 0.0f, 0.0f, 1.0f);

		if(window3D->models[currentModel3D]->modelLight){
			glEnable(GL_LIGHTING);
			glEnable(window3D->models[currentModel3D]->modelLight->currentLight);
			glEnable(GL_COLOR_MATERIAL);
			glColorMaterial(GL_FRONT, GL_AMBIENT_AND_DIFFUSE);
			glEnable(GL_NORMALIZE);
		}

		calglDisplayModel3D(window3D->models[currentModel3D]);

		if(window3D->models[currentModel3D]->infoBar){
			calglDisplayBar3D(window3D->models[currentModel3D]->infoBar);
		}

		currentModel3D = (currentModel3D+1)%window3D->noModels;
	}	glPopMatrix();

	glutSwapBuffers();
}

void calglSubReshapeWindow3D(int w, int h){
	glViewport(0, 0, w, h);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(60.0, (GLdouble)w/(GLdouble)h, window3D->globalSettings->zNear, window3D->globalSettings->zFar);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	gluLookAt(0.0f, 0.0f, 10.0f,
		0.0f, 0.0f, -1.0f,
		0.0f, 1.0f, 0.0f);

	glClearColor(0.2f, 0.2f, 1.0f, 1.0f);
	glEnable(GL_DEPTH_TEST);
}

void calglCalculatePositionAndDimensionWindow3D(struct CALWindow3D* window){
	GLint noSubWindowX = 0;
	GLint noSubWindowY = 0;
	GLint i = 0, j = 0, k = 0;

	if(window->noModels==1){
		noSubWindowX = noSubWindowY = 1;
	} else if(window->noModels==2) {
		noSubWindowX = 2;
		noSubWindowY = 1;
	} else {
		noSubWindowX = (GLint) ceil(sqrt((GLdouble) window->noModels));
		noSubWindowY = noSubWindowX;
	}

	window->sub_width = (window->globalSettings->width-SPACE*(noSubWindowY+1))/noSubWindowY;
	window->sub_height = (window->globalSettings->height-SPACE*(noSubWindowX+1))/noSubWindowX;

	for(i = 0; i<noSubWindowY && k<window->noModels; i++){
		for(j = 0; j<noSubWindowX && k<window->noModels; j++){
			window->positionsX[k] = SPACE+(window->sub_width+SPACE)*i;
			window->positionsY[k] = SPACE+(window->sub_height+SPACE)*j;		
			k++;
		}
	}
}

void calglStartProcessWindow3D(int argc, char** argv){
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);

	glClearColor(0.5f, 0.5f, 1.0f, 1.0f);
	glShadeModel (GL_SMOOTH);
	glEnable(GL_DEPTH_TEST);

	calglCreateWindow3D(argc, argv, calglGetGlobalSettings(), models3D, noModels3D);	

	calglPrintfInfoCommand3D();

	calglRedisplayAllWindow3D();

	glutMainLoop();

	calglDestroyWindow3D(window3D);
}

void calglSetfontWindow3D(struct CALWindow3D* window, char* name, int size){
	window->font_style = GLUT_BITMAP_HELVETICA_10;
	if (strcmp(name, "helvetica") == 0) {
		if (size == 12) 
			window->font_style = GLUT_BITMAP_HELVETICA_12;
		else if (size == 18)
			window->font_style = GLUT_BITMAP_HELVETICA_18;
	} else if (strcmp(name, "times roman") == 0) {
		window->font_style = GLUT_BITMAP_TIMES_ROMAN_10;
		if (size == 24)
			window->font_style = GLUT_BITMAP_TIMES_ROMAN_24;
	} else if (strcmp(name, "8x13") == 0) {
		window->font_style = GLUT_BITMAP_8_BY_13;
	} else if (strcmp(name, "9x15") == 0) {
		window->font_style = GLUT_BITMAP_9_BY_15;
	}
}

void calglDrawStringWindow3D(struct CALWindow3D* window, GLuint x, GLuint y, char* format, ...){
	va_list args;
	char buffer[255], *s;

	va_start(args, format);
	vsprintf(buffer, format, args);
	va_end(args);

	glRasterPos2i(x, y);
	for (s = buffer; *s; s++)
		glutBitmapCharacter(window->font_style , *s);
}

void calglKeyboardEventWindow3D(unsigned char key, int x, int y){
	if(key=='t'){
		x -= window3D->sub_width/2;
		y -= window3D->sub_height/2;

		if(translationOn){
			yPos[activeSubWindow]+=-(y - key_old_y)*keyboardSensitivity;
			xPos[activeSubWindow]+=(x - key_old_x)*keyboardSensitivity;

			key_old_x = x;
			key_old_y = y;
		} else {
			activeSubWindow = glutGetWindow()-2;
			key_old_x = x;
			key_old_y = y;
			translationOn = CAL_TRUE;
		}
	}

	if(key=='h'){
		calglGetGlobalSettings()->onlyModel = !calglGetGlobalSettings()->onlyModel; 
	}

	if(key=='p'){
		if(window3D->models[0]->calUpdater){
			window3D->models[0]->calUpdater->active = !window3D->models[0]->calUpdater->active;
		}
	}

	calglRedisplayAllWindow3D();
}

void calglKeyboardUpEventWindow3D(unsigned char key, int x, int y){
	if(key=='t'){
		x -= window3D->sub_width/2;
		y -= window3D->sub_height/2;

		key_old_x = x;
		key_old_y = y;
		translationOn = CAL_FALSE;
	}
}

void calglMouseWindow3D(int button, int state, int x, int y){
	if (button==0){  // Left click
		leftPressed = CAL_TRUE;
		old_x = x - window3D->sub_width/2;
		old_y = y - window3D->sub_height/2;
	} else if (button == 2) { // Right click
		rightPressed = CAL_TRUE;
		oldestY = y - window3D->sub_height/2;
	}

	if(state == GLUT_UP){
		leftPressed = CAL_FALSE;
		rightPressed = CAL_FALSE;
	} else {
		calglRedisplayAllWindow3D();
	}	
}

void calglMotionMouseWindow3D(int x, int y){
	GLfloat rot_x = 0.0f, rot_y = 0.0f;
	GLfloat transY = 0;

	if(leftPressed){
		x -= window3D->sub_width/2;
		y -= window3D->sub_height/2;

		rot_x = (float)(x - old_x) * mouseSensitivity;
		rot_y = -(float)(y - old_y) * mouseSensitivity;
		old_x = x;
		old_y = y;

		if(translationOn){
			return;
		}

		xRot[glutGetWindow()-2] -= rot_y;
		yRot[glutGetWindow()-2] += rot_x;

		/*window3D->models[glutGetWindow()-2]->modelView->xRotation -= rot_y;
		window3D->models[glutGetWindow()-2]->modelView->yRotation += rot_x;*/

		calglRedisplayAllWindow3D();		
	} else if(rightPressed){
		y -= window3D->sub_height/2;
		transY = -(y - oldestY) * mouseSensitivity;
		oldestY = y;

		zPos[glutGetWindow()-2]+=transY;

		calglRedisplayAllWindow3D();	
	}
}

void calglIdleFuncWindow3D(void){
	if(calglGetGlobalSettings()->fixedDisplay && window3D->models[0]->calUpdater->calRun->step%calglGetGlobalSettings()->fixedStep==0){
		calglRedisplayAllWindow3D();
	}
}

void calglTimeFunc3D(int value){
	glutTimerFunc(calglGetGlobalSettings()->refreshTime, calglTimeFunc3D, value);

	if(!calglGetGlobalSettings()->fixedDisplay){
		calglRedisplayAllWindow3D();
	}
}

void calglCleanDrawModelList3D(){
	if(models3D){
		free(models3D);
	}
}

void calglShowModel3D(struct CALDrawModel3D* model){
	if(!models3D){
		models3D = (struct CALDrawModel3D**) malloc(sizeof(struct CALDrawModel3D));
	}

	if(noModels3D >= capacityModels3D){
		calglIncreaseDrawModel3D();
	}

	models3D[noModels3D++] = model;
}

void calglIncreaseDrawModel3D(){
	int i = 0;
	struct CALDrawModel3D** models = NULL;

	capacityModels3D += 3;
	models = (struct CALDrawModel3D**) malloc(sizeof(struct CALDrawModel3D) * (capacityModels3D));

	for(i=0; i<(capacityModels3D-3); i++){
		models[i] = models3D[i];
	}

	free(models3D);
	models3D = models;
}

void calglPrintfInfoCommand3D(){
#ifdef WIN32
	system("cls");
#else
	system("clear");
#endif
	printf("*---------------------  Command  ---------------------*\n");
	printf("* Point mouse over a sub window                       *\n");
	printf("* Keep T key down and move mouse -> translate model   *\n");
	printf("* Left click and move mouse -> rotate model           *\n");
	printf("* Right click and move mouse -> zoom in/out model     *\n");
	printf("* Press P key -> Start/Stop Simulation                *\n");
	printf("* Press H key -> Toggle on/off Information Bar Draw   *\n");
	printf("*-----------------------------------------------------*\n");
}

void calglDisplayBar3D(struct CALGLInfoBar* infoBar){
	static GLfloat minimumDistanceX = 0;
	static GLfloat minimumDistanceY = 0;
	static GLint sub_width;
	static GLint sub_height;
	static GLfloat smallDimension = 5.0f;
	static GLfloat normalDimension = 10.0f;
	GLint i = 0;
	GLfloat internalDistance = 0.0f;

	if(calglGetGlobalSettings()->onlyModel){
		return;
	}

	sub_width = window3D->sub_width;
	sub_height = window3D->sub_height;

	minimumDistanceX = (0.1f)*sub_width;
	minimumDistanceY = (0.1f)*sub_height;

	glPushAttrib(GL_LIGHTING_BIT);{
		glDisable(GL_LIGHTING);

		glPushMatrix();{
			glDepthMask(GL_FALSE);
			glDisable(GL_DEPTH_TEST);

			glMatrixMode(GL_PROJECTION);
			glLoadIdentity();

			gluOrtho2D(0, sub_width, 0, sub_height); //left,right,bottom,top
			glMatrixMode(GL_MODELVIEW);
			glLoadIdentity();	

			if(infoBar->height >= infoBar->width){ // h >= w
				infoBar->xPosition = sub_width - minimumDistanceX - infoBar->width;
				infoBar->yPosition = sub_height - minimumDistanceY;

				glBegin(GL_QUADS);{
					switch(infoBar->infoUse){
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
					glVertex2f(infoBar->xPosition + infoBar->width, infoBar->yPosition);
					glVertex2f(infoBar->xPosition, infoBar->yPosition);

					switch(infoBar->infoUse){
					case CALGL_TYPE_INFO_USE_RED_SCALE:
						glColor3f(1.0f, 0.0f, 0.0f);
						break;
					default:
						glColor3f(0.0f, 0.0f, 0.0f);
						break;
					}
					glVertex2f(infoBar->xPosition, infoBar->yPosition - infoBar->height);
					glVertex2f(infoBar->xPosition + infoBar->width, infoBar->yPosition - infoBar->height);
				}glEnd();

				glColor3f(1.0f, 1.0f, 1.0f);
				calglPrintString3D(infoBar->xPosition + infoBar->width + normalDimension*3, infoBar->yPosition, calglGetString3D(*infoBar->max));
				calglPrintString3D(infoBar->xPosition + infoBar->width + normalDimension*3, infoBar->yPosition - infoBar->height, calglGetString3D(*infoBar->min));

				internalDistance = infoBar->height/10.0f; 

				for(i = 0; i<11; i++){
					glBegin(GL_LINES); {
						if(i%5==0){
							glVertex2f(infoBar->xPosition + infoBar->width, infoBar->yPosition - i*internalDistance);
							glVertex2f(infoBar->xPosition + infoBar->width + normalDimension*2, infoBar->yPosition - i*internalDistance);
						} else {
							glVertex2f(infoBar->xPosition + infoBar->width, infoBar->yPosition - i*internalDistance);
							glVertex2f(infoBar->xPosition + infoBar->width + smallDimension*2, infoBar->yPosition - i*internalDistance);
						}
					} glEnd();
				}
			} else { // w > h
				infoBar->xPosition = minimumDistanceX;
				infoBar->yPosition = minimumDistanceY + infoBar->height;

				glBegin(GL_QUADS);{
					switch(infoBar->infoUse){
					case CALGL_TYPE_INFO_USE_RED_SCALE:
						glColor3f(1.0f, 0.0f, 0.0f);
						break;
					default:
						glColor3f(0.0f, 0.0f, 0.0f);
						break;
					}		
					glVertex2f(infoBar->xPosition, infoBar->yPosition);
					glVertex2f(infoBar->xPosition, infoBar->yPosition - infoBar->height);					

					switch(infoBar->infoUse){
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
					glVertex2f(infoBar->xPosition + infoBar->width, infoBar->yPosition - infoBar->height);
					glVertex2f(infoBar->xPosition + infoBar->width, infoBar->yPosition);
				}glEnd();

				glColor3f(1.0f, 1.0f, 1.0f);
				calglPrintString3D(infoBar->xPosition + infoBar->width, infoBar->yPosition - infoBar->height - normalDimension*3, calglGetString3D(*infoBar->max));
				calglPrintString3D(infoBar->xPosition, infoBar->yPosition - infoBar->height - normalDimension*3, calglGetString3D(*infoBar->min));

				internalDistance = infoBar->width/10.0f; 

				for(i = 0; i<11; i++){
					glBegin(GL_LINES); {
						if(i%5==0){
							glVertex2f(infoBar->xPosition + i*internalDistance, infoBar->yPosition - infoBar->height);
							glVertex2f(infoBar->xPosition + i*internalDistance, infoBar->yPosition - infoBar->height - normalDimension*2);
						} else {
							glVertex2f(infoBar->xPosition + i*internalDistance, infoBar->yPosition - infoBar->height);
							glVertex2f(infoBar->xPosition + i*internalDistance, infoBar->yPosition - infoBar->height - smallDimension*2);
						}
					} glEnd();
				}
			}

			// Print name
			glColor3f(1.0f, 1.0f, 1.0f);
			calglPrintConstString3D(infoBar->xPosition, infoBar->yPosition + 21, infoBar->substateName);

			glDepthMask(GL_TRUE);
		}glPopMatrix();
	}glPopAttrib();

	calglSubReshapeWindow3D(sub_width, sub_height);
}

char* calglGetString3D(GLdouble number){
	char* toReturn = NULL;
	GLint tmp = (GLint) (number*100);
	GLint tmpSave = tmp;
	GLint dimension = 0;
	GLint i = 0;

	while(tmp>0){
		tmp/=10;
		dimension++;
	}
	dimension+=2;
	tmp = tmpSave;

	toReturn = (char*) malloc(sizeof(char)*dimension);

	toReturn[dimension-1] = '\0';
	for(i = dimension-2; i>=0; i--){
		if(i == dimension-4){
			toReturn[i] = ',';
		} else {
			switch(tmp%10){
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

void calglPrintString3D(GLfloat x, GLfloat y, char *string){
	int i = 0;
	//get the length of the string to display
	int len = (int) strlen(string);
	//set the position of the text in the window using the x and y coordinates
	glRasterPos2f(x, y);	
	//loop to display character by character
	for (i = 0; i < len; i++){
		glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, string[i]);
	}
	free(string);
}

void calglPrintConstString3D(GLfloat x, GLfloat y, const char *string){
	int i = 0;
	//get the length of the string to display
	int len = (int) strlen(string);
	//set the position of the text in the window using the x and y coordinates
	glRasterPos2f(x, y);	
	//loop to display character by character
	for (i = 0; i < len; i++){
		glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, string[i]);
	}
}


