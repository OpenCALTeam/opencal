#include "iso.h"
#include <stdio.h>
#include <stdlib.h>
#ifdef __APPLE__
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif
#include <time.h>


time_t start_time, end_time;

CALreal _min;
CALreal _Max;


void sciddicaTComputeExtremes(struct CALModel2D* ca2D, struct CALSubstate2Dr* Q, CALreal* m, CALreal* M)
{
	int i, j;

	//computing min and max z
	for (i=0; i<ca2D->rows; i++)
		for (j=0; j<ca2D->columns; j++)
			if (calGet2Dr(ca2D,Q,i,j) > 0)
			{
				*m = calGet2Dr(ca2D,Q,i,j);
				*M = calGet2Dr(ca2D,Q,i,j);
			}
	for (i=0; i<ca2D->rows; i++)
		for (j=0; j<ca2D->columns; j++)
		{
			if (*M < calGet2Dr(ca2D,Q,i,j) && calGet2Dr(ca2D,Q,i,j) > 0)
				*M = calGet2Dr(ca2D,Q,i,j);
			if (*m > calGet2Dr(ca2D,Q,i,j) && calGet2Dr(ca2D,Q,i,j) > 0)
				*m = calGet2Dr(ca2D,Q,i,j);
		}
}

void display(void)
{
	CALint i, j;
	CALreal	s, color;
	CALbyte b;

	glClear(GL_COLOR_BUFFER_BIT);
	glPushMatrix();

	glTranslatef(-ca[currentAC].iso->columns/2.0f, ca[currentAC].iso->rows/2.0f, 0);
	glScalef(1, -1, 1);

	sciddicaTComputeExtremes(ca[currentAC].iso, ca[currentAC].Q.value, &_min, &_Max);

	for (i=0; i<ca[currentAC].iso->rows; i++)
		for (j=0; j<ca[currentAC].iso->columns; j++)
		{
			s = calGet2Dr(ca[currentAC].iso,ca[currentAC].Q.value,i,j);

			if (s > 0)
			{
				color = (s - _min) / (_Max - _min);
				glColor3d(color,color,color);
				glRecti(j, i, j+1, i+1);
#ifdef SHOW_GRID
				b = calGet2Db(ca[currentAC].iso,ca[currentAC].Q.state,i,j);
				if (b != BLANK)
				{
					glBegin(GL_LINE_LOOP);
						glColor3d(0,1,0);
						glVertex2d(j,i);
						glVertex2d(j+1,i);
						glVertex2d(j+1,i+1);
						glVertex2d(j,i+1);
					glEnd();
				}
				if (b == TO_BE_STEADY)
				{
					glBegin(GL_LINES);
						glColor3d(1,1,0);
						glVertex2d(j,i);
						glVertex2d(j+1,i+1);
						glVertex2d(j+1,i);
						glVertex2d(j,i+1);
					glEnd();
				}
				if (b == STEADY)
				{
					glBegin(GL_LINES);
						glColor3d(0,1,0);
						glVertex2d(j,i);
						glVertex2d(j+1,i+1);
						glVertex2d(j+1,i);
						glVertex2d(j,i+1);
					glEnd();
				}
#endif
			}

		}

	glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	glColor3d(0,1,0);
	glRectd(0,0,ca[currentAC].iso->columns, ca[currentAC].iso->rows);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

	glPopMatrix();

	glutSwapBuffers();
}

void simulationRun(void)
{
	CALbyte again;

  //exectutes the global transition function, the steering function and check for the stop condition.
	again = calRunCAStep2D(ca[currentAC].isoRun);

	//simulation main loop
	ca[currentAC].isoRun->step++;

	//check for the stop condition
	if (!again)
	{
		//breaking the simulation
		end_time = time(NULL);
		glutIdleFunc(NULL);
		printf("\n");
		printf("Simulation terminated\n");
		printf("Elapsed time: %lds\n", end_time - start_time);

		//saving configuration
		printf("Saving final state to %s\n", OUTPUT_PATH);
		isoSaveConfig(&ca[currentAC]);

		//graphic rendering
		printf("step: %d; \tactive cells: %d\r", ca[currentAC].isoRun->step, ca[currentAC].isoRun->ca2D->A.size_current);
		glutPostRedisplay();
		return;
	}

#ifdef VERBOSE
	//graphic rendering
	printf("step: %d; \tactive cells: %d\r", ca[currentAC].isoRun->step, ca[currentAC].isoRun->ca2D->A.size_current);
	glutPostRedisplay();
#endif
}

void init(void)
{
	glClearColor (0.0, 0.0, 1.0, 0.0);
	glShadeModel (GL_FLAT);

	printf("iso landslide (toy) model\n");
	printf("Left click on the graphic window to start the simulation\n");
	printf("Right click on the graphic window to stop the simulation\n");
}

void reshape(int w, int h)
{
	GLfloat aspect, dim;

	glViewport (0, 0, (GLsizei) w, (GLsizei) h);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	if (ca[currentAC].iso->rows > ca[currentAC].iso->columns) dim = ca[currentAC].iso->rows * 0.5f; else dim = ca[currentAC].iso->columns * 0.5f;
	aspect = (GLfloat)w / (GLfloat)h;
	if (w <= h)
		glOrtho (-dim, dim, -dim/aspect, dim/aspect, 1.0, -1.0);
	else
		glOrtho (-dim*aspect, dim*aspect, -dim, dim, 1.0, -1.0);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
}

void mouse(int button, int state, int x, int y)
{
	switch (button) {
	case GLUT_LEFT_BUTTON:
		if (state == GLUT_DOWN)
		{
			start_time = time(NULL);
			glutIdleFunc(simulationRun);
		}
		break;
	case GLUT_MIDDLE_BUTTON:
	case GLUT_RIGHT_BUTTON:
		if (state == GLUT_DOWN)
			glutIdleFunc(NULL);
		break;
	default:
		break;
	}
}


void keyboard (unsigned char key, int x, int y)
{
	switch (key)
	{
	case 27:
		exit(0);

	default:
		break;
	}
}

int main(int argc, char** argv)
{
	int i=0;

	for (i=0; i<numAC; i++)
	{
		currentAC = i;
		isoCADef(&ca[i]);
		isoLoadConfig(&ca[i]);
	}

	currentAC = 1;

	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
	glutInitWindowSize(640, 480);
	glutInitWindowPosition(100, 100);
	glutCreateWindow(argv[0]);
	init();
	glutDisplayFunc(display);
	glutReshapeFunc(reshape);
	glutMouseFunc(mouse);
	glutKeyboardFunc(keyboard);
	glutMainLoop();

	for (i=0; i<numAC; i++)
		isoExit(&ca[i]);
	return 0;
}
