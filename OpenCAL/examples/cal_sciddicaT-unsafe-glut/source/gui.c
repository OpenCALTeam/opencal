#include "sciddicaTunsafe.h"
#include <stdlib.h>
#ifdef __APPLE__
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif
#include <time.h>


time_t start_time, end_time;

CALreal z_min;
CALreal z_Max;
CALreal h_min;
CALreal h_Max;

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
	CALreal	z, h, color;

	glClear(GL_COLOR_BUFFER_BIT);
	glPushMatrix();

	glTranslatef(-sciddicaT->columns/2.0f, sciddicaT->rows/2.0f, 0);
	glScalef(1, -1, 1);

	sciddicaTComputeExtremes(sciddicaT, Q.h, &h_min, &h_Max);

	for (i=0; i<sciddicaT->rows; i++)
		for (j=0; j<sciddicaT->columns; j++)
		{
			z = calGet2Dr(sciddicaT,Q.z,i,j);
			h = calGet2Dr(sciddicaT,Q.h,i,j);

			if (h > 0)
			{
				color = (h - h_min) / (h_Max - h_min);
				glColor3d(1,color,0);
				glRecti(j, i, j+1, i+1);
			}
			else
				if (z > 0)
				{
					color = (z - z_min) / (z_Max - z_min);
					glColor3d(color,color,color);
					glRecti(j, i, j+1, i+1);
				}
		}

	glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	glColor3d(0,1,0);
	glRectd(0,0,sciddicaT->columns, sciddicaT->rows);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

	glPopMatrix();

	glutSwapBuffers();
}

void simulationRun(void)
{
	CALbyte again;

  //exectutes the global transition function, the steering function and check for the stop condition.
	again = calRunCAStep2D(sciddicaTsimulation);

	//simulation main loop
	sciddicaTsimulation->step++;

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
		sciddicaTSaveConfig();

		//graphic rendering
		printf("step: %d; \tactive cells: %d\r", sciddicaTsimulation->step, sciddicaTsimulation->ca2D->A.size_current);
		glutPostRedisplay();
		return;
	}

#ifdef VERBOSE
	//graphic rendering
	printf("step: %d; \tactive cells: %d\r", sciddicaTsimulation->step, sciddicaTsimulation->ca2D->A.size_current);
	glutPostRedisplay();
#endif
}

void init(void)
{
	glClearColor (0.0, 0.0, 0.0, 0.0);
	glShadeModel (GL_FLAT);

	printf("sciddicaT landslide (toy) model\n");
	printf("Left click on the graphic window to start the simulation\n");
	printf("Right click on the graphic window to stop the simulation\n");
}

void reshape(int w, int h)
{
	GLfloat aspect, dim;

	glViewport (0, 0, (GLsizei) w, (GLsizei) h);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	if (sciddicaT->rows > sciddicaT->columns) dim = sciddicaT->rows * 0.5f; else dim = sciddicaT->columns * 0.5f;
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


int main(int argc, char** argv)
{
	sciddicaTCADef();
	sciddicaTComputeExtremes(sciddicaT, Q.z, &z_min, &z_Max);

	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
	glutInitWindowSize(640, 480);
	glutInitWindowPosition(100, 100);
	glutCreateWindow(argv[0]);
	init();
	glutDisplayFunc(display);
	glutReshapeFunc(reshape);
	glutMouseFunc(mouse);
	glutMainLoop();

	sciddicaTExit();
	return 0;
}
