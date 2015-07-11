from opencal import *
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
from OpenGL.constants import GLfloat
import sys


import opencal

Q = None
life3D = None
life_simulation = None
model_view= None
life_transition_function = None

def init():
	ambientLight = GLfloat_4(0.2, 0.2, 0.2, 1.0);
	diffuseLight = GLfloat_4( 0.75, 0.75, 0.75, 1.0);

	glEnable(GL_LIGHTING);
	glLightModelfv(GL_LIGHT_MODEL_AMBIENT, ambientLight);
	glLightfv(GL_LIGHT0, GL_DIFFUSE,  diffuseLight);
	glEnable(GL_LIGHT0);
	glEnable(GL_COLOR_MATERIAL);
	glColorMaterial(GL_FRONT, GL_AMBIENT_AND_DIFFUSE);

	glClearColor (0.0, 0.0, 0.0, 0.0);
	glShadeModel (GL_FLAT);
	glEnable (GL_DEPTH_TEST);
	
	print("The life 3D cellular automata model");
	print("Left click on the graphic window to start the simulation");
	print("Right click on the graphic window to stop the simulation");



class ModelView():
	def __init__(self, xrot, yrot,ztrans):
		self.x_rot = xrot;
		self.y_rot = yrot;
		self.z_trans=ztrans;


class Life_transition_function(opencal.ElementaryProcessFunctor3D):
	def run(SELF,life,i,j,k):
			s=0
			global Q
			neighbor = range(1,life.sizeof_X)
			for n in neighbor:
				s =s+opencal.calGetX3Di(life, Q, i, j,k,n);
			if (s==3) or (s==2 and opencal.calGet3Di(life, Q, i, j,k) == 1):
				opencal.calSet3Di(life, Q, i, j,k,1)
			else:
				opencal.calSet3Di(life, Q, i, j,k,0)

def lifeCADef():
		global model_view
		model_view = ModelView(0,0,0)
		global life3D		
		life3D = opencal.calCADef3D (30, 30,30, opencal.CAL_MOORE_NEIGHBORHOOD_3D, opencal.CAL_SPACE_TOROIDAL, opencal.CAL_NO_OPT)
		print("START function life")
		print(life3D.columns)
		print(life3D.rows)
		print(life3D.slices)

		global life_simulation;
		life_simulation = opencal.calRunDef3D(life3D, 1, 10, opencal.CAL_UPDATE_EXPLICIT)

		global life_transition_function
		life_transition_function = Life_transition_function()
		opencal.calAddElementaryProcess3D(life3D, life_transition_function)
		
		#add substates
		global Q
		Q = opencal.calAddSubstate3Di(life3D);
				
		opencal.calInitSubstate3Di(life3D, Q, 0);
		ri=2
		ci=2
		zi=2
		opencal.calInit3Di(life3D, Q, 0+ri, 2+ci,zi, 1);
		opencal.calInit3Di(life3D, Q, 1+ri, 0+ci,zi, 1);
		opencal.calInit3Di(life3D, Q, 1+ri, 2+ci,zi, 1);
		opencal.calInit3Di(life3D, Q, 2+ri, 1+ci,zi, 1);
		opencal.calInit3Di(life3D, Q, 2+ri, 2+ci,zi, 1);

		

def mouse(button,state,x,y):
	if (button == GLUT_LEFT_BUTTON ):
		if (state == GLUT_DOWN):
			glutIdleFunc(simulationRun);
		elif (button == GLUT_RIGHT_BUTTON): 
			if (state == GLUT_DOWN):
				glutIdleFunc(None);

def simulationRun():
	global life3D;
	global life_simulation;
	life_simulation.step=life_simulation.step+1;
	print(life_simulation.step);
	again=False;	
	if not(life_simulation is None):	
		again = opencal.calRunCAStep3D(life_simulation);
		again = again and (life_simulation.step <=5)
		if (not again):
			glutIdleFunc(None);
			print("");
			print("Simulation terminated");
			glutLeaveMainLoop();
	glutPostRedisplay();



def display():
	global Q
	global life3D

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glPushMatrix();
	global model_view
	glTranslatef(0, 0, model_view.z_trans);
	glRotatef(model_view.x_rot, 1, 0, 0);
	glRotatef(model_view.y_rot, 0, 1, 0);
	
	#Save the lighting state variables
	glPushAttrib(GL_LIGHTING_BIT);	
	glDisable(GL_LIGHTING);
	glPushMatrix();
	glColor3f(0,1,0);
	ROWS=life3D.rows
	COLS=life3D.columns
	LAYERS=life3D.slices	
	glScalef(ROWS, COLS, LAYERS);
	glutWireCube(1.0);
	glPopMatrix();
	#Restore lighting state variables
	glPopAttrib();	

	glColor3f(1,1,1);	

	for i in range(0,life3D.rows):
		for j in range(0,life3D.columns):
			for k in range (0,life3D.slices):							
				state = opencal.calGet3Di(life3D,Q,i,j,k);
				if state ==1:
					glPushMatrix();
					glTranslated(i-ROWS/2,j-COLS/2,k-LAYERS/2);
					glutSolidCube(1.0);
					glPopMatrix();

	glPopMatrix();
	glutSwapBuffers();
	
	


def reshape(w,h):
	
	global life3D	
	MAX = life3D.rows;

	if (MAX < life3D.columns):
		MAX = life3D.columns;
	if (MAX < life3D.slices):
		MAX = life3D.slices;

	lightPos = GLfloat_4(0.0, 0.0, float(2*MAX), 1.0);

	glViewport (0, 0,  w,  h);
	glMatrixMode (GL_PROJECTION);
	glLoadIdentity();
	
	HH=float(w)/float(h);
	gluPerspective(45.0, HH, 1.0, 4*MAX);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	gluLookAt (0.0, 0.0, 2*MAX, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);
	glLightfv(GL_LIGHT0, GL_POSITION, lightPos);

def life3DExit():
	global life_simulation;
	global lif3D;
	#finalizations
	opencal.calRunFinalize3D(life_simulation);
	opencal.calFinalize3D(life3D);

def specialKeys(key,x,y):
	global model_view
	xrot=5.0;
	yrot=5.0;
	ztrans=5.0;
	if (key==GLUT_KEY_DOWN):
		model_view.x_rot+=xrot;
	if (key==GLUT_KEY_UP):
		model_view.x_rot-=xrot;
	if (key==GLUT_KEY_LEFT):
		model_view.y_rot-=xrot;
	if (key==GLUT_KEY_RIGHT):
		model_view.y_rot+=xrot;

	glutPostRedisplay();

def main():
	global life_simulation;
	lifeCADef();
	
	glutInit(sys.argv)
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
	glutInitWindowSize(640, 480); 
	glutInitWindowPosition(100, 100);
	glutCreateWindow("openCAL++-PY life3D-glut");
	
	
	init();		
	
	
	glutDisplayFunc(display); 
	glutReshapeFunc(reshape); 
	glutSpecialFunc(specialKeys);
	glutMouseFunc(mouse);	
	
	glutMainLoop();
	
	life3DExit();
	return
main();
	

