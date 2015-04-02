#ifndef calglModelViewParameter_h
#define calglModelViewParameter_h

#ifdef __APPLE__
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif

#include <calgl2D.h>
#include <calgl3D.h>

struct CALGLModelViewParameter{
	GLfloat xTranslate;
	GLfloat yTranslate;
	GLfloat zTranslate;

	GLfloat xRotation;
	GLfloat yRotation;
	GLfloat zRotation;

	GLfloat xScale;
	GLfloat yScale;
	GLfloat zScale;
};

struct CALGLModelViewParameter* calglCreateModelViewParameter(GLfloat xT, GLfloat yT, GLfloat zT, 
	GLfloat xR, GLfloat yR, GLfloat zR, 
	GLfloat xS, GLfloat yS, GLfloat zS);

struct CALGLModelViewParameter* calglAutoCreateModelViewParameterFlat2D(struct CALDrawModel2D* calDrawModel);
struct CALGLModelViewParameter* calglAutoCreateModelViewParameterFlat3D(struct CALDrawModel3D* calDrawModel);
struct CALGLModelViewParameter* calglAutoCreateModelViewParameterSurface2D(struct CALDrawModel2D* calDrawModel);
struct CALGLModelViewParameter* calglAutoCreateModelViewParameterSurface3D(struct CALDrawModel3D* calDrawModel);

void calglDestroyModelViewParameter(struct CALGLModelViewParameter* calModelVieParameter);

void calglApplyModelViewParameter(struct CALGLModelViewParameter* calModelVieParameter);

#endif
