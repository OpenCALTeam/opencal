#ifndef calglLightParameter_h
#define calglLightParameter_h

#ifdef __APPLE__
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif

struct CALGLLightParameter{
	GLint currentLight;

	GLfloat* lightPosition;
	GLfloat* ambientLight;
	GLfloat* diffuseLight;
	GLfloat* specularLight;
	GLint shininess;

	GLfloat* spotDirection;
	GLfloat cutOffAngle;
};

struct CALGLLightParameter* calglCreateLightParameter(GLfloat* lightPosition,
	GLfloat* ambientLight,
	GLfloat* diffuseLight,
	GLfloat* specularLight,
	GLint shininess,
	GLfloat* spotDirection,
	GLfloat cutOffAngle);

void calglDestroyLightParameter(struct CALGLLightParameter* calLightParameter);

void calglApplyLightParameter(struct CALGLLightParameter* calLightParameter);

#endif
