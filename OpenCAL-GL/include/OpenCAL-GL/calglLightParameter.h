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
