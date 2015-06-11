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

#ifndef calglModelViewParameter_h
#define calglModelViewParameter_h

#ifdef __APPLE__
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif

#include <calgl2D.h>
#include <calgl3D.h>

struct CALDrawModel2D;

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
