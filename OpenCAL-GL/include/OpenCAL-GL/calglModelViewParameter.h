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

#include <OpenCAL-GL/calgl2D.h>
#include <OpenCAL-GL/calgl3D.h>

struct CalGlDrawModel2D;

/*! \brief Structure that contains informations about the model view matrix.
	It contains translations information.
	It contains rotations information.
	It contains scaling information.
*/
struct CALGLModelViewParameter{
	GLfloat xTranslate;	//!< x translations.
	GLfloat yTranslate;	//!< y translations.
	GLfloat zTranslate;	//!< z translations.
	GLfloat xRotation;	//!< x rotations.
	GLfloat yRotation;	//!< y rotations.
	GLfloat zRotation;	//!< z rotations.
	GLfloat xScale;		//!< x scaling.
	GLfloat yScale;		//!< y scaling.
	GLfloat zScale;		//!< z scaling.
};

/*! \brief Constructor for create a model view matrix.
*/
struct CALGLModelViewParameter* calglCreateModelViewParameter(
	GLfloat xT,		//!< x translations.
	GLfloat yT,		//!< y translations.
	GLfloat zT,		//!< z translations.
	GLfloat xR,		//!< x rotations.
	GLfloat yR,		//!< y rotations.
	GLfloat zR,		//!< z rotations.
	GLfloat xS,		//!< x scaling.
	GLfloat yS,		//!< y scaling.
	GLfloat zS		//!< z scaling.
	);

/*! \brief Function that auto-create a model view matrix.
	This version is designed for 2D cellular automata for discreet drawing.
*/
struct CALGLModelViewParameter* calglAutoCreateModelViewParameterFlat2D(
	struct CALGLDrawModel2D* calDrawModel	//!< Pointer to CALDrawModel.
	);
/*! \brief Function that auto-create a model view matrix.
	This version is designed for 3D cellular automata for discreet drawing.
*/
struct CALGLModelViewParameter* calglAutoCreateModelViewParameterFlat3D(
	struct CALGLDrawModel3D* calDrawModel	//!< Pointer to CALDrawModel.
	);
/*! \brief Function that auto-create a model view matrix.
	This version is designed for 2D cellular automata for surface drawing.
*/
struct CALGLModelViewParameter* calglAutoCreateModelViewParameterSurface2D(
	struct CALGLDrawModel2D* calDrawModel	//!< Pointer to CALDrawModel.
	);
/*! \brief Function that auto-create a model view matrix.
	This version is designed for 3D cellular automata for surface drawing.
*/
struct CALGLModelViewParameter* calglAutoCreateModelViewParameterSurface3D(
	struct CALGLDrawModel3D* calDrawModel	//!< Pointer to CALDrawModel.
	);

/*! \brief Destructor for de-allocate memory allocated before.
*/
void calglDestroyModelViewParameter(
	struct CALGLModelViewParameter* calModelVieParameter		//!< Pointer to the CALGLModelViewParameter to destroy.
	);

/*! \brief Function that replace the precedent model view matrix with the new one.
*/
void calglApplyModelViewParameter(
	struct CALGLModelViewParameter* calModelVieParameter		//!< Pointer to the new CALGLModelViewParameter.
	);

#endif
