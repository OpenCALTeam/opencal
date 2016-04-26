/*
 * Copyright (c) 2016 OpenCALTeam (https://github.com/OpenCALTeam),
 * Telesio Research Group,
 * Department of Mathematics and Computer Science,
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

#ifndef calglModelViewParameter_h
#define calglModelViewParameter_h

#ifdef __APPLE__
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif

#include <OpenCAL-GL/calgl2D.h>
#include <OpenCAL-GL/calgl3D.h>

struct CALGLDrawModel2D;

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
DllExport
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
DllExport
struct CALGLModelViewParameter* calglAutoCreateModelViewParameterFlat2D(
	struct CALGLDrawModel2D* calDrawModel	//!< Pointer to CALDrawModel.
	);
/*! \brief Function that auto-create a model view matrix.
	This version is designed for 3D cellular automata for discreet drawing.
*/
DllExport
struct CALGLModelViewParameter* calglAutoCreateModelViewParameterFlat3D(
	struct CALGLDrawModel3D* calDrawModel	//!< Pointer to CALDrawModel.
	);
/*! \brief Function that auto-create a model view matrix.
	This version is designed for 2D cellular automata for surface drawing.
*/
DllExport
struct CALGLModelViewParameter* calglAutoCreateModelViewParameterSurface2D(
	struct CALGLDrawModel2D* calDrawModel	//!< Pointer to CALDrawModel.
	);
/*! \brief Function that auto-create a model view matrix.
	This version is designed for 3D cellular automata for surface drawing.
*/
DllExport
struct CALGLModelViewParameter* calglAutoCreateModelViewParameterSurface3D(
	struct CALGLDrawModel3D* calDrawModel	//!< Pointer to CALDrawModel.
	);

/*! \brief Destructor for de-allocate memory allocated before.
*/
DllExport
void calglDestroyModelViewParameter(
	struct CALGLModelViewParameter* calModelVieParameter		//!< Pointer to the CALGLModelViewParameter to destroy.
	);

/*! \brief Function that replace the precedent model view matrix with the new one.
*/
DllExport
void calglApplyModelViewParameter(
	struct CALGLModelViewParameter* calModelVieParameter		//!< Pointer to the new CALGLModelViewParameter.
	);

#endif
