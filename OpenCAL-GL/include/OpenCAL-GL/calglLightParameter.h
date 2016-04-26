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

#ifndef calglLightParameter_h
#define calglLightParameter_h

#ifdef __APPLE__
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif

#include <OpenCAL-GL/calglCommon.h>

/*! \brief Structure that contains informations about the light that will be used in the scene.
	It contains position information.
	It contains ambient information.
	It contains diffuse information.
	It contains specular information.
	It contains spot light information.
*/
struct CALGLLightParameter{
	GLint currentLight;		//!< Current light index for OpenGL.
	GLfloat* lightPosition;	//!< Light position in the scene.
	GLfloat* ambientLight;	//!< Ambient component.
	GLfloat* diffuseLight;	//!< Diffuse component.
	GLfloat* specularLight;	//!< Specular component.
	GLint shininess;		//!< Shininess for the specular component.
	GLfloat* spotDirection;	//!< Spot direction.
	GLfloat cutOffAngle;	//!< Cut off angle for the spot light.
};

/*! \brief Constructor for create the light model.
*/
DllExport
struct CALGLLightParameter* calglCreateLightParameter(
	GLfloat* lightPosition,		//!< Light position in the scene.
	GLfloat* ambientLight,		//!< Ambient component.
	GLfloat* diffuseLight,		//!< Diffuse component.
	GLfloat* specularLight,		//!< Specular component.
	GLint shininess,			//!< Shininess for the specular component.
	GLfloat* spotDirection,		//!< Spot direction.
	GLfloat cutOffAngle			//!< Cut off angle for the spot light.
	);

/*! \brief Destructor for de-allocate memory allocated before.
*/
DllExport
void calglDestroyLightParameter(
	struct CALGLLightParameter* calLightParameter		//!< Pointer to the CALGLLightParameter to destroy.
	);

/*! \brief Function that replace the precedent model view matrix with the new one.
*/
DllExport
void calglApplyLightParameter(
	struct CALGLLightParameter* calLightParameter		//!< Pointer to the new CALGLLightParameter.
	);

#endif
