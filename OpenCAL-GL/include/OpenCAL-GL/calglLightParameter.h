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
void calglDestroyLightParameter(
	struct CALGLLightParameter* calLightParameter		//!< Pointer to the CALGLLightParameter to destroy.
	);

/*! \brief Function that replace the precedent model view matrix with the new one.
*/
void calglApplyLightParameter(
	struct CALGLLightParameter* calLightParameter		//!< Pointer to the new CALGLLightParameter.
	);

#endif
