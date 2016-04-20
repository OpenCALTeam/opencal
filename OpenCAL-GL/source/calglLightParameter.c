/*
 * Copyright (c) 2016 OpenCALTeam (https://github.com/OpenCALTeam),
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

#include <OpenCAL-GL/calglLightParameter.h>
#include <stdio.h>

struct CALGLLightParameter* calglCreateLightParameter(GLfloat* lightPosition,
	GLfloat* ambientLight,
	GLfloat* diffuseLight,
	GLfloat* specularLight,
	GLint shininess,
	GLfloat* spotDirection,
	GLfloat cutOffAngle){
		struct CALGLLightParameter* lightParameter = NULL;
		static GLint index = -1;

		index++;

		if(index >= 0 && index <=7){
			lightParameter = (struct CALGLLightParameter*) malloc(sizeof(struct CALGLLightParameter));

			switch(index){
			case 0: lightParameter->currentLight = GL_LIGHT0 ; break;
			case 1: lightParameter->currentLight = GL_LIGHT1 ; break;
			case 2: lightParameter->currentLight = GL_LIGHT2 ; break;
			case 3: lightParameter->currentLight = GL_LIGHT3 ; break;
			case 4: lightParameter->currentLight = GL_LIGHT4 ; break;
			case 5: lightParameter->currentLight = GL_LIGHT5 ; break;
			case 6: lightParameter->currentLight = GL_LIGHT6 ; break;
			case 7: lightParameter->currentLight = GL_LIGHT7 ; break;
			}

			if(lightPosition!=NULL){
				lightParameter->lightPosition = (GLfloat*) malloc(sizeof(GLfloat)*4);
				lightParameter->lightPosition[0] = lightPosition[0];
				lightParameter->lightPosition[1] = lightPosition[1];
				lightParameter->lightPosition[2] = lightPosition[2];
				lightParameter->lightPosition[3] = lightPosition[3];
			} else {
				lightParameter->lightPosition = NULL;
			}

			if(ambientLight!=NULL){
				lightParameter->ambientLight = (GLfloat*) malloc(sizeof(GLfloat)*3);
				lightParameter->ambientLight[0] = ambientLight[0];
				lightParameter->ambientLight[1] = ambientLight[1];
				lightParameter->ambientLight[2] = ambientLight[2];
			} else {
				lightParameter->ambientLight = NULL;
			}

			if(diffuseLight!=NULL){
				lightParameter->diffuseLight = (GLfloat*) malloc(sizeof(GLfloat)*3);
				lightParameter->diffuseLight[0] = diffuseLight[0];
				lightParameter->diffuseLight[1] = diffuseLight[1];
				lightParameter->diffuseLight[2] = diffuseLight[2];
			} else {
				lightParameter->diffuseLight = NULL;
			}

			if(specularLight!=NULL){
				lightParameter->specularLight = (GLfloat*) malloc(sizeof(GLfloat)*3);
				lightParameter->specularLight[0] = specularLight[0];
				lightParameter->specularLight[1] = specularLight[1];
				lightParameter->specularLight[2] = specularLight[2];
				lightParameter->shininess = shininess;
			} else {
				lightParameter->specularLight = NULL;
				lightParameter->shininess = 1;
			}

			if(spotDirection!=NULL){
				lightParameter->spotDirection = (GLfloat*) malloc(sizeof(GLfloat)*3);
				lightParameter->spotDirection[0] = spotDirection[0];
				lightParameter->spotDirection[1] = spotDirection[1];
				lightParameter->spotDirection[2] = spotDirection[2];
				lightParameter->cutOffAngle = cutOffAngle;
			} else {
				lightParameter->spotDirection = NULL;
				lightParameter->cutOffAngle = cutOffAngle = 0;
			}
		}

		return lightParameter;
}

void calglDestroyLightParameter(struct CALGLLightParameter* calLightParameter){
	if(calLightParameter){
		if(calLightParameter->lightPosition!=NULL){
			free(calLightParameter->lightPosition);
		}
		if(calLightParameter->ambientLight!=NULL){
			free(calLightParameter->ambientLight);
		}
		if(calLightParameter->diffuseLight!=NULL){
			free(calLightParameter->ambientLight);
		}
		if(calLightParameter->specularLight!=NULL){
			free(calLightParameter->ambientLight);
		}
		if(calLightParameter->spotDirection!=NULL){
			free(calLightParameter->ambientLight);
		}
		free(calLightParameter);
	}
}

void calglApplyLightParameter(struct CALGLLightParameter* calLightParameter){
	glEnable(GL_LIGHTING);
	// Enable the current light from GL_LIGHT0 to GL_LIGHT7
	glEnable(calLightParameter->currentLight);
	glEnable(GL_COLOR_MATERIAL);

	if(calLightParameter->ambientLight){
		glLightfv(calLightParameter->currentLight, GL_AMBIENT, calLightParameter->ambientLight);
	}
	if(calLightParameter->diffuseLight){
		glLightfv(calLightParameter->currentLight, GL_DIFFUSE, calLightParameter->diffuseLight);
	}
	if(calLightParameter->specularLight){
		glLightfv(calLightParameter->currentLight, GL_SPECULAR, calLightParameter->specularLight);
		glMateriali(GL_FRONT, GL_SHININESS, calLightParameter->shininess);
	}
	if(calLightParameter->spotDirection){
		glLightfv(calLightParameter->currentLight, GL_SPOT_DIRECTION, calLightParameter->spotDirection);
		glLightf(calLightParameter->currentLight, GL_SPOT_CUTOFF, calLightParameter->cutOffAngle);
	}
	if(calLightParameter->lightPosition){
		glLightfv(calLightParameter->currentLight, GL_POSITION, calLightParameter->lightPosition);
	}
	glColorMaterial(GL_FRONT, GL_AMBIENT_AND_DIFFUSE);
}
