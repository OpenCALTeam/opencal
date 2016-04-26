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

#include <OpenCAL-GL/calglInfoBar.h>
#include <stdlib.h>
#include <stdio.h>

struct CALGLInfoBar* calglCreateRelativeInfoBar2Db(const char* substateName, enum CALGL_TYPE_INFO_USE infoUse, struct CALGLDrawModel2D* calDrawModel, struct CALSubstate2Db* substate, enum CALGL_INFO_BAR_ORIENTATION orientation){
	struct CALGLInfoBar* infoBar = (struct CALGLInfoBar*) malloc(sizeof(struct CALGLInfoBar));
	struct CALNode2Db* nodeSearched = NULL;

	calglSearchSubstateDrawModel2Db(calDrawModel->byteModel, substate, &nodeSearched);

	if(nodeSearched){
		infoBar->min = &nodeSearched->min;
		infoBar->max = &nodeSearched->max;
	} else {
		free(infoBar);
		return NULL;
	}

	infoBar->substateName = substateName;
	infoBar->infoUse = infoUse;

	infoBar->orientation = orientation;
	infoBar->dimension = CALGL_INFO_BAR_DIMENSION_RELATIVE;

	infoBar->barInitialization = CAL_TRUE;
	infoBar->constWidth = 1;
	infoBar->constHeight = 1;

	return infoBar;
}
struct CALGLInfoBar* calglCreateRelativeInfoBar2Di(const char* substateName, enum CALGL_TYPE_INFO_USE infoUse, struct CALGLDrawModel2D* calDrawModel, struct CALSubstate2Di* substate, enum CALGL_INFO_BAR_ORIENTATION orientation){
	struct CALGLInfoBar* infoBar = (struct CALGLInfoBar*) malloc(sizeof(struct CALGLInfoBar));
	struct CALNode2Di* nodeSearched = NULL;

	calglSearchSubstateDrawModel2Di(calDrawModel->intModel, substate, &nodeSearched);

	if(nodeSearched){
		infoBar->min = &nodeSearched->min;
		infoBar->max = &nodeSearched->max;
	} else {
		free(infoBar);
		return NULL;
	}

	infoBar->substateName = substateName;
	infoBar->infoUse = infoUse;

	infoBar->orientation = orientation;
	infoBar->dimension = CALGL_INFO_BAR_DIMENSION_RELATIVE;

	infoBar->barInitialization = CAL_TRUE;
	infoBar->constWidth = 1;
	infoBar->constHeight = 1;

	return infoBar;
}
struct CALGLInfoBar* calglCreateRelativeInfoBar2Dr(const char* substateName, enum CALGL_TYPE_INFO_USE infoUse, struct CALGLDrawModel2D* calDrawModel, struct CALSubstate2Dr* substate, enum CALGL_INFO_BAR_ORIENTATION orientation){
	struct CALGLInfoBar* infoBar = (struct CALGLInfoBar*) malloc(sizeof(struct CALGLInfoBar));
	struct CALNode2Dr* nodeSearched = NULL;

	calglSearchSubstateDrawModel2Dr(calDrawModel->realModel, substate, &nodeSearched);

	if(nodeSearched){
		infoBar->min = &nodeSearched->min;
		infoBar->max = &nodeSearched->max;
	} else {
		free(infoBar);
		return NULL;
	}

	infoBar->substateName = substateName;
	infoBar->infoUse = infoUse;

	infoBar->orientation = orientation;
	infoBar->dimension = CALGL_INFO_BAR_DIMENSION_RELATIVE;

	infoBar->barInitialization = CAL_TRUE;
	infoBar->constWidth = 1;
	infoBar->constHeight = 1;

	return infoBar;
}
struct CALGLInfoBar* calglCreateRelativeInfoBar3Db(const char* substateName, enum CALGL_TYPE_INFO_USE infoUse, struct CALGLDrawModel3D* calDrawModel, struct CALSubstate3Db* substate, enum CALGL_INFO_BAR_ORIENTATION orientation){
	struct CALGLInfoBar* infoBar = (struct CALGLInfoBar*) malloc(sizeof(struct CALGLInfoBar));
	struct CALNode3Db* nodeSearched = NULL;

	calglSearchSubstateDrawModel3Db(calDrawModel->byteModel, substate, &nodeSearched);

	if(nodeSearched){
		infoBar->min = &nodeSearched->min;
		infoBar->max = &nodeSearched->max;
	} else {
		free(infoBar);
		return NULL;
	}

	infoBar->substateName = substateName;
	infoBar->infoUse = infoUse;

	infoBar->orientation = orientation;
	infoBar->dimension = CALGL_INFO_BAR_DIMENSION_RELATIVE;

	infoBar->barInitialization = CAL_TRUE;
	infoBar->constWidth = 1;
	infoBar->constHeight = 1;

	return infoBar;
}
struct CALGLInfoBar* calglCreateRelativeInfoBar3Di(const char* substateName, enum CALGL_TYPE_INFO_USE infoUse, struct CALGLDrawModel3D* calDrawModel, struct CALSubstate3Di* substate, enum CALGL_INFO_BAR_ORIENTATION orientation){
	struct CALGLInfoBar* infoBar = (struct CALGLInfoBar*) malloc(sizeof(struct CALGLInfoBar));
	struct CALNode3Di* nodeSearched = NULL;

	calglSearchSubstateDrawModel3Di(calDrawModel->intModel, substate, &nodeSearched);

	if(nodeSearched){
		infoBar->min = &nodeSearched->min;
		infoBar->max = &nodeSearched->max;
	} else {
		free(infoBar);
		return NULL;
	}

	infoBar->substateName = substateName;
	infoBar->infoUse = infoUse;

	infoBar->orientation = orientation;
	infoBar->dimension = CALGL_INFO_BAR_DIMENSION_RELATIVE;

	infoBar->barInitialization = CAL_TRUE;
	infoBar->constWidth = 1;
	infoBar->constHeight = 1;

	return infoBar;
}
struct CALGLInfoBar* calglCreateRelativeInfoBar3Dr(const char* substateName, enum CALGL_TYPE_INFO_USE infoUse, struct CALGLDrawModel3D* calDrawModel, struct CALSubstate3Dr* substate, enum CALGL_INFO_BAR_ORIENTATION orientation){
	struct CALGLInfoBar* infoBar = (struct CALGLInfoBar*) malloc(sizeof(struct CALGLInfoBar));
	struct CALNode3Dr* nodeSearched = NULL;

	calglSearchSubstateDrawModel3Dr(calDrawModel->realModel, substate, &nodeSearched);

	if(nodeSearched){
		infoBar->min = &nodeSearched->min;
		infoBar->max = &nodeSearched->max;
	} else {
		free(infoBar);
		return NULL;
	}

	infoBar->substateName = substateName;
	infoBar->infoUse = infoUse;

	infoBar->orientation = orientation;
	infoBar->dimension = CALGL_INFO_BAR_DIMENSION_RELATIVE;

	infoBar->barInitialization = CAL_TRUE;
	infoBar->constWidth = 1;
	infoBar->constHeight = 1;

	return infoBar;
}

struct CALGLInfoBar* calglCreateInfoBar2Db(const char* substateName, enum CALGL_TYPE_INFO_USE infoUse, struct CALGLDrawModel2D* calDrawModel, struct CALSubstate2Db* substate, GLfloat xPosition, GLfloat yPosition, GLint width, GLint height){
	struct CALGLInfoBar* infoBar = (struct CALGLInfoBar*) malloc(sizeof(struct CALGLInfoBar));
	struct CALNode2Db* nodeSearched = NULL;

	calglSearchSubstateDrawModel2Db(calDrawModel->byteModel, substate, &nodeSearched);

	if (nodeSearched){
		infoBar->min = &nodeSearched->min;
		infoBar->max = &nodeSearched->max;
	}
	else {
		free(infoBar);
		return NULL;
	}

	infoBar->substateName = substateName;
	infoBar->infoUse = infoUse;

	infoBar->dimension = CALGL_INFO_BAR_DIMENSION_ABSOLUTE;
	infoBar->xPosition = xPosition;
	infoBar->yPosition = yPosition;
	infoBar->width = width;
	infoBar->height = height;
	infoBar->orientation = width > height ? CALGL_INFO_BAR_ORIENTATION_HORIZONTAL : CALGL_INFO_BAR_ORIENTATION_VERTICAL;

	infoBar->barInitialization = CAL_TRUE;
	infoBar->constWidth = 1;
	infoBar->constHeight = 1;

	return infoBar;
}
struct CALGLInfoBar* calglCreateInfoBar2Di(const char* substateName, enum CALGL_TYPE_INFO_USE infoUse, struct CALGLDrawModel2D* calDrawModel, struct CALSubstate2Di* substate, GLfloat xPosition, GLfloat yPosition, GLint width, GLint height){
	struct CALGLInfoBar* infoBar = (struct CALGLInfoBar*) malloc(sizeof(struct CALGLInfoBar));
	struct CALNode2Di* nodeSearched = NULL;

	calglSearchSubstateDrawModel2Di(calDrawModel->intModel, substate, &nodeSearched);

	if (nodeSearched){
		infoBar->min = &nodeSearched->min;
		infoBar->max = &nodeSearched->max;
	}
	else {
		free(infoBar);
		return NULL;
	}

	infoBar->substateName = substateName;
	infoBar->infoUse = infoUse;

	infoBar->dimension = CALGL_INFO_BAR_DIMENSION_ABSOLUTE;
	infoBar->xPosition = xPosition;
	infoBar->yPosition = yPosition;
	infoBar->width = width;
	infoBar->height = height;
	infoBar->orientation = width > height ? CALGL_INFO_BAR_ORIENTATION_HORIZONTAL : CALGL_INFO_BAR_ORIENTATION_VERTICAL;

	infoBar->barInitialization = CAL_TRUE;
	infoBar->constWidth = 1;
	infoBar->constHeight = 1;

	return infoBar;
}
struct CALGLInfoBar* calglCreateInfoBar2Dr(const char* substateName, enum CALGL_TYPE_INFO_USE infoUse, struct CALGLDrawModel2D* calDrawModel, struct CALSubstate2Dr* substate, GLfloat xPosition, GLfloat yPosition, GLint width, GLint height){
	struct CALGLInfoBar* infoBar = (struct CALGLInfoBar*) malloc(sizeof(struct CALGLInfoBar));
	struct CALNode2Dr* nodeSearched = NULL;

	calglSearchSubstateDrawModel2Dr(calDrawModel->realModel, substate, &nodeSearched);

	if (nodeSearched){
		infoBar->min = &nodeSearched->min;
		infoBar->max = &nodeSearched->max;
	}
	else {
		free(infoBar);
		return NULL;
	}

	infoBar->substateName = substateName;
	infoBar->infoUse = infoUse;

	infoBar->dimension = CALGL_INFO_BAR_DIMENSION_ABSOLUTE;
	infoBar->xPosition = xPosition;
	infoBar->yPosition = yPosition;
	infoBar->width = width;
	infoBar->height = height;
	infoBar->orientation = width > height ? CALGL_INFO_BAR_ORIENTATION_HORIZONTAL : CALGL_INFO_BAR_ORIENTATION_VERTICAL;

	infoBar->barInitialization = CAL_TRUE;
	infoBar->constWidth = 1;
	infoBar->constHeight = 1;

	return infoBar;
}
struct CALGLInfoBar* calglCreateInfoBar3Db(const char* substateName, enum CALGL_TYPE_INFO_USE infoUse, struct CALGLDrawModel3D* calDrawModel, struct CALSubstate3Db* substate, GLfloat xPosition, GLfloat yPosition, GLint width, GLint height){
	struct CALGLInfoBar* infoBar = (struct CALGLInfoBar*) malloc(sizeof(struct CALGLInfoBar));
	struct CALNode3Db* nodeSearched = NULL;

	calglSearchSubstateDrawModel3Db(calDrawModel->byteModel, substate, &nodeSearched);

	if (nodeSearched){
		infoBar->min = &nodeSearched->min;
		infoBar->max = &nodeSearched->max;
	}
	else {
		free(infoBar);
		return NULL;
	}

	infoBar->substateName = substateName;
	infoBar->infoUse = infoUse;

	infoBar->dimension = CALGL_INFO_BAR_DIMENSION_ABSOLUTE;
	infoBar->xPosition = xPosition;
	infoBar->yPosition = yPosition;
	infoBar->width = width;
	infoBar->height = height;
	infoBar->orientation = width > height ? CALGL_INFO_BAR_ORIENTATION_HORIZONTAL : CALGL_INFO_BAR_ORIENTATION_VERTICAL;

	infoBar->barInitialization = CAL_TRUE;
	infoBar->constWidth = 1;
	infoBar->constHeight = 1;

	return infoBar;
}
struct CALGLInfoBar* calglCreateInfoBar3Di(const char* substateName, enum CALGL_TYPE_INFO_USE infoUse, struct CALGLDrawModel3D* calDrawModel, struct CALSubstate3Di* substate, GLfloat xPosition, GLfloat yPosition, GLint width, GLint height){
	struct CALGLInfoBar* infoBar = (struct CALGLInfoBar*) malloc(sizeof(struct CALGLInfoBar));
	struct CALNode3Di* nodeSearched = NULL;

	calglSearchSubstateDrawModel3Di(calDrawModel->intModel, substate, &nodeSearched);

	if (nodeSearched){
		infoBar->min = &nodeSearched->min;
		infoBar->max = &nodeSearched->max;
	}
	else {
		free(infoBar);
		return NULL;
	}

	infoBar->substateName = substateName;
	infoBar->infoUse = infoUse;

	infoBar->dimension = CALGL_INFO_BAR_DIMENSION_ABSOLUTE;
	infoBar->xPosition = xPosition;
	infoBar->yPosition = yPosition;
	infoBar->width = width;
	infoBar->height = height;
	infoBar->orientation = width > height ? CALGL_INFO_BAR_ORIENTATION_HORIZONTAL : CALGL_INFO_BAR_ORIENTATION_VERTICAL;

	infoBar->barInitialization = CAL_TRUE;
	infoBar->constWidth = 1;
	infoBar->constHeight = 1;

	return infoBar;
}
struct CALGLInfoBar* calglCreateInfoBar3Dr(const char* substateName, enum CALGL_TYPE_INFO_USE infoUse, struct CALGLDrawModel3D* calDrawModel, struct CALSubstate3Dr* substate, GLfloat xPosition, GLfloat yPosition, GLint width, GLint height){
	struct CALGLInfoBar* infoBar = (struct CALGLInfoBar*) malloc(sizeof(struct CALGLInfoBar));
	struct CALNode3Dr* nodeSearched = NULL;

	calglSearchSubstateDrawModel3Dr(calDrawModel->realModel, substate, &nodeSearched);

	if (nodeSearched){
		infoBar->min = &nodeSearched->min;
		infoBar->max = &nodeSearched->max;
	}
	else {
		free(infoBar);
		return NULL;
	}

	infoBar->substateName = substateName;
	infoBar->infoUse = infoUse;

	infoBar->dimension = CALGL_INFO_BAR_DIMENSION_ABSOLUTE;
	infoBar->xPosition = xPosition;
	infoBar->yPosition = yPosition;
	infoBar->width = width;
	infoBar->height = height;
	infoBar->orientation = width > height ? CALGL_INFO_BAR_ORIENTATION_HORIZONTAL : CALGL_INFO_BAR_ORIENTATION_VERTICAL;

	infoBar->barInitialization = CAL_TRUE;
	infoBar->constWidth = 1;
	infoBar->constHeight = 1;

	return infoBar;
}

void calglDestroyInfoBar(struct CALGLInfoBar* infoBar){
	if(infoBar){
		free(infoBar);
	}
}

void calglSetInfoBarConstDimension(struct CALGLInfoBar* infoBar, GLfloat width, GLfloat height){
	width > 0 ? infoBar->constWidth = (CALint)width : 1;
	height > 0 ? infoBar->constHeight = (CALint)height : 1;
}
