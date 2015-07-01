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

#include <calglInfoBar.h>
#include <stdlib.h>
#include <stdio.h>

struct CALGLInfoBar* calglCreateInfoBar2Db(const char* substateName, enum CALGL_TYPE_INFO_USE infoUse, struct CALDrawModel2D* calDrawModel, struct CALSubstate2Db* substate, enum CALGL_INFO_BAR_ORIENTATION orientation){
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

	infoBar->barInitialization = CAL_TRUE;
	infoBar->constWidth = 1;
	infoBar->constHeight = 1;

	return infoBar;
}
struct CALGLInfoBar* calglCreateInfoBar2Di(const char* substateName, enum CALGL_TYPE_INFO_USE infoUse, struct CALDrawModel2D* calDrawModel, struct CALSubstate2Di* substate, enum CALGL_INFO_BAR_ORIENTATION orientation){
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
	
	infoBar->barInitialization = CAL_TRUE;
	infoBar->constWidth = 1;
	infoBar->constHeight = 1;
	
	return infoBar;
}
struct CALGLInfoBar* calglCreateInfoBar2Dr(const char* substateName, enum CALGL_TYPE_INFO_USE infoUse, struct CALDrawModel2D* calDrawModel, struct CALSubstate2Dr* substate, enum CALGL_INFO_BAR_ORIENTATION orientation){
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
	
	infoBar->barInitialization = CAL_TRUE;
	infoBar->constWidth = 1;
	infoBar->constHeight = 1;
	
	return infoBar;
}
struct CALGLInfoBar* calglCreateInfoBar3Db(const char* substateName, enum CALGL_TYPE_INFO_USE infoUse, struct CALDrawModel3D* calDrawModel, struct CALSubstate3Db* substate, enum CALGL_INFO_BAR_ORIENTATION orientation){
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
	
	infoBar->barInitialization = CAL_TRUE;
	infoBar->constWidth = 1;
	infoBar->constHeight = 1;
	
	return infoBar;
}
struct CALGLInfoBar* calglCreateInfoBar3Di(const char* substateName, enum CALGL_TYPE_INFO_USE infoUse, struct CALDrawModel3D* calDrawModel, struct CALSubstate3Di* substate, enum CALGL_INFO_BAR_ORIENTATION orientation){
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
	
	infoBar->barInitialization = CAL_TRUE;
	infoBar->constWidth = 1;
	infoBar->constHeight = 1;
	
	return infoBar;
}
struct CALGLInfoBar* calglCreateInfoBar3Dr(const char* substateName, enum CALGL_TYPE_INFO_USE infoUse, struct CALDrawModel3D* calDrawModel, struct CALSubstate3Dr* substate, enum CALGL_INFO_BAR_ORIENTATION orientation){
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

