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

#include <OpenCAL-GL/calglCommon.h>
#include <OpenCAL-GL/calglGlobalSettings.h>
#include <stdlib.h>

#ifndef _WIN32

#include <unistd.h>

unsigned int Sleep(unsigned int usecond) {
  return usleep (usecond * 1000);
}

#endif


static struct CALGLGlobalSettings* globalSettings = NULL;

// Light
static float pos[] = {0.0f, 5.0f, 0.0f, 1.0f};
static float diff[] = {0.8f, 0.8f, 0.8f};
static float spec[] = {0.0f, 0.0f, 0.0f};
static float amb[] =  {0.4f, 0.4f, 0.4f};
static float spot[] = {0.1f, 0.1f, 0.1f};

struct CALGLGlobalSettings* calglCreateGlobalSettings(){
	struct CALGLGlobalSettings* globalSettings = NULL;

	if(globalSettings){
		return NULL;
	}

	globalSettings = (struct CALGLGlobalSettings*) malloc(sizeof(struct CALGLGlobalSettings));

	globalSettings->applicationName = "OpenCAL";
	globalSettings->iconPath = "";
	globalSettings->cellSize = 0;
	globalSettings->width = 640;
	globalSettings->height = 480;
	globalSettings->positionX = 0;
	globalSettings->positionY = 0;
	globalSettings->zNear = 1;
	globalSettings->zFar = 500;
	globalSettings->onlyModel = CAL_FALSE;
	globalSettings->lightStatus = CALGL_LIGHT_OFF;
	globalSettings->refreshTime = 100;
	globalSettings->fixedDisplay = CAL_FALSE;

	return globalSettings;
}

void calglDestroyGlobalSettings(){
	if(globalSettings){
		free(globalSettings);
	}
}

void calglInitViewer(char* applicationName, float cellSize, int width, int height, int positionX, int positionY, CALbyte enableLight, int displayStep) {
	calglSetApplicationName(applicationName);
	calglSetCellSize(cellSize);
	calglSetWindowDimension(width, height);
	calglSetWindowPosition(positionX, positionY);
	if(enableLight)
		calglEnableLights();
	if(displayStep>0) {
		calglSetDisplayStep(displayStep);
	}
}

void calglSetApplicationName(char* applicationName){
	if(!globalSettings){
		globalSettings = calglCreateGlobalSettings();
	}
	globalSettings->applicationName = applicationName;
}

void calglSetCellSize(float cellSize){
	globalSettings->cellSize = cellSize;
}

void calglSetWindowDimension(int width, int height){
	if(!globalSettings){
		globalSettings = calglCreateGlobalSettings();
	}
	globalSettings->width = width;
	globalSettings->height = height;
}

void calglSetWindowPosition(int positionX, int positionY){
	if(!globalSettings){
		globalSettings = calglCreateGlobalSettings();
	}
	globalSettings->positionX = positionX;
	globalSettings->positionY = positionY;
}

void calglSetClippingFactor(int zNear, int zFar){
	if(!globalSettings){
		globalSettings = calglCreateGlobalSettings();
	}
	globalSettings->zNear = zNear;
	globalSettings->zFar = zFar;
}

struct CALGLGlobalSettings* calglGetGlobalSettings(){
	if(!globalSettings){
		globalSettings = calglCreateGlobalSettings();
	}
	return globalSettings;
}

void calglEnableLights(){
	if(!globalSettings){
		globalSettings = calglCreateGlobalSettings();
	}
	globalSettings->lightStatus = CALGL_LIGHT_ON;
}

void calglDisableLights(){
	if(!globalSettings){
		globalSettings = calglCreateGlobalSettings();
	}
	globalSettings->lightStatus = CALGL_LIGHT_OFF;
}

enum CALGL_LIGHT calglAreLightsEnable(){
	if(!globalSettings){
		globalSettings = calglCreateGlobalSettings();
	}
	return globalSettings->lightStatus;
}

float* calglGetPositionLight(){
	return pos;
}

float* calglGetDiffuseLight(){
	return diff;
}

float* calglGetSpecularLight(){
	return spec;
}

float* calglGetAmbientLight(){
	return amb;
}

float* calglGetSpotLight(){
	return spot;
}

void calglSetRefreshTime(int time){
	if(!globalSettings){
		globalSettings = calglCreateGlobalSettings();
	}
	globalSettings->refreshTime = time;
}

void calglSetDisplayStep(int step){
	if(!globalSettings){
		globalSettings = calglCreateGlobalSettings();
	}
	globalSettings->fixedDisplay = CAL_TRUE;
	globalSettings->fixedStep = step;
}
