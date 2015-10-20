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
static float pos[] = {0.0f, 10.0f, 0.0f, 1.0f};
static float diff[] = {0.8f, 0.8f, 0.8f};
static float spec[] = {1.0f, 1.0f, 1.0f};
static float amb[] =  {0.2f, 0.2f, 0.2f};
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

void calglSetFixedDisplayStep(int step){
	if(!globalSettings){
		globalSettings = calglCreateGlobalSettings();
	}
	globalSettings->fixedDisplay = CAL_TRUE;
	globalSettings->fixedStep = step;
}



