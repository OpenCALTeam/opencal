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

#ifndef calglGlobalSettings_h
#define calglGlobalSettings_h

#include <OpenCAL/calCommon.h>
#include <OpenCAL-GL/calglCommon.h>

#ifndef _WIN32
unsigned int Sleep(unsigned int);
#endif

/*! \brief Utility structure that contains informations relevant to the whole application.
*/
struct CALGLGlobalSettings {
	char* applicationName;			//!< Application's name.
	const char* iconPath;			//!< Path to the application's icon.
	float cellSize;					//!< Cell size for the cellular automata.
	int width;						//!< Window width.
	int height;						//!< Window height.
	int positionX;					//!< Window x position.
	int positionY;					//!< Window y position.
	int zNear;						//!< zNear clipping plane.
	int zFar;						//!< zFar clipping plane.
	CALbyte onlyModel;				//!< Boolean used in display mode (0 = draw information bar and model, 1 draw only the cellular automata.
	enum CALGL_LIGHT lightStatus;	//!< Status of the light (ON/OFF).
	CALbyte fixedDisplay;			//!< Boolean for a specify a fixed call to display function.
	int fixedStep;					//!< Fixed step to the cellular automata update.
	int refreshTime;				//!< Fixed time for re-display the scene.
};

/*! \brief Constructor for create an instance of global setting.
	This is a Singleton because in the application we need a unique instance a these parameters, and we want a global point of access to this instance.
*/
DllExport
struct CALGLGlobalSettings* calglCreateGlobalSettings();

/*! \brief Destructor for de-alloate memory.
*/
DllExport
void calglDestroyGlobalSettings();

/*! \brief Function for set application parameters
*/
DllExport
void calglInitViewer(
	char* applicationName,	//!< Name of the application.
	float cellSize,			//!< Cell size.
	int width,				//!< Window width.
	int height, 			//!< Window height.
	int positionX, 			//!< Window x position.
	int positionY, 			//!< Window y position.
	CALbyte enableLight,	//!< Boolean for enabling light.
	int displayStep			//!< Value to which refresh the drawing.
	);

/*! \brief Function for set an application's name.
*/
DllExport
void calglSetApplicationName(
	char* applicationName	//!< Name of the application.
	);

/*! \brief Function for set the cell size.
*/
DllExport
void calglSetCellSize(
	float cellSize	//!< Cell size.
	);

/*! \brief Function for set the window dimensions.
*/
DllExport
void calglSetWindowDimension(
	int width,	//!< Window width.
	int height	//!< Window height.
	);

/*! \brief Function for set the window position.
*/
DllExport
void calglSetWindowPosition(
	int positionX,	//!< Window x position.
	int positionY	//!< Window y position.
	);

/*! \brief Function for set the zNear and zFar clipping parameters.
*/
DllExport
void calglSetClippingFactor(
	int zNear,	//!< Clipping zNear parameter.
	int zFar	//!< Clipping zFar parameter.
	);

/*! \brief Function to retrive the unique instance of the global settings.
*/
DllExport
struct CALGLGlobalSettings* calglGetGlobalSettings();

/*! \brief Function for enable illumination.
*/
DllExport
void calglEnableLights();

/*! \brief Function for disable illumination.
*/
DllExport
void calglDisableLights();

/*! \brief Function for verify if the light are enable.
*/
DllExport
enum CALGL_LIGHT calglAreLightsEnable();

/*! \brief Function for retrive a default value for the light position.
*/
DllExport
float* calglGetPositionLight();

/*! \brief Function for retrive a default value for the diffuse component.
*/
DllExport
float* calglGetDiffuseLight();

/*! \brief Function for retrive a default value for the specular component.
*/
DllExport
float* calglGetSpecularLight();

/*! \brief Function for retrive a default value for the ambient component.
*/
DllExport
float* calglGetAmbientLight();

/*! \brief Function for set the refresh time.
*/
DllExport
void calglSetRefreshTime(
	int time	//!< Refresh time.
	);

/*! \brief Function for set the fixed display step.
*/
DllExport
void calglSetDisplayStep(
	int step	//!< Fixed step.
	);

#endif
