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
struct CALGLGlobalSettings* calglCreateGlobalSettings();

/*! \brief Destructor for de-alloate memory.
*/
void calglDestroyGlobalSettings();

/*! \brief Function for set an application's name.
*/
void calglSetApplicationName(
	char* applicationName	//!< Name of the application.
	);

/*! \brief Function for set the cell size.
*/
void calglSetCellSize(
	float cellSize	//!< Cell size.
	);

/*! \brief Function for set the window dimensions.
*/
void calglSetWindowDimension(
	int width,	//!< Window width.
	int height	//!< Window height.
	);

/*! \brief Function for set the window position.
*/
void calglSetWindowPosition(
	int positionX,	//!< Window x position.
	int positionY	//!< Window y position.
	);

/*! \brief Function for set the zNear and zFar clipping parameters.
*/
void calglSetClippingFactor(
	int zNear,	//!< Clipping zNear parameter.
	int zFar	//!< Clipping zFar parameter.
	);

/*! \brief Function to retrive the unique instance of the global settings.
*/
struct CALGLGlobalSettings* calglGetGlobalSettings();

/*! \brief Function for enable illumination.
*/
void calglEnableLights();

/*! \brief Function for disable illumination.
*/
void calglDisableLights();

/*! \brief Function for verify if the light are enable.
*/
enum CALGL_LIGHT calglAreLightsEnable();

/*! \brief Function for retrive a default value for the light position.
*/
float* calglGetPositionLight();

/*! \brief Function for retrive a default value for the diffuse component.
*/
float* calglGetDiffuseLight();

/*! \brief Function for retrive a default value for the specular component.
*/
float* calglGetSpecularLight();

/*! \brief Function for retrive a default value for the ambient component.
*/
float* calglGetAmbientLight();

/*! \brief Function for set the refresh time.
*/
void calglSetRefreshTime(
	int time	//!< Refresh time.
	);

/*! \brief Function for set the fixed display step.
*/
void calglSetFixedDisplayStep(
	int step	//!< Fixed step.
	);

#endif
