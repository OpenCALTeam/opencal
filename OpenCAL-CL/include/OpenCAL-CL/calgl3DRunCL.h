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

#ifndef calgl3DRun_h
#define calgl3DRun_h

#include <OpenCAL-CL/calcl3D.h>
#include <time.h>
#include <pthread.h>

/*! \brief Structure that task is to update the cellular automata computation.
	This version is for 3D cellular automata.
*/
struct CALGLRun3D {
	CALbyte firstRun;			//!< Boolean for a first launch.
	CALbyte active;				//!< Boolean if it is active or not.
	CALbyte terminated;			//!< Boolean if it is terminated.
	struct CALRun3D* calRun;	//!< Reference to struct CALRun3D.
	time_t start_time;			//!< Time for which the computation is started.
	time_t end_time;			//!< Time for which the computation is ended.
	pthread_t thread;			//!< Reference to a thread variable.
	CALbyte stop;				//!< Boolean if it is stopped or not.
	CALint step;
	struct CALCLModel3D* device_CA;	//!< Reference to struct CALCLModel3D.
	CALbyte onlyOneTime;
	CALint fixedStep;
	CALint final_step;		//!< Final simulation step; if 0 the simulation becomes a loop.
	size_t * singleStepThreadNum;
	int dimNum;
	size_t * threadNumMax;
};

/*! \brief Constructor for create a struct CALGLRun3D.
*/
struct CALGLRun3D* calglRunCLDef3D(
	struct CALCLModel3D* device_CA,	//!< Reference to CALRun3D
	CALint fixedStep,
	CALint initial_step,
	CALint final_step
	);

/*! \brief Destructor for de-allocate memory.
*/
void calglDestroyUpdater3DCL(
	struct CALGLRun3D* calglRun //!< Struct to destroy.
	);

/*! \brief Main update function, it is called by the thread.
*/
void* calglFuncThreadUpdate3DCL(
	void* arg	//!< Argument which is a struct CALGLRun3D.
	);

/*! \brief Function for starting the thread.
*/
void calglStartThread3DCL(
	struct CALGLRun3D* calglRun	//!< Object which contains the thread to launch.
	);

/*! \brief Update function for updating the cellular automata computation.
*/
void calglUpdate3DCL(
	struct CALGLRun3D* calglRun	//!< Struct for retrieve the cellular automata to update.
	);

// /*! \brief Update function for saving the final state to disk.
// */
// void calglSaveStateUpdater3DCL(
// 	struct CALGLRun3D* calglRun	//!< Struct for retrieve the cellular automata data.
// 	);

#endif
