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

#ifndef calgl2DRun_h
#define calgl2DRun_h

#include <OpenCAL-CL/calcl2D.h>
#include <time.h>
#include <pthread.h>

/*! \brief Structure that task is to update the cellular automata computation.
	This version is for 2D cellular automata.
*/
struct CALGLRun2D {
	CALbyte firstRun;			//!< Boolean for a first launch.
	CALbyte active;				//!< Boolean if it is active or not.
	CALbyte terminated;			//!< Boolean if it is terminated.
	struct CALRun2D* calRun;	//!< Reference to struct CALRun2D.
	time_t start_time;			//!< Time for which the computation is started.
	time_t end_time;			//!< Time for which the computation is ended.
	pthread_t thread;			//!< Reference to a thread variable.
	CALbyte stop;				//!< Boolean if it is stopped or not.
	CALint step;
	struct CALCLModel2D* deviceCA;	//!< Reference to struct CALCLModel2D.
	CALbyte onlyOneTime;
	CALint fixedStep;
	CALint final_step;		//!< Final simulation step; if 0 the simulation becomes a loop.

};

/*! \brief Constructor for create a struct CALGLRun2D.
*/
struct CALGLRun2D* calglRunCLDef2D(
	struct CALCLModel2D* deviceCA,	//!< Reference to CALRun2D
	CALint fixedStep,
	CALint initial_step,
	CALint final_step
	);

/*! \brief Destructor for de-allocate memory.
*/
void calglDestroyUpdater2DCL(
	struct CALGLRun2D* calglRun //!< Struct to destroy.
	);

/*! \brief Main update function, it is called by the thread.
*/
void* calglFuncThreadUpdate2DCL(
	void* arg	//!< Argument which is a struct CALGLRun2D.
	);

/*! \brief Function for starting the thread.
*/
void calglStartThread2DCL(
	struct CALGLRun2D* calglRun	//!< Object which contains the thread to launch.
	);

/*! \brief Update function for updating the cellular automata computation.
*/
void calglUpdate2DCL(
	struct CALGLRun2D* calglRun	//!< Struct for retrieve the cellular automata to update.
	);

/*! \brief Update function for saving the final state to disk.
*/
void calglSaveStateUpdater2DCL(
	struct CALGLRun2D* calglRun	//!< Struct for retrieve the cellular automata data.
	);

#endif
