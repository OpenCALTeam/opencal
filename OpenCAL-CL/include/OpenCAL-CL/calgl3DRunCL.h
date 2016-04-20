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
