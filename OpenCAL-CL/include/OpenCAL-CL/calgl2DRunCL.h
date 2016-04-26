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

#ifndef calgl2DRun_h
#define calgl2DRun_h

#include <OpenCAL-CL/calcl2D.h>
#include <time.h>
#include <pthread.h>

/*! \brief Structure that task is to update the cellular automata computation.
	This version is for 2D cellular automata.
*/
struct CALGLRun2D {
	CALbyte firstRun;				//!< Boolean for a first launch.
	CALbyte active;					//!< Boolean if it is active or not.
	CALbyte terminated;				//!< Boolean if it is terminated.
	struct CALRun2D* calRun;		//!< Reference to struct CALRun2D.
	time_t start_time;				//!< Time for which the computation is started.
	time_t end_time;				//!< Time for which the computation is ended.
	pthread_t thread;				//!< Reference to a thread variable.
	CALbyte stop;					//!< Boolean if it is stopped or not.
	CALint step;					//!< Initial step.
	struct CALCLModel2D* deviceCA;	//!< Reference to struct CALCLModel2D.
	CALbyte onlyOneTime;			//!< Boolean for a first launch.
	CALint fixedStep;				//!< Fixed step for transfer memory from device to host.
	CALint final_step;				//!< Final simulation step; if 0 the simulation becomes a loop.
	size_t * singleStepThreadNum;	//!< Number of thread for single step
	int dimNum;						//!< Number of dimensions
	size_t * threadNumMax;			//!< Pointer a size_t

};

/*! \brief Constructor for create a struct CALGLRun2D.
*/
struct CALGLRun2D* calglRunCLDef2D(
	struct CALCLModel2D* deviceCA,	//!< Reference to CALRun2D
	CALint fixedStep,				//!< Fixed step for transfer memory from device to host.
	CALint initial_step,			//!< Initial step.
	CALint final_step				//!< Final step.
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

// /*! \brief Update function for saving the final state to disk.
// */
// void calglSaveStateUpdater2DCL(
// 	struct CALGLRun2D* calglRun	//!< Struct for retrieve the cellular automata data.
// 	);

#endif
