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

#ifndef calgl2DUpdater_h
#define calgl2DUpdater_h

#include <OpenCAL/cal2DRun.h>
#include <time.h>
#include <pthread.h>

/*! \brief Structure that task is to update the cellular automata computation.
	This version is for 2D cellular automata.
*/
struct CALUpdater2D {
	CALbyte firstRun;			//!< Boolean for a first launch.
	CALbyte active;				//!< Boolean if it is active or not.
	CALbyte terminated;			//!< Boolean if it is terminated.
	struct CALRun2D* calRun;	//!< Reference to struct CALRun2D.
	time_t start_time;			//!< Time for which the computation is started.
	time_t end_time;			//!< Time for which the computation is ended.
	pthread_t thread;			//!< Reference to a thread variable.
	CALbyte stop;				//!< Boolean if it is stopped or not.
};

/*! \brief Constructor for create a struct CALUpdater2D.
*/
struct CALUpdater2D* calglCreateUpdater2D(
	struct CALRun2D* calRun	//!< Reference to CALRun2D
	);

/*! \brief Destructor for de-allocate memory.
*/
void calglDestroyUpdater2D(
	struct CALUpdater2D* calUpdater //!< Struct to destroy.
	);

/*! \brief Main update function, it is called by the thread.
*/
void* calglFuncThreadUpdate2D(
	void* arg	//!< Argument which is a struct CALUpdater2D.
	);

/*! \brief Function for starting the thread.
*/
void calglStartThread2D(
	struct CALUpdater2D* calUpdater	//!< Object which contains the thread to launch.
	);

/*! \brief Update function for updating the cellular automata computation.
*/
void calglUpdate2D(
	struct CALUpdater2D* calUpdater	//!< Struct for retrieve the cellular automata to update.
	);

/*! \brief Update function for saving the final state to disk.
*/
void calglSaveStateUpdater2D(
	struct CALUpdater2D* calUpdater	//!< Struct for retrieve the cellular automata data.
	);

#endif