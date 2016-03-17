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

#ifndef calgl3DUpdater_h
#define calgl3DUpdater_h

#include <OpenCAL/cal3DRun.h>
#include <time.h>
#include <pthread.h>

/*! \brief Structure that task is to update the cellular automata computation.
	This version is for 3D cellular automata.
*/
struct CALUpdater3D {
	CALbyte firstRun;			//!< Boolean for a first launch.
	CALbyte active;				//!< Boolean if it is active or not.
	CALbyte terminated;			//!< Boolean if it is terminated.
	struct CALRun3D* calRun;	//!< Reference to struct CALRun3D.
	time_t start_time;			//!< Time for which the computation is started.
	time_t end_time;			//!< Time for which the computation is ended.
	pthread_t thread;			//!< Reference to a thread variable.
	CALbyte stop;				//!< Boolean if it is stopped or not.
	CALint step;
};

/*! \brief Constructor for create a struct CALUpdater3D.
*/
struct CALUpdater3D* calglCreateUpdater3D(
	struct CALRun3D* calRun	//!< Reference to CALRun3D
	);

/*! \brief Destructor for de-allocate memory.
*/
void calglDestroyUpdater3D(
	struct CALUpdater3D* calUpdater //!< Struct to destroy.
	);

/*! \brief Main update function, it is called by the thread.
*/
void* calglFuncThreadUpdate3D(
	void* arg	//!< Argument which is a struct CALUpdater3D.
	);

/*! \brief Function for starting the thread.
*/
void calglStartThread3D(
	struct CALUpdater3D* calUpdater	//!< Object which contains the thread to launch.
	);

/*! \brief Update function for updating the cellular automata computation.
*/
void calglUpdate3D(
	struct CALUpdater3D* calUpdater	//!< Struct for retrieve the cellular automata to update.
	);

/*! \brief Update function for saving the final state to disk.
*/
void calglSaveStateUpdater3D(
	struct CALUpdater3D* calUpdater	//!< Struct for retrieve the cellular automata data.
	);

#endif
