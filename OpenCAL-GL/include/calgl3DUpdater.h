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

#include <cal3DRun.h>
#include <time.h>
#include <pthread.h>

/* forward declaration of struct CALUpdater2D, quick fix for compiling whit gnu toolchains */

struct CALUpdater2D;

struct CALUpdater3D {
	CALbyte firstRun;
	CALbyte active;
	CALbyte terminated;
	struct CALRun3D* calRun;
	time_t start_time;
	time_t end_time;
	pthread_t thread;
	CALbyte stop;
};

/*! Constructor
*/
struct CALUpdater3D* calglCreateUpdater3D(struct CALRun3D* calRun);

/*! Destructor
*/
void calglDestroyUpdater3D(struct CALUpdater3D* calUpdater);

void* calglFuncThreadUpdate3D(void * arg);

/*TODO why CALUpdater2D and no CALUpdater3D ? */
/* void calglStartThread3D(struct CALUpdater2D* calUpdater); */
void calglStartThread3D(struct CALUpdater3D* calUpdater);

void calglUpdate3D(struct CALUpdater3D* calUpdater);

void calglSaveStateUpdater3D(struct CALUpdater3D* calUpdater);

#endif
