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

#include <cal2DRun.h>
#include <time.h>
#include <pthread.h>

struct CALUpdater2D {
	CALbyte firstRun;
	CALbyte active;
	CALbyte terminated;
	struct CALRun2D* calRun;
	time_t start_time;
	time_t end_time;
	pthread_t thread;
	CALbyte stop;
};

/*! Constructor
*/
struct CALUpdater2D* calglCreateUpdater2D(struct CALRun2D* calRun);

/*! Destructor
*/
void calglDestroyUpdater2D(struct CALUpdater2D* calUpdater);

void* calglFuncThreadUpdate2D(void * arg);

void calglStartThread2D(struct CALUpdater2D* calUpdater);

void calglUpdate2D(struct CALUpdater2D* calUpdater);

void calglSaveStateUpdater2D(struct CALUpdater2D* calUpdater);

#endif