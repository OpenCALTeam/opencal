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