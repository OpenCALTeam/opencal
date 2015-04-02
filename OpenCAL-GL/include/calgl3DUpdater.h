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
