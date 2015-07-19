#ifndef life3D_h
#define life3D_h

#include <OpenCAL/cal3D.h>
#include <OpenCAL/cal3DRun.h>


#define ROWS 65
#define COLS 65
#define LAYERS 65
#define STEPS 1000


#define VERBOSE


//cadef and rundef
extern struct CALModel3D* life3D;
extern struct CALRun3D* life3Dsimulation;

struct life3DSubstates {
	struct CALSubstate3Db *life;
};

extern struct life3DSubstates Q;


void life3DCADef();
void life3DSimulationInit(struct CALModel3D* ca);
void life3DExit();



#endif