#ifndef mbusu_h
#define mbusu_h

#include <OpenCAL/cal3D.h>
#include <OpenCAL/cal3DRun.h>

#define YOUT 29
#define YIN 0
#define XE 159
#define XW 0
#define ZSUP 129
#define ZFONDO 0

#define COLS YOUT+1
#define ROWS XE+1
#define LAYERS ZSUP+1

#define REFRESH 100
#define VERBOSE


//cadef and rundef
extern struct CALModel3D* mbusu;
extern struct CALRun3D* mbusuSimulation;

struct mbusuSubstates {
	struct CALSubstate3Dr *teta;
	struct CALSubstate3Dr *moist_cont; 
	struct CALSubstate3Dr *psi;
	struct CALSubstate3Dr *k;
	struct CALSubstate3Dr *h;
	struct CALSubstate3Dr *dqdh;
	struct CALSubstate3Dr *convergence;
	struct CALSubstate3Dr *moist_diff;
};

extern struct mbusuSubstates Q;

extern float prm_vis;


void mbusuCADef();
void mbusuSimulationInit(struct CALModel3D* ca);
void mbusuExit();



#endif