#ifndef sciddicaT_h
#define sciddicaT_h

#include <OpenCAL/cal2D.h>
#include <OpenCAL/cal2DIO.h>
#include <OpenCAL/cal2DRun.h>


#define ROWS 610
#define COLS 496
#define P_R 0.5
#define P_EPSILON 0.001
#define STEPS 4000
#define DEM_PATH "./data/dem.txt"
#define SOURCE_PATH "./data/source.txt"
#define OUTPUT_PATH "./data/width_final.txt"

#define ACTIVE_CELLS
//#define VERBOSE


//cadef and rundef
extern struct CALModel2D* sciddicaT;
extern struct CALRun2D* sciddicaTsimulation;


struct sciddicaTSubstates {
	struct CALSubstate2Dr *z;
	struct CALSubstate2Dr *h;
};

struct sciddicaTParameters {
	CALParameterr epsilon;
	CALParameterr r;
};

extern struct sciddicaTSubstates Q;
extern struct sciddicaTParameters P;


void sciddicaTCADef();
void sciddicaTLoadConfig();
void sciddicaTSaveConfig();
void sciddicaTExit();



#endif