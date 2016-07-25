#ifndef sciddicaT_h
#define sciddicaT_h

#include <OpenCAL-CPU/opencal.h>
#include <OpenCAL-CPU/calIO.h>

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
extern struct CALModel* sciddicaT;

#define NUMBER_OF_OUTFLOWS 4

struct sciddicaTSubstates {
    struct CALSubstate_r *z;
    struct CALSubstate_r *h;
    struct CALSubstate_r *f[NUMBER_OF_OUTFLOWS];
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


#endif
