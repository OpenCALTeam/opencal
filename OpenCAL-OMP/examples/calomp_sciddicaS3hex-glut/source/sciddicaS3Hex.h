#ifndef sciddicaS3Hex_h
#define sciddicaS3Hex_h

#include <OpenCAL-OMP/cal2D.h>
#include <OpenCAL-OMP/cal2DIO.h>
#include <OpenCAL-OMP/cal2DRun.h>


#define ROWS 767
#define COLS 925
#define P_ADH 0.01
#define P_RL 0.6
#define P_R 0.99
#define P_F 0.1
#define P_MT 3.5
#define P_PEF 0.015
//#define P_LTT 0
#define STEPS 2500
#define DEM_PATH "./data/dem.txt"
#define REGOLITH_PATH "./data/regolith.txt"
#define SOURCE_PATH "./data/source.txt"
#define OUTPUT_PATH "./data/width_final.txt"

//#define ACTIVE_CELLS
//#define VERBOSE


//cadef and rundef
extern struct CALModel2D* s3hex;
extern struct CALRun2D* s3hexSimulation;

#define NUMBER_OF_OUTFLOWS 7
struct sciddicaTSubstates {
	struct CALSubstate2Dr *z;	//topographic altitude
	struct CALSubstate2Dr *d;	//depth of regolith
	struct CALSubstate2Di *s;	//debris flow source
	struct CALSubstate2Dr *h;	//debris thickness
	struct CALSubstate2Dr *p;	//pseudo potential energy
    struct CALSubstate2Dr *fh[NUMBER_OF_OUTFLOWS];
    struct CALSubstate2Dr *fp[NUMBER_OF_OUTFLOWS];
};

struct sciddicaTParameters {
	CALParameterr adh;			//adhesion
	CALParameterr rl;			//run-up loss
	CALParameterr r;			//relaxation rate
	CALParameterr f;			//height threshold, related to friction angle
	CALParameterr mt;			//mobilisation threshold
	CALParameterr pef;			//progressive erosion factor
//	CALParameterr ltt;			//landslide thickness threshold
};

extern struct sciddicaTSubstates Q;
extern struct sciddicaTParameters P;


void sciddicaTCADef();
void sciddicaTLoadConfig();
void sciddicaTSaveConfig();
void sciddicaTExit();



#endif
