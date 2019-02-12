#ifndef sciddicaS3Hex_h
#define sciddicaS3Hex_h

#include <OpenCAL/cal2D.h>
#include <OpenCAL/cal2DIO.h>
#include <OpenCAL-CL/calcl2D.h>
#include <time.h>

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

#define ACTIVE_CELLS
//#define VERBOSE


//cadef and rundef
extern struct CALModel2D* s3hex;
extern struct CALCLDeviceManager * calcl_device_manager; //the device manager object
extern struct CALCLModel2D * device_CA;									//the device-side CA

#define NUMBER_OF_OUTFLOWS 7
struct sciddicaS3hexSubstates {
	struct CALSubstate2Dr *z;	//topographic altitude
	struct CALSubstate2Dr *d;	//depth of regolith
	struct CALSubstate2Di *s;	//debris flow source
	struct CALSubstate2Dr *h;	//debris thickness
	struct CALSubstate2Dr *p;	//pseudo potential energy
    struct CALSubstate2Dr *fh[NUMBER_OF_OUTFLOWS];
    struct CALSubstate2Dr *fp[NUMBER_OF_OUTFLOWS];
};

struct sciddicaS3hexParameters {
	CALParameterr adh;			//adhesion
	CALParameterr rl;			//run-up loss
	CALParameterr r;			//relaxation rate
	CALParameterr f;			//height threshold, related to friction angle
	CALParameterr mt;			//mobilisation threshold
	CALParameterr pef;			//progressive erosion factor
//	CALParameterr ltt;			//landslide thickness threshold
};

extern struct sciddicaS3hexSubstates Q;
extern struct sciddicaS3hexParameters P;

#define KERNEL_SRC "./kernel/source/"
#define KERNEL_INC "./kernel/include/"
#define KERNEL_SRC_AC "./kernelActive/source/"
#define KERNEL_INC_AC "./kernelActive/include/"

#define KERNEL_S3HEXEROSION "s3hexErosion"
#define KERNEL_S3HEXFLOWSCOMPUTATION "s3hexFlowsComputation"
#define KERNEL_S3HEXWIDTHANDPOTENTIALUPDATE "s3hexWidthAndPotentialUpdate"
#define KERNEL_S3HEXCLEAROUTFLOWS "s3hexClearOutflows"
#define KERNEL_S3HEXENERGYLOSS "s3hexEnergyLoss"
#ifdef ACTIVE_CELLS
#define KERNEL_S3HEXROMOVEINACTIVECELLS "s3hexRemoveInactiveCells"
#endif

void sciddicaS3hexCADef();
void sciddicaS3hexCALCLDef();
void sciddicaS3hexLoadConfig();
void sciddicaS3hexSaveConfig();
void sciddicaS3hexExit();





#endif
