#ifndef sciddicaT_h
#define sciddicaT_h


#include <cal2DIO.h>
#include <cal2DRun.h>


#define ROWS 610
#define COLS 496
#define P_R 0.5
#define P_EPSILON 0.001
#define STEPS  1000
//4000
#define NSIZESTEP  5



#define ACTIVE_CELLS

#ifdef _WIN32
	#define ROOT_DATA_DIR "."
#else
	#define ROOT_DATA_DIR "./CALCL_Sciddica"
#endif // _WIN32

#define DEM_PATH ROOT_DATA_DIR"/data/dem.txt"
#define SOURCE_PATH ROOT_DATA_DIR"/data/source.txt"
#define OUTPUT_PATH ROOT_DATA_DIR"/data/width_final.txt"

#define KERNEL_SRC_AC ROOT_DATA_DIR"/kernelActive/source/"
#define KERNEL_INC_AC ROOT_DATA_DIR"/kernelActive/include/"
#define KERNEL_SRC ROOT_DATA_DIR"/kernel/source/"
#define KERNEL_INC ROOT_DATA_DIR"/kernel/include/"

//cadef and rundef
extern struct CALModel2D* sciddicaT;
extern struct CALRun2D* sciddicaTsimulation;

#define NUMBER_OF_OUTFLOWS 4

struct sciddicaTSubstates {
	struct CALSubstate2Dr *z;
	struct CALSubstate2Dr *h;
	struct CALSubstate2Dr *f[NUMBER_OF_OUTFLOWS];
};

struct sciddicaTParameters {
	CALParameterr epsilon;
	CALParameterr r;
};

extern struct sciddicaTSubstates Q;
extern struct sciddicaTParameters P;


void sciddicaTCADef();
void sciddicaTLoadConfig();
void sciddicaTSaveConfig(char * outputPath);
void sciddicaTExit();



struct SciddicaTMain
{
	struct CALModel2D* M;
	struct sciddicaTSubstates Q;
	struct sciddicaTParameters P;
};

void explicitInit(struct SciddicaTMain *);


#endif
