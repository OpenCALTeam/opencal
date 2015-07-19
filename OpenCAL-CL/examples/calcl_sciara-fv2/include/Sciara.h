#ifndef CA_H_
#define CA_H_

extern "C"{
#include <OpenCAL/cal2D.h>
#include <OpenCAL/cal2DIO.h>
#include <OpenCAL/cal2DBuffer.h>
#include <OpenCAL/cal2DBufferIO.h>
#include <OpenCAL/cal2DRun.h>
}
#include "GISInfo.h"
#include "vent.h"
#include <math.h>
#include <stdlib.h>

#define NUMBER_OF_OUTFLOWS 8
#define MOORE_NEIGHBORS 9
#define VON_NEUMANN_NEIGHBORS 5
#define STEPS 1

#define SAVE_PATH "./data/2006_SAVE/2006"


#ifdef _WIN32
	#define ROOT_DATA_DIR "."
#else
	#define ROOT_DATA_DIR "."
#endif // _WIN32

#define KERNEL_SRC_AC ROOT_DATA_DIR"/kernelActive/source/"
#define KERNEL_INC_AC ROOT_DATA_DIR"/kernelActive/include/"
#define KERNEL_SRC ROOT_DATA_DIR"/kernel/source/"
#define KERNEL_INC ROOT_DATA_DIR"/kernel/include/"


typedef struct {

	struct CALSubstate2Dr * Sz;		//Altitude
	struct CALSubstate2Dr * Slt;	//Lava thickness
	struct CALSubstate2Dr * St;		//Lava temperature
	//One matrix substate
	struct CALSubstate2Dr * Sz_t0;	//Matrix of the pre-event topography
	struct CALSubstate2Di * Mv;		//Matrix of the vents
	struct CALSubstate2Db * Mb;		//Matrix of the topography bound
	struct CALSubstate2Dr * Msl;	//Matrix of the solidified lava

	//Flows Substates????
	struct CALSubstate2Dr *f[NUMBER_OF_OUTFLOWS];		//Flows Substates

} SciaraSubstates;

typedef struct{

	// Parameters
	CALreal Pclock;	//AC clock [s]
	CALreal Pc;		//cell side
	CALreal Pac;		//area of the cell
	CALreal PTsol;	//temperature of solidification
	CALreal PTvent;	//temperature of lava at vent
	// new Paramenters
	CALreal Pr_Tsol;
	CALreal Pr_Tvent;
	CALreal a;		// parametro per calcolo Pr
	CALreal b;		// parametro per calcolo Pr
	CALreal Phc_Tsol;
	CALreal Phc_Tvent;
	CALreal c;		// parametro per calcolo hc
	CALreal d;		// parametro per calcolo hc
	CALreal Pcool;
	CALreal Prho;	//density
	CALreal Pepsilon;	//emissivity
	CALreal Psigma;	//Stephen-Boltzamnn constant
	CALreal Pcv;		//Specific heat
	CALreal rad2;
	unsigned int emission_time;
	CALreal effusion_duration;


} Parameters;

typedef struct{
	//run simulation
	CALRun2D * run;
	CALint rows;
	CALint cols;

	Parameters parameters;

	CALreal elapsed_time; //tempo trascorso dall'inizio della simulazione [s]
	vector<TEmissionRate> emission_rate;
	vector<TVent> vent;
	int step;

	SciaraSubstates * substates;
	CALModel2D * model;

} Sciara;


extern Sciara *sciara;
extern int active;

void initSciara(char * demPath);
void exitSciara();
void saveConfigSciara();
void simulationInitialize(struct CALModel2D* model);


#endif /* CA_H_ */
