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


#define ROOT_DATA_DIR "."

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
	double Pclock;	//AC clock [s]
	double Pc;		//cell side
	double Pac;		//area of the cell
	double PTsol;	//temperature of solidification
	double PTvent;	//temperature of lava at vent
	// new Paramenters
	double Pr_Tsol;
	double Pr_Tvent;
	double a;		// parametro per calcolo Pr
	double b;		// parametro per calcolo Pr
	double Phc_Tsol;
	double Phc_Tvent;
	double c;		// parametro per calcolo hc
	double d;		// parametro per calcolo hc
	double Pcool;
	double Prho;	//density
	double Pepsilon;	//emissivity
	double Psigma;	//Stephen-Boltzamnn constant
	double Pcv;		//Specific heat
	double rad2;
	unsigned int emission_time;
	double effusion_duration;

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
	CALModel2D * host_CA;

} Sciara;


extern Sciara *sciara;
extern int active;

void initSciara(char * demPath);
void exitSciara();
void saveConfigSciara();
void simulationInitialize(struct CALModel2D* host_CA);


#endif /* CA_H_ */
