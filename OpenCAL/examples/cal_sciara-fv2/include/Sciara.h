#ifndef CA_H_
#define CA_H_

extern "C"{
#include <OpenCAL/cal2D.h>
#include <OpenCAL/cal2DIO.h>
#include <OpenCAL/cal2DBuffer.h>
#include <OpenCAL/cal2DBufferIO.h>
#include <OpenCAL/cal2DRun.h>
#include <OpenCAL/cal2DUnsafe.h>
}
#include "GISInfo.h"
#include "vent.h"
#include <math.h>
#include <stdlib.h>

#define NUMBER_OF_OUTFLOWS 8
#define MOORE_NEIGHBORS 9
#define VON_NEUMANN_NEIGHBORS 5
#define STEPS 500


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
	//run simulation
	CALRun2D * run;

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
	CALint rows;
	CALint cols;
	CALreal rad2;
	unsigned int emission_time;
	vector<TEmissionRate> emission_rate;
	vector<TVent> vent;
	CALreal elapsed_time; //tempo trascorso dall'inizio della simulazione [s]
	int step;

	//TODO init effusion_duration
	CALreal effusion_duration;
	SciaraSubstates * substates;
	CALModel2D * model;

} Sciara;

extern Sciara *sciara;
extern int active;

void initSciara(char const* demPath, int steps);
void runSciara();


#endif /* CA_H_ */
