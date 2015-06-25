#include "life3D.h"
#include <stdio.h>
#include <stdlib.h>

//-----------------------------------------------------------------------
//	   THE life3D(oy) cellular automaton definition section
//-----------------------------------------------------------------------

//cadef and rundef
struct CALModel3D* life3D;							//the cellular automaton
struct life3DSubstates Q;							//the substate
struct CALRun3D* life3Dsimulation;					//the simulartion run


//------------------------------------------------------------------------------
//					life3D transition function
//------------------------------------------------------------------------------
#define r1 (5)
#define r2 (26)
#define r3 (0)
#define r4 (13)

//first elementary process
void life3DTransitionFunction(struct CALModel3D* ca, int i, int j, int k)
{


	int sum = 0, n;
	CALbyte alive = calGet3Db(ca,Q.life,i,j,k);
	CALbyte nextState=alive;
	for (n=0; n<ca->sizeof_X; n++)
		sum += calGetX3Db(ca, Q.life, i, j, k, n);

	if(alive && sum >= r1 && sum <=r2)
		nextState=0;
	else if(!alive && (sum >= r2 && sum <= r4))
		nextState=1;

	calSet3Db(ca, Q.life, i, j, k, nextState);
}

//------------------------------------------------------------------------------
//					life3D simulation functions
//------------------------------------------------------------------------------
CALbyte nextBool(double probability){
    return rand() <  probability * ((double)RAND_MAX + 1.0);
}

void life3DSimulationInit(struct CALModel3D* ca)
{
	int i, j, k, state;

	//initializing substate to 0
	calInitSubstate3Db(ca, Q.life, 0);


	for(i=0;i<ca->rows;i++)
		for(j=0;j<ca->columns;j++)
			for(k=0;k<ca->slices;k++)
				calSet3Db(ca, Q.life, i, j, k, nextBool(0.65));
			
}

CALbyte life3DSimulationStopCondition(struct CALModel3D* life3D)
{
	if (life3Dsimulation->step >= STEPS)
		return CAL_TRUE;
	return CAL_FALSE;
}

//------------------------------------------------------------------------------
//					life3D CADef and runDef
//------------------------------------------------------------------------------

void life3DCADef()
{
	//cadef and rundef
	life3D = calCADef3D (ROWS, COLS, LAYERS, CAL_MOORE_NEIGHBORHOOD_3D, CAL_SPACE_TOROIDAL, CAL_NO_OPT);
	life3Dsimulation = calRunDef3D(life3D, 1, CAL_RUN_LOOP, CAL_UPDATE_IMPLICIT);

	//add transition function's elementary processes
	calAddElementaryProcess3D(life3D, life3DTransitionFunction);

	//add substates
	Q.life = calAddSubstate3Db(life3D);
		
	//simulation run setup
	calRunAddInitFunc3D(life3Dsimulation, life3DSimulationInit); calRunInitSimulation3D(life3Dsimulation);
	calRunAddStopConditionFunc3D(life3Dsimulation, life3DSimulationStopCondition);
}

//------------------------------------------------------------------------------
//					life3D finalization function
//------------------------------------------------------------------------------

void life3DExit()
{	
	//finalizations
	calRunFinalize3D(life3Dsimulation);
	calFinalize3D(life3D);
}
