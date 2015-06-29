#include "mod2CA3D.h"
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

//first elementary process
void life3DTransitionFunction(struct CALModel3D* ca, int i, int j, int k)
{
	int sum = 0, n;

	for (n=0; n<ca->sizeof_X; n++)
		sum += calGetX3Db(ca, Q.life, i, j, k, n);

	calSet3Db(ca, Q.life, i, j, k, sum%2);
}

//------------------------------------------------------------------------------
//					life3D simulation functions
//------------------------------------------------------------------------------

void life3DSimulationInit(struct CALModel3D* ca)
{
	int i, j, k, state;

	//initializing substate to 0
	calInitSubstate3Db(ca, Q.life, 0);

	//initializing a specific cell
	calSet3Db(ca, Q.life, 12, 12, 12, 1);
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
