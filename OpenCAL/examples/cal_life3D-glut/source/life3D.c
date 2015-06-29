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


/*
 * 3-D Life is "played" on an arbitrarily large three dimensional grid of cubes. Each cube represents a cell, which is either "alive"(i.e. filled in) or "dead" (i.e. not filled in). Each cube (cell) is viewed as having 26 neighbors or adjacent touching cells. A cell is "alive" or "dead", based on some transition rules i.e. on the number of living neighbors it has.

Rules: We can formalize the rules as follows: Define the transition rule as the 4-tuple of four numbers (x, y, z, d). The first two numbers dictate the fate of the living cells, and the rest dictate the fate of the dead cells.

x - indicates the fewest living neighbors a cell must have to keep from being undernourished; y - indicates the most it can have before it will be overcrouded; z - indicates the fewest living neighbors a dead cell must have to come alive; d- indicates the most it can have to come alive.

Example:

Rule 4555 represents, that a living cell dies if it has less than four or more than five living neighbors and dead cell becomes alive if it has exactly five living neighbors.

Various configurations of living cells show suprisingly complex and almost lifelike behavior, that's why the name " 3-D Life " is quiet appropriate.


 */

#define r1 (24)
#define r2 (24)
#define r3 (4)
#define r4 (4)
//first elementary process
void life3DTransitionFunction(struct CALModel3D* ca, int i, int j, int k)
{


	int sum = 0, n;
	CALbyte alive = calGet3Db(ca,Q.life,i,j,k);
	CALbyte nextState=alive;
	for (n=0; n<ca->sizeof_X; n++)
		sum += calGetX3Db(ca, Q.life, i, j, k, n);

	if(alive && (sum < r1 || sum >r2))
		nextState=0;
	else if(!alive && (sum >=r3 && sum <=r4))
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
				calSet3Db(ca, Q.life, i, j, k, nextBool(0.01));
			
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
