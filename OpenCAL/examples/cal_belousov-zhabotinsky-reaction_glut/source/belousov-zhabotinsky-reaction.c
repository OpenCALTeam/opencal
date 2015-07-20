#include "belousov-zhabotinsky-reaction.h"
#include <math.h>
#include <stdlib.h>

//-----------------------------------------------------------------------
//	   THE "belousov-zhabotinsky" cellular automaton definition section
//-----------------------------------------------------------------------

//cadef and rundef
struct CellularAutomaton zhabotinsky;

//------------------------------------------------------------------------------
//		belousov-zhabotinsky-Dewdney transition function
//------------------------------------------------------------------------------

void belousovZhabotinskyDewdneyTransitionFunction(struct CALModel2D* ca, int i, int j){
	CALint state=calGet2Di(zhabotinsky.model,zhabotinsky.Q,i,j);
	CALint numills=0,numInfected=0,numHealty=0;
	int n;
	int sum=0;
	for(n=1;n<zhabotinsky.model->sizeof_X;n++){
		CALint neighState=calGetX2Di(zhabotinsky.model,zhabotinsky.Q,i,j,n);
		sum+=neighState;
		if(neighState==QQ){
			numills++;
		}else if(neighState >1 && neighState < QQ){
			numInfected++;
		}else if(state==STATE_HEALTY)
			numHealty++;
	}

	CALint newstate =0;
	if(state==STATE_ILL){
		newstate=STATE_HEALTY;
	}else if(state==STATE_HEALTY){
		newstate=(numInfected/k1)+(numills/k2) +1;
	}else{
		newstate=sum/(9-numHealty) + G;
	}
	if(newstate>QQ)
		newstate=QQ;
//	printf("%i,%i,%i,%i,%i\n",i,j,numills,numInfected,numHealty);
	calSet2Di(zhabotinsky.model,zhabotinsky.Q,i,j,newstate);
}

//------------------------------------------------------------------------------
//					iso simulation functions
//------------------------------------------------------------------------------

void randSimulationInit(struct CALModel2D* ca)
{
	CALint i, j, state;

	srand(0);
	for (i = 0; i<ca->rows; i++)
		for (j = 0; j<ca->columns; j++)
		{
			state = (rand() % (QQ))+1;
			calInit2Di(ca, zhabotinsky.Q, i, j, state);
		}
}


void CADef(struct CellularAutomaton* ca)
{
	//cadef and rundef
	zhabotinsky.model = calCADef2D (ROWS, COLS, CAL_MOORE_NEIGHBORHOOD_2D, CAL_SPACE_TOROIDAL, CAL_NO_OPT);
	zhabotinsky.run = calRunDef2D(zhabotinsky.model, 1, CAL_RUN_LOOP, CAL_UPDATE_IMPLICIT);
	//add substates
	zhabotinsky.Q = calAddSubstate2Di(zhabotinsky.model);

	//add transition function's elementary processes
	calAddElementaryProcess2D(zhabotinsky.model, belousovZhabotinskyDewdneyTransitionFunction);

	//simulation run setup
	calRunAddInitFunc2D(zhabotinsky.run, randSimulationInit);
}

void Init(struct CellularAutomaton* ca)
{
	randSimulationInit(zhabotinsky.model);
}

void isoExit(struct CellularAutomaton* ca)
{
	//finalizations
	calRunFinalize2D(ca->run);
	calFinalize2D(ca->model);
}

//------------------------------------------------------------------------------

