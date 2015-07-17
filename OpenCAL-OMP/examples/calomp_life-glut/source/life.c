#include "life.h"
#include <math.h>
#include <stdlib.h>

//-----------------------------------------------------------------------
//	   THE "life" cellular automaton definition section
//-----------------------------------------------------------------------

//cadef and rundef
struct CellularAutomaton life;

//------------------------------------------------------------------------------
//					life transition function
//------------------------------------------------------------------------------

void lifeTransitionFunction(struct CALModel2D* ca, int i, int j)
{
	int sum = 0, n;
	for (n = 1; n<ca->sizeof_X; n++)
		sum += calGetX2Di(ca, life.Q, i, j, n);

	if ((sum == 3) || (sum == 2 && calGet2Di(ca, life.Q, i, j) == 1))
		calSet2Di(ca, life.Q, i, j, 1);
	else
		calSet2Di(ca, life.Q, i, j, 0);
	
}

//------------------------------------------------------------------------------
//					iso simulation functions
//------------------------------------------------------------------------------

void randSimulationInit(struct CALModel2D* ca)
{
	CALint i, j, state;
	
	calInitSubstate2Di(ca, life.Q, STATE_DEAD);

	srand(0);
	
	for (i = 0; i<ca->rows; i++)
		for (j = 0; j<ca->columns; j++)
		{	
			state = rand() % 2;
			calInit2Di(ca, life.Q, i, j, state);
		}			
}

//------------------------------------------------------------------------------
//					Some functions...
//------------------------------------------------------------------------------

void CADef(struct CellularAutomaton* ca)
{
	//cadef and rundef
	life.model = calCADef2D (ROWS, COLS, CAL_MOORE_NEIGHBORHOOD_2D, CAL_SPACE_TOROIDAL, CAL_NO_OPT);
	life.run = calRunDef2D(life.model, 1, CAL_RUN_LOOP, CAL_UPDATE_IMPLICIT);
	//add substates
	life.Q = calAddSubstate2Di(life.model);
	
	//add transition function's elementary processes
	calAddElementaryProcess2D(life.model, lifeTransitionFunction);

	//simulation run setup
	calRunAddInitFunc2D(life.run, randSimulationInit);
}

void Init(struct CellularAutomaton* ca)
{
	randSimulationInit(life.model);
}

void isoExit(struct CellularAutomaton* ca)
{	
	//finalizations
	calRunFinalize2D(ca->run);
	calFinalize2D(ca->model);
}

//------------------------------------------------------------------------------
