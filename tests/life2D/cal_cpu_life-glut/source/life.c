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

void lifeTransitionFunction(struct CALModel* ca, int* central_cell, int number_of_dimensions)
{
    int sum = 0, n;
    int size_of_X = calGetSizeOfX(ca);
    for (n = 1; n<size_of_X; n++)
        sum += calGetX_i(ca, life.Q, central_cell, n);

    if ((sum == 3) || (sum == 2 && calGet_i(ca, life.Q, central_cell) == 1))
        calSet_i(ca, life.Q, central_cell, 1);
    else
        calSet_i(ca, life.Q, central_cell, 0);
}

//------------------------------------------------------------------------------
//					iso simulation functions
//------------------------------------------------------------------------------

void randSimulationInit(struct CALModel* ca)
{
    calInit_i(ca, life.Q, calGetCell(life.model, 2, 4), STATE_ALIVE);
    calInit_i(ca, life.Q, calGetCell(life.model, 3, 2), STATE_ALIVE);
    calInit_i(ca, life.Q, calGetCell(life.model, 3, 4), STATE_ALIVE);
    calInit_i(ca, life.Q, calGetCell(life.model, 4, 3), STATE_ALIVE);
    calInit_i(ca, life.Q, calGetCell(life.model, 4, 4), STATE_ALIVE);
}

//------------------------------------------------------------------------------
//					Some functions...
//------------------------------------------------------------------------------

void CADef(struct CellularAutomaton* ca)
{
    //cadef
    life.model = calCADef ( calDefDimensions(2, ROWS,COLS), CAL_MOORE_NEIGHBORHOOD, CAL_SPACE_TOROIDAL, CAL_NO_OPT, 0, 0);
    //add substates
    life.Q = calAddSubstate_i(life.model, CAL_INIT_BOTH, 0);

    //add transition function's elementary processes
    calAddLocalProcess(life.model, lifeTransitionFunction);

    //simulation run setup
    calAddInitFunc(life.model, randSimulationInit);
    calForceInit(life.model);
}

void isoExit(struct CellularAutomaton* ca)
{
    //finalizations
    calFinalize(ca->model);
}

//------------------------------------------------------------------------------
