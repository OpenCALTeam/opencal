#include <cal2D.h>
#include <cal2DIO.h>
#include <cal2DRun.h>
#include <stdlib.h>

//-----------------------------------------------------------------------
//   THE LIFE CELLULAR AUTOMATON
//-----------------------------------------------------------------------

struct CALSubstate2Di *Q;
struct CALSubstate2Di *Qzito;

CALParameterr pr = 0;


void life_transition_function(struct CALModel2D* life, int i, int j)
{
	int sum = 0, n;
	for (n=1; n<life->sizeof_X; n++)
		sum += calGetX2Di(life, Q, i, j, n);

	if ((sum == 3) || (sum == 2 && calGet2Di(life, Q, i, j) == 1))
		calSet2Di(life, Q, i, j, 1);
	else
		calSet2Di(life, Q, i, j, 0);
}


void life_elementary_process_1(struct CALModel2D* life, int i, int j)
{
	calSet2Di(life, Qzito, i, j, rand() % 2);
}


void life_elementary_process_2(struct CALModel2D* life, int i, int j)
{
	calSet2Di(life, Qzito, i, j, rand() % 2);
}


void life_init(struct CALModel2D* life)
{
	//add cells to the set of active ones
	calAddActiveCell2D(life, 0, 0);
	calAddActiveCell2D(life, 0, 1);
	
	//this is needed only if one or more cells are added or eliminated from the computationally active cells
	calUpdateActiveCells2D(life);
}

void life_steering(struct CALModel2D* life)
{
	int i, j, n;

	if (life->A.cells)
		for (n=0; n<life->A.size_current; n++)
			calSet2Di(life, Qzito, life->A.cells[n].i, life->A.cells[n].j, (calGet2Di(life, Qzito, life->A.cells[n].i, life->A.cells[n].j) + 1) % 2);
	else
		for (i=0; i<life->rows; i++)
			for (j=0; j<life->columns; j++)
				calSet2Di(life, Qzito, i, j, (calGet2Di(life, Qzito, i, j) + 1) % 2);

	//this call is needed only in case CAL_UPDATE_EXPLICIT
	calUpdateSubstate2Di(life, Qzito);
	
	//this is needed only if one or more cells are added or eliminated from the computationally active cells
	calUpdateActiveCells2D(life);
}

void life_finalize(struct CALModel2D* life)
{
	//add cells to the set of active ones
	calRemoveActiveCell2D(life, 0, 0);
	calRemoveActiveCell2D(life, 0, 1);

	//this is needed only if one or more cells are added or eliminated from the computationally active cells
	calUpdateActiveCells2D(life);
}


void life()
{
	//cadef and rundef
	struct CALModel2D* life = calCADef2D (100, 100, CAL_CUSTOM_NEIGHBORHOOD_2D, CAL_SPACE_TOROIDAL, CAL_OPT_ACTIVE_CELLS);
	struct CALRun2D* life_simulation = calRunDef2D(life, 1, 1, CAL_UPDATE_EXPLICIT);

	//initialize the random number function
	srand(0);

	//add transition function's elementary processes. 
	calAddElementaryProcess2D(life, life_transition_function);
	calAddElementaryProcess2D(life, life_elementary_process_1);
	calAddElementaryProcess2D(life, life_elementary_process_2);

	//add neighbors of the Moore neighborhood
	calAddNeighbor2D(life,   0,   0);	//this is the neighbor 0 (central cell)
    calAddNeighbor2D(life, - 1,   0);	//this is the neighbor 1
    calAddNeighbor2D(life,   0, - 1);	//this is the neighbor 2
	calAddNeighbor2D(life,   0, + 1);	//this is the neighbor 3
    calAddNeighbor2D(life, + 1,   0);	//this is the neighbor 4
    calAddNeighbor2D(life, - 1, - 1);	//this is the neighbor 5
    calAddNeighbor2D(life, + 1, - 1);	//this is the neighbor 6
	calAddNeighbor2D(life, + 1, + 1);	//this is the neighbor 7
	calAddNeighbor2D(life, - 1, + 1);	//this is the neighbor 8

	//add substates
	Q = calAddSubstate2Di(life);
	Qzito = calAddSubstate2Di(life);
	
	//set the whole substate to 0
	calInitSubstate2Di(life, Q, 0);
	calInitSubstate2Di(life, Qzito, 0);
	
	//set a glider
	calInit2Di(life, Q, 0, 2, 1);
	calInit2Di(life, Q, 1, 0, 1);
	calInit2Di(life, Q, 1, 2, 1);
	calInit2Di(life, Q, 2, 1, 1);
	calInit2Di(life, Q, 2, 2, 1);
	//set another glider
	/*calInit2Di(life, Q, 49, 50, 1);
	calInit2Di(life, Q, 50, 51, 1);
	calInit2Di(life, Q, 51, 49, 1);
	calInit2Di(life, Q, 51, 50, 1);
	calInit2Di(life, Q, 51, 51, 1);*/
	//set another glider
	/*calInit2Di(life, Q, 97, 98, 1);
	calInit2Di(life, Q, 98, 99, 1);
	calInit2Di(life, Q, 99, 97, 1);
	calInit2Di(life, Q, 99, 98, 1);
	calInit2Di(life, Q, 99, 99, 1);*/


	//saving configuration
	calSaveSubstate2Di(life, Q, "./data/life_0000.txt");
	calSaveSubstate2Di(life, Qzito, "./data/zito_0000.txt");


	//simulation run
	calRunAddInitFunc2D(life_simulation, life_init);
	calRunAddSteeringFunc2D(life_simulation, life_steering);
	calRunAddFinalizeFunc2D(life_simulation, life_finalize);
	calRun2D(life_simulation);
	calRunFinalize2D(life_simulation);

	
	//saving configuration
	calSaveSubstate2Di(life, Q, "./data/life_LAST.txt");
	calSaveSubstate2Di(life, Qzito, "./data/zito_LAST.txt");
	
	//finalization
	calFinalize2D(life);
}

//-----------------------------------------------------------------------

int main()
{
	life();
	return 0;
}

//-----------------------------------------------------------------------
