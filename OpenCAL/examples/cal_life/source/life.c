#include <OpenCAL/cal2D.h>
#include <OpenCAL/cal2DIO.h>
#include <OpenCAL/cal2DRun.h>
#include <stdlib.h>

//-----------------------------------------------------------------------
//   THE LIFE CELLULAR AUTOMATON
//-----------------------------------------------------------------------

struct CALModel2D* life;
struct CALSubstate2Di *Q;
struct CALRun2D* life_simulation;


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

int main()
{
	//cadef and rundef
	life = calCADef2D(100, 100, CAL_MOORE_NEIGHBORHOOD_2D, CAL_SPACE_TOROIDAL, CAL_NO_OPT);
	life_simulation = calRunDef2D(life, 1, 1, CAL_UPDATE_EXPLICIT);

	//add substates
	Q = calAddSubstate2Di(life);

	//add transition function's elementary processes. 
	calAddElementaryProcess2D(life, life_transition_function);

	
	//set the whole substate to 0
	calInitSubstate2Di(life, Q, 0);

	//set a glider
	calInit2Di(life, Q, 0, 2, 1);
	calInit2Di(life, Q, 1, 0, 1);
	calInit2Di(life, Q, 1, 2, 1);
	calInit2Di(life, Q, 2, 1, 1);
	calInit2Di(life, Q, 2, 2, 1);

	//saving configuration
	calSaveSubstate2Di(life, Q, "./life_0000.txt");

	//simulation run
	calRun2D(life_simulation);
	
	//saving configuration
	calSaveSubstate2Di(life, Q, "./life_LAST.txt");

	//finalization
	calRunFinalize2D(life_simulation);
	calFinalize2D(life);

	return 0;
}

//-----------------------------------------------------------------------
