// Conway's game of Life Cellular Automaton

#include <OpenCAL-OMP/cal2D.h>
#include <OpenCAL-OMP/cal2DIO.h>
#include <OpenCAL-OMP/cal2DRun.h>

// declare CA, substate and simulation objects
struct CALModel2D* life;
struct CALSubstate2Di* Q;
struct CALRun2D* life_simulation;

// The cell's transition function
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
	// define of the life CA and life_simulation simulation objects
	life = calCADef2D(8, 16, CAL_MOORE_NEIGHBORHOOD_2D, CAL_SPACE_TOROIDAL, CAL_NO_OPT);
	life_simulation = calRunDef2D(life, 1, 1, CAL_UPDATE_IMPLICIT);

   	//put OpenCAL - OMP in unsafe state execution(to allow unsafe operation to be used)
	calSetUnsafe2D(life);

	// add the Q substate to the life CA
	Q = calAddSubstate2Di(life);

	// add transition function's elementary process
	calAddElementaryProcess2D(life, life_transition_function);

	// set the whole substate to 0
	calInitSubstate2Di(life, Q, 0);

	// set a glider
	calInit2Di(life, Q, 0, 2, 1);
	calInit2Di(life, Q, 1, 0, 1);
	calInit2Di(life, Q, 1, 2, 1);
	calInit2Di(life, Q, 2, 1, 1);
	calInit2Di(life, Q, 2, 2, 1);

	// save the Q substate to file
	calSaveSubstate2Di(life, Q, "./life_0000.txt");

	// simulation run
	calRun2D(life_simulation);

	// save the Q substate to file
	calSaveSubstate2Di(life, Q, "./life_LAST.txt");

	// finalize simulation and CA objects
	calRunFinalize2D(life_simulation);
	calFinalize2D(life);

	return 0;
}
