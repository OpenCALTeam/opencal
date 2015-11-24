// mod2 3D Cellular Automaton

#include <OpenCAL-OMP/cal3D.h>
#include <OpenCAL-OMP/cal3DIO.h>
#include <OpenCAL-OMP/cal3DRun.h>

#define ROWS 5
#define COLS 7
#define LAYERS 3

// declare CA, substate and simulation objects
struct CALModel3D* mod2;
struct CALSubstate3Db* Q;
struct CALRun3D* mod2_simulation;

// The cell's transition function
void mod2_transition_function(struct CALModel3D* ca, int i, int j, int k)
{
	int sum = 0, n;

	for (n=0; n<ca->sizeof_X; n++)
		sum += calGetX3Db(ca, Q, i, j, k, n);

	calSet3Db(ca, Q, i, j, k, sum%2);
}

int main()
{
	// define of the mod2 CA and mod2_simulation simulation objects
	mod2 = calCADef3D(ROWS, COLS, LAYERS, CAL_MOORE_NEIGHBORHOOD_3D, CAL_SPACE_TOROIDAL, CAL_NO_OPT);
	mod2_simulation = calRunDef3D(mod2, 1, 1, CAL_UPDATE_IMPLICIT);

	// add the Q substate to the mod2 CA
	Q = calAddSubstate3Db(mod2);

	// add transition function's elementary process
	calAddElementaryProcess3D(mod2, mod2_transition_function);

	// set the whole substate to 0
	calInitSubstate3Db(mod2, Q, 0);

	// set a seed at position (2, 3, 1)
	calInit3Db(mod2, Q, 2, 3, 1, 1);

	// save the Q substate to file
	calSaveSubstate3Db(mod2, Q, "./mod2_0000.txt");

	// simulation run
	calRun3D(mod2_simulation);

	// save the Q substate to file
	calSaveSubstate3Db(mod2, Q, "./mod2_LAST.txt");

	// finalize simulation and CA objects
	calRunFinalize3D(mod2_simulation);
	calFinalize3D(mod2);

	return 0;
}
