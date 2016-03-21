// mod2 3D Cellular Automaton

#include <OpenCAL-GL/calgl3D.h>
#include <OpenCAL-GL/calgl3DWindow.h>
#include <stdio.h>
#include <stdlib.h>

#define ROWS 25
#define COLS 25
#define LAYERS 25
#define STEPS 4000

// declare CA, substate and simulation objects
struct CALModel3D* mod2;							//the cellular automaton
struct CALSubstate3Db *Q;							//the substate Q
struct CALRun3D* mod2simulation;			//the simulartion run


// The cell's transition function (first and only elementary process)
void mod2TransitionFunction(struct CALModel3D* ca, int i, int j, int k)
{
	int sum = 0, n;

	for (n=0; n<ca->sizeof_X; n++)
		sum += calGetX3Db(ca, Q, i, j, k, n);

	calSet3Db(ca, Q, i, j, k, sum%2);
}

// Simulation init callback function used to set a seed at position (24, 0, 0)
void mod2SimulationInit(struct CALModel3D* ca)
{
	//initializing substate to 0
	calInitSubstate3Db(ca, Q, 0);
	//setting a specific cell
	calSet3Db(ca, Q, 24, 0, 0, 1);
}

// Stop condition callback function
CALbyte mod2SimulationStopCondition(struct CALModel3D* mod2)
{
	if (mod2simulation->step >= STEPS)
		return CAL_TRUE;
	return CAL_FALSE;
}

// Callback unction called just before program termination
void exitFunction(void)
{
	//finalizations
	calRunFinalize3D(mod2simulation);
	calFinalize3D(mod2);
}

// The main() function
int main(int argc, char** argv)
{
	// Declare a viewer object
	struct CALGLDrawModel3D* drawModel;

	atexit(exitFunction);

	//cadef and rundef
	mod2 = calCADef3D(ROWS, COLS, LAYERS, CAL_MOORE_NEIGHBORHOOD_3D, CAL_SPACE_TOROIDAL, CAL_NO_OPT);
	mod2simulation = calRunDef3D(mod2, 1, 4001, CAL_UPDATE_IMPLICIT);
	//add substates
	Q = calAddSubstate3Db(mod2);
	//add transition function's elementary processes
	calAddElementaryProcess3D(mod2, mod2TransitionFunction);

	//simulation run setup
	calRunAddInitFunc3D(mod2simulation, mod2SimulationInit);
	calRunInitSimulation3D(mod2simulation);	//It is required in the case the simulation main loop is explicitated; similarly for calRunFinalizeSimulation3D
	calRunAddStopConditionFunc3D(mod2simulation, mod2SimulationStopCondition);

	// Initialize the viewer
	calglInitViewer("mod2 3D CA viewer", 1.0f, 400, 400, 40, 40, CAL_TRUE, 1);
	//drawModel definition
	drawModel = calglDefDrawModel3D(CALGL_DRAW_MODE_FLAT, "3D view", mod2, mod2simulation);
	calglAdd3Db(drawModel, NULL, &Q, CALGL_TYPE_INFO_VERTEX_DATA, CALGL_TYPE_INFO_USE_NO_COLOR, CALGL_DATA_TYPE_DYNAMIC);
	calglColor3D(drawModel, 0.5f, 0.5f, 0.5f, 1.0f);
	calglAdd3Db(drawModel, Q, &Q, CALGL_TYPE_INFO_COLOR_DATA, CALGL_TYPE_INFO_USE_CURRENT_COLOR, CALGL_DATA_TYPE_DYNAMIC);
	calglAdd3Db(drawModel, Q, &Q, CALGL_TYPE_INFO_NORMAL_DATA, CALGL_TYPE_INFO_USE_NO_COLOR, CALGL_DATA_TYPE_DYNAMIC);

	// New functions for hide/display intervals of cells
	//calglHideDrawKBound3D(drawModel, 0, drawModel->calModel->slices);
	//calglDisplayDrawKBound3D(drawModel, 4, 10);
	//calglDisplayDrawKBound3D(drawModel, 20, 25);
	//calglHideDrawJBound3D(drawModel, 0, drawModel->calModel->columns);
	//calglDisplayDrawJBound3D(drawModel, 2, 6);
	//calglDisplayDrawJBound3D(drawModel, 18, 21);

	calglMainLoop3D(argc, argv);

	return 0;
}
