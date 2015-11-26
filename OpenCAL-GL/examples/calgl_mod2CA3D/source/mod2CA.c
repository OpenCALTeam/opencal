#include <OpenCAL-GL/calgl3D.h>
#include <OpenCAL-GL/calgl3DWindow.h>
#include <stdio.h>
#include <stdlib.h>

#define ROWS 25
#define COLS 25
#define LAYERS 25
#define STEPS 4000

//-----------------------------------------------------------------------
//	   THE life3D(oy) cellular automaton definition section
//-----------------------------------------------------------------------

//cadef and rundef
struct CALModel3D* life3D;							//the cellular automaton
struct life3DSubstates {
	struct CALSubstate3Db *life;
} Q;					//the substate Q.life
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
	//initializing substate to 0
	calInitSubstate3Db(ca, Q.life, 0);
	//setting a specific cell
	calSet3Db(ca, Q.life, 24, 0, 0, 1);
}

CALbyte life3DSimulationStopCondition(struct CALModel3D* life3D)
{
	if (life3Dsimulation->step >= STEPS)
		return CAL_TRUE;
	return CAL_FALSE;
}

//------------------------------------------------------------------------------
//					life3D main function
//------------------------------------------------------------------------------

int main(int argc, char** argv){
	calglInitViewer("3D life", 1.0f, 400, 400, 40, 40, CAL_TRUE, 1);

	struct CALDrawModel3D* drawModel;

	//cadef and rundef
	life3D = calCADef3D(ROWS, COLS, LAYERS, CAL_MOORE_NEIGHBORHOOD_3D, CAL_SPACE_TOROIDAL, CAL_NO_OPT);
	life3Dsimulation = calRunDef3D(life3D, 1, CAL_RUN_LOOP, CAL_UPDATE_IMPLICIT);
	//add substates
	Q.life = calAddSubstate3Db(life3D);
	//add transition function's elementary processes
	calAddElementaryProcess3D(life3D, life3DTransitionFunction);

	//simulation run setup
	calRunAddInitFunc3D(life3Dsimulation, life3DSimulationInit);
	calRunInitSimulation3D(life3Dsimulation);	//It is required in the case the simulation main loop is explicitated; similarly for calRunFinalizeSimulation3D
	calRunAddStopConditionFunc3D(life3Dsimulation, life3DSimulationStopCondition);

	//drawModel definition
	drawModel = calglDefDrawModel3D(CALGL_DRAW_MODE_FLAT, "Life", life3D, life3Dsimulation);
	calglAddToDrawModel3Db(drawModel, NULL, &Q.life, CALGL_TYPE_INFO_VERTEX_DATA, CALGL_TYPE_INFO_USE_DEFAULT, CALGL_DATA_TYPE_DYNAMIC);
	calglColor3D(drawModel, 0.5f, 0.5f, 0.5f, 1.0f);
	calglAddToDrawModel3Db(drawModel, Q.life, &Q.life, CALGL_TYPE_INFO_COLOR_DATA, CALGL_TYPE_INFO_USE_CONST_VALUE, CALGL_DATA_TYPE_DYNAMIC);
	calglAddToDrawModel3Db(drawModel, Q.life, &Q.life, CALGL_TYPE_INFO_NORMAL_DATA, CALGL_TYPE_INFO_USE_DEFAULT, CALGL_DATA_TYPE_DYNAMIC);

	// New functions for hide/display intervals of cells
	//calglHideDrawKBound3D(drawModel, 0, drawModel->calModel->slices);
	//calglDisplayDrawKBound3D(drawModel, 4, 10);
	//calglDisplayDrawKBound3D(drawModel, 20, 25);
	//calglHideDrawJBound3D(drawModel, 0, drawModel->calModel->columns);
	//calglDisplayDrawJBound3D(drawModel, 2, 6);
	//calglDisplayDrawJBound3D(drawModel, 18, 21);

	calglStartProcessWindow3D(argc, argv);

	//finalizations
	calRunFinalize3D(life3Dsimulation);
	calFinalize3D(life3D);
	return 0;
}
