// The SciddicaT debris flows CCA simulation model width_final
// a 3D graphic visualizer in OpenCAL-GL

#include <OpenCAL/cal2D.h>
#include <OpenCAL/cal2DIO.h>
#include <OpenCAL/cal2DRun.h>
#include <OpenCAL-GL/calgl2D.h>
#include <OpenCAL-GL/calgl2DWindow.h>
#include <stdlib.h>

// Some definitions...
#define P_R 0.5
#define P_EPSILON 0.001
#define NUMBER_OF_OUTFLOWS 4
#define DEM "./data/dem.txt"
#define SOURCE "./data/source.txt"
#define FINAL "./data/width_final.txt"
#define ROWS 610
#define COLUMNS 496
#define STEPS 4000

// declare CCA model (sciddicaT), substates (Q), parameters (P),
// and simulation object (sciddicaT_simulation)
struct sciddicaTSubstates {
	struct CALSubstate2Dr *z;
	struct CALSubstate2Dr *h;
	struct CALSubstate2Dr *f[NUMBER_OF_OUTFLOWS];
};

struct sciddicaTParameters {
	CALParameterr epsilon;
	CALParameterr r;
};

struct CALModel2D* sciddicaT;						//the cellular automaton
struct sciddicaTSubstates Q;						//the substates
struct sciddicaTParameters P;						//the parameters
struct CALRun2D* sciddicaTsimulation;				//the simulartion run

// The sigma_1 elementary process
void sciddicaT_flows_computation(struct CALModel2D* sciddicaT, int i, int j) {
	CALbyte eliminated_cells[5] = {CAL_FALSE, CAL_FALSE, CAL_FALSE, CAL_FALSE, CAL_FALSE};
	CALbyte again;
	CALint cells_count;
	CALreal average;
	CALreal m;
	CALreal u[5];
	CALint n;
	CALreal z, h;

	if(calGet2Dr(sciddicaT, Q.h, i, j)<=P.epsilon)
		return;

	m = calGet2Dr(sciddicaT, Q.h, i, j)-P.epsilon;
	u[0] = calGet2Dr(sciddicaT, Q.z, i, j)+P.epsilon;
	for(n = 1; n<sciddicaT->sizeof_X; n++) {
		z = calGetX2Dr(sciddicaT, Q.z, i, j, n);
		h = calGetX2Dr(sciddicaT, Q.h, i, j, n);
		u[n] = z+h;
	}

	//computes outflows
	do {
		again = CAL_FALSE;
		average = m;
		cells_count = 0;

		for(n = 0; n<sciddicaT->sizeof_X; n++)
			if(!eliminated_cells[n]) {
				average += u[n];
				cells_count++;
			}

		if(cells_count!=0)
			average /= cells_count;

		for(n = 0; n<sciddicaT->sizeof_X; n++)
			if((average<=u[n])&&(!eliminated_cells[n])) {
				eliminated_cells[n] = CAL_TRUE;
				again = CAL_TRUE;
			}

	} while(again);

	for(n = 1; n<sciddicaT->sizeof_X; n++)
		if(eliminated_cells[n])
			calSet2Dr(sciddicaT, Q.f[n-1], i, j, 0.0);
		else
			calSet2Dr(sciddicaT, Q.f[n-1], i, j, (average-u[n])*P.r);
}

// The sigma_2 elementary process
void sciddicaT_width_update(struct CALModel2D* sciddicaT, int i, int j) {
	CALreal h_next;
	CALint n;

	h_next = calGet2Dr(sciddicaT, Q.h, i, j);
	for(n = 1; n<sciddicaT->sizeof_X; n++)
		h_next += calGetX2Dr(sciddicaT, Q.f[NUMBER_OF_OUTFLOWS-n], i, j, n)-calGet2Dr(sciddicaT, Q.f[n-1], i, j);

	calSet2Dr(sciddicaT, Q.h, i, j, h_next);
}

// SciddicaT simulation init function
void sciddicaTSimulationInit(struct CALModel2D* sciddicaT) {
	CALreal z, h;
	CALint i, j;

	//initializing substates to 0
	calInitSubstate2Dr(sciddicaT, Q.f[0], 0);
	calInitSubstate2Dr(sciddicaT, Q.f[1], 0);
	calInitSubstate2Dr(sciddicaT, Q.f[2], 0);
	calInitSubstate2Dr(sciddicaT, Q.f[3], 0);

	//sciddicaT parameters setting
	P.r = P_R;
	P.epsilon = P_EPSILON;

	//sciddicaT source initialization
	for(i = 0; i<sciddicaT->rows; i++)
		for(j = 0; j<sciddicaT->columns; j++) {
			h = calGet2Dr(sciddicaT, Q.h, i, j);

			if(h>0.0) {
				z = calGet2Dr(sciddicaT, Q.z, i, j);
				calSet2Dr(sciddicaT, Q.z, i, j, z-h);
			}
		}
}

// SciddicaT steering function
void sciddicaTSteering(struct CALModel2D* sciddicaT) {
	CALreal value = 0;

	//initializing substates to 0
	calInitSubstate2Dr(sciddicaT, Q.f[0], 0);
	calInitSubstate2Dr(sciddicaT, Q.f[1], 0);
	calInitSubstate2Dr(sciddicaT, Q.f[2], 0);
	calInitSubstate2Dr(sciddicaT, Q.f[3], 0);
}

// SciddicaT stop condition function
CALbyte sciddicaTSimulationStopCondition(struct CALModel2D* sciddicaT) {
	if(sciddicaTsimulation->step>=STEPS)
		return CAL_TRUE;
	return CAL_FALSE;
}

// SciddicaT exit function
void exitFunction(void) {
	// saving configuration
	calSaveSubstate2Dr(sciddicaT, Q.h, FINAL);

	// finalizations
	calRunFinalize2D(sciddicaTsimulation);
	calFinalize2D(sciddicaT);
}


int main(int argc, char** argv) {
	struct CALGLDrawModel2D* draw_model3D = NULL;
	struct CALGLDrawModel2D* draw_model2D;

	atexit(exitFunction);

	calglInitViewer("SciddicaT OpenCAL-GL visualizer", 5, 800, 600, 10, 10, CAL_TRUE, 0);

	//cadef and rundef
	sciddicaT = calCADef2D(ROWS, COLUMNS, CAL_VON_NEUMANN_NEIGHBORHOOD_2D, CAL_SPACE_TOROIDAL, CAL_NO_OPT);
	sciddicaTsimulation = calRunDef2D(sciddicaT, 1, CAL_RUN_LOOP, CAL_UPDATE_IMPLICIT);
	//add substates
	Q.z = calAddSubstate2Dr(sciddicaT);
	Q.h = calAddSubstate2Dr(sciddicaT);
	Q.f[0] = calAddSubstate2Dr(sciddicaT);
	Q.f[1] = calAddSubstate2Dr(sciddicaT);
	Q.f[2] = calAddSubstate2Dr(sciddicaT);
	Q.f[3] = calAddSubstate2Dr(sciddicaT);
	//add transition function's elementary processes
	calAddElementaryProcess2D(sciddicaT, sciddicaT_flows_computation);
	calAddElementaryProcess2D(sciddicaT, sciddicaT_width_update);
	//load configuration
	calLoadSubstate2Dr(sciddicaT, Q.z, DEM);
	calLoadSubstate2Dr(sciddicaT, Q.h, SOURCE);
	//simulation run setup
	calRunAddInitFunc2D(sciddicaTsimulation, sciddicaTSimulationInit);
	calRunInitSimulation2D(sciddicaTsimulation);
	calRunAddSteeringFunc2D(sciddicaTsimulation, sciddicaTSteering);
	calRunAddStopConditionFunc2D(sciddicaTsimulation, sciddicaTSimulationStopCondition);

	// draw_model3D definition
	draw_model3D = calglDefDrawModel2D(CALGL_DRAW_MODE_SURFACE, "SciddicaT 3D view", sciddicaT, sciddicaTsimulation);
	// Add nodes
	calglAdd2Dr(draw_model3D, NULL, &Q.z, CALGL_TYPE_INFO_VERTEX_DATA, CALGL_TYPE_INFO_USE_NO_COLOR, CALGL_DATA_TYPE_STATIC);
	calglColor2D(draw_model3D, 0.5, 0.5, 0.5, 1.0);
	calglAdd2Dr(draw_model3D, Q.z, &Q.z, CALGL_TYPE_INFO_COLOR_DATA, CALGL_TYPE_INFO_USE_CURRENT_COLOR, CALGL_DATA_TYPE_DYNAMIC);
	calglAdd2Dr(draw_model3D, Q.z, &Q.z, CALGL_TYPE_INFO_NORMAL_DATA, CALGL_TYPE_INFO_USE_NO_COLOR, CALGL_DATA_TYPE_DYNAMIC);
	calglAdd2Dr(draw_model3D, Q.z, &Q.h, CALGL_TYPE_INFO_VERTEX_DATA, CALGL_TYPE_INFO_USE_NO_COLOR, CALGL_DATA_TYPE_DYNAMIC);
	calglAdd2Dr(draw_model3D, Q.h, &Q.h, CALGL_TYPE_INFO_COLOR_DATA, CALGL_TYPE_INFO_USE_RED_YELLOW_SCALE, CALGL_DATA_TYPE_DYNAMIC);
	calglAdd2Dr(draw_model3D, Q.h, &Q.h, CALGL_TYPE_INFO_NORMAL_DATA, CALGL_TYPE_INFO_USE_NO_COLOR, CALGL_DATA_TYPE_DYNAMIC);
	// InfoBar
	//calglRelativeInfoBar2Dr(draw_model3D, Q.h, "Debris thickness", CALGL_TYPE_INFO_USE_RED_SCALE, CALGL_INFO_BAR_ORIENTATION_VERTICAL);
	calglInfoBar2Dr(draw_model3D, Q.h, "Debris thickness", CALGL_TYPE_INFO_USE_RED_SCALE, 20, 120, 300, 40);

	// Set offset between substates
	calglSetHeightOffset2D(draw_model3D, 0.0f);

	// Hide/display intervals of cells
	//	calglHideDrawJBound2D(draw_model3D, 0, draw_model3D->calModel->columns);
	//	calglDisplayDrawJBound2D(draw_model3D, 300, draw_model3D->calModel->columns);
	//	calglHideDrawIBound2D(draw_model3D, 100, 150);

	draw_model2D = calglDefDrawModel2D(CALGL_DRAW_MODE_FLAT, "SciddicaT 2D view", sciddicaT, sciddicaTsimulation);
	draw_model2D->realModel = draw_model3D->realModel;
	calglInfoBar2Dr(draw_model2D, Q.h, "Debris thickness", CALGL_TYPE_INFO_USE_RED_SCALE, 20, 200, 50, 150);

	calglSetLayoutOrientation2D(CALGL_LAYOUT_ORIENTATION_VERTICAL);

	calglSetDisplayStep(100);

	calglMainLoop2D(argc, argv);

	return 0;
}
