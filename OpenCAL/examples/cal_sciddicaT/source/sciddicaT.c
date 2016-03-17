// The SciddicaT debris flows XCA simulation model

#include <OpenCAL/cal2D.h>
#include <OpenCAL/cal2DIO.h>
#include <OpenCAL/cal2DRun.h>
#include <stdlib.h>
#include <time.h>

// Some definitions...
#define ROWS 610
#define COLS 496
#define P_R 0.5
#define P_EPSILON 0.001
#define STEPS 4000
#define DEM_PATH "./data/dem.txt"
#define SOURCE_PATH "./data/source.txt"
#define OUTPUT_PATH "./data/width_final.txt"
#define NUMBER_OF_OUTFLOWS 4

// Declare XCA model (sciddicaT), substates (Q), parameters (P),
// and simulation object (sciddicaT_simulation)
struct CALModel2D* sciddicaT;

struct sciddicaTSubstates {
	struct CALSubstate2Dr *z;
	struct CALSubstate2Dr *h;
	struct CALSubstate2Dr *f[NUMBER_OF_OUTFLOWS];
} Q;

struct sciddicaTParameters {
	CALParameterr epsilon;
	CALParameterr r;
} P;

struct CALRun2D* sciddicaT_simulation;

// The sigma_1 elementary process
void sciddicaT_flows_computation(struct CALModel2D* sciddicaT, int i, int j)
{
	CALbyte eliminated_cells[5]={CAL_FALSE,CAL_FALSE,CAL_FALSE,CAL_FALSE,CAL_FALSE};
	CALbyte again;
	CALint cells_count;
	CALreal average;
	CALreal m;
	CALreal u[5];
	CALint n;
	CALreal z, h;

	if (calGet2Dr(sciddicaT, Q.h, i, j) <= P.epsilon)
		return;

	m = calGet2Dr(sciddicaT, Q.h, i, j) - P.epsilon;
	u[0] = calGet2Dr(sciddicaT, Q.z, i, j) + P.epsilon;
	for (n=1; n<sciddicaT->sizeof_X; n++)
	{
		z = calGetX2Dr(sciddicaT, Q.z, i, j, n);
		h = calGetX2Dr(sciddicaT, Q.h, i, j, n);
		u[n] = z + h;
	}

	//computes outflows
	do{
		again = CAL_FALSE;
		average = m;
		cells_count = 0;

		for (n=0; n<sciddicaT->sizeof_X; n++)
			if (!eliminated_cells[n]){
				average += u[n];
				cells_count++;
			}

			if (cells_count != 0)
				average /= cells_count;

			for (n=0; n<sciddicaT->sizeof_X; n++)
				if( (average<=u[n]) && (!eliminated_cells[n]) ){
					eliminated_cells[n]=CAL_TRUE;
					again=CAL_TRUE;
				}
	}while (again);

	for (n=1; n<sciddicaT->sizeof_X; n++)
		if (eliminated_cells[n])
			calSet2Dr(sciddicaT, Q.f[n-1], i, j, 0.0);
		else
			calSet2Dr(sciddicaT, Q.f[n-1], i, j, (average-u[n])*P.r);
}

// The sigma_2 elementary process
void sciddicaT_width_update(struct CALModel2D* sciddicaT, int i, int j)
{
	CALreal h_next;
	CALint n;

	h_next = calGet2Dr(sciddicaT, Q.h, i, j);
	for(n=1; n<sciddicaT->sizeof_X; n++)
		h_next +=  calGetX2Dr(sciddicaT, Q.f[NUMBER_OF_OUTFLOWS - n], i, j, n) - calGet2Dr(sciddicaT, Q.f[n-1], i, j);

	calSet2Dr(sciddicaT, Q.h, i, j, h_next);
}

// SciddicaT simulation init function
void sciddicaT_simulation_init(struct CALModel2D* sciddicaT)
{
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
	for (i=0; i<sciddicaT->rows; i++)
		for (j=0; j<sciddicaT->columns; j++)
		{
			h = calGet2Dr(sciddicaT, Q.h, i, j);

			if ( h > 0.0 ) {
				z = calGet2Dr(sciddicaT, Q.z, i, j);
				calSet2Dr(sciddicaT, Q.z, i, j, z-h);
			}
		}
}

// SciddicaT steering function
void sciddicaTSteering(struct CALModel2D* sciddicaT)
{
	// set flow to 0 everywhere
	calInitSubstate2Dr(sciddicaT, Q.f[0], 0);
	calInitSubstate2Dr(sciddicaT, Q.f[1], 0);
	calInitSubstate2Dr(sciddicaT, Q.f[2], 0);
	calInitSubstate2Dr(sciddicaT, Q.f[3], 0);
}

int main()
{
	time_t start_time, end_time;

	// define of the sciddicaT CA and sciddicaT_simulation simulation objects
  sciddicaT = calCADef2D (ROWS, COLS, CAL_VON_NEUMANN_NEIGHBORHOOD_2D, CAL_SPACE_TOROIDAL, CAL_NO_OPT);
	sciddicaT_simulation = calRunDef2D(sciddicaT, 1, STEPS, CAL_UPDATE_IMPLICIT);

	// add transition function's sigma_1 and sigma_2 elementary processes
	calAddElementaryProcess2D(sciddicaT, sciddicaT_flows_computation);
	calAddElementaryProcess2D(sciddicaT, sciddicaT_width_update);

	// add substates
	Q.z = calAddSubstate2Dr(sciddicaT);
	Q.h = calAddSubstate2Dr(sciddicaT);
	Q.f[0] = calAddSubstate2Dr(sciddicaT);
	Q.f[1] = calAddSubstate2Dr(sciddicaT);
	Q.f[2] = calAddSubstate2Dr(sciddicaT);
	Q.f[3] = calAddSubstate2Dr(sciddicaT);

	// load configuration
	calLoadSubstate2Dr(sciddicaT, Q.z, DEM_PATH);
	calLoadSubstate2Dr(sciddicaT, Q.h, SOURCE_PATH);

	// simulation run
	calRunAddInitFunc2D(sciddicaT_simulation, sciddicaT_simulation_init);
	calRunAddSteeringFunc2D(sciddicaT_simulation, sciddicaTSteering);
	printf ("Starting simulation...\n");
	start_time = time(NULL);
	calRun2D(sciddicaT_simulation);
	end_time = time(NULL);
	printf ("Simulation terminated.\nElapsed time: %lds\n", end_time-start_time);

	// saving configuration
	calSaveSubstate2Dr(sciddicaT, Q.h, OUTPUT_PATH);

	// finalizations
	calRunFinalize2D(sciddicaT_simulation);
	calFinalize2D(sciddicaT);

	return 0;
}
