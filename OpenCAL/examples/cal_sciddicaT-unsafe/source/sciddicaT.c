#include <cal2D.h>
#include <cal2DIO.h>
#include <cal2DRun.h>
#include <cal2DUnsafe.h>
#include <stdlib.h>
#include <time.h>

//-----------------------------------------------------------------------
//   THE sciddicaT (Toy model) CELLULAR AUTOMATON
//-----------------------------------------------------------------------

#define ROWS 610
#define COLS 496
#define P_R 0.5
#define P_EPSILON 0.001
#define STEPS 4000
#define DEM_PATH "./data/dem.txt"
#define SOURCE_PATH "./data/source.txt"
#define OUTPUT_PATH "./data/width_final.txt"


#define NUMBER_OF_OUTFLOWS 4

struct sciddicaTSubstates {
	struct CALSubstate2Dr *z;
	struct CALSubstate2Dr *h;
} Q;

struct sciddicaTParameters {
	CALParameterr epsilon;
	CALParameterr r;
} P;


//transition function
void sciddicaT_transition_function(struct CALModel2D* sciddicaT, int i, int j)
{
	CALbyte eliminated_cells[5]={CAL_FALSE,CAL_FALSE,CAL_FALSE,CAL_FALSE,CAL_FALSE};
	CALbyte again;
	CALint cells_count;
	CALreal average;
	CALreal m;
	CALreal u[5];
	CALint n;
	CALreal z, h;
	CALreal f;


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
		if (!eliminated_cells[n])
		{
			f = (average-u[n])*P.r;
			calSet2Dr (sciddicaT,Q.h,i,j,   calGetNext2Dr (sciddicaT,Q.h,i,j)   - f );
			calSetX2Dr(sciddicaT,Q.h,i,j,n, calGetNextX2Dr(sciddicaT,Q.h,i,j,n) + f );

#ifdef ACTIVE_CELLS
			//adds the cell (i, j, n) to the set of active ones
            calAddActiveCellX2D(sciddicaT, i, j, n);
#endif		
		}
}


void sciddicaT_remove_inactive_cells(struct CALModel2D* sciddicaT, int i, int j)
{
#ifdef ACTIVE_CELLS
	if (calGet2Dr(sciddicaT, Q.h, i, j) <= P.epsilon)
		calRemoveActiveCell2D(sciddicaT,i,j);	
#endif
}


void sciddicaT_simulation_init(struct CALModel2D* sciddicaT)
{
	CALreal z, h;
	CALint i, j;

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
				calSetCurrent2Dr(sciddicaT, Q.z, i, j, z-h);

#ifdef ACTIVE_CELLS
                //adds the cell (i, j) to the set of active ones
                calAddActiveCell2D(sciddicaT, i, j);
#endif
			}
		}
}


int main()
{
	time_t start_time, end_time;
	
	//cadef and rundef
	struct CALModel2D* sciddicaT = calCADef2D (ROWS, COLS, CAL_VON_NEUMANN_NEIGHBORHOOD_2D, CAL_SPACE_TOROIDAL, CAL_OPT_ACTIVE_CELLS);
	struct CALRun2D* sciddicaT_simulation = calRunDef2D(sciddicaT, 1, STEPS, CAL_UPDATE_IMPLICIT);

	//add transition function's elementary processes
	calAddElementaryProcess2D(sciddicaT, sciddicaT_transition_function);
	calAddElementaryProcess2D(sciddicaT, sciddicaT_remove_inactive_cells);

	//add substates
	Q.z = calAddSingleLayerSubstate2Dr(sciddicaT);
	Q.h = calAddSubstate2Dr(sciddicaT);

	//load configuration
	calLoadSubstate2Dr(sciddicaT, Q.z, DEM_PATH);
	calLoadSubstate2Dr(sciddicaT, Q.h, SOURCE_PATH);
	
	//simulation run
	calRunAddInitFunc2D(sciddicaT_simulation, sciddicaT_simulation_init);
	printf ("Starting simulation...\n");
	start_time = time(NULL);
	calRun2D(sciddicaT_simulation);
	end_time = time(NULL);
	printf ("Simulation terminated.\nElapsed time: %d\n", end_time-start_time);
	

	//saving configuration
	calSaveSubstate2Dr(sciddicaT, Q.h, OUTPUT_PATH);
	
	//finalizations
	calRunFinalize2D(sciddicaT_simulation);
	calFinalize2D(sciddicaT);

	return 0;
}

