#include "sciddicaTunsafe.h"

#include <OpenCAL-OMP/cal2DUnsafe.h>
#include <stdlib.h>

//-----------------------------------------------------------------------
//	   THE sciddicaT(oy) cellular automaton definition section
//-----------------------------------------------------------------------

//cadef and rundef
struct CALModel2D* sciddicaT;						//the cellular automaton
struct sciddicaTSubstates Q;						//the substates
struct sciddicaTParameters P;						//the parameters
struct CALRun2D* sciddicaTsimulation;				//the simulartion run


//------------------------------------------------------------------------------
//					sciddicaT transition function
//------------------------------------------------------------------------------

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
			calAddNext2Dr(sciddicaT,Q.h,i,j,-f);
			calAddNextX2Dr(sciddicaT,Q.h,i,j,n,f);

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

//------------------------------------------------------------------------------
//					sciddicaT simulation functions
//------------------------------------------------------------------------------

void sciddicaTSimulationInit(struct CALModel2D* sciddicaT)
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

CALbyte sciddicaTSimulationStopCondition(struct CALModel2D* sciddicaT)
{
	if (sciddicaTsimulation->step >= STEPS)
		return CAL_TRUE;
	return CAL_FALSE;
}


//------------------------------------------------------------------------------
//					sciddicaT CADef and runDef
//------------------------------------------------------------------------------

void sciddicaTCADef()
{
	//cadef and rundef
	sciddicaT = calCADef2D (ROWS, COLS, CAL_VON_NEUMANN_NEIGHBORHOOD_2D, CAL_SPACE_TOROIDAL, CAL_OPT_ACTIVE_CELLS);
	sciddicaTsimulation = calRunDef2D(sciddicaT, 1, CAL_RUN_LOOP, CAL_UPDATE_IMPLICIT);

	//put OpenCAL - OMP in unsafe state execution(to allow unsafe operation to be used)
	calSetUnsafe2D(sciddicaT);

	//add transition function's elementary processes
	calAddElementaryProcess2D(sciddicaT, sciddicaT_transition_function);
	calAddElementaryProcess2D(sciddicaT, sciddicaT_remove_inactive_cells);

	//add substates
	Q.z = calAddSingleLayerSubstate2Dr(sciddicaT);
	Q.h = calAddSubstate2Dr(sciddicaT);

	//load configuration
	sciddicaTLoadConfig();

	//simulation run setup
	calRunAddInitFunc2D(sciddicaTsimulation, sciddicaTSimulationInit); calRunInitSimulation2D(sciddicaTsimulation);
	calRunAddStopConditionFunc2D(sciddicaTsimulation, sciddicaTSimulationStopCondition);
}

//------------------------------------------------------------------------------
//					sciddicaT I/O functions
//------------------------------------------------------------------------------

void sciddicaTLoadConfig()
{
	//load configuration
	calLoadSubstate2Dr(sciddicaT, Q.z, DEM_PATH);
	calLoadSubstate2Dr(sciddicaT, Q.h, SOURCE_PATH);
}

void sciddicaTSaveConfig()
{
	calSaveSubstate2Dr(sciddicaT, Q.h, OUTPUT_PATH);
}

//------------------------------------------------------------------------------
//					sciddicaT finalization function
//------------------------------------------------------------------------------


void sciddicaTExit()
{
	//finalizations
	calRunFinalize2D(sciddicaTsimulation);
	calFinalize2D(sciddicaT);
}
