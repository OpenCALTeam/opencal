#include "sciddicaS3Hex.h"

#include <OpenCAL/cal2DUnsafe.h>
#include <stdlib.h>

//-----------------------------------------------------------------------
//	   THE s3hex(oy) cellular automaton definition section
//-----------------------------------------------------------------------

//cadef and rundef
struct CALModel2D* s3hex;						//the cellular automaton
struct sciddicaTSubstates Q;						//the substates
struct sciddicaTParameters P;						//the parameters
struct CALRun2D* s3hexSimulation;				//the simulartion run


//------------------------------------------------------------------------------
//					s3hex transition function
//------------------------------------------------------------------------------


void doErosion(struct CALModel2D* s3hex, int i, int j, CALreal	erosion_depth)
{
	CALreal z, d, h, p, runup;

	z = calGet2Dr(s3hex,Q.z,i,j);
	d = calGet2Dr(s3hex,Q.d,i,j);
	h = calGet2Dr(s3hex,Q.h,i,j);
	p = calGet2Dr(s3hex,Q.p,i,j);

	if (h > 0)
		runup =  p/h + erosion_depth;
	else
		runup = erosion_depth;

	calSetCurrent2Dr(s3hex,Q.z,i,j, (z - erosion_depth));
	calSetCurrent2Dr(s3hex,Q.d,i,j, (d - erosion_depth));
	calSet2Dr(s3hex,Q.h,i,j, (h + erosion_depth));
	calSet2Dr(s3hex,Q.p,i,j, (h + erosion_depth)*runup);
}


//transition function
void s3hexErosion(struct CALModel2D* s3hex, int i, int j)
{
	CALint s;
	CALreal d, p, erosion_depth;

	d = calGet2Dr(s3hex,Q.d,i,j);
	if (d > 0)
	{
		s = calGet2Di(s3hex,Q.s,i,j);
		if (s <  -1)
			calSetCurrent2Di(s3hex,Q.s,i,j,s+1);
		if (s == -1) {
			calSetCurrent2Di(s3hex,Q.s,i,j,0);
			doErosion(s3hex,i,j,d);
#ifdef ACTIVE_CELLS
			calAddActiveCell2D(s3hex,i,j);
#endif
		}

		p = calGet2Dr(s3hex,Q.p,i,j);
		if (p > P.mt) {
			erosion_depth = p * P.pef;
			if (erosion_depth > d)
				erosion_depth = d;
			doErosion(s3hex,i,j,erosion_depth);
		}
	}
}

void s3hexFlowsComputation(struct CALModel2D* s3hex, int i, int j)
{
	CALbyte eliminated_cells[7]={CAL_FALSE,CAL_FALSE,CAL_FALSE,CAL_FALSE,CAL_FALSE, CAL_FALSE, CAL_FALSE};
	CALbyte again;
	CALint cells_count;
	CALreal average;
	CALreal m;
	CALreal u[7], delta_H[7], delta_z[7];
	CALint n;
	CALreal z_0, h_0, z_n, h_n, runup_0, z_0_plus_runup_0, sum;
	CALreal f;


	if (calGet2Dr(s3hex,Q.h,i,j) <= P.adh)
		return;

	z_0 = calGet2Dr(s3hex, Q.z, i, j);
	h_0 = calGet2Dr(s3hex, Q.h, i, j);
	runup_0 = calGet2Dr(s3hex, Q.p, i, j) / h_0;
	z_0_plus_runup_0 = z_0 + runup_0;

	m = runup_0;
	u[0] = z_0;
	delta_z[0] = 0;
	delta_H[0] = 0;
	for (n=1; n<s3hex->sizeof_X; n++)
	{
		z_n = calGetX2Dr(s3hex, Q.z, i, j, n);
		h_n = calGetX2Dr(s3hex, Q.h, i, j, n);

		u[n] = z_n + h_n;
		delta_z[n] = z_0 - z_n;
		delta_H[n] = z_0_plus_runup_0 - u[n];
	}

	for (n=1; n<s3hex->sizeof_X; n++)
		eliminated_cells[n] = (delta_H[n] < P.f);
	//computes outflows
	do{
		again = CAL_FALSE;
		average = m;
		cells_count = 0;

		for (n=0; n<s3hex->sizeof_X; n++)
			if (!eliminated_cells[n]){
				average += u[n];
				cells_count++;
			}

			if (cells_count != 0)
				average /= cells_count;

			for (n=0; n<s3hex->sizeof_X; n++)
				if( (average<=u[n]) && (!eliminated_cells[n]) ){
					eliminated_cells[n]=CAL_TRUE;
					again=CAL_TRUE;
				}

	}while (again);


	sum = 0;
	for (n=0; n<s3hex->sizeof_X; n++)
		if (!eliminated_cells[n])
			sum += average - u[n];

	for (n=1; n<s3hex->sizeof_X; n++)
		if (!eliminated_cells[n])
		{
			//f = (h_0 - P.adh) * ((average-u[n])/sum) * P.r;
			f = h_0 * ((average-u[n])/sum) * P.r;
			calSet2Dr (s3hex,Q.h,i,j,   calGetNext2Dr (s3hex,Q.h,i,j)   - f );
			calSetX2Dr(s3hex,Q.h,i,j,n, calGetNextX2Dr(s3hex,Q.h,i,j,n) + f );

			calSet2Dr (s3hex,Q.p,i,j,   calGetNext2Dr (s3hex,Q.p,i,j)   - runup_0 * f );
			calSetX2Dr(s3hex,Q.p,i,j,n, calGetNextX2Dr(s3hex,Q.p,i,j,n) + (z_0_plus_runup_0 - u[n]) * f );

#ifdef ACTIVE_CELLS
			//adds the cell (i, j, n) to the set of active ones
            calAddActiveCellX2D(s3hex, i, j, n);
#endif
		}
}


void s3hexEnergyLoss(struct CALModel2D* s3hex, int i, int j)
{
	CALreal h, runup;

	if (calGet2Dr(s3hex,Q.h,i,j) <= P.adh)
		return;

	h = calGet2Dr(s3hex,Q.h,i,j);
	if (h > P.adh) {
		runup = calGet2Dr(s3hex,Q.p,i,j) / h - P.rl;
		if (runup < h)
			runup = h;
		calSet2Dr(s3hex,Q.p,i,j,h*runup);
	}
}



void s3hexRomoveInactiveCells(struct CALModel2D* s3hex, int i, int j)
{
#ifdef ACTIVE_CELLS
	if (calGet2Dr(s3hex,Q.h,i,j) <= P.adh)
		calRemoveActiveCell2D(s3hex,i,j);
#endif
}

//------------------------------------------------------------------------------
//					s3hex simulation functions
//------------------------------------------------------------------------------

void sciddicaTSimulationInit(struct CALModel2D* s3hex)
{
	int i, j;

	//s3hex parameters setting
	P.adh = P_ADH;
	P.rl = P_RL;
	P.r = P_R;
	P.f = P_F;
	P.mt = P_MT;
	P.pef = P_PEF;
//	P.ltt = P_LTT;

	//initializing debris source
	calInitSubstate2Dr(s3hex, Q.h, 0);
	calInitSubstate2Dr(s3hex, Q.p, 0);

#ifdef ACTIVE_CELLS
	for (i=0; i<s3hex->rows; i++)
		for (j=0; j<s3hex->columns; j++)
			if (calGet2Dr(s3hex,Q.h,i,j) > P.adh) {
				calAddActiveCell2D(s3hex,i,j);
	}
#endif

	//Substates and active cells update
	calUpdate2D(s3hex);
}

void sciddicaTransitionFunction(struct CALModel2D* s3hex)
{
	calApplyElementaryProcess2D(s3hex, s3hexErosion);
	calUpdateSubstate2Dr(s3hex, Q.h);
	calUpdateSubstate2Dr(s3hex, Q.p);

	calApplyElementaryProcess2D(s3hex, s3hexFlowsComputation);
	calUpdateActiveCells2D(s3hex);
	calUpdateSubstate2Dr(s3hex, Q.h);
	calUpdateSubstate2Dr(s3hex, Q.p);

	calApplyElementaryProcess2D(s3hex, s3hexRomoveInactiveCells);
	calUpdateActiveCells2D(s3hex);

	calApplyElementaryProcess2D(s3hex, s3hexEnergyLoss);
	calUpdateSubstate2Dr(s3hex, Q.p);
}

CALbyte sciddicaTSimulationStopCondition(struct CALModel2D* s3hex)
{
	if (s3hexSimulation->step >= STEPS)
		return CAL_TRUE;
	return CAL_FALSE;
}


//------------------------------------------------------------------------------
//					s3hex CADef and runDef
//------------------------------------------------------------------------------

void sciddicaTCADef()
{
	//cadef and rundef
	s3hex = calCADef2D (ROWS, COLS, CAL_HEXAGONAL_NEIGHBORHOOD_2D, CAL_SPACE_TOROIDAL, CAL_OPT_ACTIVE_CELLS);
	s3hexSimulation = calRunDef2D(s3hex, 1, CAL_RUN_LOOP, CAL_UPDATE_EXPLICIT);

	//add transition function's elementary processes
	calAddElementaryProcess2D(s3hex, s3hexErosion);
	calAddElementaryProcess2D(s3hex, s3hexFlowsComputation);
	calAddElementaryProcess2D(s3hex, s3hexRomoveInactiveCells);
	calAddElementaryProcess2D(s3hex, s3hexEnergyLoss);

	//add substates
	Q.z = calAddSingleLayerSubstate2Dr(s3hex);
	Q.d = calAddSingleLayerSubstate2Dr(s3hex);
	Q.s = calAddSingleLayerSubstate2Di(s3hex);
	Q.h = calAddSubstate2Dr(s3hex);
	Q.p = calAddSubstate2Dr(s3hex);

	//load configuration
	sciddicaTLoadConfig();

	//simulation run setup
	calRunAddInitFunc2D(s3hexSimulation, sciddicaTSimulationInit); calRunInitSimulation2D(s3hexSimulation);
	calRunAddGlobalTransitionFunc2D(s3hexSimulation, sciddicaTransitionFunction);
	calRunAddStopConditionFunc2D(s3hexSimulation, sciddicaTSimulationStopCondition);
}

//------------------------------------------------------------------------------
//					s3hex I/O functions
//------------------------------------------------------------------------------

void sciddicaTLoadConfig()
{
	//load configuration
	calLoadSubstate2Dr(s3hex, Q.z, DEM_PATH);
	calLoadSubstate2Dr(s3hex, Q.d, REGOLITH_PATH);
	calLoadSubstate2Di(s3hex, Q.s, SOURCE_PATH);
}

void sciddicaTSaveConfig()
{
	calSaveSubstate2Dr(s3hex, Q.h, OUTPUT_PATH);
}

//------------------------------------------------------------------------------
//					s3hex finalization function
//------------------------------------------------------------------------------


void sciddicaTExit()
{
	//finalizations
	calRunFinalize2D(s3hexSimulation);
	calFinalize2D(s3hex);
}
