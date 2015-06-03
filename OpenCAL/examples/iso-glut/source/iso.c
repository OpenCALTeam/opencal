#include "iso.h"
#include <math.h>
#include <stdlib.h>

//-----------------------------------------------------------------------
//	   THE "iso" cellular automaton definition section
//-----------------------------------------------------------------------

//cadef and rundef
struct CALModel2D* iso;			//the cellular automaton
struct isoSubstates Q;			//the substates
CALParameterr exponent = EXP;
struct CALRun2D* isoRun;		//the simulartion run

struct Coordinates
{
	CALint row;
	CALint col;
};

//------------------------------------------------------------------------------
//					iso transition function
//------------------------------------------------------------------------------

CALbyte not_yet_considered(struct Coordinates c, struct Coordinates* cells, CALint dim)
{
	CALint n;

	for (n=1; n<iso->sizeof_X; n++)
		if (c.row == cells[n].row && c.col == cells[n].col)
			return CAL_FALSE;

	return CAL_TRUE;

}

void isoSignalDiffusion(struct CALModel2D* iso, int i, int j)
{
	CALint n;
	CALbyte state;
	CALint flux_count_candidate = NODATA, max_flux_count_index = 0;
	CALreal source_row = NODATA, source_col = NODATA;
	CALreal distance_from_source = 0;

	CALbyte to_update = CAL_FALSE;
	CALreal d, inv_d, sum_inv_d = 0, sum_u = 0;
	CALint count;

	struct Coordinates c;
	struct Coordinates* sources = (struct Coordinates*)malloc(sizeof(struct Coordinates)*iso->sizeof_X);
	for (n=0; n<iso->sizeof_X; n++)
	{
		sources[n].row = NODATA;
		sources[n].col = NODATA;
	}

#ifdef BOUND
	if (i==0 || i==iso->rows-1 || j==0 || j==iso->columns-1)
		return;
#endif

	if (calGet2Db(iso,Q.state,i,j)==BLANK)
	{
		//compute the flux_count_candidate = ( max{Q.flux_count[n], n=1,2,...} + 1 )
		flux_count_candidate = NODATA;
		max_flux_count_index = 0;
		for (n=1; n<iso->sizeof_X; n++)
		{
			state = calGetX2Db(iso,Q.state,i,j,n);
			if (state != BLANK)
			{
				c.row = calGetX2Di(iso,Q.source_row,i,j,n);
				c.col = calGetX2Di(iso,Q.source_col,i,j,n);

				if (not_yet_considered(c, sources, iso->sizeof_X))
				{
					sources[n].row = c.row;
					sources[n].col = c.col;
				}

				if (flux_count_candidate < calGetX2Di(iso,Q.flux_count,i,j,n))
				{
					flux_count_candidate = calGetX2Di(iso,Q.flux_count,i,j,n);
					max_flux_count_index = n;
				}
			}
		}
		if (flux_count_candidate == NODATA)
			return;
		else
			flux_count_candidate++;

		count = 0;
		for (n=1; n<iso->sizeof_X; n++)
			if (sources[n].row != NODATA)
				count++;
		
		if (count > 1)
			to_update = CAL_TRUE;
		else
		{
		//compute the distance from the source and compare it with flux_count_candidate
		source_row = (CALreal)calGetX2Di(iso,Q.source_row,i,j,max_flux_count_index);
		source_col = (CALreal)calGetX2Di(iso,Q.source_col,i,j,max_flux_count_index);
		distance_from_source = sqrt( (source_row - i)*(source_row - i) + (source_col - j)*(source_col - j) );

		//the case in which the cell is not elaborated
		if (distance_from_source > flux_count_candidate)
			return;
		}

		if (to_update)
		{
			for (n=1; n<iso->sizeof_X; n++)
			{
				if (sources[n].row != NODATA)
				{
					d = sqrt( (double)(i-sources[n].row)*(i-sources[n].row) + (double)(j-sources[n].col)*(j-sources[n].col) );
					inv_d = 1 / d;
					sum_u += calGetX2Dr(iso, Q.value, i, j, n) * pow(inv_d,exponent);
					sum_inv_d += pow(inv_d,exponent);
				}
				calSet2Db(iso, Q.state, i, j, TO_BE_STEADY);
				calSet2Dr(iso, Q.value, i, j, sum_u/sum_inv_d);
				calSet2Di(iso, Q.source_row, i, j, i);
				calSet2Di(iso, Q.source_col, i, j, j);
				calSet2Di(iso, Q.flux_count, i, j, 0);
			}
		}
		else
		{
			calSet2Db(iso, Q.state, i, j, UNSTEADY);
			calSet2Dr(iso, Q.value, i, j, calGetX2Dr(iso,Q.value,i,j,max_flux_count_index));
			calSet2Di(iso, Q.flux_count, i, j, flux_count_candidate);
			calSet2Di(iso, Q.source_row, i, j, (CALint)source_row);
			calSet2Di(iso, Q.source_col, i, j, (CALint)source_col);
		}

	}

	free(sources);
}

void isoComputeValue(struct CALModel2D* iso, int i, int j)
{
	CALint n;
	CALreal d, inv_d, sum_inv_d, sum_u;
	CALreal source_row = NODATA, source_col = NODATA;
	CALbyte to_fix_stall = CAL_FALSE;

	struct Coordinates c;
	struct Coordinates* sources = (struct Coordinates*)malloc(sizeof(struct Coordinates)*iso->sizeof_X);
	for (n=0; n<iso->sizeof_X; n++)
	{
		sources[n].row = NODATA;
		sources[n].col = NODATA;
	}

#ifdef BOUND
	if (i==0 || i==iso->rows-1 || j==0 || j==iso->columns-1)
		return;
#endif
	
	if (calGet2Db(iso,Q.state,i,j)==UNSTEADY)
	{
		source_row = (CALreal)calGet2Di(iso,Q.source_row,i,j);
		source_col = (CALreal)calGet2Di(iso,Q.source_col,i,j);
		d = sqrt( (double)(i-source_row)*(i-source_row) + (double)(j-source_col)*(j-source_col) );
		inv_d = 1 / d;
		sum_u = calGet2Dr(iso, Q.value, i, j) * inv_d;
		sum_inv_d = inv_d;

		for (n=1; n<iso->sizeof_X; n++) //ATTENZIONE: OGNI SORGENTE DEVE ESSERE CONSIDERATA UNA SOLA VOLTA.
			if (calGetX2Db(iso,Q.state,i,j,n) != BLANK)
			{
				source_row = (CALreal)calGetX2Di(iso,Q.source_row,i,j,n);
				source_col = (CALreal)calGetX2Di(iso,Q.source_col,i,j,n);
				
				//if (calGetX2Dr(iso, Q.value, i, j, n) != calGet2Dr(iso, Q.value, i, j))
				if (calGetX2Di(iso, Q.source_row, i, j, n) != calGet2Di(iso, Q.source_row, i, j) || 
					calGetX2Di(iso, Q.source_col, i, j, n) != calGet2Di(iso, Q.source_col, i, j) )
				{
					//Controlla se la sorgente è già stata considerata, cioè se è già nell'array sources
					c.row = source_row;
					c.col = source_col;
					if (not_yet_considered(c, sources, iso->sizeof_X))
					{
						d = sqrt( (double)(i-source_row)*(i-source_row) + (double)(j-source_col)*(j-source_col) );
						inv_d = 1 / d;
						sum_u += calGetX2Dr(iso, Q.value, i, j, n) * pow(inv_d,exponent);
						sum_inv_d += pow(inv_d,exponent);

						sources[n].row = c.row;
						sources[n].col = c.col;

						to_fix_stall = CAL_TRUE;
					}
				}
			}
	}

	if (to_fix_stall)
	{
		calSet2Db(iso, Q.state, i, j, STEADY);
		calSet2Dr(iso, Q.value, i, j, sum_u/sum_inv_d);
		calSet2Di(iso, Q.source_row, i, j, i);
		calSet2Di(iso, Q.source_col, i, j, j);
		calSet2Di(iso, Q.flux_count, i, j, 0);
	}

	free(sources);
}

//------------------------------------------------------------------------------
//					iso simulation functions
//------------------------------------------------------------------------------

void isoSimulationInit(struct CALModel2D* iso)
{
	CALint i, j;

	calInitSubstate2Db(iso, Q.state, BLANK);
	calInitSubstate2Di(iso, Q.source_row, NODATA);
	calInitSubstate2Di(iso, Q.source_col, NODATA);
	calInitSubstate2Di(iso, Q.flux_count, NODATA);

	for (i=0; i<iso->rows; i++)
		for (j=0; j<iso->columns; j++)
			if (calGet2Dr(iso,Q.value,i,j) != 0)
			{
				calSet2Db(iso,Q.state,i,j,STEADY);
				calSet2Di(iso,Q.source_row,i,j,i);
				calSet2Di(iso,Q.source_col,i,j,j);
				calSet2Di(iso,Q.flux_count,i,j,0);
			}
}

void isoSteering(struct CALModel2D* iso)
{
	CALint i, j, nodata_count = 0;

	for (i=istart; i<istop; i++)
		for (j=jstart; j<jstop; j++)
			if (calGet2Db(iso,Q.state,i,j) == BLANK)
				nodata_count++;

	if (nodata_count == 0)
	{
		for (i=istart; i<istop; i++)
			for (j=jstart; j<jstop; j++)
				isoComputeValue(iso,i,j);

		calUpdate2D(iso);/**/

		for (i=istart; i<istop; i++)
			for (j=jstart; j<jstop; j++)
			{
				if (calGet2Db(iso,Q.state,i,j) == UNSTEADY)
				{
					calSet2Db(iso, Q.state, i, j, BLANK);
					calSet2Dr(iso, Q.value, i, j, NODATA);
					calSet2Di(iso, Q.source_row, i, j, NODATA);
					calSet2Di(iso, Q.source_col, i, j, NODATA);
					calSet2Di(iso, Q.flux_count, i, j, NODATA);
				}
			}
	}
}

void isoIverseDistanceWeighting(struct CALModel2D* iso)
{
	CALint i, j, k, l;
	CALreal inv_d, sum_inv_d, sum_u;
	
	for (i=0; i<iso->rows; i++)
		for (j=0; j<iso->columns; j++)
			if (calGet2Db(iso,Q.state,i,j) == BLANK)
			{
				sum_u = 0;
				sum_inv_d  = 0;
				for (k=0; k<iso->rows; k++)
					for (l=0; l<iso->columns; l++)
						if (calGet2Db(iso,Q.state,k,l) == STEADY)
						{
							inv_d = 1.0 / sqrt( (double)(i-k)*(i-k) + (double)(j-l)*(j-l) );
							sum_inv_d  += inv_d*inv_d;
							sum_u += inv_d*inv_d * calGet2Dr(iso,Q.value,k,l);
							calSet2Dr(iso,Q.value,i,j,sum_u/sum_inv_d);
							calSet2Db(iso,Q.state,i,j,STEADY);
						}
			}
}

CALbyte isoSimulationStopCondition(struct CALModel2D* iso)
{
	CALint i, j;
	CALreal value;
	CALbyte STOP_CONDITION = CAL_TRUE;


#ifdef DEBUG
	return CAL_TRUE;
#endif
	
	value = calGet2Dr(iso,Q.value,istart,jstart);
	for (i=istart; i<istop; i++)
		for (j=jstart; j<jstop; j++)
			if (/*calGet2Dr(iso,Q.value,i,j) != value || */calGet2Db(iso,Q.state,i,j) != STEADY)
			{
				STOP_CONDITION = CAL_FALSE;
				goto return_condition;
			}

#ifdef VERBOSE
	printf("Simulation done\n\n");
#endif

return_condition: return STOP_CONDITION;
}

//------------------------------------------------------------------------------
//					iso CADef and runDef
//------------------------------------------------------------------------------

void isoCADef()
{
	//cadef and rundef
	iso = calCADef2D (ROWS, COLS, CAL_MOORE_NEIGHBORHOOD_2D, CAL_SPACE_TOROIDAL, CAL_NO_OPT);
	isoRun = calRunDef2D(iso, 1, CAL_RUN_LOOP, CAL_UPDATE_IMPLICIT);
	//add substates
	Q.value = calAddSubstate2Dr(iso);
	Q.state = calAddSubstate2Db(iso);
	Q.source_row = calAddSubstate2Di(iso);
	Q.source_col = calAddSubstate2Di(iso);
	Q.flux_count = calAddSubstate2Di(iso);
	
	//add transition function's elementary processes
#ifndef IDW
	calAddElementaryProcess2D(iso, isoSignalDiffusion);
	//calAddElementaryProcess2D(iso, isoComputeValue);
#endif

	//simulation run setup
	calRunAddInitFunc2D(isoRun, isoSimulationInit);
	//simulation steering
#ifndef IDW
	calRunAddSteeringFunc2D(isoRun, isoSteering);
#else
	calRunAddSteeringFunc2D(isoRun, isoIverseDistanceWeighting);
#endif
	//simulation stop condition
	calRunAddStopConditionFunc2D(isoRun, isoSimulationStopCondition);
}

//------------------------------------------------------------------------------
//					iso I/O functions
//------------------------------------------------------------------------------

void isoLoadConfig()
{
	//load configuration
	calLoadSubstate2Dr(iso, Q.value, SOURCE_PATH);

	isoRun->init(iso); //it calls isoSimulationInit
	calUpdate2D(iso);
}

void isoSaveConfig()
{
	calSaveSubstate2Dr(iso, Q.value, OUTPUT_PATH);
	calSaveSubstate2Db(iso, Q.state, OUTPUT_PATH_STATE);
}

//------------------------------------------------------------------------------
//					iso finalization function
//------------------------------------------------------------------------------

void isoExit()
{	
	//finalizations
	calRunFinalize2D(isoRun);
	calFinalize2D(iso);
}
