#include "iso.h"
#include <math.h>
#include <stdlib.h>

//-----------------------------------------------------------------------
//	   THE "iso" cellular automaton definition section
//-----------------------------------------------------------------------

//cadef and rundef

CALint currentAC;
struct CellularAutomata ca[numAC];

struct Coordinates {
	CALint row;
	CALint col;
};

CALParameterr exponent = EXP;	//exponent for the IDW algorithm
//------------------------------------------------------------------------------
//					iso transition function
//------------------------------------------------------------------------------

CALbyte not_yet_considered(struct Coordinates c, struct Coordinates* cells, CALint dim)
{
	CALint n;

	for (n=1; n<ca[currentAC].iso->sizeof_X; n++)
		if (c.row == cells[n].row && c.col == cells[n].col)
			return CAL_FALSE;

	return CAL_TRUE;

}

void isoSignalDiffusion(struct CALModel2D* iso, int i, int j)
{
	CALint n;
	CALbyte state_n;
	CALint flux_count_candidate = NODATA, max_flux_count_index = 0;
	CALreal src_row = NODATA, src_col = NODATA;
	CALreal distance_from_source = 0;

	CALbyte to_update = CAL_FALSE;
	CALreal d, inv_d, sum_inv_d = 0, sum_u = 0;
	CALint count;

	struct Coordinates c;
	struct Coordinates* sources_coords = (struct Coordinates*)malloc(sizeof(struct Coordinates)*iso->sizeof_X);
	for (n=0; n<iso->sizeof_X; n++)	{
		sources_coords[n].row = NODATA;
		sources_coords[n].col = NODATA;
	}

#ifdef BOUND
	if (i==0 || i==iso->rows-1 || j==0 || j==iso->columns-1) {
		free(sources_coords);
		return;
	}
#endif

	if (calGet2Db(iso,ca[currentAC].Q.state,i,j)==BLANK) //if (calGet2Db(iso,Q.state,i,j)!=STEADY)
	{
		//compute the flux_count_candidate = ( max{Q.flux_count[n], n=1,2,...} + 1 )
		flux_count_candidate = NODATA;
		max_flux_count_index = 0;
		for (n=1; n<iso->sizeof_X; n++)
		{
			state_n = calGetX2Db(iso,ca[currentAC].Q.state,i,j,n);
			if (state_n != BLANK)
			{
				if (state_n != TO_BE_STEADY)
				{
					c.row = calGetX2Di(iso,ca[currentAC].Q.source_row,i,j,n);
					c.col = calGetX2Di(iso,ca[currentAC].Q.source_col,i,j,n);
				}
				else
				{
					c.row = calGetX2Di(iso,ca[currentAC].Q.next_source_row,i,j,n);
					c.col = calGetX2Di(iso,ca[currentAC].Q.next_source_col,i,j,n);
				}

				if (not_yet_considered(c, sources_coords, iso->sizeof_X))
				{
					sources_coords[n].row = c.row;
					sources_coords[n].col = c.col;
				}

				if (flux_count_candidate < calGetX2Di(iso,ca[currentAC].Q.flux_count,i,j,n))
				{
					flux_count_candidate = calGetX2Di(iso,ca[currentAC].Q.flux_count,i,j,n);
					max_flux_count_index = n;
				}
			}
		}
		if (flux_count_candidate == NODATA)
		{
			free(sources_coords);
			return;
		}
		else
			flux_count_candidate++;


		count = 0;
		for (n=1; n<iso->sizeof_X; n++)
			if (sources_coords[n].row != NODATA)
				count++;

#ifdef TWO_SRC
		if (count > 1)
#else
		if (count > (ca[currentAC].initial_sources<=2?1:2))
#endif	
			to_update = CAL_TRUE;
		else
		{
			//compute the distance from the source and compare it with flux_count_candidate
			src_row = (CALreal)calGetX2Di(iso,ca[currentAC].Q.source_row,i,j,max_flux_count_index);
			src_col = (CALreal)calGetX2Di(iso,ca[currentAC].Q.source_col,i,j,max_flux_count_index);
			distance_from_source = sqrt( (src_row - i)*(src_row - i) + (src_col - j)*(src_col - j) );

			//the case in which the cell is not elaborated
			if (distance_from_source > flux_count_candidate)
			{
				free(sources_coords);
				return;
			}
		}

		if (to_update)
		{
			for (n=1; n<iso->sizeof_X; n++)
			{
				if (sources_coords[n].row != NODATA)
				{
					d = sqrt( (double)(i-sources_coords[n].row)*(i-sources_coords[n].row) + (double)(j-sources_coords[n].col)*(j-sources_coords[n].col) );
					inv_d = 1 / d;
					
					sum_u += calGetX2Dr(iso, ca[currentAC].Q.value, i, j, n) * pow(inv_d,exponent);
					sum_inv_d += pow(inv_d,exponent);
					
					/*sum_u += calGetX2Dr(iso, Q.value, i, j, n);
					sum_inv_d++;*/
				}
			}

			calSet2Db(iso, ca[currentAC].Q.state, i, j, TO_BE_STEADY);
			calSet2Dr(iso, ca[currentAC].Q.value, i, j, sum_u/sum_inv_d);
			calSet2Di(iso, ca[currentAC].Q.next_source_row, i, j, i);
			calSet2Di(iso, ca[currentAC].Q.next_source_col, i, j, j);
			calSet2Di(iso, ca[currentAC].Q.flux_count, i, j, 0);
		}
		else
		{
			calSet2Db(iso, ca[currentAC].Q.state, i, j, UNSTEADY);
			calSet2Dr(iso, ca[currentAC].Q.value, i, j, calGetX2Dr(iso,ca[currentAC].Q.value,i,j,max_flux_count_index));
			calSet2Di(iso, ca[currentAC].Q.flux_count, i, j, flux_count_candidate);
			calSet2Di(iso, ca[currentAC].Q.source_row, i, j, (CALint)src_row);
			calSet2Di(iso, ca[currentAC].Q.source_col, i, j, (CALint)src_col);
		}

	}

	free(sources_coords);
}

void isoFixBoundLocally(struct CALModel2D* iso, int i, int j)
{
	if (i>0 && i<iso->rows-1 && j==0) 
		calSet2Dr(iso,ca[currentAC].Q.value,i,0,calGet2Dr(iso,ca[currentAC].Q.value,i,1));

	if (i>0 && i<iso->rows-1 && j==iso->columns-1)
		calSet2Dr(iso,ca[currentAC].Q.value,i,iso->columns-1,calGet2Dr(iso,ca[currentAC].Q.value,i,iso->columns-2));

	if (i==0 && j>0 && j<iso->columns-1)
		calSet2Dr(iso,ca[currentAC].Q.value,0,j,calGet2Dr(iso,ca[currentAC].Q.value,1,j));

	if (i==iso->rows-1 && j>0 && j<iso->columns-1)
		calSet2Dr(iso,ca[currentAC].Q.value,iso->rows-1,j,calGet2Dr(iso,ca[currentAC].Q.value,iso->rows-2,j));

	calSet2Dr(iso,ca[currentAC].Q.value,0,0,calGet2Dr(iso,ca[currentAC].Q.value,1,1));
	calSet2Dr(iso,ca[currentAC].Q.value,0,iso->columns-1,calGet2Dr(iso,ca[currentAC].Q.value,1,iso->columns-2));
	calSet2Dr(iso,ca[currentAC].Q.value,iso->rows-1,0,calGet2Dr(iso,ca[currentAC].Q.value,iso->rows-2,1));
	calSet2Dr(iso,ca[currentAC].Q.value,iso->rows-1,iso->columns-1,calGet2Dr(iso,ca[currentAC].Q.value,iso->rows-2,iso->columns-2));
}

//------------------------------------------------------------------------------
//					iso simulation functions
//------------------------------------------------------------------------------

void isoSimulationInit(struct CALModel2D* iso)
{
	CALint i, j;

	calInitSubstate2Db(iso, ca[currentAC].Q.state, BLANK);
	calInitSubstate2Di(iso, ca[currentAC].Q.source_row, NODATA);
	calInitSubstate2Di(iso, ca[currentAC].Q.source_col, NODATA);
	calInitSubstate2Di(iso, ca[currentAC].Q.flux_count, NODATA);

	ca[currentAC].initial_sources = 0;
	for (i=0; i<iso->rows; i++)
		for (j=0; j<iso->columns; j++)
			if (calGet2Dr(iso,ca[currentAC].Q.value,i,j) != 0)
			{
				ca[currentAC].initial_sources++;
				calSet2Db(iso,ca[currentAC].Q.state,i,j,STEADY);
				calSet2Di(iso,ca[currentAC].Q.source_row,i,j,i);
				calSet2Di(iso,ca[currentAC].Q.source_col,i,j,j);
				calSet2Di(iso,ca[currentAC].Q.flux_count,i,j,0);
			}
}

void isoFixUnsteady3(struct CALModel2D* iso, int i, int j)
{
	CALint n;
	CALbyte state_n;
	CALint flux_count_candidate = NODATA, max_flux_count_index = 0;
	CALreal src_row = NODATA, src_col = NODATA;
	CALreal distance_from_source = 0;

	CALbyte to_update = CAL_FALSE;
	CALreal d, inv_d, sum_inv_d = 0, sum_u = 0;
	CALint count;

	struct Coordinates c;
	struct Coordinates* sources_coords = (struct Coordinates*)malloc(sizeof(struct Coordinates)*iso->sizeof_X);
	for (n=0; n<iso->sizeof_X; n++)	{
		sources_coords[n].row = NODATA;
		sources_coords[n].col = NODATA;
	}

#ifdef BOUND
	if (i==0 || i==iso->rows-1 || j==0 || j==iso->columns-1) {
		free(sources_coords);
		return;
	}
#endif

	if (calGet2Db(iso,ca[currentAC].Q.state,i,j)==UNSTEADY)
	{
		count = 0;
		for (n=1; n<iso->sizeof_X; n++)
		{
			state_n = calGetX2Db(iso,ca[currentAC].Q.state,i,j,n);
			if (state_n != BLANK && state_n != TO_BE_STEADY)
			{
				if (state_n == TO_BE_STEADY)
				{
					c.row = calGetX2Di(iso,ca[currentAC].Q.next_source_row,i,j,n);
					c.col = calGetX2Di(iso,ca[currentAC].Q.next_source_col,i,j,n);
				}
				else
				{
					c.row = calGetX2Di(iso,ca[currentAC].Q.source_row,i,j,n);
					c.col = calGetX2Di(iso,ca[currentAC].Q.source_col,i,j,n);
				}

				if (not_yet_considered(c, sources_coords, iso->sizeof_X)) {
					sources_coords[n].row = c.row;
					sources_coords[n].col = c.col;
					
					count++;
				}
			}
		}
		
		if (count > 2)
		{
			for (n=1; n<iso->sizeof_X; n++)
			{
				if (sources_coords[n].row != NODATA)
				{
					d = sqrt( (double)(i-sources_coords[n].row)*(i-sources_coords[n].row) + (double)(j-sources_coords[n].col)*(j-sources_coords[n].col) );
					inv_d = 1 / d;
					
					sum_u += calGetX2Dr(iso, ca[currentAC].Q.value, i, j, n) * pow(inv_d,exponent);
					sum_inv_d += pow(inv_d,exponent);
					
					/*sum_u += calGetX2Dr(iso, Q.value, i, j, n);
					sum_inv_d++;*/
				}
			}

			calSet2Db(iso, ca[currentAC].Q.state, i, j, TO_BE_STEADY);
			calSet2Dr(iso, ca[currentAC].Q.value, i, j, sum_u/sum_inv_d);
			calSet2Di(iso, ca[currentAC].Q.next_source_row, i, j, i);
			calSet2Di(iso, ca[currentAC].Q.next_source_col, i, j, j);
			calSet2Di(iso, ca[currentAC].Q.flux_count, i, j, 0);
		}
	}

	free(sources_coords);
}

void isoSteering(struct CALModel2D* iso)
{
	CALint i, j, nodata_count = 0;

#ifdef BOUND
	for (i=istart; i<istop; i++)
		for (j=jstart; j<jstop; j++)
			if (calGet2Db(iso,ca[currentAC].Q.state,i,j) == BLANK)
				nodata_count++;

	if (nodata_count == 0)
	{
		for (i=istart; i<istop; i++)
			for (j=jstart; j<jstop; j++)
				isoFixUnsteady3(iso, i, j);

		calUpdate2D(iso);

		for (i=istart; i<istop; i++)
			for (j=jstart; j<jstop; j++)
			{
				if (calGet2Db(iso,ca[currentAC].Q.state,i,j) == UNSTEADY)
				{
					calSet2Db(iso, ca[currentAC].Q.state, i, j, BLANK);
					calSet2Dr(iso, ca[currentAC].Q.value, i, j, NODATA);
					calSet2Di(iso, ca[currentAC].Q.source_row, i, j, NODATA);
					calSet2Di(iso, ca[currentAC].Q.source_col, i, j, NODATA);
					calSet2Di(iso, ca[currentAC].Q.flux_count, i, j, NODATA);
				}

				if (calGet2Db(iso,ca[currentAC].Q.state,i,j) == TO_BE_STEADY)
				{
					calSet2Db(iso, ca[currentAC].Q.state, i, j, STEADY);
					calSet2Di(iso, ca[currentAC].Q.source_row, i, j, calGet2Di(iso, ca[currentAC].Q.next_source_row, i, j));
					calSet2Di(iso, ca[currentAC].Q.source_col, i, j, calGet2Di(iso, ca[currentAC].Q.next_source_col, i, j));
				}
			}

		calUpdate2D(iso);

		for (i=0; i<iso->rows; i++)
			for (j=0; j<iso->columns; j++)
				isoFixBoundLocally(iso, i, j);
	}
#else
	for (i=0; i<iso->rows; i++)
			for (j=0; j<iso->columns; j++)
			if (calGet2Db(iso,Q.state,i,j) == BLANK)
				nodata_count++;

	if (nodata_count == 0)
	{
		for (i=0; i<iso->rows; i++)
			for (j=0; j<iso->columns; j++)
			{
				if (calGet2Db(iso,Q.state,i,j) == UNSTEADY)
				{
					calSet2Db(iso, ca[currentAC].Q.state, i, j, BLANK);
					calSet2Dr(iso, ca[currentAC].Q.value, i, j, NODATA);
					calSet2Di(iso, ca[currentAC].Q.source_row, i, j, NODATA);
					calSet2Di(iso, ca[currentAC].Q.source_col, i, j, NODATA);
					calSet2Di(iso, ca[currentAC].Q.flux_count, i, j, NODATA);
				}
			}
	}
#endif
}

void isoIverseDistanceWeighting(struct CALModel2D* iso)
{
	CALint i, j, k, l;
	CALreal inv_d, sum_inv_d, sum_u;
	
	for (i=0; i<iso->rows; i++)
		for (j=0; j<iso->columns; j++)
			if (calGet2Db(iso,ca[currentAC].Q.state,i,j) == BLANK)
			{
				sum_u = 0;
				sum_inv_d  = 0;
				for (k=0; k<iso->rows; k++)
					for (l=0; l<iso->columns; l++)
						if (calGet2Db(iso,ca[currentAC].Q.state,k,l) == STEADY)
						{
							inv_d = 1.0 / sqrt( (double)(i-k)*(i-k) + (double)(j-l)*(j-l) );
							sum_inv_d  += inv_d * pow(inv_d,exponent);
							sum_u +=  calGet2Dr(iso,ca[currentAC].Q.value,k,l) * inv_d* pow(inv_d,exponent);
							calSet2Dr(iso,ca[currentAC].Q.value,i,j,sum_u/sum_inv_d);
							calSet2Db(iso,ca[currentAC].Q.state,i,j,STEADY);
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
	
	value = calGet2Dr(iso,ca[currentAC].Q.value,istart,jstart);
	for (i=istart; i<istop; i++)
		for (j=jstart; j<jstop; j++)
			if (/*calGet2Dr(iso,Q.value,i,j) != value || */calGet2Db(iso,ca[currentAC].Q.state,i,j) != STEADY)
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

void isoCADef(struct CellularAutomata* ca)
{
	//cadef and rundef
	ca->iso = calCADef2D (ROWS, COLS, CAL_MOORE_NEIGHBORHOOD_2D, CAL_SPACE_TOROIDAL, CAL_NO_OPT);
	ca->isoRun = calRunDef2D(ca->iso, 1, CAL_RUN_LOOP, CAL_UPDATE_IMPLICIT);
	//add substates
	ca->Q.value = calAddSubstate2Dr(ca->iso);
	ca->Q.state = calAddSubstate2Db(ca->iso);
	ca->Q.flux_count = calAddSubstate2Di(ca->iso);
	ca->Q.source_row = calAddSubstate2Di(ca->iso);
	ca->Q.source_col = calAddSubstate2Di(ca->iso);
	ca->Q.next_source_row = calAddSubstate2Di(ca->iso);
	ca->Q.next_source_col = calAddSubstate2Di(ca->iso);

	//add transition function's elementary processes
#ifndef IDW
	calAddElementaryProcess2D(ca->iso, isoSignalDiffusion);
	//calAddElementaryProcess2D(iso, isoFixUnsteady3);
	calAddElementaryProcess2D(ca->iso, isoFixBoundLocally);
#endif

	//simulation run setup
	calRunAddInitFunc2D(ca->isoRun, isoSimulationInit);
	//simulation steering
#ifndef IDW
	calRunAddSteeringFunc2D(ca->isoRun, isoSteering);
#else
	calRunAddSteeringFunc2D(ca->isoRun, isoIverseDistanceWeighting);
#endif
	//simulation stop condition
	calRunAddStopConditionFunc2D(ca->isoRun, isoSimulationStopCondition);
}

//------------------------------------------------------------------------------
//					iso I/O functions
//------------------------------------------------------------------------------

void isoLoadConfig(struct CellularAutomata* ca)
{
	//load configuration
	calLoadSubstate2Dr(ca->iso, ca->Q.value, SOURCE_PATH);

	ca->isoRun->init(ca->iso); //it calls isoSimulationInit
	calUpdate2D(ca->iso);
}

void isoSaveConfig(struct CellularAutomata* ca)
{
	calSaveSubstate2Dr(ca->iso, ca->Q.value, OUTPUT_PATH);
	calSaveSubstate2Db(ca->iso, ca->Q.state, OUTPUT_PATH_STATE);
}

//------------------------------------------------------------------------------
//					iso finalization function
//------------------------------------------------------------------------------

void isoExit(struct CellularAutomata* ca)
{	
	//finalizations
	calRunFinalize2D(ca->isoRun);
	calFinalize2D(ca->iso);
}

//------------------------------------------------------------------------------