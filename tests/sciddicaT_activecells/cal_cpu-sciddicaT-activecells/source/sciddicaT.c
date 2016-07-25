#include "sciddicaT.h"
#include <stdlib.h>

//-----------------------------------------------------------------------
//	   THE sciddicaT(oy) cellular automaton definition section
//-----------------------------------------------------------------------

//cadef and rundef
struct CALModel* sciddicaT;						//the cellular automaton
struct sciddicaTSubstates Q;						//the substates
struct sciddicaTParameters P;						//the parameters

//------------------------------------------------------------------------------
//					sciddicaT transition function
//------------------------------------------------------------------------------

//first elementary process
void sciddicaT_flows_computation(struct CALModel* sciddicaT, CALIndices cell, int numb_of_dim)
{
    CALbyte eliminated_cells[5]={CAL_FALSE,CAL_FALSE,CAL_FALSE,CAL_FALSE,CAL_FALSE};
    CALbyte again;
    CALint cells_count;
    CALreal average;
    CALreal m;
    CALreal u[5];
    CALint n;
    CALreal z, h;


    if (calGet_r(sciddicaT, Q.h, cell) <= P.epsilon)
        return;

    m = calGet_r(sciddicaT, Q.h, cell) - P.epsilon;
    u[0] = calGet_r(sciddicaT, Q.z, cell) + P.epsilon;
    for (n=1; n<calGetSizeOfX(sciddicaT); n++)
    {
        z = calGetX_r(sciddicaT, Q.z, cell, n);
        h = calGetX_r(sciddicaT, Q.h, cell, n);
        u[n] = z + h;
    }

    //computes outflows
    do{
        again = CAL_FALSE;
        average = m;
        cells_count = 0;

        for (n=0; n<calGetSizeOfX(sciddicaT); n++)
            if (!eliminated_cells[n]){
                average += u[n];
                cells_count++;
            }

            if (cells_count != 0)
                average /= cells_count;

            for (n=0; n<calGetSizeOfX(sciddicaT); n++)
                if( (average<=u[n]) && (!eliminated_cells[n]) ){
                    eliminated_cells[n]=CAL_TRUE;
                    again=CAL_TRUE;
                }

    }while (again);

    for (n=1; n<calGetSizeOfX(sciddicaT); n++)
        if (eliminated_cells[n])
            calSet_r(sciddicaT, Q.f[n-1], cell, 0.0);
        else
        {
            calSet_r(sciddicaT, Q.f[n-1], cell, (average-u[n])*P.r);

#ifdef ACTIVE_CELLS
            //adds the cell (i, j, n) to the set of active ones
            calAddActiveCellX(sciddicaT, cell, n);
#endif
        }
}

//second (and last) elementary process
void sciddicaT_width_update(struct CALModel* sciddicaT, CALIndices cell, int numb_of_dim)
{
    CALreal h_next;
    CALint n;

    h_next = calGet_r(sciddicaT, Q.h, cell);
    for(n=1; n<calGetSizeOfX(sciddicaT); n++)
        h_next +=  calGetX_r(sciddicaT, Q.f[NUMBER_OF_OUTFLOWS - n], cell, n) - calGet_r(sciddicaT, Q.f[n-1], cell);

    calSet_r(sciddicaT, Q.h, cell, h_next);
}

void sciddicaT_remove_inactive_cells(struct CALModel* sciddicaT, CALIndices cell, int numb_of_dim)
{
#ifdef ACTIVE_CELLS
    if (calGet_r(sciddicaT, Q.h, cell) <= P.epsilon)
        calRemoveActiveCell(sciddicaT, cell);
#endif
}

//------------------------------------------------------------------------------
//					sciddicaT simulation functions
//------------------------------------------------------------------------------

void sciddicaTSimulationInit(struct CALModel* sciddicaT)
{
    CALreal z, h;
    CALint i, j;

    //sciddicaT parameters setting
    P.r = P_R;
    P.epsilon = P_EPSILON;

    //sciddicaT source initialization
    for (i = 0; i < ROWS; i++)
        for (j = 0; j < COLS; j++)
        {
            h = calGet_r(sciddicaT, Q.h, calGetCell(sciddicaT, i, j));

            if ( h > 0.0 ) {
                z = calGet_r(sciddicaT, Q.z, calGetCell(sciddicaT, i, j));
                calSet_r(sciddicaT, Q.z, calGetCell(sciddicaT, i, j), z-h);

#ifdef ACTIVE_CELLS
                //adds the cell (i, j) to the set of active ones
                calAddActiveCell(sciddicaT, calGetCell(sciddicaT, i, j));
#endif
            }
        }
}

void sciddicaTResetFlows(struct CALModel* sciddicaT)
{
    //initializing substates to 0
    calInitSubstate_r(sciddicaT, Q.f[0], CAL_INIT_NEXT, 0);
    calInitSubstate_r(sciddicaT, Q.f[1], CAL_INIT_NEXT, 0);
    calInitSubstate_r(sciddicaT, Q.f[2], CAL_INIT_NEXT, 0);
    calInitSubstate_r(sciddicaT, Q.f[3], CAL_INIT_NEXT, 0);
}

CALbyte sciddicaTSimulationStopCondition(struct CALModel* sciddicaT)
{
    if (calGetCurrentStep(sciddicaT) >= STEPS)
        return CAL_TRUE;
    return CAL_FALSE;
}


//------------------------------------------------------------------------------
//					sciddicaT CADef and runDef
//------------------------------------------------------------------------------

void sciddicaTCADef()
{
    //cadef and rundef
    sciddicaT = calCADef(calDefDimensions(2, ROWS, COLS), CAL_VON_NEUMANN_NEIGHBORHOOD, CAL_SPACE_TOROIDAL, CAL_OPT_ACTIVE_CELLS, 0, 0);

    //add transition function's elementary processes
    calAddLocalProcess(sciddicaT, sciddicaT_flows_computation);
    calAddLocalProcess(sciddicaT, sciddicaT_width_update);
    calAddLocalProcess(sciddicaT, sciddicaT_remove_inactive_cells);
    calAddGlobalProcess(sciddicaT, sciddicaTResetFlows);

    //add substates
    Q.z = calAddSubstate_r(sciddicaT, CAL_INIT_BOTH, 0);
    Q.h = calAddSubstate_r(sciddicaT, CAL_INIT_BOTH, 0);
    Q.f[0] = calAddSubstate_r(sciddicaT, CAL_INIT_BOTH, 0);
    Q.f[1] = calAddSubstate_r(sciddicaT, CAL_INIT_BOTH, 0);
    Q.f[2] = calAddSubstate_r(sciddicaT, CAL_INIT_BOTH, 0);
    Q.f[3] = calAddSubstate_r(sciddicaT, CAL_INIT_BOTH, 0);

    //load configuration
    sciddicaTLoadConfig();

    //simulation run setup
    calAddInitFunc(sciddicaT,sciddicaTSimulationInit);
    calAddStopCondition(sciddicaT, sciddicaTSimulationStopCondition);
}

//------------------------------------------------------------------------------
//					sciddicaT I/O functions
//------------------------------------------------------------------------------

void sciddicaTLoadConfig()
{
    //load configuration
    calLoadSubstate_r(sciddicaT, Q.z, DEM_PATH);
    calLoadSubstate_r(sciddicaT, Q.h, SOURCE_PATH);
}

void sciddicaTSaveConfig()
{
    calSaveSubstate_r(sciddicaT, Q.h, OUTPUT_PATH);
}
