// The SciddicaT debris flows model with the active cells optimization

#include <OpenCAL/cal2D.h>
#include <OpenCAL/cal2DIO.h>
#include <OpenCAL/cal2DRun.h>
#include <OpenCAL/cal2DUnsafe.h>
#include <OpenCALTime.h>
#include <stdlib.h>
#include <time.h>

// Some definitions...
#define ROWS 610
#define COLS 496
#define P_R 0.5
#define P_EPSILON 0.001
#define STEPS 4000
#define DEM_PATH "./testData/sciddicaT-data/dem.txt"
#define SOURCE_PATH "./testData/sciddicaT-data/source.txt"
#define OUTPUT_PATH "./data/width_final.txt"
#define NUMBER_OF_OUTFLOWS 4

// declare CCA model (sciddicaT), substates (Q), parameters (P),
// and simulation object (sciddicaT_simulation)
struct sciddicaTSubstates {
    struct CALSubstate2Dr *z;
    struct CALSubstate2Dr *h;
    struct CALSubstate2Dr *f[NUMBER_OF_OUTFLOWS];
} Q;

struct sciddicaTParameters {
    CALParameterr epsilon;
    CALParameterr r;
} P;

int numberOfLoops;

// The sigma_1 elementary process
void sciddicaTFlowsComputation(struct CALModel2D* sciddicaT, int i, int j)
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

    int k;
    for(k = 0; k < numberOfLoops; k++){
        m = calGet2Dr(sciddicaT, Q.h, i, j) - P.epsilon;
        u[0] = calGet2Dr(sciddicaT, Q.z, i, j) + P.epsilon;
        for (n=1; n<sciddicaT->sizeof_X; n++)
        {
            z = calGetX2Dr(sciddicaT, Q.z, i, j, n);
            h = calGetX2Dr(sciddicaT, Q.h, i, j, n);
            u[n] = z + h;
        }

        //computes outflows and updates debris thickness
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
            {
                calSet2Dr(sciddicaT, Q.f[n-1], i, j, (average-u[n])*P.r);
                calAddActiveCellX2D(sciddicaT, i, j, n);
            }
    }

}

// The sigma_2 elementary process
void sciddicaTWidthUpdate(struct CALModel2D* sciddicaT, int i, int j)
{
    CALreal h_next;
    CALint n;

    int k;
    for(k = 0; k < numberOfLoops; k++){
        h_next = calGet2Dr(sciddicaT, Q.h, i, j);
        for(n=1; n<sciddicaT->sizeof_X; n++)
            h_next +=  calGetX2Dr(sciddicaT, Q.f[NUMBER_OF_OUTFLOWS - n], i, j, n) - calGet2Dr(sciddicaT, Q.f[n-1], i, j);

        calSet2Dr(sciddicaT, Q.h, i, j, h_next);
    }
}

// The sigma_3 elementary process
void sciddicaTRemoveInactiveCells(struct CALModel2D* sciddicaT, int i, int j)
{
    if (calGet2Dr(sciddicaT, Q.h, i, j) <= P.epsilon)
        calRemoveActiveCell2D(sciddicaT,i,j);
}


void sciddicaTSimulationInit(struct CALModel2D* sciddicaT)
{
    CALreal z, h;
    CALint i, j;

    //sciddicaT parameters setting
    P.r = P_R;
    P.epsilon = P_EPSILON;

    //initializing substates to 0
    calInitSubstate2Dr(sciddicaT, Q.f[0], 0);
    calInitSubstate2Dr(sciddicaT, Q.f[1], 0);
    calInitSubstate2Dr(sciddicaT, Q.f[2], 0);
    calInitSubstate2Dr(sciddicaT, Q.f[3], 0);

    //sciddicaT source initialization
    for (i=0; i<sciddicaT->rows; i++)
        for (j=0; j<sciddicaT->columns; j++)
        {
            h = calGet2Dr(sciddicaT, Q.h, i, j);

            if ( h > 0.0 ) {
                z = calGet2Dr(sciddicaT, Q.z, i, j);
                calSetCurrent2Dr(sciddicaT, Q.z, i, j, z-h);

                //adds the cell (i, j) to the set of active ones
                calAddActiveCell2D(sciddicaT, i, j);
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


int main(int argc, char** argv)
{
    // read from argv the number of steps
    int steps;
    if (sscanf (argv[2], "%i", &steps)!=1 && steps >=0) {
        printf ("number of steps is not an integer");
        exit(-1);
    }
    // read from argv the number of steps
    if (sscanf (argv[3], "%i", &numberOfLoops)!=1 && numberOfLoops >=0) {
        printf ("number of loops is not an integer");
        exit(-1);
    }
    // define of the sciddicaT CA and sciddicaT_simulation simulation objects
    struct CALModel2D* sciddicaT = calCADef2D (ROWS, COLS, CAL_VON_NEUMANN_NEIGHBORHOOD_2D, CAL_SPACE_TOROIDAL, CAL_OPT_ACTIVE_CELLS);
    struct CALRun2D* sciddicaT_simulation = calRunDef2D(sciddicaT, 1, steps, CAL_UPDATE_IMPLICIT);

    // add transition function's sigma_1 and sigma_2 elementary processes
    calAddElementaryProcess2D(sciddicaT, sciddicaTFlowsComputation);
    calAddElementaryProcess2D(sciddicaT, sciddicaTWidthUpdate);
    calAddElementaryProcess2D(sciddicaT, sciddicaTRemoveInactiveCells);

    // add substates
    Q.z = calAddSingleLayerSubstate2Dr(sciddicaT);
    Q.h = calAddSubstate2Dr(sciddicaT);
    Q.f[0] = calAddSubstate2Dr(sciddicaT);
    Q.f[1] = calAddSubstate2Dr(sciddicaT);
    Q.f[2] = calAddSubstate2Dr(sciddicaT);
    Q.f[3] = calAddSubstate2Dr(sciddicaT);

    // load configuration
    calLoadSubstate2Dr(sciddicaT, Q.z, DEM_PATH);
    calLoadSubstate2Dr(sciddicaT, Q.h, SOURCE_PATH);

    // simulation run
    calRunAddInitFunc2D(sciddicaT_simulation, sciddicaTSimulationInit);
    calRunAddSteeringFunc2D(sciddicaT_simulation, sciddicaTSteering);
    struct OpenCALTime * opencalTime= (struct OpenCALTime *)malloc(sizeof(struct OpenCALTime));
    startTime(opencalTime);
    calRun2D(sciddicaT_simulation);
    endTime(opencalTime);
    free(opencalTime);

    // saving configuration
    calSaveSubstate2Dr(sciddicaT, Q.h, "./testsout/other/1.txt");

    // finalizations
    calRunFinalize2D(sciddicaT_simulation);
    calFinalize2D(sciddicaT);

    return 0;
}
