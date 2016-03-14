#include "sciddicaT.h"

#define ROWS 610
#define COLS 496
time_t start_time, end_time;

int main()
{
    time_t start_time, end_time;

    //define coordinates of cellular space
    int * coordinates = new int[2] {ROWS, COLS};
    // dimension of cellular space
    int dimension = 2;
    SciddicaTModel* sciddicaTModel = new SciddicaTModel (coordinates, dimension);

    printf ("Starting simulation...\n");
    start_time = time(NULL);
    sciddicaTModel->sciddicaTRun();
    end_time = time(NULL);
    printf ("Simulation terminated.\nElapsed time: %d\n", end_time-start_time);

    //saving configuration
    sciddicaTModel->sciddicaTSaveConfig();
    //deallocates memory
    delete sciddicaTModel;
    return 0;
}


