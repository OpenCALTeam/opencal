#include "sciddicaT.h"
#include <time.h>


#define ROWS 610
#define COLS 496
time_t start_time, end_time;

int main()
{
    time_t start_time, end_time;

    int * coordinates = new int[2] {ROWS, COLS};
    int dimension = 2;
    SciddicaTModel* sciddicaTModel = new SciddicaTModel (coordinates, dimension);

    printf ("Starting simulation...\n");
    start_time = time(NULL);
    sciddicaTModel->sciddicaTRun();
    end_time = time(NULL);
    printf ("Simulation terminated.\nElapsed time: %d\n", end_time-start_time);

    sciddicaTModel->sciddicaTSaveConfig();
    delete sciddicaTModel;


    return 0;
}


