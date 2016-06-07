#include "sciddicaT.h"

#define ROWS 610
#define COLS 496
time_t start_time, end_time;

int main(int argc, char** argv)
{
//    time_t start_time, end_time;

//    int version=1;
//    if (sscanf (argv[1], "%i", &version)!=1 && version >=0) {
//        printf ("error - not an integer");
//        exit(-1);
//     }

    //define coordinates of cellular space
    std::array<COORD_TYPE,2> coords  = {ROWS, COLS};
    // dimension of cellular space
//    int dimension = 2;
    SciddicaTModel* sciddicaTModel = new SciddicaTModel (coords);


//    printf ("Starting simulation...\n");
//    start_time = time(NULL);
    sciddicaTModel->sciddicaTRun();
//    end_time = time(NULL);
//    printf ("Simulation terminated.\nElapsed time: %ld\n", end_time-start_time);

    //saving configuration
    sciddicaTModel->sciddicaTSaveConfig();
    //deallocates memory
    delete sciddicaTModel;
    return 0;
}
