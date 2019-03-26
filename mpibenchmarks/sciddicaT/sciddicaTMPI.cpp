#include <mpi.h>
#include <stdio.h>
#include <vector>
#include<string>
#include<iostream>
#include <utility>
#include<OpenCAL-CL/calclMultiNode.h>
//#define ACTIVE_CELLS

#define OUTPUT_PATH "./data/width_final_"
#define NUMBER_OF_OUTFLOWS 4


//test local
// #define DEM_PATH "/home/mpiuser/git/sciddicaT-local/testData/testLocal/dem.txt"
// #define SOURCE_PATH "/home/mpiuser/git/sciddicaT-local/testData/testLocal/source.txt"


//STANDARD
#define DEM_PATH "./data/dem_standard.txt"
#define SOURCE_PATH "./data/source_standard.txt"
//STRESS TEST R
//#define DEM_PATH "./testData/stress_test_R/dem_stress_test_R.txt"
//#define SOURCE_PATH "./testData/stress_test_R/source_stress_test_R.txt"

//STRESS TEST RD Rick
//#define DEM_PATH "/home/mpiuser/git/sciddicaT/testData/stress_test_RD/etnad.txt"
//#define SOURCE_PATH "/home/mpiuser/git/sciddicaT/testData/stress_test_RD/sourced.txt"

//STRESS TEST RD 
//#define DEM_PATH "/home/mpiuser/git/sciddicaT/testData/stress_test_R/dem_stress_test_RD.txt"
//#define SOURCE_PATH "/home/mpiuser/git/sciddicaT/testData/stress_test_R/source_stress_test_RD.txt"


#ifdef ACTIVE_CELLS
#define KERNEL_SRC "./kernelActive/source/"
#define KERNEL_INC "./kernelActive/include/"
#else
#define KERNEL_SRC "./kernel_sciddicaT/source/"
#define KERNEL_INC "./kernel_sciddicaT/include/"

#endif


#define P_R 0.5
#define P_EPSILON 0.001
#define KERNEL_ELEM_PROC_FLOW_COMPUTATION "flowsComputation"
#define KERNEL_ELEM_PROC_WIDTH_UPDATE "widthUpdate"
#define KERNEL_STEERING  "steering"


// Declare XCA model (host_CA), substates (Q), parameters (P)
struct CALModel2D* host_CA;
struct sciddicaTSubstates {
    struct CALSubstate2Dr *f[NUMBER_OF_OUTFLOWS];
    struct CALSubstate2Dr *z;
    struct CALSubstate2Dr *h;
} Q;

struct sciddicaTParameters {
    CALParameterr epsilon;
    CALParameterr r;
} P;
CALCLmem bufferEpsilonParameter;
CALCLmem bufferRParameter;


// SciddicaT simulation init function
void sciddicaTSimulationInit(struct CALModel2D* host_CA) {
    CALreal z, h;
    CALint i, j;

    //initializing substates to 0
    calInitSubstate2Dr(host_CA, Q.f[0], 0);
    calInitSubstate2Dr(host_CA, Q.f[1], 0);
    calInitSubstate2Dr(host_CA, Q.f[2], 0);
    calInitSubstate2Dr(host_CA, Q.f[3], 0);

    //sciddicaT parameters sett1ing
    P.r = P_R;
    P.epsilon = P_EPSILON;

    //sciddicaT source initialization
    for (i = 0; i < host_CA->rows; i++)
        for (j = 0; j < host_CA->columns; j++) {
            h = calGet2Dr(host_CA, Q.h, i, j);

            if (h > 0.0) {
                z = calGet2Dr(host_CA, Q.z, i, j);
                calSet2Dr(host_CA, Q.z, i, j, z - h);

#ifdef ACTIVE_CELLS
                //adds the cell (i, j) to the set of active ones
                calAddActiveCell2D(host_CA, i, j);
#endif
            }
        }
}

void init( struct CALCLMultiDevice* multidevice, const Node& mynode){

CALModel2D* host_CA;
#ifdef ACTIVE_CELLS
    host_CA = calCADef2D(mynode.workload, mynode.columns, CAL_VON_NEUMANN_NEIGHBORHOOD_2D, CAL_SPACE_TOROIDAL, CAL_OPT_ACTIVE_CELLS_NAIVE);
#else
    host_CA = calCADef2D(mynode.workload, mynode.columns, CAL_VON_NEUMANN_NEIGHBORHOOD_2D, CAL_SPACE_TOROIDAL, CAL_NO_OPT);
#endif
    // Add substates
    Q.f[0] = calAddSubstate2Dr(host_CA);
    Q.f[1] = calAddSubstate2Dr(host_CA);
    Q.f[2] = calAddSubstate2Dr(host_CA);
    Q.f[3] = calAddSubstate2Dr(host_CA);
    Q.z = calAddSubstate2Dr(host_CA);
    Q.h = calAddSubstate2Dr(host_CA);

    // Load configuration
    calNodeLoadSubstate2Dr(host_CA, Q.z, DEM_PATH, mynode);//TODO offset e workload
    calNodeLoadSubstate2Dr(host_CA, Q.h, SOURCE_PATH, mynode);//TODO offset e workload

    // Initialization
    sciddicaTSimulationInit(host_CA);
    calUpdate2D(host_CA);

    // Define a device-side CAs
    int borderSize=1;
    calclMultiDeviceCADef2D(multidevice, host_CA, KERNEL_SRC, KERNEL_INC, borderSize, mynode.devices);

    // Extract kernels from program
    calclMultiDeviceAddElementaryProcess2D(multidevice, KERNEL_ELEM_PROC_FLOW_COMPUTATION);
    calclMultiDeviceAddElementaryProcess2D(multidevice, KERNEL_ELEM_PROC_WIDTH_UPDATE);

    bufferEpsilonParameter = calclCreateBuffer(multidevice->context, &P.epsilon, sizeof(CALParameterr));
    bufferRParameter = calclCreateBuffer(multidevice->context, &P.r, sizeof(CALParameterr));

    calclMultiDeviceAddSteeringFunc2D(multidevice,KERNEL_STEERING);
    calclMultiDeviceSetKernelArg2D(multidevice,KERNEL_ELEM_PROC_FLOW_COMPUTATION, 0,sizeof(CALCLmem), &bufferEpsilonParameter);
    calclMultiDeviceSetKernelArg2D(multidevice,KERNEL_ELEM_PROC_FLOW_COMPUTATION, 1, sizeof(CALCLmem), &bufferRParameter);
}

// void finalize(struct CALCLMultiDevice* multidevice){
//     // Saving results
//     int rank;
//     MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//     std::cout<<"sono il processo "<<rank<<" finalizzo\n";
//     std::string s = OUTPUT_PATH + std::to_string(rank) + ".txt";
//     calSaveSubstate2Dr(multidevice->device_models[0]->host_CA, Q.h, (char*)s.c_str());
// }

void finalize(struct CALCLMultiDevice* multidevice){
    // Saving results
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::cout<<"sono il processo "<<rank<<" finalizzo\n";
    std::string s = OUTPUT_PATH + std::to_string(rank) + ".txt";
    calSaveSubstate2Dr(multidevice->device_models[0]->host_CA, Q.h, (char*)s.c_str());
}

int main(int argc, char** argv){

    CALDistributedDomain2D domain = calDomainPartition2D(argc,argv);

    MultiNode mn(domain, init, finalize);
    mn.allocateAndInit();
    mn.run(4000);

    return 0;
}
