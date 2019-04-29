#include <mpi.h>
#include <stdio.h>
#include <vector>
#include<string>
#include<iostream>
#include <utility>
#include <OpenCAL-CL/calclMultiNode3D.h>
//#define ACTIVE_CELLS
#ifdef ACTIVE_CELLS
#define KERNEL_SRC "./kernel_mod2_active/source/"
#define KERNEL_INC "./kernel_mod2_active/source/"
#else
#define KERNEL_SRC "./kernel_mod2/source/"
#define KERNEL_INC "./kernel_mod2/include/"
#endif

#define KERNEL  "mod2TransitionFunction"
#define OUTPUT_PATH "./output"


struct CALModel3D* mod2;
struct CALSubstate3Di* Q;

//void mod2TransitionFunction(struct CALModel3D* ca, int i, int j, int k)
//{
//    int sum = 0, n;

//    for (n=0; n<ca->sizeof_X; n++)
//        sum += calGetX3Db(ca, Q, i, j, k, n);

//    calSet3Db(ca, Q, i, j, k, sum%2);
//}

void init( struct CALCLMultiDevice3D* multidevice, const Node& mynode){
    #ifdef ACTIVE_CELLS
        mod2 = calCADef3D(mynode.rows, mynode.columns, mynode.workload, CAL_MOORE_NEIGHBORHOOD_3D, CAL_SPACE_TOROIDAL, CAL_OPT_ACTIVE_CELLS_NAIVE);
    #else
        mod2 = calCADef3D(mynode.rows, mynode.columns, mynode.workload, CAL_MOORE_NEIGHBORHOOD_3D, CAL_SPACE_TOROIDAL, CAL_NO_OPT);
    #endif
    // add the Q substate to the mod2 CA
    Q = calAddSubstate3Di(mod2);

    // set the whole substate to 0
    calInitSubstate3Di(mod2, Q, 0);

    // set a seed at position (2, 3, 1)
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if(rank ==0)
    {
       calInit3Di(mod2, Q, 2, 3, 1, 1);
#ifdef ACTIVE_CELLS
        //adds the cell (i, j) to the set of active ones
        calAddActiveCell3D(mod2, 2, 3, 1);
        printf("ActiveCells = %d \n", mod2->A->size_current);
        calUpdateActiveCells3D(mod2);
#endif
    }
//    int l = 1;
//    if(rank ==1){
//        l = 3;
//    }
//        for (int k =0; k < mynode.workload; ++k) {
//            for (int i = 0; i < mynode.rows; ++i) {
//                for (int j = 0; j < mynode.columns; ++j) {
//                    calSetCurrent3Di(mod2, Q, i, j, k, (k+l));
//                }
//            }
//        }


    std::string s = OUTPUT_PATH + std::to_string(rank) + "_0000.txt";
    calSaveSubstate3Di(mod2, Q, (char*)s.c_str());

    // Define a device-side CAs
    int borderSize=1;
    calclMultiDeviceCADef3D(multidevice, mod2, KERNEL_SRC, KERNEL_INC, borderSize, mynode.devices);

    // add transition function's elementary process
    calclMultiDeviceAddElementaryProcess3D(multidevice, KERNEL);

}

void finalize(struct CALCLMultiDevice3D* multidevice){
         // Saving results
         int rank;
         MPI_Comm_rank(MPI_COMM_WORLD, &rank);
         std::cout<<"sono il processo "<<rank<<" finalizzo\n";
         std::string s = OUTPUT_PATH + std::to_string(rank) + "_LAST.txt";
         calSaveSubstate3Di(multidevice->device_models[0]->host_CA, Q, (char*)s.c_str());

}

int main(int argc, char** argv){

    CALDistributedDomain3D domain = calDomainPartition3D(argc,argv);
    MultiNode3D mn(domain, init, finalize);
    mn.allocateAndInit();
    mn.run(1);
    return 0;
}
