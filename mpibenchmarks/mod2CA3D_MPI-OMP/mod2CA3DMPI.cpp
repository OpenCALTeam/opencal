#include <mpi.h>
#include <stdio.h>
#include <vector>
#include<string>
#include<iostream>
#include <utility>
#include <OpenCAL-OMP/cal3DMultiNode.h>
#define ACTIVE_CELLS


#define OUTPUT_PATH "./output"


struct CALModel3D* mod2;
struct CALSubstate3Di* Q;

void mod2TransitionFunction(struct CALModel3D* ca, int i, int j, int k)
{
   int sum = 0, n;

   for (n=0; n<ca->sizeof_X; n++)
       sum += calGetX3Di(ca, Q, i, j, k, n);

   calSet3Di(ca, Q, i, j, k, sum%2);
   if(sum%2){
      calAddActiveCell3D(ca, i, j, k);
   }

//    if(k == 1 && i == 1 && j == 3){
//        printf("sum = %d, calGet3Di(ca, Q, i, j, k) = %d \n", sum, ca->Q->current, Q, i, j, k));
//    }
}

void init( MultiNode* multinode, Node& mynode){
    int borderSize = 1;
    #ifdef ACTIVE_CELLS
        mod2 = calCADef3DMN(mynode.rows, mynode.columns, mynode.workload, CAL_MOORE_NEIGHBORHOOD_3D, CAL_SPACE_TOROIDAL, CAL_OPT_ACTIVE_CELLS_NAIVE, borderSize);
    #else
        mod2 = calCADef3DMN(mynode.rows, mynode.columns, mynode.workload, CAL_MOORE_NEIGHBORHOOD_3D, CAL_SPACE_TOROIDAL, CAL_NO_OPT, borderSize);
    #endif
    // add the Q substate to the mod2 CA
    Q = calAddSubstate3Di(mod2);

    calAddElementaryProcess3D(mod2, mod2TransitionFunction);

    // set the whole substate to 0
    calInitSubstate3Di(mod2, Q, 0);

    // set a seed at position (2, 3, 1)
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if(rank ==0)
    {
       calInit3Di(mod2, Q, 2, 3, 1+borderSize, 1);
#ifdef ACTIVE_CELLS
        //adds the cell (i, j) to the set of active ones
        calAddActiveCell3D(mod2, 2, 3, 1);
        printf("ActiveCells = %d \n", mod2->A->size_current);
        calUpdateActiveCells3D(mod2);
#endif
    }
    std::string s = OUTPUT_PATH + std::to_string(rank) + "_0000.txt";
    calNodeSaveSubstate3Di(mod2, Q, (char*)s.c_str(), mynode);

    struct CALRun3D * host_simulation = calRunDef3D(mod2, 1, 1, CAL_UPDATE_IMPLICIT);
    multinode->setRunSimulation(host_simulation);

}

void finalize(MultiNode* multinode, Node& mynode){
    // Saving results
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::cout<<"sono il processo "<<rank<<" finalizzo\n";
    std::string s = OUTPUT_PATH + std::to_string(rank) + "_LAST.txt";
    calNodeSaveSubstate3Di(multinode->host_CA, Q, (char*)s.c_str(), mynode);

}

int main(int argc, char** argv){

    CALDistributedDomain3D domain = calDomainPartition3D(argc,argv);
    MultiNode mn(domain, init, finalize);
    mn.allocateAndInit();
    mn.run(20);
    return 0;
}
