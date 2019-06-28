#ifndef CALCLMULTINODE3D_H_
#define CALCLMULTINODE3D_H_

#include <mpi.h>
#include <stdio.h>
#include <vector>
#include <string>
#include <iostream>
#include <utility>
#include <sys/time.h>

extern "C"
{
#include <OpenCAL-OMP/cal3D.h>
#include <OpenCAL-OMP/cal3DRun.h>
};

#include <OpenCAL-OMP/cal3DMultiNodeCommon.h>

#include <string.h>

typedef struct BorderMapper
{

    size_t bufDIMreal; //!< Number of CALreal substates
    size_t bufDIMint;  //!< Number of CALint substates
    size_t bufDIMbyte; //!< Number of CALbyte substates
    //size_t bufDIMflags;

    CALreal *realBorder_OUT; //!< Array containing all the CALreal substates
    CALint *intBorder_OUT;   //!< Array containing all the CALint substates
    CALbyte *byteBorder_OUT; //!< Array containing all the CALbyte substates
    CALbyte *flagsBorder_OUT;

} borderMapper;

typedef void (*CALCallbackFuncMNInit3D)(struct MultiNode *ca3D, Node &mynode);
typedef void (*CALCallbackFuncMNFinalize3D)(struct MultiNode *ca3D, Node &mynode);

//MULTINODE---------------
//template <class F_INIT,class F_FINALIZE>
class MultiNode
{
  public:
    //only used to kep track of the communication overhead
    double comm_total = 0;
    double comp_total = 0;
    double kernel_total = 0;
    double kernel_communication = 0;
    timeval start_comm, end_comm;
    timeval start_comp, end_comp;
    timeval start_kernel_communication, end_kernel_communication;
    timeval start_kernel, end_kernel;

    CALDistributedDomain3D c;
    int rank;
    CALModel3D *host_CA;
    CALRun3D *ca_simulation;

    CALCallbackFuncMNInit3D init;
    CALCallbackFuncMNFinalize3D finalize;

    CALreal *realNodeGhosts = 0;
    CALint *intNodeGhosts = 0;
    CALbyte *byteNodeGhosts = 0;
    CALbyte *flagsNodeGhosts = 0;
    BorderMapper borderMapper;
    int borderSizeInCells;

    bool exchange_full_border;

    MultiNode(CALDistributedDomain3D _c, CALCallbackFuncMNInit3D i, CALCallbackFuncMNFinalize3D f) : c(_c),
                                                                                                     init(i), finalize(f)
    {
        MPI_Init(NULL, NULL);
        int world_rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

        rank = world_rank;
        MPI_Barrier(MPI_COMM_WORLD);
    }

    void setModel(CALModel3D *ca)
    {
        host_CA = ca;
        borderSizeInCells = host_CA->borderSizeInRows * host_CA->columns* host_CA->rows;

    }

    int GetNodeRank()
    {
        return rank;
    }

    bool checkWorkloads() { return true; };

    void setRunSimulation(CALRun3D *ca_sim)
    {
        ca_simulation = ca_sim;
        host_CA = ca_sim->ca3D;
        borderSizeInCells = host_CA->borderSizeInRows * host_CA->columns* host_CA->rows;
    }

    void allocateAndInit()
    {
        Node mynode = c.nodes[rank];

        init(this, mynode);

        exchange_full_border = c.is_full_exchange();

        const int rnumSubstate = host_CA->sizeof_pQr_array;
        const int inumSubstate = host_CA->sizeof_pQi_array;
        const int bnumSubstate = host_CA->sizeof_pQb_array;
        realNodeGhosts = 0;
        intNodeGhosts = 0;
        byteNodeGhosts = 0;
        flagsNodeGhosts = 0;
        realNodeGhosts = (CALreal *)malloc(rnumSubstate * borderSizeInCells * 2* sizeof(CALreal));
        intNodeGhosts = (CALint *)malloc(inumSubstate * borderSizeInCells * 2* sizeof(CALint));
        byteNodeGhosts = (CALbyte *)malloc(bnumSubstate * borderSizeInCells * 2* sizeof(CALbyte));
        flagsNodeGhosts = (CALbyte *)malloc(borderSizeInCells * 2* sizeof(CALbyte));
        MPI_Barrier(MPI_COMM_WORLD);
        
        borderMapper.bufDIMreal = sizeof(CALreal) * borderSizeInCells * rnumSubstate * 2;
        borderMapper.bufDIMint = sizeof(CALint)   * borderSizeInCells  * inumSubstate * 2;
        borderMapper.bufDIMbyte = sizeof(CALbyte) * borderSizeInCells * bnumSubstate * 2;
        
        borderMapper.realBorder_OUT = (CALreal *)malloc(borderMapper.bufDIMreal);
        borderMapper.intBorder_OUT = (CALint *)malloc(borderMapper.bufDIMint);
        borderMapper.byteBorder_OUT = (CALbyte *)malloc(borderMapper.bufDIMbyte);
        borderMapper.flagsBorder_OUT = (CALbyte *)malloc(sizeof(CALbyte) * borderSizeInCells * 2);

    }

    void _finalize()
    {
        free(realNodeGhosts);
        free(byteNodeGhosts);
        free(intNodeGhosts);
        free(flagsNodeGhosts);
        finalize(this, c.nodes[rank]);
    }

    void calCopyBuffer3DbM(CALbyte* M_src, CALbyte* M_dest, int rows, int columns, int slices)
    { 
         int tn;
         int ttotal;
         size_t size;

         int start;
         int chunk;

         size = rows * columns * slices;

     #pragma omp parallel private (start, chunk, tn, ttotal)
         {
             ttotal = CAL_GET_NUM_THREADS();

             tn = CAL_GET_THREAD_NUM();
             chunk = size / ttotal;
             start = tn * chunk;

             if (tn == ttotal - 1)
                 chunk = size - start;

             memcpy(M_dest + start+borderSizeInCells, M_src + start+borderSizeInCells,  sizeof(CALbyte) * chunk);
         }        
        // const CALint borderSizeInCells = borderSizeInRows * host_CA->columns;
        //memcpy(M_dest+borderSizeInCells, M_src+borderSizeInCells, sizeof(CALbyte)*rows*columns*slices);
    }
    void calCopyBuffer3DiM(CALint* M_src, CALint* M_dest, int rows, int columns, int slices)
    {
         int tn;
         int ttotal;
         size_t size;

         int start;
         int chunk;

         size = rows * columns * slices;

     #pragma omp parallel private (start, chunk, tn, ttotal)
         {
             ttotal = CAL_GET_NUM_THREADS();

             tn = CAL_GET_THREAD_NUM();
             chunk = size / ttotal;

             start = tn * chunk;

             if (tn == ttotal - 1)
                 chunk = size - start;

             memcpy(M_dest + start+borderSizeInCells, M_src + start+borderSizeInCells,
                    sizeof(CALint) * chunk);
         }
        
    //    const CALint borderSizeInCells = borderSizeInRows * host_CA->columns;
     //  memcpy(M_dest+borderSizeInCells, M_src+borderSizeInCells, sizeof(CALint)*rows*columns*slices);

    }
    void calCopyBuffer3DrM(CALreal* M_src, CALreal* M_dest, int rows, int columns, int slices)
    {
         int tn;
         int ttotal;
         size_t size;

         int start;
         int chunk;

         size = rows * columns * slices;

     #pragma omp parallel private (start, chunk, tn, ttotal)
         {
             ttotal = CAL_GET_NUM_THREADS();

             tn = CAL_GET_THREAD_NUM();
             chunk = size / ttotal;

             start = tn * chunk;

             if (tn == ttotal - 1)
                 chunk = size - start;

             memcpy(M_dest + start+borderSizeInCells, M_src + start+borderSizeInCells,
                    sizeof(CALreal) * chunk);
         }

      //const CALint borderSizeInCells = borderSizeInRows * host_CA->columns;  
     // memcpy(M_dest+borderSizeInCells, M_src+borderSizeInCells, sizeof(CALreal)*rows*columns*slices);
    }

    void calUpdateSubstate3DbM(struct CALModel3D* ca3D, struct CALSubstate3Db* Q) {
    // if ( (ca3D->OPTIMIZATION == CAL_OPT_ACTIVE_CELLS && ca3D->contiguousLinkedList->size_current > 0 ) ||
    //      ( ca3D->OPTIMIZATION == CAL_OPT_ACTIVE_CELLS_NAIVE && ca3D->A->size_current > 0) )
    //     calCopyBufferActiveCells3Db(Q->next, Q->current, ca3D);
    // else
        calCopyBuffer3DbM(Q->next, Q->current, ca3D->rows, ca3D->columns, ca3D->slices-(host_CA->borderSizeInRows*2));
    }

    void calUpdateSubstate3DiM(struct CALModel3D* ca3D, struct CALSubstate3Di* Q) {
    // if ( (ca3D->OPTIMIZATION == CAL_OPT_ACTIVE_CELLS && ca3D->contiguousLinkedList->size_current > 0 ) ||
    //      ( ca3D->OPTIMIZATION == CAL_OPT_ACTIVE_CELLS_NAIVE && ca3D->A->size_current > 0) )
    //     calCopyBufferActiveCells3Di(Q->next, Q->current, ca3D);
    // else
        calCopyBuffer3DiM(Q->next, Q->current, ca3D->rows, ca3D->columns, ca3D->slices-(host_CA->borderSizeInRows*2));
    }

    void calUpdateSubstate3DrM(struct CALModel3D* ca3D, struct CALSubstate3Dr* Q) {
    // if ( (ca3D->OPTIMIZATION == CAL_OPT_ACTIVE_CELLS && ca3D->contiguousLinkedList->size_current > 0 ) ||
    //      ( ca3D->OPTIMIZATION == CAL_OPT_ACTIVE_CELLS_NAIVE && ca3D->A->size_current > 0) )
    //     calCopyBufferActiveCells3Dr(Q->next, Q->current, ca3D);
    // else
        calCopyBuffer3DrM(Q->next, Q->current, ca3D->rows, ca3D->columns, ca3D->slices-(host_CA->borderSizeInRows*2));

    }

    void calUpdate3DM(struct CALModel3D *ca3D)
    {
        int i;

        if (ca3D->OPTIMIZATION == CAL_OPT_ACTIVE_CELLS_NAIVE)
            calUpdateActiveCellsNaive3D(ca3D);


        //updating substates
        for (i = 0; i < ca3D->sizeof_pQb_array; i++)
            calUpdateSubstate3DbM(ca3D, ca3D->pQb_array[i]);

        for (i = 0; i < ca3D->sizeof_pQi_array; i++)
            calUpdateSubstate3DiM(ca3D, ca3D->pQi_array[i]);

        for (i = 0; i < ca3D->sizeof_pQr_array; i++)
            calUpdateSubstate3DrM(ca3D, ca3D->pQr_array[i]);
    }

    void calApplyElementaryProcess3DM(struct CALModel3D *ca3D,             //!< Pointer to the cellular automaton structure.
                                      CALCallbackFunc3D elementary_process //!< Pointer to a transition function's elementary process.
    )
    {
        int i, j, k;


        if (ca3D->OPTIMIZATION == CAL_OPT_ACTIVE_CELLS_NAIVE && ca3D->A->size_current > 0) //Computationally active cells optimization(naive).
            calApplyElementaryProcessActiveCellsNaive3D( ca3D, elementary_process);
        // else if(ca3D->OPTIMIZATION == CAL_OPT_ACTIVE_CELLS && ca3D->contiguousLinkedList->size_current > 0) //Computationally active cells optimization(optimal).
        //     calApplyElementaryProcessActiveCellsCLL3D(ca3D, elementary_process);
        // else //Standart cicle of the transition function
#pragma omp parallel for private (k, i, j)
		for (i=0; i<ca3D->rows; i++)
			for (j=0; j<ca3D->columns; j++)
                for (k=host_CA->borderSizeInRows; k<ca3D->slices-host_CA->borderSizeInRows; k++)
                elementary_process(ca3D, i, j, k);
    }

    void calGlobalTransitionFunction3DM(struct CALModel3D *ca3D)
    {
        //The global transition function.
        //It applies transition function elementary processes sequentially.
        //Note that a substates' update is performed after each elementary process.
        int b;

        for (b = 0; b < ca3D->num_of_elementary_processes; b++)
        {
            //applying the b-th elementary process
            calApplyElementaryProcess3DM(ca3D, ca3D->elementary_processes[b]);
            //updating substates
            calUpdate3DM(ca3D);

            double TotalTime = 0;
            int ntComuunication = 0;
            handleBorderNodes(TotalTime, ntComuunication);
            if (ca3D->OPTIMIZATION == CAL_OPT_ACTIVE_CELLS_NAIVE){
                handleFlagsMultiNode();
                calUpdateActiveCellsNaive3D(ca3D);
            }
        }
    }

    void run(int STEPS)
    {

        int rank;

        double T1, T2, /* start/end times per rep */
            sumT = 0,  /* sum of all reps times */
            deltaT;    /* time for one rep */
        double avgT;
        double TotalTime = 0;
        int ntComuunication = 0;

        timeval start, end;

        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        gettimeofday(&start, NULL);

        //----------MEASURE TIME---------
        gettimeofday(&start_comm, NULL);
        handleBorderNodes(TotalTime, ntComuunication);
        gettimeofday(&end_comm, NULL);

        comm_total += 1000.0 * (end_comm.tv_sec - start_comm.tv_sec) +
                      (end_comm.tv_usec - start_comm.tv_usec) / 1000.0;

        int totalSteps = STEPS;
        while (STEPS--)
        {
            if (ca_simulation->globalTransition)
            {
                ca_simulation->globalTransition(ca_simulation->ca3D);
                if (ca_simulation->UPDATE_MODE == CAL_UPDATE_IMPLICIT)
                    calUpdate3DM(ca_simulation->ca3D);
            }
            else
                calGlobalTransitionFunction3DM(ca_simulation->ca3D);
            //No explicit substates and active cells updates are needed in this case

            if (ca_simulation->steering)
            {
                ca_simulation->steering(ca_simulation->ca3D);
                if (ca_simulation->UPDATE_MODE == CAL_UPDATE_IMPLICIT)
                    calUpdate3DM(ca_simulation->ca3D);
                handleBorderNodes(TotalTime, ntComuunication);
            }

            if (ca_simulation->stopCondition)
                if (ca_simulation->stopCondition(ca_simulation->ca3D))
                    break;

            gettimeofday(&start_comm, NULL);

            sumT += deltaT;
            gettimeofday(&end_comm, NULL);
            comm_total += 1000.0 * (end_comm.tv_sec - start_comm.tv_sec) +
                          (end_comm.tv_usec - start_comm.tv_usec) / 1000.0;

            gettimeofday(&end_comp, NULL);
            comp_total += 1000.0 * (end_comp.tv_sec - start_comp.tv_sec) +
                          (end_comp.tv_usec - start_comp.tv_usec) / 1000.0;

        } //STEPS

        gettimeofday(&end, NULL);
        unsigned long long t = 1000 * (end.tv_sec - start.tv_sec) +
                               (end.tv_usec - start.tv_usec) / 1000;

        double TotalCommTime = TotalTime + (kernel_communication / 1000);

        // end = time(NULL);
/*        printf("Rank=%d, Elapsed time: %llu ms\n", rank, t);
        printf("Rank=%d, Elapsed Communication time: %f s\n", rank, TotalCommTime);
        printf("Rank=%d, Elapsed MPI communication : %f s\n", rank, TotalTime);
        printf("Rank=%d, Elapsed Host Device PCI-E communication: %f ms\n", rank, kernel_communication);
        printf("Rank=%d, Elapsed time kernels: %f ms\n", rank, kernel_total);
        // three elementary procecces
        avgT = (TotalTime) / (ntComuunication);
        printf("Rank=%d, Avg Latency Time = %f s\n", rank, avgT);
        printf("Rank=%d, number of time send : %d \n", rank, ntComuunication);
*/
        MPI_Barrier(MPI_COMM_WORLD);
        _finalize();
        MPI_Finalize();
    } //run function

    void handleBorderNodes(double &T, int &ntComuunication)
    {
        if (host_CA->borderSizeInRows <= 0)
            return;
        handleBorderNodesR(T, ntComuunication);
        handleBorderNodesI(T, ntComuunication);
        handleBorderNodesB(T, ntComuunication);
    }

    void handleBorderNodesR(double &T, int &ntComuunication)
    {
        const MPI_Datatype DATATYPE = MPI_DOUBLE;
        if (!c.is_full_exchange())
        {

            CALint prev, next;
            CALreal *send_offset;
            CALreal *recv_offset = realNodeGhosts;

            //const CALint borderSizeInCells = borderSizeInRows * host_CA->columns;
            const int fullSize = host_CA->columns * host_CA->rows * host_CA->slices;
            const int numSubstates = host_CA->sizeof_pQr_array;
            const CALint count = (numSubstates * borderSizeInCells);

            if (numSubstates <= 0)
                return;
    
           for (int i = 0; i < numSubstates; ++i)
            {
                memcpy(borderMapper.realBorder_OUT + i * borderSizeInCells, host_CA->pQr_array[i]->current + (borderSizeInCells), sizeof(CALreal) * borderSizeInCells);
                memcpy(borderMapper.realBorder_OUT + (numSubstates + i) * borderSizeInCells, host_CA->pQr_array[i]->current + ((fullSize - borderSizeInCells*2)), sizeof(CALreal) * borderSizeInCells);
            }

            
 
    
            double T1, T2, deltaT;

            for (int i = 0; i < 2; i++)
            {

                next = ((rank + 1) + c.nodes.size()) % c.nodes.size();
                prev = ((rank - 1) + c.nodes.size()) % c.nodes.size();
                if (i == 1)
                    std::swap(next, prev);

                send_offset = borderMapper.realBorder_OUT;
                send_offset += (i == 0 ? 1 : 0) * count;

                recv_offset = realNodeGhosts;
                recv_offset += (i == 0 ? 0 : 1) * count;

                if (rank % 2 == 0)
                {
                    ntComuunication++;
                    T1 = MPI_Wtime();
                    
                    MPI_Send(send_offset, count, DATATYPE, next, i, MPI_COMM_WORLD);

                    MPI_Recv(recv_offset, count, DATATYPE, prev, i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    T2 = MPI_Wtime();
                    deltaT = T2 - T1;
                    T += deltaT;
                }
                else
                {

                    MPI_Recv(recv_offset, count, DATATYPE, prev, i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                    MPI_Send(send_offset, count, DATATYPE, next, i, MPI_COMM_WORLD);
                }
            }

            

            for (int i = 0; i < numSubstates; i++)
            {

                memcpy(host_CA->pQr_array[i]->current,(realNodeGhosts + i * borderSizeInCells),  sizeof(CALreal) * borderSizeInCells);
                memcpy(host_CA->pQr_array[i]->current + (fullSize - borderSizeInCells), (realNodeGhosts + numSubstates * borderSizeInCells + (i * borderSizeInCells)), sizeof(CALreal) * borderSizeInCells);
                memcpy(host_CA->pQr_array[i]->next,(realNodeGhosts + i * borderSizeInCells),  sizeof(CALreal) * borderSizeInCells);
                memcpy(host_CA->pQr_array[i]->next + (fullSize - borderSizeInCells), (realNodeGhosts + numSubstates * borderSizeInCells + (i * borderSizeInCells)), sizeof(CALreal) * borderSizeInCells);

            }


        }
    }

    void handleBorderNodesI(double &T, int &ntComuunication)
    {
        const MPI_Datatype DATATYPE = MPI_INT;
        if (!c.is_full_exchange())
        {

            CALint prev, next;
            CALint *send_offset;
            CALint *recv_offset = intNodeGhosts;

            //const CALint borderSizeInCells = borderSizeInRows * host_CA->columns;
            const int fullSize = host_CA->columns * host_CA->rows * host_CA->slices;
            const int numSubstates = host_CA->sizeof_pQi_array;
            const CALint count = (numSubstates * borderSizeInCells);
            double T1, T2, deltaT;

            if (numSubstates <= 0)
                return;
    
           for (int i = 0; i < numSubstates; ++i)
            {
                memcpy(borderMapper.intBorder_OUT + i * borderSizeInCells, host_CA->pQi_array[i]->current + (borderSizeInCells), sizeof(CALint) * borderSizeInCells);
                memcpy(borderMapper.intBorder_OUT + (numSubstates + i) * borderSizeInCells, host_CA->pQi_array[i]->current + ((fullSize - borderSizeInCells*2)), sizeof(CALint)* borderSizeInCells);
            }

                


            for (int i = 0; i < 2; i++)
            {

                next = ((rank + 1) + c.nodes.size()) % c.nodes.size();
                prev = ((rank - 1) + c.nodes.size()) % c.nodes.size();

                if (i == 1)
                    std::swap(next, prev);

                send_offset = borderMapper.intBorder_OUT;
                send_offset += (i == 0 ? 1 : 0) * count;

                recv_offset = intNodeGhosts;
                recv_offset += (i == 0 ? 0 : 1) * count;

                if (rank % 2 == 0)
                {

                    ntComuunication++;
                    T1 = MPI_Wtime();

                    MPI_Send(send_offset, count, DATATYPE, next, i, MPI_COMM_WORLD);

                    MPI_Recv(recv_offset, count, DATATYPE, prev, i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    T2 = MPI_Wtime();
                    deltaT = T2 - T1;
                    T += deltaT;
                }
                else
                {

                    MPI_Recv(recv_offset, count, DATATYPE, prev, i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    

                    MPI_Send(send_offset, count, DATATYPE, next, i, MPI_COMM_WORLD);
                }
            }
            for (int i = 0; i < numSubstates; i++)
            {

                memcpy(host_CA->pQi_array[i]->current,(intNodeGhosts + i * borderSizeInCells),  sizeof(CALint) * borderSizeInCells);
                memcpy(host_CA->pQi_array[i]->current + (fullSize - borderSizeInCells), (intNodeGhosts + numSubstates * borderSizeInCells + (i * borderSizeInCells)), sizeof(CALint) * borderSizeInCells);
                memcpy(host_CA->pQi_array[i]->next,(intNodeGhosts + i * borderSizeInCells),  sizeof(CALint) * borderSizeInCells);
                memcpy(host_CA->pQi_array[i]->next + (fullSize - borderSizeInCells), (intNodeGhosts + numSubstates * borderSizeInCells + (i * borderSizeInCells)), sizeof(CALint) * borderSizeInCells);

            }

        }
    }

    void handleBorderNodesB(double &T, int &ntComuunication)
    {
        const MPI_Datatype DATATYPE = MPI_CHAR;
        if (!c.is_full_exchange())
        {

            CALint prev, next;
            CALbyte *send_offset;
            CALbyte *recv_offset = byteNodeGhosts;

            //const CALint borderSizeInCells = borderSizeInRows * host_CA->columns;
            const int fullSize = host_CA->columns * host_CA->rows * host_CA->slices;
            const int numSubstates = host_CA->sizeof_pQb_array;
            const CALint count = (numSubstates * borderSizeInCells);

            if (numSubstates <= 0)
                return;

            for (int i = 0; i < numSubstates; ++i)
            {
                memcpy(borderMapper.byteBorder_OUT + i * borderSizeInCells, host_CA->pQb_array[i]->current + (borderSizeInCells), sizeof(CALbyte) * borderSizeInCells);
                memcpy(borderMapper.byteBorder_OUT + (numSubstates + i) * borderSizeInCells, host_CA->pQb_array[i]->current + ((fullSize - borderSizeInCells*2)), sizeof(CALbyte) * borderSizeInCells);
            }

            double T1, T2, deltaT;
            for (int i = 0; i < 2; i++)
            {

                next = ((rank + 1) + c.nodes.size()) % c.nodes.size();
                prev = ((rank - 1) + c.nodes.size()) % c.nodes.size();
                if (i == 1)
                    std::swap(next, prev);

                send_offset = borderMapper.byteBorder_OUT;
                send_offset += (i == 0 ? 1 : 0) * count;

                recv_offset = byteNodeGhosts;
                recv_offset += (i == 0 ? 0 : 1) * count;

                if (rank % 2 == 0)
                {

                    ntComuunication++;
                    T1 = MPI_Wtime();

                    MPI_Send(send_offset, count, DATATYPE, next, i, MPI_COMM_WORLD);

                    MPI_Recv(recv_offset, count, DATATYPE, prev, i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    T2 = MPI_Wtime();
                    deltaT = T2 - T1;
                    T += deltaT;
                }
                else
                {

                    MPI_Recv(recv_offset, count, DATATYPE, prev, i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    MPI_Send(send_offset, count, DATATYPE, next, i, MPI_COMM_WORLD);
                }
            }


            for (int i = 0; i < numSubstates; i++)
            {

                memcpy(host_CA->pQb_array[i]->current,(byteNodeGhosts + i * borderSizeInCells),  sizeof(CALbyte) * borderSizeInCells);
                memcpy(host_CA->pQb_array[i]->current + (fullSize - borderSizeInCells), (byteNodeGhosts + numSubstates * borderSizeInCells + (i * borderSizeInCells)), sizeof(CALbyte) * borderSizeInCells);
                memcpy(host_CA->pQb_array[i]->next,(byteNodeGhosts + i * borderSizeInCells),  sizeof(CALbyte) * borderSizeInCells);
                memcpy(host_CA->pQb_array[i]->next + (fullSize - borderSizeInCells), (byteNodeGhosts + numSubstates * borderSizeInCells + (i * borderSizeInCells)), sizeof(CALbyte) * borderSizeInCells);

            }
        }
    }


    void handleFlagsMultiNode() {
      const MPI_Datatype DATATYPE = MPI_CHAR;      
      if (!c.is_full_exchange()) {

        CALint prev, next;
        CALbyte* send_offset;
        CALbyte* recv_offset = flagsNodeGhosts;
        const CALint count = (borderSizeInCells);

        memcpy(borderMapper.flagsBorder_OUT, host_CA->A->flags, sizeof(CALbyte) * borderSizeInCells);
        memcpy(borderMapper.flagsBorder_OUT+ borderSizeInCells, host_CA->A->flags + (host_CA->rows*host_CA->columns)-(borderSizeInCells) , sizeof(CALbyte) * borderSizeInCells);

        next = ((rank + 1) + c.nodes.size()) % c.nodes.size();
        prev = ((rank - 1) + c.nodes.size()) % c.nodes.size();
        
        
        for (int i = 0; i < 2; i++) {

          if (i == 1) std::swap(next, prev);

          send_offset = borderMapper.flagsBorder_OUT;
          send_offset += (i == 0 ? 1 : 0) * count;

          recv_offset = flagsNodeGhosts;
          recv_offset += (i == 0 ? 0 : 1) * count;

          if (rank % 2 == 0) 
            {
              MPI_Send(send_offset, count, DATATYPE, next, i, MPI_COMM_WORLD);
              MPI_Recv(recv_offset, count, DATATYPE, prev, i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            } 
          else 
            {
              MPI_Recv(recv_offset, count, DATATYPE, prev, i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
              MPI_Send(send_offset, count, DATATYPE, next, i, MPI_COMM_WORLD);
            }

          //MPI_Barrier(MPI_COMM_WORLD);
            

        }  // for

        for(int i = 0; i < borderSizeInCells; i++){
            if(flagsNodeGhosts[i] == CAL_TRUE && host_CA->A->flags[borderSizeInCells+i] == CAL_FALSE){
                host_CA->A->size_next[CAL_GET_THREAD_NUM()]++;
                host_CA->A->flags[borderSizeInCells+i] = CAL_TRUE;
            }
         }

        int start = (host_CA->rows* host_CA->columns)-(borderSizeInCells*2);
        for(int i = 0; i < borderSizeInCells; i++){
            if(flagsNodeGhosts[borderSizeInCells+i] == CAL_TRUE && host_CA->A->flags[start+i] == CAL_FALSE)
            {
                host_CA->A->size_next[CAL_GET_THREAD_NUM()]++;
                host_CA->A->flags[start+i] = CAL_TRUE;
            }
         }

        int start2 = (host_CA->rows* host_CA->columns)-(borderSizeInCells);
        for(int i = 0; i < borderSizeInCells; i++){
            if(host_CA->A->flags[i] )
            {
                host_CA->A->size_next[CAL_GET_THREAD_NUM()]--;
                host_CA->A->flags[i] = CAL_FALSE;
            }
            if(host_CA->A->flags[start2+i] )
            {
                host_CA->A->size_next[CAL_GET_THREAD_NUM()]--;
                host_CA->A->flags[start2+i] = CAL_FALSE;
            }
        }
      }// if full excahnge
    }//function
};
//END MULTINODE--------

#endif /*CALCLMULTINODE3D_H_*/
