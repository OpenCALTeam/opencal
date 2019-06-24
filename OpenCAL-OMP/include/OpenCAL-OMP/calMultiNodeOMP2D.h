#ifndef CALCLMULTINODE2D_H_
#define CALCLMULTINODE2D_H_

#include <mpi.h>
#include <stdio.h>
#include <vector>
#include <string>
#include <iostream>
#include <utility>
#include <sys/time.h>

extern "C"
{
#include <OpenCAL-OMP/cal2D.h>
#include <OpenCAL-OMP/cal2DRun.h>
};

#include <OpenCAL-OMP/calMultiNodeCommon.h>

#include <string.h>

typedef struct BorderMapper
{

    size_t bufDIMreal; //!< Number of CALreal substates
    size_t bufDIMint;  //!< Number of CALint substates
    size_t bufDIMbyte; //!< Number of CALbyte substates
                       // size_t bufDIMflags;

    CALreal *realBorder_OUT; //!< Array containing all the CALreal substates
    CALint *intBorder_OUT;   //!< Array containing all the CALint substates
    CALbyte *byteBorder_OUT; //!< Array containing all the CALbyte substates
    //CALbyte *flagsBorder_OUT;

} borderMapper;

typedef void (*CALCallbackFuncMNInit2D)(struct MultiNode *ca2D, const Node &mynode);
typedef void (*CALCallbackFuncMNFinalize2D)(struct MultiNode *ca2D, const Node &mynode);

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

    CALDistributedDomain2D c;
    int rank;
    CALModel2D *host_CA;
    CALRun2D *ca_simulation;

    CALCallbackFuncMNInit2D init;
    CALCallbackFuncMNFinalize2D finalize;

    CALreal *realNodeGhosts = 0;
    CALint *intNodeGhosts = 0;
    CALbyte *byteNodeGhosts = 0;
    CALbyte *flagsNodeGhosts = 0;
    BorderMapper borderMapper;

    bool exchange_full_border;

    int borderSize = 1;

    MultiNode(CALDistributedDomain2D _c, CALCallbackFuncMNInit2D i, CALCallbackFuncMNFinalize2D f) : c(_c),
                                                                                                     init(i), finalize(f)
    {
        MPI_Init(NULL, NULL);
        int world_rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

        rank = world_rank;
        MPI_Barrier(MPI_COMM_WORLD);
    }

    void setBorderSize(int b)
    {
        borderSize = b;
    }

    void setModel(CALModel2D *ca)
    {
        host_CA = ca;
    }

    int GetNodeRank()
    {
        return rank;
    }

    bool checkWorkloads() { return true; };

    void setRunSimulation(CALRun2D *ca_sim)
    {
        ca_simulation = ca_sim;
        host_CA = ca_sim->ca2D;
    }

    void allocateAndInit()
    {
        Node mynode = c.nodes[rank];

        init(this, mynode);

        exchange_full_border = c.is_full_exchange();

        const CALint sizeBorder = borderSize * host_CA->columns;
        const int rnumSubstate = host_CA->sizeof_pQr_array;
        const int inumSubstate = host_CA->sizeof_pQi_array;
        const int bnumSubstate = host_CA->sizeof_pQb_array;
        realNodeGhosts = 0;
        intNodeGhosts = 0;
        byteNodeGhosts = 0;
        realNodeGhosts = (CALreal *)calloc(rnumSubstate * sizeBorder * 2, sizeof(CALreal));
        intNodeGhosts = (CALint *)calloc(inumSubstate * sizeBorder * 2, sizeof(CALint));
        byteNodeGhosts = (CALbyte *)calloc(bnumSubstate * sizeBorder * 2, sizeof(CALbyte));
        flagsNodeGhosts = (CALbyte *)calloc(sizeBorder * 2, sizeof(CALbyte));
        MPI_Barrier(MPI_COMM_WORLD);
        borderMapper.bufDIMbyte = sizeof(CALbyte) * host_CA->columns * host_CA->sizeof_pQb_array * borderSize * 2;
        borderMapper.bufDIMreal = sizeof(CALreal) * host_CA->columns * host_CA->sizeof_pQr_array * borderSize * 2;
        borderMapper.bufDIMint = sizeof(CALint) * host_CA->columns * host_CA->sizeof_pQi_array * borderSize * 2;

        borderMapper.byteBorder_OUT = (CALbyte *)malloc(borderMapper.bufDIMbyte);
        borderMapper.realBorder_OUT = (CALreal *)malloc(borderMapper.bufDIMreal);
        borderMapper.intBorder_OUT = (CALint *)malloc(borderMapper.bufDIMint);
    }

    void _finalize()
    {
        free(realNodeGhosts);
        free(byteNodeGhosts);
        free(intNodeGhosts);
        free(flagsNodeGhosts);
        finalize(this, c.nodes[rank]);
    }

    void calApplyElementaryProcess2DM(struct CALModel2D *ca2D,             //!< Pointer to the cellular automaton structure.
                                      CALCallbackFunc2D elementary_process //!< Pointer to a transition function's elementary process.
    )
    {
        int i, j;

        // if (ca2D->OPTIMIZATION == CAL_OPT_ACTIVE_CELLS_NAIVE && ca2D->A->size_current > 0) //Computationally active cells optimization(naive).
        //     calApplyElementaryProcessActiveCellsNaive2D( ca2D, elementary_process);
        // else if(ca2D->OPTIMIZATION == CAL_OPT_ACTIVE_CELLS && ca2D->contiguousLinkedList->size_current > 0) //Computationally active cells optimization(optimal).
        //     calApplyElementaryProcessActiveCellsCLL2D(ca2D, elementary_process);
        // else //Standart cicle of the transition function
        for (i = borderSize; i < ca2D->rows - borderSize; i++)
            for (j = 0; j < ca2D->columns; j++)
                elementary_process(ca2D, i, j);
    }

    void calCopyBuffer2DbM(CALbyte* M_src, CALbyte* M_dest, int rows, int columns)
    { 
        const CALint sizeBorder = borderSize * host_CA->columns;
        memcpy(M_dest+sizeBorder, M_src+sizeBorder, sizeof(CALbyte)*rows*columns);
    }
    void calCopyBuffer2DiM(CALint* M_src, CALint* M_dest, int rows, int columns)
    {
       const CALint sizeBorder = borderSize * host_CA->columns;
       memcpy(M_dest+sizeBorder, M_src+sizeBorder, sizeof(CALint)*rows*columns);
    }
    void calCopyBuffer2DrM(CALreal* M_src, CALreal* M_dest, int rows, int columns)
    {
      const CALint sizeBorder = borderSize * host_CA->columns;  
      memcpy(M_dest+sizeBorder, M_src+sizeBorder, sizeof(CALreal)*rows*columns);
    }

    void calUpdateSubstate2DbM(struct CALModel2D* ca2D, struct CALSubstate2Db* Q) {
    // if ( (ca2D->OPTIMIZATION == CAL_OPT_ACTIVE_CELLS && ca2D->contiguousLinkedList->size_current > 0 ) ||
    //      ( ca2D->OPTIMIZATION == CAL_OPT_ACTIVE_CELLS_NAIVE && ca2D->A->size_current > 0) )
    //     calCopyBufferActiveCells2Db(Q->next, Q->current, ca2D);
    // else
        calCopyBuffer2DbM(Q->next, Q->current, ca2D->rows-borderSize, ca2D->columns);
    }

    void calUpdateSubstate2DiM(struct CALModel2D* ca2D, struct CALSubstate2Di* Q) {
    // if ( (ca2D->OPTIMIZATION == CAL_OPT_ACTIVE_CELLS && ca2D->contiguousLinkedList->size_current > 0 ) ||
    //      ( ca2D->OPTIMIZATION == CAL_OPT_ACTIVE_CELLS_NAIVE && ca2D->A->size_current > 0) )
    //     calCopyBufferActiveCells2Di(Q->next, Q->current, ca2D);
    // else
        calCopyBuffer2DiM(Q->next, Q->current, ca2D->rows-borderSize, ca2D->columns);
    }

    void calUpdateSubstate2DrM(struct CALModel2D* ca2D, struct CALSubstate2Dr* Q) {
    // if ( (ca2D->OPTIMIZATION == CAL_OPT_ACTIVE_CELLS && ca2D->contiguousLinkedList->size_current > 0 ) ||
    //      ( ca2D->OPTIMIZATION == CAL_OPT_ACTIVE_CELLS_NAIVE && ca2D->A->size_current > 0) )
    //     calCopyBufferActiveCells2Dr(Q->next, Q->current, ca2D);
    // else
        calCopyBuffer2DrM(Q->next, Q->current, ca2D->rows-borderSize, ca2D->columns);

    }

    void calUpdate2DM(struct CALModel2D *ca2D)
    {
        int i;

        //updating substates
        for (i = 0; i < ca2D->sizeof_pQb_array; i++)
            calUpdateSubstate2DbM(ca2D, ca2D->pQb_array[i]);

        for (i = 0; i < ca2D->sizeof_pQi_array; i++)
            calUpdateSubstate2DiM(ca2D, ca2D->pQi_array[i]);

        for (i = 0; i < ca2D->sizeof_pQr_array; i++)
            calUpdateSubstate2DrM(ca2D, ca2D->pQr_array[i]);
    }

    void calGlobalTransitionFunction2DM(struct CALModel2D *ca2D)
    {
        //The global transition function.
        //It applies transition function elementary processes sequentially.
        //Note that a substates' update is performed after each elementary process.
        int b;

        for (b = 0; b < ca2D->num_of_elementary_processes; b++)
        {
            //applying the b-th elementary process
            calApplyElementaryProcess2DM(ca2D, ca2D->elementary_processes[b]);
            //updating substates
            calUpdate2DM(ca2D);

            double TotalTime = 0;
            int ntComuunication = 0;
            handleBorderNodes(TotalTime, ntComuunication);
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
                ca_simulation->globalTransition(ca_simulation->ca2D);
                if (ca_simulation->UPDATE_MODE == CAL_UPDATE_IMPLICIT)
                    calUpdate2DM(ca_simulation->ca2D);
            }
            else
                calGlobalTransitionFunction2DM(ca_simulation->ca2D);
            //No explicit substates and active cells updates are needed in this case

            if (ca_simulation->steering)
            {
                ca_simulation->steering(ca_simulation->ca2D);
                if (ca_simulation->UPDATE_MODE == CAL_UPDATE_IMPLICIT)
                    calUpdate2DM(ca_simulation->ca2D);
                handleBorderNodes(TotalTime, ntComuunication);
            }

            if (ca_simulation->stopCondition)
                if (ca_simulation->stopCondition(ca_simulation->ca2D))
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
        if (borderSize <= 0)
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

            const CALint sizeBorder = borderSize * host_CA->columns;
            const int fullSize = host_CA->columns * host_CA->rows;
            const int numSubstates = host_CA->sizeof_pQr_array;
            const CALint count = (numSubstates * sizeBorder);

            if (numSubstates <= 0)
                return;
    
           for (int i = 0; i < numSubstates; ++i)
            {
                memcpy(borderMapper.realBorder_OUT + i * sizeBorder, host_CA->pQr_array[i]->current + (sizeBorder), sizeof(CALreal) * sizeBorder);
                memcpy(borderMapper.realBorder_OUT + (numSubstates + i) * sizeBorder, host_CA->pQr_array[i]->current + ((fullSize - sizeBorder*2)), sizeof(CALreal) * sizeBorder);
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

                    // std::cout << std::endl
                    //           << rank << " invia a " << next << std::endl;

                    // //LOG
                    // for (int i = 0; i < count*2; ++i)
                    // {
                    //     if (i % (sizeBorder) == 0)
                    //     {
                    //         std::cout << std::endl;
                    //     }
                    //     std::cout << borderMapper.realBorder_OUT[i] << " ";
                        
                    // }
                    
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

                memcpy(host_CA->pQr_array[i]->current,(realNodeGhosts + i * sizeBorder),  sizeof(CALreal) * sizeBorder);

                memcpy(host_CA->pQr_array[i]->current + (fullSize - sizeBorder), (realNodeGhosts + numSubstates * sizeBorder + (i * sizeBorder)), sizeof(CALreal) * sizeBorder);
                memcpy(host_CA->pQr_array[i]->next,(realNodeGhosts + i * sizeBorder),  sizeof(CALreal) * sizeBorder);

                memcpy(host_CA->pQr_array[i]->next + (fullSize - sizeBorder), (realNodeGhosts + numSubstates * sizeBorder + (i * sizeBorder)), sizeof(CALreal) * sizeBorder);

            }
            // if(rank == 1){
            //    std::cout << std::endl
            //                   << rank << " riceve da " << prev << std::endl;

            //         //LOG
            //         for (int i = 0; i < count*2; ++i)
            //         {
            //            if (i % (count) == 0)
            //             {
            //                 std::cout << std::endl;
            //             } 
            //             std::cout << realNodeGhosts[i]<< " ";
                        
            //         }
            //         std::cout << std::endl;
            //         std::cout <<  std::endl;
            //     for (int i = 0; i < host_CA->rows*host_CA->columns; ++i)
            //          {
            //            if (i % (host_CA->columns) == 0)
            //             {
            //                 std::cout << std::endl;
            //             } 
            //             std::cout << host_CA->pQr_array[0]->current[i]<< " ";
                        
            //         }


            // }


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

            const CALint sizeBorder = borderSize * host_CA->columns;
            const int fullSize = host_CA->columns * (host_CA->rows);
            const int numSubstates = host_CA->sizeof_pQi_array;
            const CALint count = (numSubstates * sizeBorder);
            double T1, T2, deltaT;

            if (numSubstates <= 0)
                return;

            for (int i = 0; i < numSubstates; ++i)
            {
                memcpy(borderMapper.intBorder_OUT + i * sizeBorder, host_CA->pQi_array[i]->current + (i * fullSize + sizeBorder), sizeof(CALint) * sizeBorder);
               /* if(rank == 0){
                    std::cout <<  "fullsize " <<fullSize << std::endl; 
                    std::cout <<  "sizeBorder " << sizeBorder << std::endl;
                }*/
                memcpy(borderMapper.intBorder_OUT + (numSubstates + i) * sizeBorder, host_CA->pQi_array[i]->current + (i * fullSize + (fullSize - sizeBorder*2)), sizeof(CALint) * sizeBorder);
            }

            for (int i = 0; i < 2; i++)
            {

                next = ((rank + 1) + c.nodes.size()) % c.nodes.size();
                prev = ((rank - 1) + c.nodes.size()) % c.nodes.size();

               /*if (rank == 0)
                {
                    std::cout << std::endl
                              << rank << " invia a " << next << std::endl;

                    //LOG
                    for (int i = 0; i < sizeBorder * 2 * numSubstates; ++i)
                    {
                        if (i % (sizeBorder) == 0)
                        {
                            std::cout << std::endl;
                        }
                        std::cout << borderMapper.intBorder_OUT[i] << " ";
                    }
                }*/

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

                    // std::cout << std::endl
                    //           << rank << " invia a " << next << std::endl;

                    // //LOG
                    // for (int i = 0; i < count; ++i)
                    // {
                    //     std::cout << borderMapper.intBorder_OUT[i] << " ";
                    //     if (i % (sizeBorder * 2) == 0)
                    //     {
                    //         std::cout << std::endl;
                    //     }
                    // }

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
                memcpy(host_CA->pQi_array[i]->current + (i * fullSize),(intNodeGhosts + i * sizeBorder), sizeof(CALint) * sizeBorder);

                memcpy(host_CA->pQi_array[i]->current + ((i + 1) * fullSize - sizeBorder),(intNodeGhosts + numSubstates * sizeBorder + (i * sizeBorder)),  sizeof(CALint) * sizeBorder);
            }
            if(rank == 1){
               std::cout << std::endl
                              << rank << " riceve da " << prev << std::endl;

                    //LOG
                    for (int i = 0; i < count*2; ++i)
                    {
                       if (i % (count) == 0)
                        {
                            std::cout << std::endl;
                        } 
                        std::cout << intNodeGhosts[i]<< " ";
                        
                    }
                    std::cout << std::endl;
                    std::cout <<  std::endl;
                for (int i = 0; i < host_CA->rows*host_CA->columns; ++i)
                     {
                       if (i % (host_CA->columns) == 0)
                        {
                            std::cout << std::endl;
                        } 
                        std::cout << host_CA->pQi_array[0]->current[i]<< " ";
                        
                    }


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

            const CALint sizeBorder = borderSize * host_CA->columns;
            const int fullSize = host_CA->columns * host_CA->rows + sizeBorder;
            const int numSubstates = host_CA->sizeof_pQb_array;
            const CALint count = (numSubstates * sizeBorder);

            if (numSubstates <= 0)
                return;

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

                memcpy((byteNodeGhosts + i * sizeBorder), host_CA->pQb_array[i] + (i * fullSize), sizeof(CALbyte) * sizeBorder);

                memcpy((byteNodeGhosts + numSubstates * sizeBorder + (i * sizeBorder)), host_CA->pQb_array[i] + ((i + 1) * fullSize - sizeBorder), sizeof(CALbyte) * sizeBorder);
            }
        }
    }
};
//END MULTINODE--------

#endif /*CALCLMULTINODE2D_H_*/
