#ifndef CALCLMULTINODE_H_
#define CALCLMULTINODE_H_

#include <mpi.h>
#include <stdio.h>
#include <vector>
#include<string>
#include<iostream>
#include <utility>
#include <OpenCAL-CL/calclMultiDevice2D.h>
#include <sys/time.h>
extern "C"{
#include <OpenCAL/cal2DIO.h>
#include <OpenCAL/cal2DBuffer.h>
#include <OpenCAL/cal2DBufferIO.h>
}
typedef void (* CALCallbackFuncMNInit2D)(struct CALCLMultiDevice* ca2D, const Node& mynode);
typedef void (* CALCallbackFuncMNFinalize2D)(struct CALCLMultiDevice* ca2D);

CALbyte calNodeLoadMatrix2Dr(CALreal* M, const int rows, const int columns, const char* path,  int read_offset = 0, const int write_offset = 0)
{
    FILE *f = NULL;
    f = fopen(path, "r");

    if ( !f )
        return CAL_FALSE;

    //skip #offset rows
    const int _s = 20*2*columns; //assuming 20char per number + spaces
    char tmp[_s];
    while(read_offset--)
        fgets(tmp,sizeof(char)*_s,f);

    calfLoadMatrix2Dr(M+write_offset, rows, columns, f);

    fclose(f);

    return CAL_TRUE;
}

CALbyte calNodeLoadMatrix2Di(CALint* M, const int rows, const int columns, const char* path,  int read_offset = 0, const int write_offset = 0)
{
    FILE *f = NULL;
    f = fopen(path, "r");

    if ( !f )
        return CAL_FALSE;

    //skip #offset rows
    const int _s = 20*2*columns; //assuming 20char per number + spaces
    char tmp[_s];
    while(read_offset--)
        fgets(tmp,sizeof(char)*_s,f);

    calfLoadMatrix2Di(M+write_offset, rows, columns, f);

    fclose(f);

    return CAL_TRUE;
}


CALbyte calNodeLoadMatrix2Db(CALbyte* M, const int rows, const int columns, const char* path,  int read_offset = 0, const int write_offset = 0)
{
    FILE *f = NULL;
    f = fopen(path, "r");

    if ( !f )
        return CAL_FALSE;

    //skip #offset rows
    const int _s = 20*2*columns; //assuming 20char per number + spaces
    char tmp[_s];
    while(read_offset--)
        fgets(tmp,sizeof(char)*_s,f);

    calfLoadMatrix2Db(M+write_offset, rows, columns, f);

    fclose(f);

    return CAL_TRUE;
}

CALbyte calNodeLoadSubstate2Dr(CALModel2D* ca2D, struct CALSubstate2Dr* Q, char* path,const Node& mynode) {
    int write_offset =0;
    CALbyte return_state = calNodeLoadMatrix2Dr(Q->current, ca2D->rows, ca2D->columns, path,mynode.offset,write_offset);
    if (Q->next)
        calCopyBuffer2Dr(Q->current, Q->next, ca2D->rows, ca2D->columns);
    return return_state;
}

CALbyte calNodeLoadSubstate2Di(CALModel2D* ca2D, struct CALSubstate2Di* Q, char* path,const Node& mynode) {
    int write_offset =0;
    CALbyte return_state = calNodeLoadMatrix2Di(Q->current, ca2D->rows, ca2D->columns, path,mynode.offset,write_offset);
    if (Q->next)
        calCopyBuffer2Di(Q->current, Q->next, ca2D->rows, ca2D->columns);
    return return_state;
}

CALbyte calNodeLoadSubstate2Db(CALModel2D* ca2D, struct CALSubstate2Db* Q, char* path,const Node& mynode) {
    int write_offset =0;
    CALbyte return_state = calNodeLoadMatrix2Db(Q->current, ca2D->rows, ca2D->columns, path,mynode.offset,write_offset);
    if (Q->next)
        calCopyBuffer2Db(Q->current, Q->next, ca2D->rows, ca2D->columns);
    return return_state;
}





// CALbyte calNodeSaveMatrix2Dr(CALreal* M, int rows, int columns, char* path, int readoffset, int writeoffset)
// {

// 	FILE *f;
// 	f = fopen(path, "w");
//     fclose(f);
//     int nbytesofarow = 4962;
//     MPI_Barrier(MPI_COMM_WORLD);
//     fclose(f);
//     f = fopen(path, "a");

// 	if ( !f )
// 		return CAL_FALSE;

//     printf("readoffset*nbytesofarow = %d\n", readoffset*nbytesofarow);
//     fseek(f,readoffset*nbytesofarow, SEEK_SET);

// 	calfSaveMatrix2Dr(M, rows, columns, f);

// 	fclose(f);

// 	return CAL_TRUE;
// }

// CALbyte calNodeSaveSubstate2Dr(CALModel2D* ca2D, struct CALSubstate2Dr* Q, char* path,const Node& mynode) {
//     int write_offset =0;
//     CALbyte return_state = calNodeSaveMatrix2Dr(Q->current, ca2D->rows, ca2D->columns, path,mynode.offset,write_offset);
//     return return_state;
// }




//MULTINODE---------------
//template <class F_INIT,class F_FINALIZE>
class MultiNode{
public:
//only used to kep track of the communication overhead
    double comm_total=0; 
    double comp_total=0;
    double kernel_total=0;
    double kernel_communication=0;
    timeval start_comm, end_comm;
    timeval start_comp, end_comp;
    timeval start_kernel_communication, end_kernel_communication;
    timeval start_kernel, end_kernel;

    CALDistributedDomain2D c;
    CALCallbackFuncMNInit2D init;
    CALCallbackFuncMNFinalize2D finalize;
    int rank;
    CALCLMultiDevice* multidevice;

    CALreal* realNodeGhosts=0;
    CALint * intNodeGhosts=0;
    CALbyte* byteNodeGhosts=0;
    CALbyte* flagsNodeGhosts=0;


    MultiNode(CALDistributedDomain2D _c,CALCallbackFuncMNInit2D i, CALCallbackFuncMNFinalize2D f):c(_c),
        init(i), finalize(f) {
        MPI_Init(NULL, NULL);
        int world_rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

        rank = world_rank;

        multidevice=nullptr;
        multidevice = (CALCLMultiDevice*)malloc(sizeof(CALCLMultiDevice));
                
        Node mynode = c.nodes[rank];
        auto devices = mynode.devices;

        // calclPrintPlatformsAndDevices(calcl_device_manager);
        struct CALCLDeviceManager * calcl_device_manager = calclCreateManager();

        calclSetNumDevice(multidevice,devices.size());
        for(auto& d : devices){
            calclAddDevice(multidevice,calclGetDevice(calcl_device_manager, d.num_platform , d.num_device) ,  d.workload);
        }
        printf("Costruttore  rank = %d \n", rank);
        MPI_Barrier(MPI_COMM_WORLD);
    }

    int GetNodeRank() {
        return rank;
    }

    bool checkWorkloads(){return true;};

    void allocateAndInit(){
        Node mynode = c.nodes[rank];
        // auto devices = mynode.devices;

        // // calclPrintPlatformsAndDevices(calcl_device_manager);
        // struct CALCLDeviceManager * calcl_device_manager = calclCreateManager();

        // calclSetNumDevice(multidevice,devices.size());
        // for(auto& d : devices){
        //     calclAddDevice(multidevice,calclGetDevice(calcl_device_manager, d.num_platform , d.num_device) ,  d.workload);
        // }
        
        init(multidevice,mynode);

        multidevice->exchange_full_border = c.is_full_exchange();

        CALCLModel2D* last_gpu = multidevice->device_models[multidevice->num_devices-1];
        const CALint sizeBorder = last_gpu->borderSize*last_gpu->columns;
        const int rnumSubstate = last_gpu->host_CA->sizeof_pQr_array;
        const int inumSubstate = last_gpu->host_CA->sizeof_pQi_array;
        const int bnumSubstate = last_gpu->host_CA->sizeof_pQb_array;
        realNodeGhosts=0;
        intNodeGhosts=0;
        byteNodeGhosts=0;
        realNodeGhosts = (CALreal*)calloc(rnumSubstate*sizeBorder*2,sizeof(CALreal));
        intNodeGhosts  = (CALint*)calloc(inumSubstate*sizeBorder*2,sizeof(CALint));
        byteNodeGhosts = (CALbyte*)calloc(bnumSubstate*sizeBorder*2,sizeof(CALbyte));
        flagsNodeGhosts = (CALbyte*)calloc(sizeBorder*2,sizeof(CALbyte));
        MPI_Barrier(MPI_COMM_WORLD);
    }

    void _finalize(){
        free(realNodeGhosts);
        free(byteNodeGhosts);
        free(intNodeGhosts);
        free(flagsNodeGhosts);
        finalize(multidevice);
    }


    void run(int STEPS){

    int rank;
    timeval start, end;


    double T1, T2,              /* start/end times per rep */
    sumT=0,                   /* sum of all reps times */
    deltaT;                 /* time for one rep */
    double avgT;
    double TotalTime=0;
    int ntComuunication=0;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    //printf("rank %d\n", rank);


 //MPI_Barrier(MPI_COMM_WORLD);
   // if(rank == 0)
        gettimeofday(&start, NULL);

       

        
        int dimNum;
        for (int gpu = 0; gpu < multidevice->num_devices; ++gpu) {
          multidevice->singleStepThreadNums[gpu] =
              computekernelLaunchParams(multidevice, gpu, &dimNum);
            printf("%d  %d\n",gpu,multidevice->singleStepThreadNums[gpu][0]);
            printf("%d  %d\n",gpu,multidevice->singleStepThreadNums[gpu][1]);
        }


//----------MEASURE TIME---------
        gettimeofday(&start_comm, NULL);
        handleBorderNodes(TotalTime,ntComuunication);
        gettimeofday(&end_comm, NULL);

         comm_total+= 1000.0 * (end_comm.tv_sec - start_comm.tv_sec) +
                                 (end_comm.tv_usec - start_comm.tv_usec) / 1000.0;
        //handleFlagsMultiNode();
         //----------------------------------
         for (int gpu = 0; gpu < multidevice->num_devices; ++gpu) {
           for (int el_proc = 0;
                el_proc < multidevice->device_models[0]->elementaryProcessesNum;
                el_proc++) {
             struct CALCLModel2D* calclmodel2D = multidevice->device_models[gpu];
             size_t* singleStepThreadNum = multidevice->singleStepThreadNums[gpu];

             cl_int err;

             if (calclmodel2D->kernelInitSubstates != NULL)
               calclSetReductionParameters2D(calclmodel2D,
                                             calclmodel2D->kernelInitSubstates);
             if (calclmodel2D->kernelStopCondition != NULL)
               calclSetReductionParameters2D(calclmodel2D,
                                             calclmodel2D->kernelStopCondition);
             if (calclmodel2D->kernelSteering != NULL)
               calclSetReductionParameters2D(calclmodel2D,
                                             calclmodel2D->kernelSteering);

             int i = 0;

             calclSetReductionParameters2D(
                 calclmodel2D, calclmodel2D->elementaryProcesses[el_proc]);
           }
         }

        int totalSteps= STEPS;   
        while(STEPS--){
            //debug("step \n");
gettimeofday(&start_comp, NULL);

        for (int j = 0; j < multidevice->device_models[0]->elementaryProcessesNum; j++) {
                
                //MPI_Barrier(MPI_COMM_WORLD);
                //printf("rank =  %d\n",rank );
                gettimeofday(&start_kernel, NULL);
               
                
                  calcl_executeElementaryProcess(multidevice, j,
                                               dimNum /*elementary process*/,this);
                
                MPI_Barrier(MPI_COMM_WORLD);

                // if (multidevice->num_devices != 1 || c.nodes.size() != 1) {

                // barrier tutte hanno finito
                for (int gpu = 0; gpu < multidevice->num_devices; ++gpu) {
                    clFinish(multidevice->device_models[gpu]->queue);
                  }

                gettimeofday(&end_kernel, NULL);

                kernel_total+= 1000.0 * (end_kernel.tv_sec - start_kernel.tv_sec) +
                                 (end_kernel.tv_usec - start_kernel.tv_usec) / 1000.0;


                  //----------MEASURE TIME---------
                  gettimeofday(&start_comm, NULL);
                  
                  // Read from the substates and set ghost borders
                  gettimeofday(&start_kernel_communication, NULL);
                 for (int gpu = 0; gpu < multidevice->num_devices; ++gpu) {
                    calclGetBorderFromDeviceToHost2D(
                        multidevice->device_models[gpu]);
                  }
                  

                  // scambia bordi
                  // Write from the ghost borders to the substates
                  calclMultiDeviceUpdateHalos2D(
                      multidevice, multidevice->exchange_full_border);
                  gettimeofday(&end_kernel_communication, NULL);
                   handleBorderNodes(TotalTime, ntComuunication);


                  gettimeofday(&end_comm, NULL);
                  comm_total += 1000.0 * (end_comm.tv_sec - start_comm.tv_sec) +
                                (end_comm.tv_usec - start_comm.tv_usec) / 1000.0;
                  kernel_communication += 1000.0 * (end_kernel_communication.tv_sec - start_kernel_communication.tv_sec) +
                                (end_kernel_communication.tv_usec - start_kernel_communication.tv_usec) / 1000.0;
                //----------------------------------     
             //   }
        }
       


/*
//vanno scambiati i ghost. dentro l'arraydelle celle attive (non real)
// le prime e ultime righe vanno mandate alle Device vicine (esattamente nello stesso modo
//con cui vengono scambiati i sottostati) 
// i dati ricevuto dalle Device vicine non possono essere direttamente mpacchiati 
//dentro la matrice delle celle attive ma vanno "mergiati" perchè le celle possono essere
//attivate da una Device vicina ma anche da me stesso 
___________
|HALO00    |   bordo01 e la prima riga di Device1 vanno mergiate prima dentro la prima riga di GPu1
|__________|   bordo10 e la ultima riga di Device0 vanno mergiate prima dentro l'ultima riga di Device0
|          |
|  Device0 |   bordo00 e l'ultima riga di Device1 vanno mergiate dentro ultima riga di gpu1 e cosi via
|          |
|          |  
|----------|
|HALO01    |
|__________|
___________
|HALO10    |
|__________|
|          |
|  Device1 |   
|          |
|          |  
|----------|
|HALO10    |
|__________|
*/


// #if 0
            // STEERING------------------------------
            struct CALCLModel2D* calclmodel2DFirst = multidevice->device_models[0];

            if (calclmodel2DFirst->kernelSteering != NULL) {
gettimeofday(&start_kernel, NULL);
              for (int gpu = 0; gpu < multidevice->num_devices; ++gpu) {
                

                struct CALCLModel2D* calclmodel2D =
                    multidevice->device_models[gpu];
                size_t* singleStepThreadNum = multidevice->singleStepThreadNums[gpu];

                if (calclmodel2D->kernelSteering != NULL) {
                    if (singleStepThreadNum[0] > 0) 
                  calclKernelCall2D(calclmodel2D, calclmodel2D->kernelSteering,
                                    dimNum, singleStepThreadNum, NULL, NULL);
                  CALbyte activeCells =
                      calclmodel2D->opt == CAL_OPT_ACTIVE_CELLS_NAIVE;
                  if (activeCells == CAL_TRUE)
                    if (singleStepThreadNum[0] > 0) 
                    calclKernelCall2D(calclmodel2D,
                                      calclmodel2D->kernelUpdateSubstate,
                                      dimNum, singleStepThreadNum, NULL, NULL);
                  else
                    copySubstatesBuffers2D(calclmodel2D);

                }

                
              }
    
              //----------------------------------------

             // if(multidevice->num_devices != 1 || c.nodes.size() != 1){
                for (int gpu = 0; gpu < multidevice->num_devices; ++gpu) {
                  clFinish(multidevice->device_models[gpu]->queue);
                }

                gettimeofday(&end_kernel, NULL);

                kernel_total+= 1000.0 * (end_kernel.tv_sec - start_kernel.tv_sec) +
                                 (end_kernel.tv_usec - start_kernel.tv_usec) / 1000.0;
 //----------MEASURE TIME---------
                  gettimeofday(&start_comm, NULL);
                  gettimeofday(&start_kernel_communication, NULL);
                for (int gpu = 0; gpu < multidevice->num_devices; ++gpu) {
                  calclGetBorderFromDeviceToHost2D(
                      multidevice->device_models[gpu]);
                }

                // scambia bordi
                calclMultiDeviceUpdateHalos2D(
                    multidevice, multidevice->exchange_full_border);
                gettimeofday(&end_kernel_communication, NULL);
                  kernel_communication += 1000.0 * (end_kernel_communication.tv_sec - start_kernel_communication.tv_sec) +
                                (end_kernel_communication.tv_usec - start_kernel_communication.tv_usec) / 1000.0;
 
                 gettimeofday(&end_comm, NULL);
                  comm_total += 1000.0 * (end_comm.tv_sec - start_comm.tv_sec) +
                                (end_comm.tv_usec - start_comm.tv_usec) / 1000.0;
                //----------------------------------     
              //}
            }  // Steering
// #endif
           // if (multidevice->num_devices != 1 || c.nodes.size() != 1) {
              //----------MEASURE TIME---------
              gettimeofday(&start_comm, NULL);

               T1 = MPI_Wtime();
               handleBorderNodes(TotalTime, ntComuunication);
               T2 = MPI_Wtime();
               deltaT = T2 - T1;
               //printf("%4d  %8.8f  %8.8f  %2.8f\n", STEPS, T1, T2, deltaT);

               sumT += deltaT;
              gettimeofday(&end_comm, NULL);
              comm_total += 1000.0 * (end_comm.tv_sec - start_comm.tv_sec) +
                            (end_comm.tv_usec - start_comm.tv_usec) / 1000.0;
              //----------------------------------
          //  }

 gettimeofday(&end_comp, NULL);
                  comp_total += 1000.0 * (end_comp.tv_sec - start_comp.tv_sec) +
                                (end_comp.tv_usec - start_comp.tv_usec) / 1000.0;


        }//STEPS
    

        //handleBorderNodes();
        calclMultiDeviceToNode(multidevice);
 

        //if (rank == 0) {
          gettimeofday(&end, NULL);
          unsigned long long t = 1000 * (end.tv_sec - start.tv_sec) +
                                 (end.tv_usec - start.tv_usec) / 1000;

          double TotalCommTime = TotalTime + (kernel_communication/1000);              

          // end = time(NULL);
          printf("Rank=%d, Elapsed time: %llu ms\n", rank,t);
          printf("Rank=%d, Elapsed Communication time: %f s\n", rank,TotalCommTime);
          printf("Rank=%d, Elapsed MPI communication : %f s\n" ,rank, TotalTime);
          printf("Rank=%d, Elapsed Host Device PCI-E communication: %f ms\n", rank,kernel_communication);
          printf("Rank=%d, Elapsed time kernels: %f ms\n", rank,kernel_total);
          // three elementary procecces
          avgT = (TotalTime)/(ntComuunication);
          printf("Rank=%d, Avg Latency Time = %f s\n",rank, avgT);
          printf("Rank=%d, number of time send : %d \n" ,rank, ntComuunication);
         

      //  }
        MPI_Barrier(MPI_COMM_WORLD);
        _finalize();
        MPI_Finalize();
    }//run function

    void handleBorderNodes(double & T, int & ntComuunication){
        if(multidevice->device_models[0]->borderSize<=0)
            return;
        handleBorderNodesR(T,ntComuunication);
        handleBorderNodesI(T,ntComuunication);
        handleBorderNodesB(T,ntComuunication);
    }


    void handleBorderNodesR(double& T, int &ntComuunication){
        const MPI_Datatype DATATYPE = MPI_DOUBLE;
        if(!c.is_full_exchange()){

            CALint prev,next;
            CALCLModel2D* gpu_to_use = multidevice->device_models[0];
            CALreal* send_offset;
            CALreal* recv_offset = realNodeGhosts;

            const CALint sizeBorder = gpu_to_use->borderSize*gpu_to_use->columns;
            const int numSubstates = gpu_to_use->host_CA->sizeof_pQr_array;
            const CALint count = (numSubstates*sizeBorder);

            if(numSubstates <= 0)
                return;
            double T1, T2, deltaT;

            for(int i=0;i<2;i++){

                next=((rank+1)+c.nodes.size())%c.nodes.size();
                prev=((rank-1)+c.nodes.size())%c.nodes.size();
                if(i==1)
                    std::swap(next,prev);


                //this should be multidevice->num_devices-1 and 0 during the two iterations
                CALint indexgpu = (i+multidevice->num_devices-1)%multidevice->num_devices;
                gpu_to_use = multidevice->device_models[indexgpu];


                send_offset = gpu_to_use->borderMapper.realBorder_OUT;
                send_offset+=(i==0 ? 1 : 0)*count;

                recv_offset = realNodeGhosts;
                recv_offset+= (i==0 ? 0: 1)*count;

                if(rank % 2 == 0){
                    //MPI send
                    ntComuunication++;
                    T1 = MPI_Wtime();

                    // printf("I'm %d:  sedning to %d \n" ,  rank , next);
                    //cerca convenzione per i nomi dei tags
                    MPI_Send(send_offset,count, DATATYPE , next , i ,MPI_COMM_WORLD);

                    // printf("I'm %d:  receiving from  %d \n" ,  rank , prev);
                    MPI_Recv(recv_offset , count , DATATYPE, prev, i, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
                    T2 = MPI_Wtime(); 
                    deltaT = T2 - T1;
                    T += deltaT;
                    //send to rank+1
                    //receive rank-1
                }else{

                    // printf("I'm %d:  receiving from  %d \n" ,  rank , prev);
                    MPI_Recv(recv_offset , count , DATATYPE, prev, i ,MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    // printf("I'm %d:  sedning to %d \n" ,  rank , next);
                    MPI_Send(send_offset , count, DATATYPE , next , i ,MPI_COMM_WORLD);

                    //receice from rank-1;
                    //send to rank+1
                }

                //MPI_Barrier(MPI_COMM_WORLD);


            }




            //memory has been exchanged between mpi processes. Now it'0s time to foward that memory to
            //the right Devices
            cl_int err;
            for(int i =0; i < numSubstates; i++){
                CALCLModel2D* m = multidevice->device_models[0];
                //upper ghost
                err = clEnqueueWriteBuffer(m->queue,
                                           m->bufferCurrentRealSubstate,
                                           CL_TRUE,
                                           (i*m->fullSize)*sizeof(CALreal),
                                           sizeof(CALreal)*sizeBorder,
                                           realNodeGhosts +(i*sizeBorder),
                                           0,
                                           NULL,
                                           NULL);
                calclHandleError(err);
                m = multidevice->device_models[multidevice->num_devices-1];
                //lower ghost
                err = clEnqueueWriteBuffer(m->queue,
                                           m->bufferCurrentRealSubstate,
                                           CL_TRUE,
                                           ((i+1)*m->fullSize-sizeBorder)*sizeof(CALreal),
                                           sizeof(CALreal)*sizeBorder,
                                           realNodeGhosts + numSubstates*sizeBorder + (i*sizeBorder),
                                           0,
                                           NULL,
                                           NULL);
                calclHandleError(err);

            }


        }
    }





    void handleBorderNodesI(double& T, int &ntComuunication){
        const MPI_Datatype DATATYPE = MPI_INT;
        if(!c.is_full_exchange()){

            CALint prev,next;
            CALCLModel2D* gpu_to_use = multidevice->device_models[0];
            CALint* send_offset;
            CALint* recv_offset = intNodeGhosts;

            const CALint sizeBorder = gpu_to_use->borderSize*gpu_to_use->columns;
            const int numSubstates = gpu_to_use->host_CA->sizeof_pQi_array;
            const CALint count = (numSubstates*sizeBorder);

            if(numSubstates <= 0)
                return;
            double T1, T2, deltaT;

            for(int i=0;i<2;i++){

                next=((rank+1)+c.nodes.size())%c.nodes.size();
                prev=((rank-1)+c.nodes.size())%c.nodes.size();
                if(i==1)
                    std::swap(next,prev);


                //this should be multidevice->num_devices-1 and 0 during the two iterations
                CALint indexgpu = (i+multidevice->num_devices-1)%multidevice->num_devices;
                gpu_to_use = multidevice->device_models[indexgpu];


                send_offset = gpu_to_use->borderMapper.intBorder_OUT;
                send_offset+=(i==0 ? 1 : 0)*count;

                recv_offset = intNodeGhosts;
                recv_offset+= (i==0 ? 0: 1)*count;

                if(rank % 2 == 0){
                    //MPI send

                    ntComuunication++;
                    T1 = MPI_Wtime();

                    // printf("I'm %d:  sedning to %d \n" ,  rank , next);
                    //cerca convenzione per i nomi dei tags
                    MPI_Send(send_offset,count, DATATYPE , next , i ,MPI_COMM_WORLD);

                    // printf("I'm %d:  receiving from  %d \n" ,  rank , prev);
                    MPI_Recv(recv_offset , count , DATATYPE, prev, i, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
                    T2 = MPI_Wtime(); 
                    deltaT = T2 - T1;
                    T += deltaT;


                    //send to rank+1
                    //receive rank-1
                }else{

                    // printf("I'm %d:  receiving from  %d \n" ,  rank , prev);
                    MPI_Recv(recv_offset , count , DATATYPE, prev, i ,MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    // printf("I'm %d:  sedning to %d \n" ,  rank , next);
                    MPI_Send(send_offset , count, DATATYPE , next , i ,MPI_COMM_WORLD);

                    /*  if(rank==1){
                    std::cout<<std::endl;
                    for (int i = 0; i < 2*count; ++i) {
                        if(i%gpu_to_use->columns==0)
                            std::cout<<std::endl;
                        std::cout << intNodeGhosts[i] << " ";
                    }
                    std::cout<<std::endl;
                }*/
                    //receice from rank-1;
                    //send to rank+1
                }

                //MPI_Barrier(MPI_COMM_WORLD);


            }
            //memory has been exchanged between mpi processes. Now it'0s time to foward that memory to
            //the right Devices
            cl_int err;
            for(int i =0; i < numSubstates; i++){
                CALCLModel2D* m = multidevice->device_models[0];
                //upper ghost
                err = clEnqueueWriteBuffer(m->queue,
                                           m->bufferCurrentIntSubstate,
                                           CL_TRUE,
                                           (i*m->fullSize)*sizeof(CALint),
                                           sizeof(CALint)*sizeBorder,
                                           intNodeGhosts +(i*sizeBorder),
                                           0,
                                           NULL,
                                           NULL);
                calclHandleError(err);
                m = multidevice->device_models[multidevice->num_devices-1];
                //lower ghost
                err = clEnqueueWriteBuffer(m->queue,
                                           m->bufferCurrentIntSubstate,
                                           CL_TRUE,
                                           ((i+1)*m->fullSize-sizeBorder)*sizeof(CALint),
                                           sizeof(CALint)*sizeBorder,
                                           intNodeGhosts + numSubstates*sizeBorder + (i*sizeBorder),
                                           0,
                                           NULL,
                                           NULL);
                calclHandleError(err);

            }


        }
    }



    void handleBorderNodesB(double& T, int &ntComuunication){
        const MPI_Datatype DATATYPE = MPI_CHAR;
        if(!c.is_full_exchange()){

            CALint prev,next;
            CALCLModel2D* gpu_to_use = multidevice->device_models[0];
            CALbyte* send_offset;
            CALbyte* recv_offset = byteNodeGhosts;

            const CALint sizeBorder = gpu_to_use->borderSize*gpu_to_use->columns;
            const int numSubstates = gpu_to_use->host_CA->sizeof_pQb_array;
            const CALint count = (numSubstates*sizeBorder);

            if(numSubstates <= 0)
                return;

            double T1, T2, deltaT;
            for(int i=0;i<2;i++){

                next=((rank+1)+c.nodes.size())%c.nodes.size();
                prev=((rank-1)+c.nodes.size())%c.nodes.size();
                if(i==1)
                    std::swap(next,prev);


                //this should be multidevice->num_devices-1 and 0 during the two iterations
                CALint indexgpu = (i+multidevice->num_devices-1)%multidevice->num_devices;
                gpu_to_use = multidevice->device_models[indexgpu];


                send_offset = gpu_to_use->borderMapper.byteBorder_OUT;
                send_offset+=(i==0 ? 1 : 0)*count;

                recv_offset = byteNodeGhosts;
                recv_offset+= (i==0 ? 0: 1)*count;

                if(rank % 2 == 0){
                    //MPI send

                    ntComuunication++;
                    T1 = MPI_Wtime();

                    // printf("I'm %d:  sedning to %d \n" ,  rank , next);
                    //cerca convenzione per i nomi dei tags
                    MPI_Send(send_offset,count, DATATYPE , next , i ,MPI_COMM_WORLD);

                    // printf("I'm %d:  receiving from  %d \n" ,  rank , prev);
                    MPI_Recv(recv_offset , count , DATATYPE, prev, i, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
                    T2 = MPI_Wtime(); 
                    deltaT = T2 - T1;
                    T += deltaT;


                    //send to rank+1
                    //receive rank-1
                }else{

                    // printf("I'm %d:  receiving from  %d \n" ,  rank , prev);
                    MPI_Recv(recv_offset , count , DATATYPE, prev, i ,MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    // printf("I'm %d:  sedning to %d \n" ,  rank , next);
                    MPI_Send(send_offset , count, DATATYPE , next , i ,MPI_COMM_WORLD);

                    //receice from rank-1;
                    //send to rank+1
                }

                //MPI_Barrier(MPI_COMM_WORLD);


            }




            //memory has been exchanged between mpi processes. Now it'0s time to foward that memory to
            //the right Devices
            cl_int err;
            for(int i =0; i < numSubstates; i++){
                CALCLModel2D* m = multidevice->device_models[0];
                //upper ghost
                err = clEnqueueWriteBuffer(m->queue,
                                           m->bufferCurrentByteSubstate,
                                           CL_TRUE,
                                           (i*m->fullSize)*sizeof(CALbyte),
                                           sizeof(CALbyte)*sizeBorder,
                                           byteNodeGhosts +(i*sizeBorder),
                                           0,
                                           NULL,
                                           NULL);
                calclHandleError(err);
                m = multidevice->device_models[multidevice->num_devices-1];
                //lower ghost
                err = clEnqueueWriteBuffer(m->queue,
                                           m->bufferCurrentByteSubstate,
                                           CL_TRUE,
                                           ((i+1)*m->fullSize-sizeBorder)*sizeof(CALbyte),
                                           sizeof(CALbyte)*sizeBorder,
                                           byteNodeGhosts + numSubstates*sizeBorder + (i*sizeBorder),
                                           0,
                                           NULL,
                                           NULL);
                calclHandleError(err);

            }


        }
    }

    void handleFlagsMultiNode() {
      const MPI_Datatype DATATYPE = MPI_CHAR;      
      if (!c.is_full_exchange()) {

        CALint prev, next;
        CALCLModel2D* gpu_to_use = multidevice->device_models[0];
        CALbyte* send_offset;
        CALbyte* recv_offset = flagsNodeGhosts;
        const CALint sizeBorder = gpu_to_use->borderSize * gpu_to_use->columns;
        const CALint count = (sizeBorder);
       // printf("rank %d --> multidevice->singleStepThreadNums[prev] %d",rank, multidevice->singleStepThreadNums[prev]);
       // printf("rank %d --> multidevice->singleStepThreadNums[next] %d",rank, multidevice->singleStepThreadNums[next]);

        for (int i = 0; i < 2; i++) {

          next = ((rank + 1) + c.nodes.size()) % c.nodes.size();
          prev = ((rank - 1) + c.nodes.size()) % c.nodes.size();

          if (i == 1) std::swap(next, prev);

          // this should be multidevice->num_devices-1 and 0 during the two
          // iterations
          CALint indexgpu =
              (i + multidevice->num_devices - 1) % multidevice->num_devices;
          // printf("indexgpu %d \n", indexgpu);
          gpu_to_use = multidevice->device_models[indexgpu];

          send_offset = gpu_to_use->borderMapper.flagsBorder_OUT;
          send_offset += (i == 0 ? 1 : 0) * count;

          recv_offset = flagsNodeGhosts;
          recv_offset += (i == 0 ? 0 : 1) * count;

          if (rank % 2 == 0) {
            // MPI send
            // printf("I'm %d:  sedning to %d \n", rank, next);
            // cerca convenzione per i nomi dei tags
            MPI_Send(send_offset, count, DATATYPE, next, i, MPI_COMM_WORLD);

             //printf("I'm %d:  receiving from  %d \n" ,  rank , prev);
            MPI_Recv(recv_offset, count, DATATYPE, prev, i, MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);

            // send to rank+1
            // receive rank-1
          } else {
            //printf("I'm %d:  receiving to %d \n", rank, prev);
            MPI_Recv(recv_offset, count, DATATYPE, prev, i, MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);

           //  printf("I'm %d:  sedning to %d \n" ,  rank , next);
            MPI_Send(send_offset, count, DATATYPE, next, i, MPI_COMM_WORLD);
            // receice from rank-1;
            // send to rank+1
          }

          MPI_Barrier(MPI_COMM_WORLD);
          // printf("barrier\n");
            

        }  // for

// calcolo le due gpu che devono ricevere i flagsNodeGhosts
        cl_int err;

        CALCLModel2D* m = multidevice->device_models[0];
        // upper ghost
        err = clEnqueueWriteBuffer(
            m->queue,
            m->borderMapper.mergeflagsBorder, 
            CL_TRUE,
            0, 
            sizeof(CALbyte) * sizeBorder,
            flagsNodeGhosts, 
            0, 
            NULL, 
            NULL);
        calclHandleError(err);
        m = multidevice->device_models[multidevice->num_devices - 1];
        // lower ghost
        err = clEnqueueWriteBuffer(
            m->queue,
            m->borderMapper.mergeflagsBorder,
            CL_TRUE,
            sizeBorder * sizeof(CALbyte),
            sizeof(CALbyte) * sizeBorder,
            flagsNodeGhosts + sizeBorder,
            0,
            NULL,
            NULL);
        calclHandleError(err);

      }// if full excahnge
    }//function


void calclStreamCompactionMulti(struct CALCLMultiDevice* multidevice, MultiNode * mn){

//----------MEASURE TIME---------
gettimeofday(&(mn->start_comm), NULL);


  cl_int err;
  // Read from substates and set flags borders
  //printf(" before read bufferActiveCellsFlags \n");

  for (int gpu = 0; gpu < multidevice->num_devices; ++gpu) {
    struct CALCLModel2D* calclmodel2D = multidevice->device_models[gpu];
    CALCLqueue queue = calclmodel2D->queue;

    cl_int err;
    int dim = calclmodel2D->fullSize;

    int sizeBorder = calclmodel2D->borderSize * calclmodel2D->columns;
    //printf(" read first border OK bufferActiveCellsFlags gpu = %d  %d\n", gpu, sizeof(CALbyte) * sizeBorder );
    err = clEnqueueReadBuffer(queue, calclmodel2D->bufferActiveCellsFlags,
                              CL_TRUE, 0, sizeof(CALbyte) * sizeBorder,
                              calclmodel2D->borderMapper.flagsBorder_OUT, 0,
                              NULL, NULL);
    calclHandleError(err);
    //printf(" read first border OK bufferActiveCellsFlags gpu = %d\n", gpu);

    err = clEnqueueReadBuffer(
        queue, calclmodel2D->bufferActiveCellsFlags, CL_TRUE,
        ((dim - sizeBorder)) * sizeof(CALbyte), sizeof(CALbyte) * sizeBorder,
        calclmodel2D->borderMapper.flagsBorder_OUT + sizeBorder, 0, NULL, NULL);
    calclHandleError(err);
    // printf(" read last border OK bufferActiveCellsFlags gpu = %d \n",gpu);
  }

 mn->handleFlagsMultiNode();

  //printf(" after read bufferActiveCellsFlags \n"); 
  //printf(" before write bufferActiveCellsFlags \n"); 
  for (int gpu = 0; gpu < multidevice->num_devices; ++gpu) {

    struct CALCLModel2D* calclmodel2D = multidevice->device_models[gpu];
    struct CALCLModel2D* calclmodel2DPrev = NULL;
    const int gpuP =
        ((gpu - 1) + multidevice->num_devices) % multidevice->num_devices;
    const int gpuN =
        ((gpu + 1) + multidevice->num_devices) % multidevice->num_devices;

    if (calclmodel2D->host_CA->T == CAL_SPACE_TOROIDAL || ((gpu - 1) >= 0)) {

      calclmodel2DPrev = multidevice->device_models[gpuP];
    }

    struct CALCLModel2D* calclmodel2DNext = NULL;
    if (calclmodel2D->host_CA->T == CAL_SPACE_TOROIDAL ||
        ((gpu + 1) < multidevice->num_devices)) {
      calclmodel2DNext = multidevice->device_models[gpuN];
    }

    int dim = calclmodel2D->fullSize;

    const int sizeBorder = calclmodel2D->borderSize * calclmodel2D->columns;

    const CALbyte activeCells = calclmodel2D->opt == CAL_OPT_ACTIVE_CELLS_NAIVE;
    if (activeCells == CAL_TRUE) {
      // copy border flags from DevicePrev and Device next to a mergeflagsBorder
      if (calclmodel2DPrev != NULL &&
          (multidevice->exchange_full_border || gpu != 0)) {
        err = clEnqueueWriteBuffer(
            calclmodel2D->queue, calclmodel2D->borderMapper.mergeflagsBorder,
            CL_TRUE, 0, sizeof(CALbyte) * sizeBorder,
            calclmodel2DPrev->borderMapper.flagsBorder_OUT + sizeBorder, 0,
            NULL, NULL);

        calclHandleError(err);
      }
      if (calclmodel2DNext != NULL && (multidevice->exchange_full_border ||
                                       gpu != multidevice->num_devices - 1)) {
        err = clEnqueueWriteBuffer(
            calclmodel2D->queue, calclmodel2D->borderMapper.mergeflagsBorder,
            CL_TRUE, ((sizeBorder)) * sizeof(CALbyte),
            sizeof(CALbyte) * sizeBorder,
            calclmodel2DNext->borderMapper.flagsBorder_OUT, 0, NULL, NULL);
        calclHandleError(err);
      }
    }
  }
  gettimeofday(&(mn->end_comm), NULL);

  mn->comm_total+= 1000 * (mn->end_comm.tv_sec - mn->start_comm.tv_sec) +
                      (mn->end_comm.tv_usec - mn->start_comm.tv_usec) / 1000;

//----------------------------------   

//questo pezzo di codice va spostato nella parte multinodo magari in una funzione a se stante
  // bisogna mettere degli if di guard di modo che tutte ste cose (che hanno un overhead) siano eseguite SOLO
  // se l'ottimizza<ione delle celle attive è enabled

//se mergebuffer è completo dei bordiche vengono dagli altri nodi quello che ci st aqui sotto
//funziona

//facciamo finta che flagsNodeGhosts siano pieni dei dati ricevuti dai vicini
// bisogna fare due write:
// 1) la prima metà di flagsNodeGhosts sulla prima meta di mergeflagsBorder della prima Device del nodo
// 2) la seconda meta di flagsNodeGhosts sulla seconda metà di mergeflagsBorder dell'ultima Device del nodo


// a questo punto mergeBufer contiene i bordi delle Device adiacenti (anche se esse stanno su altri nodi)
// quindi il merging può essere fatto. Qui sotto funziona.
  for (int gpu = 0; gpu < multidevice->num_devices; ++gpu) {
    struct CALCLModel2D* calclmodel2D = multidevice->device_models[gpu];
    const CALbyte activeCells = calclmodel2D->opt == CAL_OPT_ACTIVE_CELLS_NAIVE;
    if (activeCells) {
      size_t singleNumThreadsMerge =
          calclmodel2D->borderSize * 2 * calclmodel2D->columns;
      
      //    printf("gpu=%d, before streamcompact --> %d\n", gpu, 
       //      multidevice->singleStepThreadNums[gpu][0]);
      calclKernelCall2D(calclmodel2D,  calclmodel2D->kernelMergeFlags, 1,
                        &(singleNumThreadsMerge), NULL, NULL);
      //printf("gpu=%d, launch kernelSetDiffFlags\n",gpu,multidevice->singleStepThreadNums[gpu][0]);
      //calclKernelCall2D(calclmodel2D, calclmodel2D->kernelSetDiffFlags, 1,
      //                  &(singleNumThreadsMerge), NULL, NULL);

      clFinish(multidevice->device_models[gpu]->queue);

      calclComputeStreamCompaction2D(calclmodel2D);
      calclResizeThreadsNum2D(calclmodel2D,
                              multidevice->singleStepThreadNums[gpu]);
    

     // printf("gpu=%d,after streamcompact --> %d\n", gpu, multidevice->singleStepThreadNums[gpu][0]);
    clFinish(multidevice->device_models[gpu]->queue);
    }
  }
}


void calcl_executeElementaryProcess(struct CALCLMultiDevice* multidevice,
                                    const int el_proc, int dimNum, MultiNode * mn) {


    
    for (int gpu = 0; gpu < multidevice->num_devices; ++gpu) {
        struct CALCLModel2D * calclmodel2D = multidevice->device_models[gpu];
        size_t* singleStepThreadNum = multidevice->singleStepThreadNums[gpu];

        cl_int err;


        CALbyte activeCells = calclmodel2D->opt == CAL_OPT_ACTIVE_CELLS_NAIVE;
        if (activeCells == CAL_TRUE) {

          if (singleStepThreadNum[0] > 0)
            calclKernelCall2D(calclmodel2D,
                              calclmodel2D->elementaryProcesses[el_proc],
                              dimNum, singleStepThreadNum, NULL, NULL);
//clFinish(multidevice->device_models[gpu]->queue);
         //printf("rank %d --> gpu=%d, before streamcompact el_proc num = %d --> %d\n",mn->rank,  gpu, el_proc, singleStepThreadNum[0]);

         // if (singleStepThreadNum[0] > 0) {
             /*calclComputeStreamCompaction2D(calclmodel2D);
            calclResizeThreadsNum2D(calclmodel2D, singleStepThreadNum);*/
            
            calclStreamCompactionMulti(multidevice,mn);
            //if(mn->rank ==1)
            //printf("rank %d --> gpu=%d, after streamcompact el_proc num = %d --> %d\n",mn->rank,  gpu, el_proc, singleStepThreadNum[0]);

            //printf("gpu=%d,after streamcompact el_proc num = %d --> %d \n", gpu,             el_proc, singleStepThreadNum[0]);
        //  }
        
//clFinish(multidevice->device_models[gpu]->queue);
          if (singleStepThreadNum[0] > 0) {
            //printf("gpu=%d,before update --> %d \n", gpu,  singleStepThreadNum[0]);
            calclKernelCall2D(calclmodel2D, calclmodel2D->kernelUpdateSubstate,
                              dimNum, singleStepThreadNum, NULL, NULL);
            //printf("gpu=%d,after update --> %d \n", gpu,singleStepThreadNum[0]);
              
          }




        } else {  // == CAL_TRUE

          calclKernelCall2D(calclmodel2D,
                            calclmodel2D->elementaryProcesses[el_proc], dimNum,
                            singleStepThreadNum, calclmodel2D->workGroupDimensions, NULL);
          copySubstatesBuffers2D(calclmodel2D);
        }  // == CAL_TRUE
//clFinish(multidevice->device_models[gpu]->queue);
    }  // Devices
    //printf("\n");
}


};
//END MULTINODE--------




#endif /*CALCLMULTINODE_H_*/
