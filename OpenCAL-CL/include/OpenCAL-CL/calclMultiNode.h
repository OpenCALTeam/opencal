#ifndef CALCLMULTINODE_H_
#define CALCLMULTINODE_H_

#include <mpi.h>
#include <stdio.h>
#include <vector>
#include<string>
#include<iostream>
#include <utility>
#include <OpenCAL-CL/calclMultiGPU2D.h>
#include <sys/time.h>
extern "C"{
#include <OpenCAL/cal2DIO.h>
#include <OpenCAL/cal2DBuffer.h>
#include <OpenCAL/cal2DBufferIO.h>
}


CALbyte calLoadMatrix2DrMulti(CALreal* M, const int rows, const int columns, const char* path,  int read_offset = 0, const int write_offset = 0)
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

CALbyte calLoadMatrix2DiMulti(CALint* M, const int rows, const int columns, const char* path,  int read_offset = 0, const int write_offset = 0)
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



CALbyte calLoadSubstate2DrMulti(CALModel2D* ca2D, struct CALSubstate2Dr* Q, char* path,int read_offset = 0, const int write_offset = 0) {
    CALbyte return_state = calLoadMatrix2DrMulti(Q->current, ca2D->rows, ca2D->columns, path,read_offset,write_offset);
    if (Q->next)
        calCopyBuffer2Dr(Q->current, Q->next, ca2D->rows, ca2D->columns);
    return return_state;
}

CALbyte calLoadSubstate2DiMulti(CALModel2D* ca2D, struct CALSubstate2Di* Q, char* path,int read_offset = 0, const int write_offset = 0) {
    CALbyte return_state = calLoadMatrix2DiMulti(Q->current, ca2D->rows, ca2D->columns, path,read_offset,write_offset);
    if (Q->next)
        calCopyBuffer2Di(Q->current, Q->next, ca2D->rows, ca2D->columns);
    return return_state;
}


//MULTINODE---------------
template <class F_INIT,class F_FINALIZE>
class MultiNode{
public:
    Cluster c;
    F_INIT *init;
    F_FINALIZE *finalize;
    int rank;
    CALCLMultiGPU* multigpu;

    CALreal* realNodeGhosts=0;
    CALint * intNodeGhosts=0;
    CALbyte* byteNodeGhosts=0;
    CALbyte* flagsNodeGhosts=0;


    MultiNode(Cluster _c, int _rank,F_INIT *i, F_FINALIZE *f):c(_c), rank(_rank),
        init(i), finalize(f) {
        multigpu=nullptr;
        multigpu = (CALCLMultiGPU*)malloc(sizeof(CALCLMultiGPU));

    }

    bool checkWorkloads(){return true;};

    void allocateAndInit(){

        init(multigpu,&c);

        CALCLModel2D* last_gpu = multigpu->device_models[multigpu->num_devices-1];
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
    }

    void _finalize(){
        free(realNodeGhosts);
        free(byteNodeGhosts);
        free(intNodeGhosts);
        free(flagsNodeGhosts);
        finalize(multigpu);
    }


    void run(int STEPS){

    int rank;
    timeval start, end;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    //printf("rank %d\n", rank);


    if(rank == 0)
        gettimeofday(&start, NULL);

        MPI_Barrier(MPI_COMM_WORLD);

        
        int dimNum;
        for (int gpu = 0; gpu < multigpu->num_devices; ++gpu) {
          multigpu->singleStepThreadNums[gpu] =
              computekernelLaunchParams(multigpu, gpu, &dimNum);
            //printf("%d %d\n",gpu,multigpu->singleStepThreadNums[gpu][0]);
        }



        handleBorderNodes();
        handleFlagsMultiNode();

        
        int totalSteps= STEPS;   
        while(STEPS--){
            //debug("step \n");

        for (int j = 0; j < multigpu->device_models[0]->elementaryProcessesNum; j++) {
                
                MPI_Barrier(MPI_COMM_WORLD);
                printf("rank =  %d\n",rank );
                calcl_executeElementaryProcess(multigpu, j,
                                               dimNum /*elementary process*/,this);
                MPI_Barrier(MPI_COMM_WORLD);

                           // barrier tutte hanno finito
            for (int gpu = 0; gpu < multigpu->num_devices; ++gpu) {
              clFinish(multigpu->device_models[gpu]->queue);
            }

            // Read from the substates and set ghost borders
            for (int gpu = 0; gpu < multigpu->num_devices; ++gpu) {
              calclGetBorderFromDeviceToHost2D(multigpu->device_models[gpu]);
            }


            // scambia bordi
            // Write from the ghost borders to the substates
            calclMultiGPUHandleBordersMultiNode(multigpu,
                                                multigpu->exchange_full_border);

          handleBorderNodes();
        }
       

//vanno scambiati i ghost. dentro l'arraydelle celle attive (non real)
// le prime e ultime righe vanno mandate alle GPU vicine (esattamente nello stesso modo
//con cui vengono scambiati i sottostati) 
// i dati ricevuto dalle GPU vicine non possono essere direttamente mpacchiati 
//dentro la matrice delle celle attive ma vanno "mergiati" perchè le celle possono essere
//attivate da una GPU vicina ma anche da me stesso 
/*
_________
|BORDO00|   bordo01 e la prima riga di GPU1 vanno mergiate prima dentro la prima riga di GPu1
|_______|   bordo10 e la ultima riga di GPU0 vanno mergiate prima dentro l'ultima riga di GPU0
|       |
|  GPU0 |   bordo00 e l'ultima riga di GPU1 vanno mergiate dentro ultima riga di gpu1 e cosi via
|       |
|       |  
|-------|
|BORDO01|
|_______|
_________
|BORDO10|
|_______|
|       |
|  GPU1 |   
|       |
|       |  
|-------|
|BORDO10|
|_______|
*/


//#if 0
            // STEERING------------------------------
            struct CALCLModel2D* calclmodel2DFirst = multigpu->device_models[0];

            if (calclmodel2DFirst->kernelSteering != NULL) {

              for (int gpu = 0; gpu < multigpu->num_devices; ++gpu) {
                struct CALCLModel2D* calclmodel2D =
                    multigpu->device_models[gpu];
                size_t* singleStepThreadNum = multigpu->singleStepThreadNums[gpu];

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

              for (int gpu = 0; gpu < multigpu->num_devices; ++gpu) {
                clFinish(multigpu->device_models[gpu]->queue);
              }

              for (int gpu = 0; gpu < multigpu->num_devices; ++gpu) {
                calclGetBorderFromDeviceToHost2D(multigpu->device_models[gpu]);
              }

              // scambia bordi
              calclMultiGPUHandleBordersMultiNode(
                  multigpu, multigpu->exchange_full_border);
            }//Steering
//#endif
            handleBorderNodes();

        }//STEPS
    

        //handleBorderNodes();
        calclDevicesToNode(multigpu);

        if (rank == 0) {
          gettimeofday(&end, NULL);
          unsigned long long t = 1000 * (end.tv_sec - start.tv_sec) +
                                 (end.tv_usec - start.tv_usec) / 1000;

          // end = time(NULL);
          printf("Elapsed time: %llu ms\n", t);
        }

        _finalize();
    }//run function

    void handleBorderNodes(){
        handleBorderNodesR();
        handleBorderNodesI();
        handleBorderNodesB();
    }


    void handleBorderNodesR(){
        const MPI_Datatype DATATYPE = MPI_DOUBLE;
        if(!c.is_full_exchange()){

            CALint prev,next;
            CALCLModel2D* gpu_to_use = multigpu->device_models[0];
            CALreal* send_offset;
            CALreal* recv_offset = realNodeGhosts;

            const CALint sizeBorder = gpu_to_use->borderSize*gpu_to_use->columns;
            const int numSubstates = gpu_to_use->host_CA->sizeof_pQr_array;
            const CALint count = (numSubstates*sizeBorder);

            if(numSubstates <= 0)
                return;

            for(int i=0;i<2;i++){

                next=((rank+1)+c.nodes.size())%c.nodes.size();
                prev=((rank-1)+c.nodes.size())%c.nodes.size();
                if(i==1)
                    std::swap(next,prev);


                //this should be multigpu->num_devices-1 and 0 during the two iterations
                CALint indexgpu = (i+multigpu->num_devices-1)%multigpu->num_devices;
                gpu_to_use = multigpu->device_models[indexgpu];


                send_offset = gpu_to_use->borderMapper.realBorder_OUT;
                send_offset+=(i==0 ? 1 : 0)*count;

                recv_offset = realNodeGhosts;
                recv_offset+= (i==0 ? 0: 1)*count;

                if(rank % 2 == 0){
                    //MPI send


                    // printf("I'm %d:  sedning to %d \n" ,  rank , next);
                    //cerca convenzione per i nomi dei tags
                    MPI_Send(send_offset,count, DATATYPE , next , i ,MPI_COMM_WORLD);

                    // printf("I'm %d:  receiving from  %d \n" ,  rank , prev);
                    MPI_Recv(recv_offset , count , DATATYPE, prev, i, MPI_COMM_WORLD,MPI_STATUS_IGNORE);

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

                MPI_Barrier(MPI_COMM_WORLD);


            }




            //memory has been exchanged between mpi processes. Now it'0s time to foward that memory to
            //the right GPUs
            cl_int err;
            for(int i =0; i < numSubstates; i++){
                CALCLModel2D* m = multigpu->device_models[0];
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
                m = multigpu->device_models[multigpu->num_devices-1];
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





    void handleBorderNodesI(){
        const MPI_Datatype DATATYPE = MPI_INT;
        if(!c.is_full_exchange()){

            CALint prev,next;
            CALCLModel2D* gpu_to_use = multigpu->device_models[0];
            CALint* send_offset;
            CALint* recv_offset = intNodeGhosts;

            const CALint sizeBorder = gpu_to_use->borderSize*gpu_to_use->columns;
            const int numSubstates = gpu_to_use->host_CA->sizeof_pQi_array;
            const CALint count = (numSubstates*sizeBorder);

            if(numSubstates <= 0)
                return;

            for(int i=0;i<2;i++){

                next=((rank+1)+c.nodes.size())%c.nodes.size();
                prev=((rank-1)+c.nodes.size())%c.nodes.size();
                if(i==1)
                    std::swap(next,prev);


                //this should be multigpu->num_devices-1 and 0 during the two iterations
                CALint indexgpu = (i+multigpu->num_devices-1)%multigpu->num_devices;
                gpu_to_use = multigpu->device_models[indexgpu];


                send_offset = gpu_to_use->borderMapper.intBorder_OUT;
                send_offset+=(i==0 ? 1 : 0)*count;

                recv_offset = intNodeGhosts;
                recv_offset+= (i==0 ? 0: 1)*count;

                if(rank % 2 == 0){
                    //MPI send


                    // printf("I'm %d:  sedning to %d \n" ,  rank , next);
                    //cerca convenzione per i nomi dei tags
                    MPI_Send(send_offset,count, DATATYPE , next , i ,MPI_COMM_WORLD);

                    // printf("I'm %d:  receiving from  %d \n" ,  rank , prev);
                    MPI_Recv(recv_offset , count , DATATYPE, prev, i, MPI_COMM_WORLD,MPI_STATUS_IGNORE);



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

                MPI_Barrier(MPI_COMM_WORLD);


            }
            //memory has been exchanged between mpi processes. Now it'0s time to foward that memory to
            //the right GPUs
            cl_int err;
            for(int i =0; i < numSubstates; i++){
                CALCLModel2D* m = multigpu->device_models[0];
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
                m = multigpu->device_models[multigpu->num_devices-1];
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



    void handleBorderNodesB(){
        const MPI_Datatype DATATYPE = MPI_CHAR;
        if(!c.is_full_exchange()){

            CALint prev,next;
            CALCLModel2D* gpu_to_use = multigpu->device_models[0];
            CALbyte* send_offset;
            CALbyte* recv_offset = byteNodeGhosts;

            const CALint sizeBorder = gpu_to_use->borderSize*gpu_to_use->columns;
            const int numSubstates = gpu_to_use->host_CA->sizeof_pQb_array;
            const CALint count = (numSubstates*sizeBorder);

            if(numSubstates <= 0)
                return;

            for(int i=0;i<2;i++){

                next=((rank+1)+c.nodes.size())%c.nodes.size();
                prev=((rank-1)+c.nodes.size())%c.nodes.size();
                if(i==1)
                    std::swap(next,prev);


                //this should be multigpu->num_devices-1 and 0 during the two iterations
                CALint indexgpu = (i+multigpu->num_devices-1)%multigpu->num_devices;
                gpu_to_use = multigpu->device_models[indexgpu];


                send_offset = gpu_to_use->borderMapper.byteBorder_OUT;
                send_offset+=(i==0 ? 1 : 0)*count;

                recv_offset = byteNodeGhosts;
                recv_offset+= (i==0 ? 0: 1)*count;

                if(rank % 2 == 0){
                    //MPI send


                    // printf("I'm %d:  sedning to %d \n" ,  rank , next);
                    //cerca convenzione per i nomi dei tags
                    MPI_Send(send_offset,count, DATATYPE , next , i ,MPI_COMM_WORLD);

                    // printf("I'm %d:  receiving from  %d \n" ,  rank , prev);
                    MPI_Recv(recv_offset , count , DATATYPE, prev, i, MPI_COMM_WORLD,MPI_STATUS_IGNORE);



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

                MPI_Barrier(MPI_COMM_WORLD);


            }




            //memory has been exchanged between mpi processes. Now it'0s time to foward that memory to
            //the right GPUs
            cl_int err;
            for(int i =0; i < numSubstates; i++){
                CALCLModel2D* m = multigpu->device_models[0];
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
                m = multigpu->device_models[multigpu->num_devices-1];
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
        CALCLModel2D* gpu_to_use = multigpu->device_models[0];
        CALbyte* send_offset;
        CALbyte* recv_offset = flagsNodeGhosts;
        const CALint sizeBorder = gpu_to_use->borderSize * gpu_to_use->columns;
        const CALint count = (sizeBorder);
       // printf("rank %d --> multigpu->singleStepThreadNums[prev] %d",rank, multigpu->singleStepThreadNums[prev]);
       // printf("rank %d --> multigpu->singleStepThreadNums[next] %d",rank, multigpu->singleStepThreadNums[next]);

        for (int i = 0; i < 2; i++) {

          next = ((rank + 1) + c.nodes.size()) % c.nodes.size();
          prev = ((rank - 1) + c.nodes.size()) % c.nodes.size();

          if (i == 1) std::swap(next, prev);

          // this should be multigpu->num_devices-1 and 0 during the two
          // iterations
          CALint indexgpu =
              (i + multigpu->num_devices - 1) % multigpu->num_devices;
          // printf("indexgpu %d \n", indexgpu);
          gpu_to_use = multigpu->device_models[indexgpu];

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

        CALCLModel2D* m = multigpu->device_models[0];
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
        m = multigpu->device_models[multigpu->num_devices - 1];
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









};
//END MULTINODE--------

#endif /*CALCLMULTINODE_H_*/
