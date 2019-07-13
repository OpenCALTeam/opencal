#ifndef CALCLMULTINODE3D_H_
#define CALCLMULTINODE3D_H_


#include <mpi.h>
#include <stdio.h>
#include <vector>
#include<string>
#include<iostream>
#include <utility>
#include <OpenCAL-CL/calclMultiDevice3D.h>
#include <sys/time.h>
extern "C"{
#include <OpenCAL/cal3DIO.h>
#include <OpenCAL/cal3DBuffer.h>
#include <OpenCAL/cal3DBufferIO.h>
#include <OpenCAL/cal3DUnsafe.h>
}
typedef void (* CALCallbackFuncMNInit3D)(struct CALCLMultiDevice3D* ca3D, const Node& mynode);
typedef void (* CALCallbackFuncMNFinalize3D)(struct CALCLMultiDevice3D* ca3D);
typedef bool (* CALCallbackFuncMNStopCondition3D)(struct CALCLMultiDevice3D* ca3D);

CALbyte calNodeLoadMatrix3Dr(CALreal* M, const int rows, const int columns, const int slices,const char* path,  int read_offset = 0, const int write_offset = 0)
{
    FILE *f = NULL;
    f = fopen(path, "r");

    if ( !f )
        return CAL_FALSE;
    // read_offset number of layers from top
    int numberofrowstoskip = read_offset*rows;
    //skip #offset rows
    const int _s = 20*2*columns; //assuming 20char per number + spaces
    char tmp[_s];
    while(numberofrowstoskip--)
        fgets(tmp,sizeof(char)*_s,f);

    calfLoadBuffer3Dr(M+write_offset, rows, columns,slices, f);

    fclose(f);

    return CAL_TRUE;
}

CALbyte calNodeLoadSubstate3Dr(CALModel3D* ca3D, struct CALSubstate3Dr* Q, char* path,const Node& mynode) {
    int write_offset =0;
    CALbyte return_state = calNodeLoadMatrix3Dr(Q->current, ca3D->rows, ca3D->columns, ca3D->slices, path,mynode.offset,write_offset);
    if (Q->next)
        calCopyBuffer3Dr(Q->current, Q->next, ca3D->rows, ca3D->columns, ca3D->slices);
    return return_state;
}

CALbyte calNodeLoadMatrix3Di(CALint* M, const int rows, const int columns, const int slices, const char* path,  int read_offset = 0, const int write_offset = 0)
{
    FILE *f = NULL;
    f = fopen(path, "r");

    if ( !f )
        return CAL_FALSE;
    // read_offset number of layers from top
    int numberofrowstoskip = read_offset*rows;
    //skip #offset rows
    const int _s = 20*2*columns; //assuming 20char per number + spaces
    char tmp[_s];
    while(numberofrowstoskip--)
        fgets(tmp,sizeof(char)*_s,f);

    calfLoadBuffer3Di(M+write_offset, rows, columns, slices, f);

    fclose(f);

    return CAL_TRUE;
}

CALbyte calNodeLoadSubstate3Di(CALModel3D* ca3D, struct CALSubstate3Di* Q, char* path,const Node& mynode) {
    int write_offset =0;
    CALbyte return_state = calNodeLoadMatrix3Di(Q->current, ca3D->rows, ca3D->columns, ca3D->slices, path,mynode.offset,write_offset);
    if (Q->next)
        calCopyBuffer3Di(Q->current, Q->next, ca3D->rows, ca3D->columns, ca3D->slices);
    return return_state;
}

CALbyte calNodeLoadMatrix3Db(CALbyte* M, const int rows, const int columns,  const int slices, const char* path,  int read_offset = 0, const int write_offset = 0)
{
    FILE *f = NULL;
    f = fopen(path, "r");

    if ( !f )
        return CAL_FALSE;
    // read_offset number of layers from top
    int numberofrowstoskip = read_offset*rows;
    //skip #offset rows
    const int _s = 20*2*columns; //assuming 20char per number + spaces
    char tmp[_s];
    while(numberofrowstoskip--)
        fgets(tmp,sizeof(char)*_s,f);

    calfLoadBuffer3Db(M+write_offset, rows, columns, slices, f);

    fclose(f);

    return CAL_TRUE;
}

CALbyte calNodeLoadSubstate3Db(CALModel3D* ca3D, struct CALSubstate3Db* Q, char* path,const Node& mynode) {
    int write_offset =0;
    CALbyte return_state = calNodeLoadMatrix3Db(Q->current, ca3D->rows, ca3D->columns, ca3D->slices, path,mynode.offset,write_offset);
    if (Q->next)
        calCopyBuffer3Db(Q->current, Q->next, ca3D->rows, ca3D->columns, ca3D->slices);
    return return_state;
}


class MultiNode3D{
public:
    //only used to kep track of the communication overhead

    double start_kernel_computation, end_kernel_computation;
    double start_kernel_communication, end_kernel_communication;
    double start_kernel_streamcompaction_communication, end_kernel_streamcompaction_communication;
    double start_kernel_streamcompaction_computation, end_kernel_streamcompaction_computation;
    double start_communication, end_communication;
   
    double kernel_total_communication=0;
    double kernel_total_computation=0;

    double total_kernel_streamcompaction_communication=0;
    double total_kernel_streamcompaction_computation=0;

    double totalCommunicationTimeMPI = 0;
    int totalNumberofCommunicationMPI = 0;

    double total_communication=0;
    double elapsedTime=0;
    
    double * gatherElapsedTime;

    CALDistributedDomain3D c;
    CALCallbackFuncMNInit3D init;
    CALCallbackFuncMNFinalize3D finalize;
    CALCallbackFuncMNStopCondition3D stopCondition;
    int checkStopCondition;

    int rank;
    CALCLMultiDevice3D* multidevice;

    CALreal* realNodeGhosts=0;
    CALint * intNodeGhosts=0;
    CALbyte* byteNodeGhosts=0;
    CALbyte* flagsNodeGhosts=0;

    MultiNode3D(CALDistributedDomain3D _c,CALCallbackFuncMNInit3D i, CALCallbackFuncMNFinalize3D f):c(_c),
        init(i), finalize(f) {
        MPI_Init(NULL, NULL);
        stopCondition = NULL;
        checkStopCondition = 1;
        int world_rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

        rank = world_rank;

        multidevice=nullptr;
        multidevice = (CALCLMultiDevice3D*)malloc(sizeof(CALCLMultiDevice3D));

        Node mynode = c.nodes[rank];
        auto devices = mynode.devices;

        // calclPrintPlatformsAndDevices(calcl_device_manager);
        struct CALCLDeviceManager * calcl_device_manager = calclCreateManager();

        calclSetNumDevice(multidevice,devices.size());
        for(auto& d : devices){
            calclAddDevice(multidevice,calclGetDevice(calcl_device_manager, d.num_platform , d.num_device) ,  d.workload);
        }
    }

    void setStopCondition(CALCallbackFuncMNStopCondition3D f, int n){
        stopCondition = f;
        checkStopCondition = n;
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

        CALCLModel3D* last_gpu = multidevice->device_models[multidevice->num_devices-1];
        const CALint sizeBorder = last_gpu->borderSize*last_gpu->columns*last_gpu->rows;
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
        //MPI_Barrier(MPI_COMM_WORLD);
    }

    void _finalize(){
        free(realNodeGhosts);
        free(byteNodeGhosts);
        free(intNodeGhosts);
        free(flagsNodeGhosts);
        finalize(multidevice);
    }

    void handleBorderNodes(double & T, int & ntComuunication){
        if(multidevice->device_models[0]->borderSize<=0 )
            return;
        handleBorderNodesR(T,ntComuunication);
        handleBorderNodesI(T,ntComuunication);
        handleBorderNodesB(T,ntComuunication);
    }

    void handleBorderNodesR(double& T, int &ntComuunication){
        const MPI_Datatype DATATYPE = MPI_DOUBLE;
        if(!c.is_full_exchange()){

            CALint prev,next;
            CALCLModel3D* gpu_to_use = multidevice->device_models[0];
            CALreal* send_offset;
            CALreal* recv_offset = realNodeGhosts;

            const CALint sizeBorder = gpu_to_use->borderSize*gpu_to_use->columns*gpu_to_use->rows;
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
                CALCLModel3D* m = multidevice->device_models[0];
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
            CALCLModel3D* gpu_to_use = multidevice->device_models[0];
            CALint* send_offset;
            CALint* recv_offset = intNodeGhosts;

            const CALint sizeBorder = gpu_to_use->borderSize*gpu_to_use->columns*gpu_to_use->rows;
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
                CALCLModel3D* m = multidevice->device_models[0];
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
            CALCLModel3D* gpu_to_use = multidevice->device_models[0];
            CALbyte* send_offset;
            CALbyte* recv_offset = byteNodeGhosts;

            const CALint sizeBorder = gpu_to_use->borderSize*gpu_to_use->columns*gpu_to_use->rows;
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

                    //                 printf("I'm %d:  sedning to %d \n" ,  rank , next);
                    //cerca convenzione per i nomi dei tags
                    MPI_Send(send_offset,count, DATATYPE , next , i ,MPI_COMM_WORLD);

                    //                   printf("I'm %d:  receiving from  %d \n" ,  rank , prev);
                    MPI_Recv(recv_offset , count , DATATYPE, prev, i, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
                    T2 = MPI_Wtime();
                    deltaT = T2 - T1;
                    T += deltaT;


                    //                                        for (int bor = 0;bor < count; bor++) {
                    //                                            if(bor != 0 && bor%(gpu_to_use->host_CA->columns) ==0 )
                    //                                                printf("\n");
                    //                                            if(bor != 0 && bor%(gpu_to_use->host_CA->rows*
                    //                                                                gpu_to_use->host_CA->columns) ==0 )
                    //                                                printf("\n\n");
                    //                                            printf(" %d ",recv_offset[bor]);
                    //                                        }
                    //                                        printf("\n\n");
                    //send to rank+1
                    //receive rank-1
                }else{

                    //printf("I'm %d:  receiving from  %d \n" ,  rank , prev);
                    MPI_Recv(recv_offset , count , DATATYPE, prev, i ,MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    //printf("I'm %d:  sedning to %d \n" ,  rank , next);
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
                CALCLModel3D* m = multidevice->device_models[0];
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
            CALCLModel3D* gpu_to_use = multidevice->device_models[0];
            CALbyte* send_offset;
            CALbyte* recv_offset = flagsNodeGhosts;
            const CALint sizeBorder = gpu_to_use->borderSize * gpu_to_use->columns* gpu_to_use->rows;
            const CALint count = (sizeBorder);
            //printf("rank %d --> multidevice->singleStepThreadNums[prev] %d",rank, multidevice->singleStepThreadNums[prev]);
            //printf("rank %d --> multidevice->singleStepThreadNums[next] %d",rank, multidevice->singleStepThreadNums[next]);

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

                    // printf("I'm %d:  receiving from  %d \n" ,  rank , prev);
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

                //MPI_Barrier(MPI_COMM_WORLD);
                // printf("barrier\n");


            }  // for

            // if(rank ==1){
                
            //     for(int i = 0; i < 9600; i++)
            //     {
            //        if(i%4800==0 && i != 0)
            //        printf("\n");

            //        printf(" %d ", flagsNodeGhosts[i]);
            //     }
                
            // }

            // calcolo le due gpu che devono ricevere i flagsNodeGhosts
            cl_int err;

            CALCLModel3D* m = multidevice->device_models[0];
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


    void calclStreamCompactionMulti(struct CALCLMultiDevice3D* multidevice, MultiNode3D * mn){

        //----------MEASURE TIME---------
        mn->start_kernel_streamcompaction_communication = MPI_Wtime();

        cl_int err;
        // Read from substates and set flags borders
        //printf(" before read bufferActiveCellsFlags \n");

        for (int gpu = 0; gpu < multidevice->num_devices; ++gpu) {
            struct CALCLModel3D* calclmodel3D = multidevice->device_models[gpu];
            CALCLqueue queue = calclmodel3D->queue;

            cl_int err;
            int dim = calclmodel3D->fullSize;

            int sizeBorder = calclmodel3D->borderSize * calclmodel3D->columns* calclmodel3D->rows;
            //printf(" read first border OK bufferActiveCellsFlags gpu = %d  %d\n", gpu, sizeof(CALbyte) * sizeBorder );
            err = clEnqueueReadBuffer(queue, calclmodel3D->bufferActiveCellsFlags,
                                      CL_TRUE, 0, sizeof(CALbyte) * sizeBorder,
                                      calclmodel3D->borderMapper.flagsBorder_OUT, 0,
                                      NULL, NULL);
            calclHandleError(err);
            //printf(" read first border OK bufferActiveCellsFlags gpu = %d\n", gpu);

            err = clEnqueueReadBuffer(
                        queue, calclmodel3D->bufferActiveCellsFlags, CL_TRUE,
                        ((dim - sizeBorder)) * sizeof(CALbyte), sizeof(CALbyte) * sizeBorder,
                        calclmodel3D->borderMapper.flagsBorder_OUT + sizeBorder, 0, NULL, NULL);
            calclHandleError(err);
            // printf(" read last border OK bufferActiveCellsFlags gpu = %d \n",gpu);
        }

        mn->handleFlagsMultiNode();

        //printf(" after read bufferActiveCellsFlags \n");
        //printf(" before write bufferActiveCellsFlags \n");
        for (int gpu = 0; gpu < multidevice->num_devices; ++gpu) {

            struct CALCLModel3D* calclmodel3D = multidevice->device_models[gpu];
            struct CALCLModel3D* calclmodel3DPrev = NULL;
            const int gpuP =
                    ((gpu - 1) + multidevice->num_devices) % multidevice->num_devices;
            const int gpuN =
                    ((gpu + 1) + multidevice->num_devices) % multidevice->num_devices;

            if (calclmodel3D->host_CA->T == CAL_SPACE_TOROIDAL || ((gpu - 1) >= 0)) {

                calclmodel3DPrev = multidevice->device_models[gpuP];
            }

            struct CALCLModel3D* calclmodel3DNext = NULL;
            if (calclmodel3D->host_CA->T == CAL_SPACE_TOROIDAL ||
                    ((gpu + 1) < multidevice->num_devices)) {
                calclmodel3DNext = multidevice->device_models[gpuN];
            }

            int dim = calclmodel3D->fullSize;

            const int sizeBorder = calclmodel3D->borderSize * calclmodel3D->columns* calclmodel3D->rows;



            /*const CALbyte activeCells = calclmodel3D->opt == CAL_OPT_ACTIVE_CELLS_NAIVE;
            if (activeCells == CAL_TRUE) {
                // copy border flags from DevicePrev and Device next to a mergeflagsBorder
                if (calclmodel3DPrev != NULL &&
                        (multidevice->exchange_full_border || gpu != 0)) {
                    err = clEnqueueWriteBuffer(
                                calclmodel3D->queue, calclmodel3D->borderMapper.mergeflagsBorder,
                                CL_TRUE, 0, sizeof(CALbyte) * sizeBorder,
                                calclmodel3DPrev->borderMapper.flagsBorder_OUT + sizeBorder, 0,
                                NULL, NULL);

                    calclHandleError(err);
                }
                if (calclmodel3DNext != NULL && (multidevice->exchange_full_border ||
                                                 gpu != multidevice->num_devices - 1)) {
                    err = clEnqueueWriteBuffer(
                                calclmodel3D->queue, calclmodel3D->borderMapper.mergeflagsBorder,
                                CL_TRUE, ((sizeBorder)) * sizeof(CALbyte),
                                sizeof(CALbyte) * sizeBorder,
                                calclmodel3DNext->borderMapper.flagsBorder_OUT, 0, NULL, NULL);
                    calclHandleError(err);
                }
            }*/
        }
        mn->end_kernel_streamcompaction_communication = MPI_Wtime();

        mn->total_kernel_streamcompaction_communication += mn->end_kernel_streamcompaction_communication - mn->start_kernel_streamcompaction_communication;

        mn->total_communication += mn->total_kernel_streamcompaction_communication;


        mn->start_kernel_streamcompaction_computation = MPI_Wtime();

        for (int gpu = 0; gpu < multidevice->num_devices; ++gpu) {
            struct CALCLModel3D* calclmodel3D = multidevice->device_models[gpu];
            const CALbyte activeCells = calclmodel3D->opt == CAL_OPT_ACTIVE_CELLS_NAIVE;
            if (activeCells) {
                size_t singleNumThreadsMerge =
                        calclmodel3D->borderSize * 2 * calclmodel3D->columns* calclmodel3D->rows;

                //printf("rank %d, gpu=%d, singleNumThreadsMerget --> %d\n",rank, gpu, singleNumThreadsMerge);
                //      multidevice->singleStepThreadNums[gpu][0]);

                calclKernelCall3D(calclmodel3D,  calclmodel3D->kernelMergeFlags, 1,
                                  &(singleNumThreadsMerge), calclmodel3D->workGroupDimensions);
                //printf("gpu=%d, launch calclmodel3D->streamCompactionThreadsNum %d\n",gpu,calclmodel3D->streamCompactionThreadsNum);
                //calclKernelCall3D(calclmodel3D, calclmodel3D->kernelSetDiffFlags, 1,
                //                  &(singleNumThreadsMerge), NULL, NULL);

                clFinish(multidevice->device_models[gpu]->queue);

                calclComputeStreamCompaction3D(calclmodel3D);

                calclResizeThreadsNum3D(calclmodel3D,
                                        multidevice->singleStepThreadNums[gpu]);


                //printf("rank %d, gpu=%d,after streamcompact --> %d\n", rank, gpu, multidevice->singleStepThreadNums[gpu][0]);
                clFinish(multidevice->device_models[gpu]->queue);
            }
        }
        mn->end_kernel_streamcompaction_computation = MPI_Wtime();

        mn->total_kernel_streamcompaction_computation += mn->end_kernel_streamcompaction_computation - mn->start_kernel_streamcompaction_computation;

    }

    void calcl_executeElementaryProcess(struct CALCLMultiDevice3D* multidevice,
                                        const int el_proc, int dimNum, MultiNode3D * mn) {



        for (int gpu = 0; gpu < multidevice->num_devices; ++gpu) {
            struct CALCLModel3D * calclmodel3D = multidevice->device_models[gpu];
            size_t* singleStepThreadNum = multidevice->singleStepThreadNums[gpu];

            cl_int err;
            //printf("rank %d --> gpu=%d, after elementaryProcesses el_proc num = %d --> %d\n",mn->rank,  gpu, el_proc, singleStepThreadNum[0]);


            CALbyte activeCells = calclmodel3D->opt == CAL_OPT_ACTIVE_CELLS_NAIVE;
            if (activeCells == CAL_TRUE) {
               // printf("rank %d --> gpu=%d, before elementaryProcesses el_proc num = %d --> %d\n",mn->rank,  gpu, el_proc, singleStepThreadNum[0]);
                if (singleStepThreadNum[0] > 0)
                {
                    calclKernelCall3D(calclmodel3D,
                                      calclmodel3D->elementaryProcesses[el_proc],
                                      dimNum, singleStepThreadNum, calclmodel3D->workGroupDimensions);
                 //clFinish(multidevice->device_models[gpu]->queue);
                //printf("rank %d --> gpu=%d, before streamcompact el_proc num = %d --> %d\n",mn->rank,  gpu, el_proc, singleStepThreadNum[0]);
                }

                calclStreamCompactionMulti(multidevice,mn);
                
               // if (singleStepThreadNum[0] > 0) {
                   // calclComputeStreamCompaction3D(calclmodel3D);
                   // calclResizeThreadsNum3D(calclmodel3D, singleStepThreadNum);
                //}
                
                //if(mn->rank ==1)
                //printf("rank %d --> gpu=%d, after streamcompact el_proc num = %d --> %d\n",mn->rank,  gpu, el_proc, singleStepThreadNum[0]);

                //printf("gpu=%d,after streamcompact el_proc num = %d --> %d \n", gpu,             el_proc, singleStepThreadNum[0]);
                //  }

                //clFinish(multidevice->device_models[gpu]->queue);
                if (singleStepThreadNum[0] > 0) {
                    //printf("gpu=%d,before update --> %d \n", gpu,  singleStepThreadNum[0]);
                    calclKernelCall3D(calclmodel3D, calclmodel3D->kernelUpdateSubstate,
                                      dimNum, singleStepThreadNum, calclmodel3D->workGroupDimensions);
                    //printf("gpu=%d,after update --> %d \n", gpu,singleStepThreadNum[0]);

                }




            } else {  // == CAL_TRUE

                calclKernelCall3D(calclmodel3D,
                                  calclmodel3D->elementaryProcesses[el_proc], dimNum,
                                  singleStepThreadNum, calclmodel3D->workGroupDimensions);

                calclCopySubstatesBuffers3D(calclmodel3D);
            }  // == CAL_TRUE
            //clFinish(multidevice->device_models[gpu]->queue);
        }  // Devices
        //printf("\n");
    }

    void run(int STEPS){

        int rank;
        double start, end;

        bool stop = false;

        MPI_Barrier(MPI_COMM_WORLD);

        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        
        int dimNum;
        for (int gpu = 0; gpu < multidevice->num_devices; ++gpu) {
            multidevice->singleStepThreadNums[gpu] =
                    computekernelLaunchParams(multidevice, gpu, &dimNum);
        }


        //----------MEASURE TIME---------
        start_communication = MPI_Wtime();
        handleBorderNodes(totalCommunicationTimeMPI,totalNumberofCommunicationMPI);
        end_communication = MPI_Wtime();

        total_communication += end_communication - start_communication;
        
        //----------------------------------
        for (int gpu = 0; gpu < multidevice->num_devices; ++gpu) {
            for (int el_proc = 0;
                 el_proc < multidevice->device_models[0]->elementaryProcessesNum;
                 el_proc++) {
                struct CALCLModel3D* calclmodel3D = multidevice->device_models[gpu];
                size_t* singleStepThreadNum = multidevice->singleStepThreadNums[gpu];

                cl_int err;

                if (calclmodel3D->kernelInitSubstates != NULL)
                    calclSetReductionParameters3D(calclmodel3D,
                                                  &calclmodel3D->kernelInitSubstates);
                if (calclmodel3D->kernelStopCondition != NULL)
                    calclSetReductionParameters3D(calclmodel3D,
                                                  &calclmodel3D->kernelStopCondition);
                if (calclmodel3D->kernelSteering != NULL)
                    calclSetReductionParameters3D(calclmodel3D,
                                                  &calclmodel3D->kernelSteering);

                int i = 0;

                calclSetReductionParameters3D(
                            calclmodel3D, &calclmodel3D->elementaryProcesses[el_proc]);
            }
        }

        int totalSteps= STEPS;

        start = MPI_Wtime();

        while(STEPS-- && !stop){

            for (int j = 0; j < multidevice->device_models[0]->elementaryProcessesNum; j++) {

                start_kernel_computation =  MPI_Wtime();

                calcl_executeElementaryProcess(multidevice, j,
                                               dimNum /*elementary process*/,this);

                // if (multidevice->num_devices != 1 || c.nodes.size() != 1) {

                // barrier tutte hanno finito
                for (int gpu = 0; gpu < multidevice->num_devices; ++gpu) {
                    clFinish(multidevice->device_models[gpu]->queue);
                }

                end_kernel_computation =  MPI_Wtime();
                kernel_total_computation += end_kernel_computation - start_kernel_computation;

                //----------MEASURE TIME---------
                start_communication =  MPI_Wtime();

                // Read from the substates and set ghost borders
                // start_kernel_communication =  MPI_Wtime();
                // for (int gpu = 0; gpu < multidevice->num_devices; ++gpu) {
                //     calclGetBorderFromDeviceToHost3D(
                //                 multidevice->device_models[gpu]);
                // }

                //scambia bordi
                //Write from the ghost borders to the substates
                // calclMultiDeviceUpdateHalos3D(
                //             multidevice, multidevice->exchange_full_border);
                // end_kernel_communication =  MPI_Wtime(); 
                // handleBorderNodes(totalCommunicationTimeMPI,totalNumberofCommunicationMPI);

                // end_communication =  MPI_Wtime();

                // total_communication += end_communication-start_communication;
                // kernel_total_communication += end_kernel_communication- start_kernel_communication;
                //----------------------------------
                //   }
            }

            // STEERING------------------------------
            struct CALCLModel3D* calclmodel3DFirst = multidevice->device_models[0];
            
            if (calclmodel3DFirst->kernelSteering != NULL) {
                start_kernel_computation =  MPI_Wtime();
                for (int gpu = 0; gpu < multidevice->num_devices; ++gpu) {


                    struct CALCLModel3D* calclmodel3D =
                            multidevice->device_models[gpu];
                    size_t* singleStepThreadNum = multidevice->singleStepThreadNums[gpu];

                    if (calclmodel3D->kernelSteering != NULL) {
                        if (singleStepThreadNum[0] > 0)
                            calclKernelCall3D(calclmodel3D, calclmodel3D->kernelSteering,
                                              dimNum, singleStepThreadNum, calclmodel3D->workGroupDimensions);
                        CALbyte activeCells =
                                calclmodel3D->opt == CAL_OPT_ACTIVE_CELLS_NAIVE;
                        if (activeCells == CAL_TRUE)
                            if (singleStepThreadNum[0] > 0)
                                calclKernelCall3D(calclmodel3D,
                                                  calclmodel3D->kernelUpdateSubstate,
                                                  dimNum, singleStepThreadNum, calclmodel3D->workGroupDimensions);
                            else
                                calclCopySubstatesBuffers3D(calclmodel3D);

                    }


                }
                //----------------------------------------

                for (int gpu = 0; gpu < multidevice->num_devices; ++gpu) {
                    clFinish(multidevice->device_models[gpu]->queue);
                }
                end_kernel_computation =  MPI_Wtime();
                kernel_total_computation += end_kernel_computation - start_kernel_computation;

                //----------MEASURE TIME---------

                // start_communication = MPI_Wtime();
                // start_kernel_communication = MPI_Wtime();
                // for (int gpu = 0; gpu < multidevice->num_devices; ++gpu) {
                //     calclGetBorderFromDeviceToHost3D(
                //                 multidevice->device_models[gpu]);
                // }

                // // scambia bordi
                // calclMultiDeviceUpdateHalos3D(
                //             multidevice, multidevice->exchange_full_border);
                
                // end_communication = MPI_Wtime();
                // end_kernel_communication = MPI_Wtime();
                // kernel_total_communication += end_kernel_communication - start_kernel_communication;

                // end_communication = MPI_Wtime();
                // total_communication += end_communication - start_communication;
                //----------------------------------
                //}
            }  // Steering

            
            if (calclmodel3DFirst->kernelStopCondition != NULL) {
                start_kernel_computation =  MPI_Wtime();
                for (int gpu = 0; gpu < multidevice->num_devices; ++gpu) {


                    struct CALCLModel3D* calclmodel3D =
                            multidevice->device_models[gpu];
                    size_t* singleStepThreadNum = multidevice->singleStepThreadNums[gpu];

                    if (calclmodel3D->kernelStopCondition != NULL) {
                        if (singleStepThreadNum[0] > 0){
                        //    stop = checkStopCondition3D(calclmodel3D, dimNum, singleStepThreadNum);
                        //     printf("rank %d, stop %d\n",rank,  stop);
                        }
                    }


                }
                for (int gpu = 0; gpu < multidevice->num_devices; ++gpu) {
                    clFinish(multidevice->device_models[gpu]->queue);
                }
                
                end_kernel_computation =  MPI_Wtime();
                kernel_total_communication += end_kernel_computation - start_kernel_computation;
            }
            
            if ( stopCondition != NULL && STEPS%checkStopCondition==0) {
                 calclMultiDeviceToNode(multidevice);
                  stop = stopCondition(multidevice);

            }

            start_communication = MPI_Wtime();
            start_kernel_communication = MPI_Wtime();
            for (int gpu = 0; gpu < multidevice->num_devices; ++gpu) {
                calclGetBorderFromDeviceToHost3D(
                            multidevice->device_models[gpu]);
            }
            // scambia bordi
            calclMultiDeviceUpdateHalos3D(
                        multidevice, multidevice->exchange_full_border);
            
            end_kernel_communication = MPI_Wtime();
            kernel_total_communication += end_kernel_communication - start_kernel_communication;
            
            end_communication = MPI_Wtime();
            total_communication += end_communication - start_communication;

            //----------MEASURE TIME---------
            start_communication = MPI_Wtime();

            handleBorderNodes(totalCommunicationTimeMPI,totalNumberofCommunicationMPI);

            end_communication =  MPI_Wtime();
            total_communication += end_communication - start_communication;
            //----------------------------------
            //  }

        }//STEPS

        end = MPI_Wtime();
        elapsedTime = end -start;

        start_communication = MPI_Wtime();
        //handleBorderNodes();
        calclMultiDeviceToNode(multidevice);
        end_communication =  MPI_Wtime();
        total_communication += end_communication - start_communication;

        MPI_Barrier(MPI_COMM_WORLD);

        double maxTotalCommunicationTimeMPI = 0;
        MPI_Reduce(&totalCommunicationTimeMPI, &maxTotalCommunicationTimeMPI, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

        double sumTotalCommunicationTimeMPI = 0;
        MPI_Reduce(&totalCommunicationTimeMPI, &sumTotalCommunicationTimeMPI, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

        double maxTotalCommunicationTime = 0;
        MPI_Reduce(&total_communication, &maxTotalCommunicationTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

       
        int gsize;
        MPI_Comm_size( MPI_COMM_WORLD, &gsize);
        if(rank == 0)
        {
            gatherElapsedTime = (double *)malloc(gsize*sizeof(double));
        }

        MPI_Gather(&elapsedTime, 1, MPI_DOUBLE, gatherElapsedTime, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);



        double max_kernel_total_communication=0;
        MPI_Reduce(&kernel_total_communication, &max_kernel_total_communication, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

        double max_kernel_total_computation=0;
        MPI_Reduce(&kernel_total_computation, &max_kernel_total_computation, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

        double max_total_kernel_streamcompaction_communication=0;
        MPI_Reduce(&total_kernel_streamcompaction_communication, &max_total_kernel_streamcompaction_communication, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        
        double max_total_kernel_streamcompaction_computation=0;
        MPI_Reduce(&total_kernel_streamcompaction_computation, &max_total_kernel_streamcompaction_computation, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

        if (rank == 0) {

            printf("\n");
            printf("-------------------------- Time info---------------------\n");
            printf("\n");
            double max=gatherElapsedTime[0];
            for(int n = 0; n < gsize; n++)
                if(gatherElapsedTime[n] > max)
                    max = gatherElapsedTime[n];
            
            printf("%f [s] - max elapsed time\n", max);

            for(int n = 0; n < gsize; n++)
                printf("Rank=%d, %f [s] - elapsed time\n", n, gatherElapsedTime[n]);

            printf("\n");
            printf("-------------------------- MPI Communication info---------------------\n");
            printf("\n");
            printf("Max Elapsed MPI communication : %f s\n", maxTotalCommunicationTimeMPI);
            printf("MPI Total Number of Communication per process: %d\n", totalNumberofCommunicationMPI);
            printf("MPI Total Communication Time per process: %f s\n", sumTotalCommunicationTimeMPI);

            double avgTotalCommunicationTime = sumTotalCommunicationTimeMPI/gsize;
            printf("Avg MPI Total Communication Time per process: %f s\n", avgTotalCommunicationTime);

            double avgMessageTime = avgTotalCommunicationTime/totalNumberofCommunicationMPI;
            printf("Avg MPI Communication Time per message: %f s\n", avgMessageTime);
            printf("\n");

            printf("\n");
            printf("-------------------------- Kenrel info---------------------\n");
            printf("\n");
            
            printf("max_kernel_total_communication : %f s\n", max_kernel_total_communication);
            printf("max_kernel_total_computation : %f s\n", max_kernel_total_computation);
            printf("max_total_kernel_streamcompaction_communication : %f s\n", max_total_kernel_streamcompaction_communication);
            printf("max_total_kernel_streamcompaction_computation : %f s\n", max_total_kernel_streamcompaction_computation);

        }
        MPI_Barrier(MPI_COMM_WORLD);
        _finalize();
        MPI_Finalize();
    }//run function


};

#endif CALCLMULTINODE3D_H_
