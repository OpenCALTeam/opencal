/*
 * Copyright (c) 2016 OpenCALTeam (https://github.com/OpenCALTeam),
 * Telesio Research Group,
 * Department of Mathematics and Computer Science,
 * University of Calabria, Italy.
 *
 * This file is part of OpenCAL (Open Computing Abstraction Layer).
 *
 * OpenCAL is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3 of
 * the License, or (at your option) any later version.
 *
 * OpenCAL is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with OpenCAL. If not, see <http://www.gnu.org/licenses/>.
 */

/*! \file calclMultiGPU2D.h
 *\brief calclMultiGPU2D contains structures and functions that allow to run parallel CA simulation using Opencl and OpenCAL.
 *
 *	calclMultiGPU2D contains structures that allows easily to transfer data of a CALModel2D instance from host to GPUs.
 *	It's possible to setup a CA simulation by only defining kernels for elementary processes, and optionally
 *	initialization, steering and stop condition. Moreover, user can avoid to use the simulation cycle provided
 *	by the library and define his own simulation cycle.
 */

#ifndef CALCLMULTIGPU_H_
#define CALCLMULTIGPU_H_

extern "C"{
#include <OpenCAL-CL/calcl2D.h>
}

#include <OpenCAL-CL/calclCluster.h>

typedef struct kernelID{
    char name[MAX_FNAME_SIZE];
    int index;
}kernelID;


template <class F_INIT,class F_FINALIZE>
class MultiNode;


struct CALCLMultiGPU{

    struct CALCLModel2D ** device_models;
    CALint num_devices;
    CALCLcontext context;
    CALCLdevice * devices;
    CALCLprogram * programs;
    CALint * workloads;
    cl_event * kernel_events;
    CALint pos_device;
    CALbyte exchange_full_border;

    C_vector kernelsID;

    size_t** singleStepThreadNums;

};


void calclSetNumDevice(struct CALCLMultiGPU* multigpu, const CALint _num_devices);

void calclAddDevice(struct CALCLMultiGPU* multigpu,const CALCLdevice device, const CALint workload);

int calclCheckWorkload(struct CALCLMultiGPU* multigpu);

void calclMultiGPUDef2D(struct CALCLMultiGPU* mulitgpu,struct CALModel2D *host_CA ,char* kernel_src,char* kernel_inc, const CALint borderSize , const std::vector<Device>& devices, const CALbyte exchange_full_border);

void calclMultiGPURun2D(struct CALCLMultiGPU* multigpu, CALint init_step, CALint final_step);

void calclAddElementaryProcessMultiGPU2D(struct CALCLMultiGPU* multigpu, char * kernelName);

void calclAddSteeringFuncMultiGPU2D(struct CALCLMultiGPU* multigpu, char * kernelName);

void calclMultiGPUFinalize(struct CALCLMultiGPU* mulitgpu);


//void computekernelLaunchParams(struct CALCLMultiGPU* multigpu,  size_t*&  singleStepThreadNum, int *dim);
size_t* computekernelLaunchParams(struct CALCLMultiGPU* , const int,int*);

void calclSetKernelArgMultiGPU2D(struct CALCLMultiGPU * multigpu,const char * kernel, cl_uint arg_index, size_t arg_size, const void *arg_value);


void calclDevicesToNode(struct CALCLMultiGPU* multigpu);

void calclMultiGPUHandleBordersMultiNode(struct CALCLMultiGPU* multigpu,const CALbyte exchange_full_border);

//void calclStreamCompactionMulti(struct CALCLMultiGPU* multigpu);

void calclSetWorkGroupDimensionsMultiGPU(struct CALCLMultiGPU* multigpu, int m, int n);


template <class F_INIT,class F_FINALIZE>
void calclStreamCompactionMulti(struct CALCLMultiGPU* multigpu,MultiNode<F_INIT,F_FINALIZE> * mn){

//----------MEASURE TIME---------
gettimeofday(&(mn->start_comm), NULL);


  cl_int err;
  // Read from substates and set flags borders
  //printf(" before read bufferActiveCellsFlags \n");

  for (int gpu = 0; gpu < multigpu->num_devices; ++gpu) {
    struct CALCLModel2D* calclmodel2D = multigpu->device_models[gpu];
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
  for (int gpu = 0; gpu < multigpu->num_devices; ++gpu) {

    struct CALCLModel2D* calclmodel2D = multigpu->device_models[gpu];
    struct CALCLModel2D* calclmodel2DPrev = NULL;
    const int gpuP =
        ((gpu - 1) + multigpu->num_devices) % multigpu->num_devices;
    const int gpuN =
        ((gpu + 1) + multigpu->num_devices) % multigpu->num_devices;

    if (calclmodel2D->host_CA->T == CAL_SPACE_TOROIDAL || ((gpu - 1) >= 0)) {

      calclmodel2DPrev = multigpu->device_models[gpuP];
    }

    struct CALCLModel2D* calclmodel2DNext = NULL;
    if (calclmodel2D->host_CA->T == CAL_SPACE_TOROIDAL ||
        ((gpu + 1) < multigpu->num_devices)) {
      calclmodel2DNext = multigpu->device_models[gpuN];
    }

    int dim = calclmodel2D->fullSize;

    const int sizeBorder = calclmodel2D->borderSize * calclmodel2D->columns;

    const CALbyte activeCells = calclmodel2D->opt == CAL_OPT_ACTIVE_CELLS_NAIVE;
    if (activeCells == CAL_TRUE) {
      // copy border flags from GPUPrev and GPU next to a mergeflagsBorder
      if (calclmodel2DPrev != NULL &&
          (multigpu->exchange_full_border || gpu != 0)) {
        err = clEnqueueWriteBuffer(
            calclmodel2D->queue, calclmodel2D->borderMapper.mergeflagsBorder,
            CL_TRUE, 0, sizeof(CALbyte) * sizeBorder,
            calclmodel2DPrev->borderMapper.flagsBorder_OUT + sizeBorder, 0,
            NULL, NULL);

        calclHandleError(err);
      }
      if (calclmodel2DNext != NULL && (multigpu->exchange_full_border ||
                                       gpu != multigpu->num_devices - 1)) {
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
// 1) la prima metà di flagsNodeGhosts sulla prima meta di mergeflagsBorder della prima GPU del nodo
// 2) la seconda meta di flagsNodeGhosts sulla seconda metà di mergeflagsBorder dell'ultima GPU del nodo


// a questo punto mergeBufer contiene i bordi delle GPU adiacenti (anche se esse stanno su altri nodi)
// quindi il merging può essere fatto. Qui sotto funziona.
  for (int gpu = 0; gpu < multigpu->num_devices; ++gpu) {
    struct CALCLModel2D* calclmodel2D = multigpu->device_models[gpu];
    const CALbyte activeCells = calclmodel2D->opt == CAL_OPT_ACTIVE_CELLS_NAIVE;
    if (activeCells) {
      size_t singleNumThreadsMerge =
          calclmodel2D->borderSize * 2 * calclmodel2D->columns;
      
      //    printf("gpu=%d, before streamcompact --> %d\n", gpu, 
       //      multigpu->singleStepThreadNums[gpu][0]);
      calclKernelCall2D(calclmodel2D,  calclmodel2D->kernelMergeFlags, 1,
                        &(singleNumThreadsMerge), NULL, NULL);
      //printf("gpu=%d, launch kernelSetDiffFlags\n",gpu,multigpu->singleStepThreadNums[gpu][0]);
      //calclKernelCall2D(calclmodel2D, calclmodel2D->kernelSetDiffFlags, 1,
      //                  &(singleNumThreadsMerge), NULL, NULL);

      clFinish(multigpu->device_models[gpu]->queue);

      calclComputeStreamCompaction2D(calclmodel2D);
      calclResizeThreadsNum2D(calclmodel2D,
                              multigpu->singleStepThreadNums[gpu]);
    

     // printf("gpu=%d,after streamcompact --> %d\n", gpu, multigpu->singleStepThreadNums[gpu][0]);
    clFinish(multigpu->device_models[gpu]->queue);
    }
  }
}



template <class F_INIT,class F_FINALIZE>
void calcl_executeElementaryProcess(struct CALCLMultiGPU* multigpu,
                                    const int el_proc, int dimNum, MultiNode<F_INIT,F_FINALIZE> * mn) {


    
    for (int gpu = 0; gpu < multigpu->num_devices; ++gpu) {
        struct CALCLModel2D * calclmodel2D = multigpu->device_models[gpu];
        size_t* singleStepThreadNum = multigpu->singleStepThreadNums[gpu];

        cl_int err;


        CALbyte activeCells = calclmodel2D->opt == CAL_OPT_ACTIVE_CELLS_NAIVE;
        if (activeCells == CAL_TRUE) {

          if (singleStepThreadNum[0] > 0)
            calclKernelCall2D(calclmodel2D,
                              calclmodel2D->elementaryProcesses[el_proc],
                              dimNum, singleStepThreadNum, NULL, NULL);
//clFinish(multigpu->device_models[gpu]->queue);
         //printf("rank %d --> gpu=%d, before streamcompact el_proc num = %d --> %d\n",mn->rank,  gpu, el_proc, singleStepThreadNum[0]);

         // if (singleStepThreadNum[0] > 0) {
             /*calclComputeStreamCompaction2D(calclmodel2D);
            calclResizeThreadsNum2D(calclmodel2D, singleStepThreadNum);*/
            
            calclStreamCompactionMulti(multigpu,mn);
            //if(mn->rank ==1)
            //printf("rank %d --> gpu=%d, after streamcompact el_proc num = %d --> %d\n",mn->rank,  gpu, el_proc, singleStepThreadNum[0]);

            //printf("gpu=%d,after streamcompact el_proc num = %d --> %d \n", gpu,             el_proc, singleStepThreadNum[0]);
        //  }
        
//clFinish(multigpu->device_models[gpu]->queue);
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
//clFinish(multigpu->device_models[gpu]->queue);
    }  // GPUs
    //printf("\n");
}




#endif /* CALCLMULTIGPU_H_ */
