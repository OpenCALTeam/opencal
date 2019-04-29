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
extern "C"{
  #include <OpenCAL/cal2DBuffer.h>
  #include <OpenCAL-CL/calcl2D.h>
}
#include <OpenCAL-CL/calclMultiDevice2D.h>
//#include <OpenCAL-CL/calclMultiNode.h>


/******************************************************************************
 *              PRIVATE FUNCTIONS
 ******************************************************************************/
void calclGetSubstatesFromDevice2D(struct CALCLModel2D* calclmodel2D, const CALint workload, const CALint offset, const CALint borderSize) {

    CALCLqueue queue = calclmodel2D->queue;

    cl_int err;
    // every calclmodel get results from device to host and use an offset = sizeof(CALreal)*borderSize*calclmodel2D->columns
    // to insert the results to calclmodel2D->substateMapper.Substate_current_OUT because calclmodel2D->substateMapper.Substate_current_OUT
    // dimension is equal to rows*cols while  calclmodel2D->bufferCurrentRealSubstate dimension is equal to rows*cols+ sizeBoder * 2* cols

    //    for (int i = 0; i < calclmodel2D->fullSize*calclmodel2D->host_CA->sizeof_pQr_array; ++i) {
    //            if(i%calclmodel2D->columns == 0 && i !=0)
    //                printf("\n");
    //            printf("%f ", calclmodel2D->substateMapper.realSubstate_current_OUT[i]);
    //        }

    //        printf("\n");
    //        printf("\n");

    //        printf("\n");
    //        printf("\n");
    //        printf("\n");
    //        printf("%d \n",calclmodel2D->host_CA->sizeof_pQr_array);
    //        printf("\n");



    if(calclmodel2D->host_CA->sizeof_pQr_array > 0) {
        err = clEnqueueReadBuffer(queue,
                                  calclmodel2D->bufferCurrentRealSubstate,
                                  CL_TRUE, //blocking call
                                  0,
                                  calclmodel2D->substateMapper.bufDIMreal,
                                  calclmodel2D->substateMapper.realSubstate_current_OUT,
                                  0,
                                  NULL,
                                  NULL);
        calclHandleError(err);
    }





    if(calclmodel2D->host_CA->sizeof_pQi_array > 0) {
        err = clEnqueueReadBuffer(queue,
                                  calclmodel2D->bufferCurrentIntSubstate,
                                  CL_TRUE,
                                  0,
                                  calclmodel2D->substateMapper.bufDIMint,
                                  calclmodel2D->substateMapper.intSubstate_current_OUT,
                                  0,
                                  NULL,
                                  NULL);
        calclHandleError(err);
    }

    if(calclmodel2D->host_CA->sizeof_pQb_array > 0) {
        err = clEnqueueReadBuffer(queue,
                                  calclmodel2D->bufferCurrentByteSubstate,
                                  CL_TRUE,
                                  0,
                                  calclmodel2D->substateMapper.bufDIMbyte,
                                  calclmodel2D->substateMapper.byteSubstate_current_OUT,
                                  0,
                                  NULL,
                                  NULL);
        calclHandleError(err);
    }

}

void calclMultiDeviceMapperToSubstates2D(struct CALModel2D * host_CA, CALCLSubstateMapper * mapper,const size_t realSize, const CALint offset, int borderSize) {

    int ssNum_r = host_CA->sizeof_pQr_array;
    int ssNum_i = host_CA->sizeof_pQi_array;
    int ssNum_b = host_CA->sizeof_pQb_array;

    long int outIndex = borderSize * host_CA->columns;

    int i;
    unsigned int j;

    for (i = 0; i < ssNum_r; i++) {
        for (j = 0; j < realSize; j++)
            host_CA->pQr_array[i]->current[j+offset*host_CA->columns] = mapper->realSubstate_current_OUT[outIndex++];
        outIndex=outIndex+2*borderSize* host_CA->columns;
    }

    outIndex = borderSize * host_CA->columns;

    for (i = 0; i < ssNum_i; i++) {
        for (j = 0; j < realSize; j++)
            host_CA->pQi_array[i]->current[j+offset*host_CA->columns] = mapper->intSubstate_current_OUT[outIndex++];
        outIndex=outIndex+2*borderSize* host_CA->columns;
    }

    outIndex = borderSize * host_CA->columns;

    for (i = 0; i < ssNum_b; i++) {
        for (j = 0; j < realSize; j++)
            host_CA->pQb_array[i]->current[j+offset*host_CA->columns] = mapper->byteSubstate_current_OUT[outIndex++];
        outIndex=outIndex+2*borderSize* host_CA->columns;
    }

}

int vector_search_char(vector *v,const char * search) {

    for (int i = 0; i < v->total; i++) {
        struct kernelID * tmp = (kernelID*)vector_get(v,i);
        if( strcmp(tmp->name,search) == 0) {
            return tmp->index;
        }
    }

    return -1;

}

void calclCopyGhostb(struct CALModel2D * host_CA,CALbyte* tmpGhostCellsb,int offset,int workload,int borderSize) {

    size_t count = 0;
    if(host_CA->T == CAL_SPACE_TOROIDAL || offset-1 >= 0) {
        for (int i = 0; i < host_CA->sizeof_pQb_array; ++i)
            for (int b = 0; b < borderSize; ++b)
                for (int j = 0; j < host_CA->columns; ++j)
                    tmpGhostCellsb[count++] =  calGet2Db(host_CA,host_CA->pQb_array[i],((offset+b)+host_CA->rows)%host_CA->rows,j);

    } else {
        count += host_CA->sizeof_pQb_array*host_CA->columns;
    }

    int lastRow = workload+offset-borderSize;
    if(host_CA->T == CAL_SPACE_TOROIDAL || lastRow < host_CA->rows) {
        for (int i = 0; i < host_CA->sizeof_pQb_array; ++i)
            for (int b = 0; b < borderSize; ++b)
                for (int j = 0; j < host_CA->columns; ++j)
                    tmpGhostCellsb[count++] =  calGet2Db(host_CA,host_CA->pQb_array[i],((lastRow+b)+host_CA->rows)%host_CA->rows,j);

    }
}

void calclCopyGhosti(struct CALModel2D * host_CA,CALint* tmpGhostCellsi, int offset, int workload, int borderSize) {

    size_t count = 0;
    if(host_CA->T == CAL_SPACE_TOROIDAL || offset-1 >= 0) {
        for (int i = 0; i < host_CA->sizeof_pQi_array; ++i)
            for (int b = 0; b < borderSize; ++b)
                for (int j = 0; j < host_CA->columns; ++j)
                    tmpGhostCellsi[count++] =  calGet2Di(host_CA,host_CA->pQi_array[i],((offset+b)+host_CA->rows)%host_CA->rows,j);
    } else {
        count += host_CA->sizeof_pQi_array*host_CA->columns;
    }

    int lastRow = workload+offset-borderSize;
    if(host_CA->T == CAL_SPACE_TOROIDAL || lastRow < host_CA->rows) {
        for (int i = 0; i < host_CA->sizeof_pQi_array; ++i)
            for (int b = 0; b < borderSize; ++b)
                for (int j = 0; j < host_CA->columns; ++j)
                    tmpGhostCellsi[count++] =  calGet2Di(host_CA,host_CA->pQi_array[i],((lastRow+b)+host_CA->rows)%host_CA->rows,j);

    }
}

void calclCopyGhostr(struct CALModel2D * host_CA,CALreal* tmpGhostCellsr,int offset,int workload, int borderSize) {

    size_t count = 0;
    if(host_CA->T == CAL_SPACE_TOROIDAL || offset-1 >= 0) {
        for (int i = 0; i < host_CA->sizeof_pQr_array; ++i)
            for (int b = 0; b < borderSize; ++b)
                for (int j = 0; j < host_CA->columns; ++j)
                    tmpGhostCellsr[count++] =  calGet2Dr(host_CA,host_CA->pQr_array[i],((offset+b)+host_CA->rows)%host_CA->rows,j);

    } else {
        count += host_CA->sizeof_pQr_array*host_CA->columns;  //times bordeSize!!!!!
    }

    int lastRow = workload+offset-borderSize;
    if(host_CA->T == CAL_SPACE_TOROIDAL || lastRow < host_CA->rows) {
        for (int i = 0; i < host_CA->sizeof_pQr_array; ++i)
            for (int b = 0; b < borderSize; ++b)
                for (int j = 0; j < host_CA->columns; ++j)
                    tmpGhostCellsr[count++] =  calGet2Dr(host_CA,host_CA->pQr_array[i],((lastRow+b)+host_CA->rows)%host_CA->rows,j);

    }
}

void calclMultiDeviceSetKernelArg2D(struct CALCLMultiDevice2D * multidevice,const char * kernel, cl_uint arg_index, size_t arg_size, const void *arg_value) {
    int index = vector_search_char(&multidevice->kernelsID,kernel);
    assert(index != -1);

    for (int gpu = 0; gpu < multidevice->num_devices; ++gpu) {
        //CALCLkernel * k = &multidevice->device_models[gpu]->elementaryProcesses[index];
        clSetKernelArg(multidevice->device_models[gpu]->elementaryProcesses[index], MODEL_ARGS_NUM + arg_index, arg_size, arg_value);
    }
}

void calclMultiDeviceAddSteeringFunc2D(struct CALCLMultiDevice2D* multidevice, char* kernelName) {


    struct kernelID * kernel = (kernelID*)malloc(sizeof(struct kernelID));
    kernel->index = vector_total(&multidevice->kernelsID);
    memset(kernel->name,'\0',sizeof(kernel->name));
    strcpy(kernel->name,kernelName);

    VECTOR_ADD(multidevice->kernelsID, kernel);

    for (int i = 0; i < multidevice->num_devices; i++) {

        CALCLprogram p=multidevice->programs[i];
        CALCLkernel kernel = calclGetKernelFromProgram(p,kernelName);
        calclAddSteeringFunc2D(multidevice->device_models[i],kernel);
    }
}

/******************************************************************************
 *              PUBLIC MULTIDEVICE FUNCTIONS
 ******************************************************************************/

void calclSetNumDevice(struct CALCLMultiDevice2D* multidevice, const CALint _num_devices) {
    multidevice->num_devices = _num_devices;
    multidevice->devices = (CALCLdevice*)malloc(sizeof(CALCLdevice)*multidevice->num_devices);
    multidevice->programs = (CALCLprogram*)malloc(sizeof(CALCLprogram)*multidevice->num_devices);
    multidevice->workloads = (CALint*)malloc(sizeof(CALint)*multidevice->num_devices);
    multidevice->device_models = (struct CALCLModel2D**)malloc(sizeof(struct CALCLModel2D*)*multidevice->num_devices);
    multidevice->pos_device = 0;
}

void calclAddDevice(struct CALCLMultiDevice2D* multidevice,const CALCLdevice device, const CALint workload) {
    multidevice->devices[multidevice->pos_device] = device;
    multidevice->workloads[multidevice->pos_device] = workload;
    multidevice->pos_device++;
}

int calclCheckWorkload(struct CALCLMultiDevice2D* multidevice) {
    int tmpsum=0;
    for (int i = 0; i < multidevice->num_devices; ++i) {
        tmpsum +=multidevice->workloads[i];
    }
    return tmpsum;
}

void calclMultiDeviceUpdateHalos2D(struct CALCLMultiDevice2D* multidevice) {

//se il bordo da scmabiare ha raggio zero non ci sta bisogno di fare alcuno scambio quindi semplicemente ritorno
//assumiamo che tutti abbiano lo stesso raggio, quindi semplicemente prendo bordersize dal modello zero
  //if(!multidevice->device_models[0]->borderSize)
 //   return;

    cl_int err;

    for (int gpu = 0; gpu < multidevice->num_devices; ++gpu) {

        struct CALCLModel2D * calclmodel2D = multidevice->device_models[gpu];
        struct CALCLModel2D * calclmodel2DPrev = NULL;
        const int gpuP = ((gpu-1)+multidevice->num_devices)%multidevice->num_devices;
        const int gpuN = ((gpu+1)+multidevice->num_devices)%multidevice->num_devices;

        if(calclmodel2D->host_CA->T == CAL_SPACE_TOROIDAL || ((gpu-1) >= 0) ) {

            calclmodel2DPrev = multidevice->device_models[gpuP];
        }


        struct CALCLModel2D * calclmodel2DNext = NULL;
        if(calclmodel2D->host_CA->T == CAL_SPACE_TOROIDAL || ((gpu + 1) < multidevice->num_devices) ) {
            calclmodel2DNext = multidevice->device_models[gpuN];
        }



        int dim = calclmodel2D->fullSize;


        const int sizeBorder = calclmodel2D->borderSize*calclmodel2D->columns;

        int numSubstate = calclmodel2D->host_CA->sizeof_pQr_array;
        for (int i = 0; i < numSubstate; ++i) {

            if(calclmodel2DPrev != NULL) {
                err = clEnqueueWriteBuffer(calclmodel2D->queue,
                                           calclmodel2D->bufferCurrentRealSubstate,
                                           CL_TRUE,
                                           (i*dim)*sizeof(CALreal),
                                           sizeof(CALreal)*sizeBorder,
                                           calclmodel2DPrev->borderMapper.realBorder_OUT +(numSubstate*sizeBorder) + i * sizeBorder,
                                           0,
                                           NULL,
                                           NULL);
                calclHandleError(err);
            }

            if(calclmodel2DNext != NULL) {
                err = clEnqueueWriteBuffer(calclmodel2D->queue,
                                           calclmodel2D->bufferCurrentRealSubstate,
                                           CL_TRUE,
                                           (i * dim + (dim - sizeBorder) )*sizeof(CALreal),
                                           sizeof(CALreal)*sizeBorder,
                                           calclmodel2DNext->borderMapper.realBorder_OUT + i * sizeBorder,
                                           0,
                                           NULL,
                                           NULL);
                calclHandleError(err);
            }


        }

        numSubstate = calclmodel2D->host_CA->sizeof_pQi_array;


        for (int i = 0; i < numSubstate; ++i) {


            if(calclmodel2DPrev != NULL) {
                err = clEnqueueWriteBuffer(calclmodel2D->queue,
                                           calclmodel2D->bufferCurrentIntSubstate,
                                           CL_TRUE,
                                           (i*dim)*sizeof(CALint),
                                           sizeof(CALint)*sizeBorder,
                                           calclmodel2DPrev->borderMapper.intBorder_OUT +(numSubstate*sizeBorder) + i * sizeBorder,
                                           0,
                                           NULL,
                                           NULL);
                calclHandleError(err);
            }
            if(calclmodel2DNext != NULL) {
                err = clEnqueueWriteBuffer(calclmodel2D->queue,
                                           calclmodel2D->bufferCurrentIntSubstate,
                                           CL_TRUE,
                                           (i * dim + (dim - sizeBorder) )*sizeof(CALint),
                                           sizeof(CALint)*sizeBorder,
                                           calclmodel2DNext->borderMapper.intBorder_OUT + i * sizeBorder,
                                           0,
                                           NULL,
                                           NULL);
                calclHandleError(err);
            }

        }


        numSubstate = calclmodel2D->host_CA->sizeof_pQb_array;
        for (int i = 0; i < numSubstate; ++i) {

            if(calclmodel2DPrev != NULL) {
                err = clEnqueueWriteBuffer(calclmodel2D->queue,
                                           calclmodel2D->bufferCurrentByteSubstate,
                                           CL_TRUE,
                                           (i*dim)*sizeof(CALbyte),
                                           sizeof(CALbyte)*sizeBorder,
                                           calclmodel2DPrev->borderMapper.byteBorder_OUT +(numSubstate*sizeBorder) + i * sizeBorder,
                                           0,
                                           NULL,
                                           NULL);
                calclHandleError(err);
            }
            if(calclmodel2DNext != NULL) {
                err = clEnqueueWriteBuffer(calclmodel2D->queue,
                                           calclmodel2D->bufferCurrentByteSubstate,
                                           CL_TRUE,
                                           (i * dim + (dim - sizeBorder) )*sizeof(CALbyte),
                                           sizeof(CALbyte)*sizeBorder,
                                           calclmodel2DNext->borderMapper.byteBorder_OUT + i * sizeBorder,
                                           0,
                                           NULL,
                                           NULL);
                calclHandleError(err);
            }

        }

    }


}

void calclMultiDeviceUpdateHalos2D(struct CALCLMultiDevice2D* multidevice,const CALbyte exchange_full_border) {

  cl_int err;

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

    int numSubstate = calclmodel2D->host_CA->sizeof_pQr_array;
    for (int i = 0; i < numSubstate; ++i) {

      if (calclmodel2DPrev != NULL && (exchange_full_border || gpu != 0)) {
        err = clEnqueueWriteBuffer(
            calclmodel2D->queue, calclmodel2D->bufferCurrentRealSubstate,
            CL_TRUE, (i * dim) * sizeof(CALreal), sizeof(CALreal) * sizeBorder,
            calclmodel2DPrev->borderMapper.realBorder_OUT +
                (numSubstate * sizeBorder) + i * sizeBorder,
            0, NULL, NULL);
        calclHandleError(err);
      }

      if (calclmodel2DNext != NULL &&
          (exchange_full_border || gpu != multidevice->num_devices - 1)) {
        err = clEnqueueWriteBuffer(
            calclmodel2D->queue, calclmodel2D->bufferCurrentRealSubstate,
            CL_TRUE, (i * dim + (dim - sizeBorder)) * sizeof(CALreal),
            sizeof(CALreal) * sizeBorder,
            calclmodel2DNext->borderMapper.realBorder_OUT + i * sizeBorder, 0,
            NULL, NULL);
        calclHandleError(err);
      }
    }

    numSubstate = calclmodel2D->host_CA->sizeof_pQi_array;

    for (int i = 0; i < numSubstate; ++i) {

      if (calclmodel2DPrev != NULL && (exchange_full_border || gpu != 0)) {
        err = clEnqueueWriteBuffer(
            calclmodel2D->queue, calclmodel2D->bufferCurrentIntSubstate,
            CL_TRUE, (i * dim) * sizeof(CALint), sizeof(CALint) * sizeBorder,
            calclmodel2DPrev->borderMapper.intBorder_OUT +
                (numSubstate * sizeBorder) + i * sizeBorder,
            0, NULL, NULL);
        calclHandleError(err);
      }
      if (calclmodel2DNext != NULL &&
          (exchange_full_border || gpu != multidevice->num_devices - 1)) {
        err = clEnqueueWriteBuffer(
            calclmodel2D->queue, calclmodel2D->bufferCurrentIntSubstate,
            CL_TRUE, (i * dim + (dim - sizeBorder)) * sizeof(CALint),
            sizeof(CALint) * sizeBorder,
            calclmodel2DNext->borderMapper.intBorder_OUT + i * sizeBorder, 0,
            NULL, NULL);
        calclHandleError(err);
      }
    }

    numSubstate = calclmodel2D->host_CA->sizeof_pQb_array;
    for (int i = 0; i < numSubstate; ++i) {

      if (calclmodel2DPrev != NULL && (exchange_full_border || gpu != 0)) {
        err = clEnqueueWriteBuffer(
            calclmodel2D->queue, calclmodel2D->bufferCurrentByteSubstate,
            CL_TRUE, (i * dim) * sizeof(CALbyte), sizeof(CALbyte) * sizeBorder,
            calclmodel2DPrev->borderMapper.byteBorder_OUT +
                (numSubstate * sizeBorder) + i * sizeBorder,
            0, NULL, NULL);
        calclHandleError(err);
      }
      if (calclmodel2DNext != NULL &&
          (exchange_full_border || gpu != multidevice->num_devices - 1)) {
        err = clEnqueueWriteBuffer(
            calclmodel2D->queue, calclmodel2D->bufferCurrentByteSubstate,
            CL_TRUE, (i * dim + (dim - sizeBorder)) * sizeof(CALbyte),
            sizeof(CALbyte) * sizeBorder,
            calclmodel2DNext->borderMapper.byteBorder_OUT + i * sizeBorder, 0,
            NULL, NULL);
        calclHandleError(err);
      }
    }

  }//GPUs
}

void calclMultiDeviceGetHalos(struct CALCLMultiDevice2D* multidevice, int offset, int gpu) {
    struct CALCLModel2D * calclmodel2D = multidevice->device_models[gpu];

    calclCopyGhostb(calclmodel2D->host_CA, calclmodel2D->borderMapper.byteBorder_OUT, offset, multidevice->workloads[gpu], calclmodel2D->borderSize);

    calclCopyGhosti(calclmodel2D->host_CA, calclmodel2D->borderMapper.intBorder_OUT, offset, multidevice->workloads[gpu], calclmodel2D->borderSize);

    calclCopyGhostr(calclmodel2D->host_CA, calclmodel2D->borderMapper.realBorder_OUT, offset, multidevice->workloads[gpu], calclmodel2D->borderSize);
}

void calclMultiDeviceCADef2D(struct CALCLMultiDevice2D* multidevice,
                        struct CALModel2D* host_CA, char* kernel_src,
                        char* kernel_inc, const CALint borderSize,
                        const std::vector<Device>& devices) {
  
  assert(host_CA->rows == calclCheckWorkload(multidevice));
  multidevice->context =
      calclCreateContext(multidevice->devices, multidevice->num_devices);



  //manual initialization vector::resize causes crash
  multidevice->singleStepThreadNums = (size_t**)calloc(multidevice->num_devices,sizeof(size_t*));
  //for (int i = 0; i < multidevice->num_devices; ++i) 
    //multidevice->singleStepThreadNums[i]=0;  

  for (int i = 0; i < multidevice->num_devices; ++i) {
    const cl_uint offset = devices[i].offset;
    multidevice->programs[i] = calclLoadProgram2D(
        multidevice->context, multidevice->devices[i], kernel_src, kernel_inc);

    multidevice->device_models[i] = calclCADef2D(
        host_CA, multidevice->context, multidevice->programs[i], multidevice->devices[i],
        multidevice->workloads[i], offset, borderSize);  // offset

    calclMultiDeviceGetHalos(multidevice, offset, i);

    cl_int err;
    setParametersReduction(err, multidevice->device_models[i]);
  }
//considera una barriera qui
    calclMultiDeviceUpdateHalos2D(multidevice, multidevice->exchange_full_border);

    vector_init(&multidevice->kernelsID);



}

void calclMultiDeviceToNode(struct CALCLMultiDevice2D* multidevice) {

    for (int gpu = 0; gpu < multidevice->num_devices; ++gpu) {
        calclGetSubstatesFromDevice2D(multidevice->device_models[gpu],
                                               multidevice->workloads[gpu],
                                               multidevice->device_models[gpu]->offset,
                                               multidevice->device_models[gpu]->borderSize);
        calclMultiDeviceMapperToSubstates2D(multidevice->device_models[gpu]->host_CA,
                                         &multidevice->device_models[gpu]->substateMapper,
                                         multidevice->device_models[gpu]->realSize,
                                         multidevice->device_models[gpu]->offset,
                                         multidevice->device_models[gpu]->borderSize);

    }

}

size_t* computekernelLaunchParams(struct CALCLMultiDevice2D* multidevice, const int gpu,int *dim) {
   size_t* singleStepThreadNum;
    if (multidevice->device_models[0]->opt == CAL_NO_OPT) {
        singleStepThreadNum = (size_t*) malloc(sizeof(size_t) * 2);
        (singleStepThreadNum)[0] = multidevice->device_models[gpu]->rows;
        (singleStepThreadNum)[1] = multidevice->device_models[gpu]->columns;
        *dim = 2;
    } else {
        singleStepThreadNum = (size_t*) malloc(sizeof(size_t));
        singleStepThreadNum[0] = multidevice->device_models[gpu]->num_active_cells; 
        *dim = 1;
    }
    return singleStepThreadNum;
}

void calclMultiDeviceSetWorkGroupSize2D(struct CALCLMultiDevice2D* multidevice, int m, int n) {
    for (int gpu = 0; gpu < multidevice->num_devices; ++gpu) {
        struct CALCLModel2D* calclmodel2D = multidevice->device_models[gpu];
        calclSetWorkGroupDimensions2D(calclmodel2D, m, n);

    }
}

void calclMultiDeviceRun2D(struct CALCLMultiDevice2D* multidevice, CALint init_step, CALint final_step) {

    int steps = init_step;

    size_t * threadNumMax = (size_t*) malloc(sizeof(size_t) * 2);
    threadNumMax[0] = multidevice->device_models[0]->rows;
    threadNumMax[1] = multidevice->device_models[0]->columns;
    size_t * singleStepThreadNum;
    int dimNum;

    if (multidevice->device_models[0]->opt == CAL_NO_OPT) {
        singleStepThreadNum = (size_t*) malloc(sizeof(size_t) * 2);
        singleStepThreadNum[0] = threadNumMax[0];
        singleStepThreadNum[1] = threadNumMax[1];
        dimNum = 2;
    } else {
        singleStepThreadNum = (size_t*) malloc(sizeof(size_t));
        singleStepThreadNum[0] = multidevice->device_models[0]->host_CA->A->size_current;
        dimNum = 1;
    }


    while (steps <= (int) final_step || final_step == CAL_RUN_LOOP) {


        //calcola dimNum e ThreadsNum

        for (int j = 0; j < multidevice->device_models[0]->elementaryProcessesNum; j++) {

            for (int gpu = 0; gpu < multidevice->num_devices; ++gpu) {
                struct CALCLModel2D * calclmodel2D = multidevice->device_models[gpu];
                CALbyte activeCells = calclmodel2D->opt == CAL_OPT_ACTIVE_CELLS_NAIVE;
                cl_int err;

                setParametersReduction(err, calclmodel2D);

                if (calclmodel2D->kernelInitSubstates != NULL)
                    calclSetReductionParameters2D(calclmodel2D, calclmodel2D->kernelInitSubstates);
                if (calclmodel2D->kernelStopCondition != NULL)
                    calclSetReductionParameters2D(calclmodel2D, calclmodel2D->kernelStopCondition);
                if (calclmodel2D->kernelSteering != NULL)
                    calclSetReductionParameters2D(calclmodel2D, calclmodel2D->kernelSteering);

                int i = 0;

                for (i = 0; i < calclmodel2D->elementaryProcessesNum; i++) {
                    calclSetReductionParameters2D(calclmodel2D, calclmodel2D->elementaryProcesses[i]);
                }

                if (activeCells == CAL_TRUE) {
                    for (j = 0; j < calclmodel2D->elementaryProcessesNum; j++) {
                        if(singleStepThreadNum[0] > 0)
                            calclKernelCall2D(calclmodel2D, calclmodel2D->elementaryProcesses[j], dimNum, singleStepThreadNum,
                                              NULL,NULL);
                        if(singleStepThreadNum[0] > 0) {
                            calclComputeStreamCompaction2D(calclmodel2D);
                            calclResizeThreadsNum2D(calclmodel2D, singleStepThreadNum);
                        }
                        if(singleStepThreadNum[0] > 0)
                            calclKernelCall2D(calclmodel2D, calclmodel2D->kernelUpdateSubstate, dimNum, singleStepThreadNum, NULL,NULL);
                        calclGetBorderFromDeviceToHost2D(calclmodel2D);
                    }

                    calclExecuteReduction2D(calclmodel2D, calclmodel2D->roundedDimensions);

                } else {
                    for (j = 0; j < calclmodel2D->elementaryProcessesNum; j++) {
                        calclKernelCall2D(calclmodel2D, calclmodel2D->elementaryProcesses[j], dimNum, singleStepThreadNum,
                                          calclmodel2D->workGroupDimensions,NULL);
                        copySubstatesBuffers2D(calclmodel2D);
                        calclGetBorderFromDeviceToHost2D(calclmodel2D);


                    }
                    calclExecuteReduction2D(calclmodel2D, calclmodel2D->roundedDimensions);
                    //copySubstatesBuffers2D(calclmodel2D);

                }

                //                calclKernelCall2D(calclmodel2D, calclmodel2D->elementaryProcesses[j], dimNum, singleStepThreadNum,
                //                                  NULL, NULL);



            }

            // barrier tutte hanno finito
            for (int gpu = 0; gpu < multidevice->num_devices; ++gpu) {
                clFinish(multidevice->device_models[gpu]->queue);
            }

            for (int gpu = 0; gpu < multidevice->num_devices; ++gpu) {
                calclGetBorderFromDeviceToHost2D(multidevice->device_models[gpu]);
            }

            //scambia bordi
            calclMultiDeviceUpdateHalos2D(multidevice);

            for (int gpu = 0; gpu < multidevice->num_devices; ++gpu) {
                struct CALCLModel2D * calclmodel2D = multidevice->device_models[gpu];
                CALbyte activeCells = calclmodel2D->opt == CAL_OPT_ACTIVE_CELLS_NAIVE;
                if (activeCells == CAL_TRUE) {
                    if (calclmodel2D->kernelSteering != NULL) {
                        calclKernelCall2D(calclmodel2D, calclmodel2D->kernelSteering, dimNum, singleStepThreadNum, NULL,NULL);
                        calclKernelCall2D(calclmodel2D, calclmodel2D->kernelUpdateSubstate, dimNum, singleStepThreadNum, NULL,NULL);
                    }
                } else
                {
                    if (calclmodel2D->kernelSteering != NULL) {
                        calclKernelCall2D(calclmodel2D, calclmodel2D->kernelSteering, dimNum, singleStepThreadNum, NULL, NULL);
                        copySubstatesBuffers2D(calclmodel2D);
                    }
                }
            }

            for (int gpu = 0; gpu < multidevice->num_devices; ++gpu) {
                clFinish(multidevice->device_models[gpu]->queue);
            }

            for (int gpu = 0; gpu < multidevice->num_devices; ++gpu) {
                calclGetBorderFromDeviceToHost2D(multidevice->device_models[gpu]);
            }

            //scambia bordi
            calclMultiDeviceUpdateHalos2D(multidevice);


        }//for elementary process

        steps++;

    }// while


    for (int gpu = 0; gpu < multidevice->num_devices; ++gpu) {
        calclGetSubstatesFromDevice2D(multidevice->device_models[gpu],
                                               multidevice->workloads[gpu],
                                               multidevice->device_models[gpu]->offset,
                                               multidevice->device_models[gpu]->borderSize);
        calclMultiDeviceMapperToSubstates2D(multidevice->device_models[gpu]->host_CA,
                                         &multidevice->device_models[gpu]->substateMapper,
                                         multidevice->device_models[gpu]->realSize,
                                         multidevice->device_models[gpu]->offset,
                                         multidevice->device_models[gpu]->borderSize);

    }






}

void calclMultiDeviceAddElementaryProcess2D(struct CALCLMultiDevice2D* multidevice, char * kernelName) {
    struct kernelID * kernel = (kernelID*)malloc(sizeof(struct kernelID));
    kernel->index = vector_total(&multidevice->kernelsID);
    memset(kernel->name,'\0',sizeof(kernel->name));
    strcpy(kernel->name,kernelName);

    VECTOR_ADD(multidevice->kernelsID, kernel);

    for (int i = 0; i < multidevice->num_devices; i++) {

        CALCLprogram p=multidevice->programs[i];
        CALCLkernel kernel = calclGetKernelFromProgram(p,kernelName);
        calclAddElementaryProcess2D(multidevice->device_models[i],kernel);
    }
}

void calclMultiDeviceFinalize2D(struct CALCLMultiDevice2D* multidevice) {

    free(multidevice->devices);
    free(multidevice->programs);
    // free(multidevice->kernel_events);
    for (int i = 0; i < multidevice->num_devices; ++i) {
        calclFinalize2D(multidevice->device_models[i]);
    }

    vector_free(&multidevice->kernelsID);
    free(multidevice);
}


