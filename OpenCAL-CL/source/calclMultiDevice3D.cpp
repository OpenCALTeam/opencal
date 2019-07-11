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
#include <OpenCAL/cal3DBuffer.h>
#include <OpenCAL-CL/calcl3D.h>
}
#include <OpenCAL-CL/calclMultiDevice3D.h>
//#include <OpenCAL-CL/calclMultiNode.h>


/******************************************************************************
 *              PRIVATE FUNCTIONS
 ******************************************************************************/
void calclGetSubstatesFromDevice3D(struct CALCLModel3D* calclmodel, const CALint workload, const CALint offset, const CALint borderSize) {

    CALCLqueue queue = calclmodel->queue;

    cl_int err;
    // every calclmodel get results from device to host and use an offset = sizeof(CALreal)*borderSize*calclmodel3D->columns
    // to insert the results to calclmodel3D->substateMapper.Substate_current_OUT because calclmodel3D->substateMapper.Substate_current_OUT
    // dimension is equal to rows*cols while  calclmodel3D->bufferCurrentRealSubstate dimension is equal to rows*cols+ sizeBoder * 2* cols



    if(calclmodel->host_CA->sizeof_pQr_array > 0) {
        err = clEnqueueReadBuffer(queue,
                                  calclmodel->bufferCurrentRealSubstate,
                                  CL_TRUE, //blocking call
                                  0,
                                  calclmodel->substateMapper.bufDIMreal,
                                  calclmodel->substateMapper.realSubstate_current_OUT,
                                  0,
                                  NULL,
                                  NULL);
        calclHandleError(err);
    }





    if(calclmodel->host_CA->sizeof_pQi_array > 0) {
        err = clEnqueueReadBuffer(queue,
                                  calclmodel->bufferCurrentIntSubstate,
                                  CL_TRUE,
                                  0,
                                  calclmodel->substateMapper.bufDIMint,
                                  calclmodel->substateMapper.intSubstate_current_OUT,
                                  0,
                                  NULL,
                                  NULL);
        calclHandleError(err);
    }

    if(calclmodel->host_CA->sizeof_pQb_array > 0) {
        err = clEnqueueReadBuffer(queue,
                                  calclmodel->bufferCurrentByteSubstate,
                                  CL_TRUE,
                                  0,
                                  calclmodel->substateMapper.bufDIMbyte,
                                  calclmodel->substateMapper.byteSubstate_current_OUT,
                                  0,
                                  NULL,
                                  NULL);
        calclHandleError(err);
    }


}

void calclMultiDeviceMapperToSubstates3D(struct CALModel3D * host_CA, CALCLSubstateMapper * mapper,const size_t realSize, const CALint offset, int borderSize) {

    int ssNum_r = host_CA->sizeof_pQr_array;
    int ssNum_i = host_CA->sizeof_pQi_array;
    int ssNum_b = host_CA->sizeof_pQb_array;

    int dimLayer =  host_CA->columns * host_CA->rows;
    long int outIndex = borderSize * dimLayer;

    int i;
    unsigned int j;

    for (i = 0; i < ssNum_r; i++) {
        for (j = 0; j < realSize; j++)
            host_CA->pQr_array[i]->current[j+offset*dimLayer] = mapper->realSubstate_current_OUT[outIndex++];
        outIndex=outIndex+2*borderSize*dimLayer;
    }

    outIndex = borderSize *dimLayer;

    for (i = 0; i < ssNum_i; i++) {
        for (j = 0; j < realSize; j++)
            host_CA->pQi_array[i]->current[j+offset*dimLayer] = mapper->intSubstate_current_OUT[outIndex++];
        outIndex=outIndex+2*borderSize*dimLayer;
    }

    outIndex = borderSize *dimLayer;

    for (i = 0; i < ssNum_b; i++) {
        for (j = 0; j < realSize; j++)
            host_CA->pQb_array[i]->current[j+offset*dimLayer] = mapper->byteSubstate_current_OUT[outIndex++];
        outIndex=outIndex+2*borderSize*dimLayer;
    }

}

int vector_search_char3D(vector *v,const char * search) {

    for (int i = 0; i < v->total; i++) {
        struct kernelID * tmp = (kernelID*)vector_get(v,i);
        if( strcmp(tmp->name,search) == 0) {
            return tmp->index;
        }
    }

    return -1;

}

void calclCopyGhostb(struct CALModel3D * host_CA,CALbyte* tmpGhostCellsb,int offset,int workload,int borderSize) {

    size_t count = 0;
    if(host_CA->T == CAL_SPACE_TOROIDAL || offset-1 >= 0) {
        for (int s = 0; s < host_CA->sizeof_pQb_array; ++s)
            for (int b = 0; b < borderSize; ++b)
                for (int i = 0; i < host_CA->rows; ++i)
                    for (int j = 0; j < host_CA->columns; ++j)
                        tmpGhostCellsb[count++] =  calGet3Db(host_CA,host_CA->pQb_array[s],i,j, ((offset+b)+host_CA->slices)%host_CA->slices);

    } else {
        count += host_CA->sizeof_pQb_array*host_CA->columns*host_CA->rows;
    }

    int lastRow = workload+offset-borderSize;
    if(host_CA->T == CAL_SPACE_TOROIDAL || lastRow < host_CA->slices) {
        for (int s = 0; s < host_CA->sizeof_pQb_array; ++s)
            for (int b = 0; b < borderSize; ++b)
                for (int i = 0; i < host_CA->rows; ++i)
                    for (int j = 0; j < host_CA->columns; ++j)
                        tmpGhostCellsb[count++] =  calGet3Db(host_CA,host_CA->pQb_array[s],i,j,((lastRow+b)+host_CA->slices)%host_CA->slices);

    }

}

void calclCopyGhosti(struct CALModel3D * host_CA,CALint* tmpGhostCellsi, int offset, int workload, int borderSize) {

    size_t count = 0;
    if(host_CA->T == CAL_SPACE_TOROIDAL || offset-1 >= 0) {
        for (int s = 0; s < host_CA->sizeof_pQi_array; ++s)
            for (int b = 0; b < borderSize; ++b)
                for (int i = 0; i < host_CA->rows; ++i)
                    for (int j = 0; j < host_CA->columns; ++j)
                        tmpGhostCellsi[count++] =  calGet3Di(host_CA,host_CA->pQi_array[s],i, j, ((offset+b)+host_CA->slices)%host_CA->slices);
    } else {
        count += host_CA->sizeof_pQi_array*host_CA->columns*host_CA->rows;
    }

    int lastRow = workload+offset-borderSize;
    if(host_CA->T == CAL_SPACE_TOROIDAL || lastRow < host_CA->rows) {
        for (int s = 0; s < host_CA->sizeof_pQi_array; ++s)
            for (int b = 0; b < borderSize; ++b)
                for (int i = 0; i < host_CA->rows; ++i)
                    for (int j = 0; j < host_CA->columns; ++j)
                        tmpGhostCellsi[count++] =  calGet3Di(host_CA,host_CA->pQi_array[s],i,j, ((lastRow+b)+host_CA->slices)%host_CA->slices);

    }
}

void calclCopyGhostr(struct CALModel3D * host_CA,CALreal* tmpGhostCellsr,int offset,int workload, int borderSize) {

    size_t count = 0;
    if(host_CA->T == CAL_SPACE_TOROIDAL || offset-1 >= 0) {
        for (int s = 0; s < host_CA->sizeof_pQr_array; ++s)
            for (int b = 0; b < borderSize; ++b)
                for (int i = 0; i < host_CA->rows; ++i)
                    for (int j = 0; j < host_CA->columns; ++j)
                        tmpGhostCellsr[count++] =  calGet3Dr(host_CA,host_CA->pQr_array[s],i,j, ((offset+b)+host_CA->slices)%host_CA->slices);

    } else {
        count += host_CA->sizeof_pQr_array*host_CA->columns*host_CA->rows;  //times bordeSize!!!!!
    }

    int lastRow = workload+offset-borderSize;
    if(host_CA->T == CAL_SPACE_TOROIDAL || lastRow < host_CA->rows) {
        for (int s = 0; s < host_CA->sizeof_pQr_array; ++s)
            for (int b = 0; b < borderSize; ++b)
                for (int i = 0; i < host_CA->rows; ++i)
                    for (int j = 0; j < host_CA->columns; ++j)
                        tmpGhostCellsr[count++] =  calGet3Dr(host_CA,host_CA->pQr_array[s],i,j,((lastRow+b)+host_CA->slices)%host_CA->slices);

    }
}





/******************************************************************************
 *              PUBLIC MULTIDEVICE FUNCTIONS
 ******************************************************************************/

void calclSetNumDevice(struct CALCLMultiDevice3D* multidevice, const CALint _num_devices) {
    multidevice->num_devices = _num_devices;
    multidevice->devices = (CALCLdevice*)malloc(sizeof(CALCLdevice)*multidevice->num_devices);
    multidevice->programs = (CALCLprogram*)malloc(sizeof(CALCLprogram)*multidevice->num_devices);
    multidevice->workloads = (CALint*)malloc(sizeof(CALint)*multidevice->num_devices);
    multidevice->device_models = (struct CALCLModel3D**)malloc(sizeof(struct CALCLModel3D*)*multidevice->num_devices);
    multidevice->pos_device = 0;
}

void calclAddDevice(struct CALCLMultiDevice3D* multidevice,const CALCLdevice device, const CALint workload) {
    multidevice->devices[multidevice->pos_device] = device;
    multidevice->workloads[multidevice->pos_device] = workload;
    multidevice->pos_device++;
}

int calclCheckWorkload(struct CALCLMultiDevice3D* multidevice) {
    int tmpsum=0;
    for (int i = 0; i < multidevice->num_devices; ++i) {
        tmpsum +=multidevice->workloads[i];
    }
    return tmpsum;
}

//void calclMultiDeviceUpdateHalos3D(struct CALCLMultiDevice3D* multidevice) {

//    //se il bordo da scmabiare ha raggio zero non ci sta bisogno di fare alcuno scambio quindi semplicemente ritorno
//    //assumiamo che tutti abbiano lo stesso raggio, quindi semplicemente prendo bordersize dal modello zero
//    //if(!multidevice->device_models[0]->borderSize)
//    //   return;

//    cl_int err;

//    for (int gpu = 0; gpu < multidevice->num_devices; ++gpu) {

//        struct CALCLModel3D * calclmodel3D = multidevice->device_models[gpu];
//        struct CALCLModel3D * calclmodel3DPrev = NULL;
//        const int gpuP = ((gpu-1)+multidevice->num_devices)%multidevice->num_devices;
//        const int gpuN = ((gpu+1)+multidevice->num_devices)%multidevice->num_devices;

//        if(calclmodel3D->host_CA->T == CAL_SPACE_TOROIDAL || ((gpu-1) >= 0) ) {

//            calclmodel3DPrev = multidevice->device_models[gpuP];
//        }


//        struct CALCLModel3D * calclmodel3DNext = NULL;
//        if(calclmodel3D->host_CA->T == CAL_SPACE_TOROIDAL || ((gpu + 1) < multidevice->num_devices) ) {
//            calclmodel3DNext = multidevice->device_models[gpuN];
//        }



//        int dim = calclmodel3D->fullSize;


//        const int sizeBorder = calclmodel3D->borderSize*calclmodel3D->columns*calclmodel3D->rows;

//        int numSubstate = calclmodel3D->host_CA->sizeof_pQr_array;
//        for (int i = 0; i < numSubstate; ++i) {

//            if(calclmodel3DPrev != NULL) {
//                err = clEnqueueWriteBuffer(calclmodel3D->queue,
//                                           calclmodel3D->bufferCurrentRealSubstate,
//                                           CL_TRUE,
//                                           (i*dim)*sizeof(CALreal),
//                                           sizeof(CALreal)*sizeBorder,
//                                           calclmodel3DPrev->borderMapper.realBorder_OUT +(numSubstate*sizeBorder) + i * sizeBorder,
//                                           0,
//                                           NULL,
//                                           NULL);
//                calclHandleError(err);
//            }

//            if(calclmodel3DNext != NULL) {
//                err = clEnqueueWriteBuffer(calclmodel3D->queue,
//                                           calclmodel3D->bufferCurrentRealSubstate,
//                                           CL_TRUE,
//                                           (i * dim + (dim - sizeBorder) )*sizeof(CALreal),
//                                           sizeof(CALreal)*sizeBorder,
//                                           calclmodel3DNext->borderMapper.realBorder_OUT + i * sizeBorder,
//                                           0,
//                                           NULL,
//                                           NULL);
//                calclHandleError(err);
//            }


//        }

//        numSubstate = calclmodel3D->host_CA->sizeof_pQi_array;


//        for (int i = 0; i < numSubstate; ++i) {


//            if(calclmodel3DPrev != NULL) {
//                err = clEnqueueWriteBuffer(calclmodel3D->queue,
//                                           calclmodel3D->bufferCurrentIntSubstate,
//                                           CL_TRUE,
//                                           (i*dim)*sizeof(CALint),
//                                           sizeof(CALint)*sizeBorder,
//                                           calclmodel3DPrev->borderMapper.intBorder_OUT +(numSubstate*sizeBorder) + i * sizeBorder,
//                                           0,
//                                           NULL,
//                                           NULL);
//                calclHandleError(err);
//            }
//            if(calclmodel3DNext != NULL) {
//                err = clEnqueueWriteBuffer(calclmodel3D->queue,
//                                           calclmodel3D->bufferCurrentIntSubstate,
//                                           CL_TRUE,
//                                           (i * dim + (dim - sizeBorder) )*sizeof(CALint),
//                                           sizeof(CALint)*sizeBorder,
//                                           calclmodel3DNext->borderMapper.intBorder_OUT + i * sizeBorder,
//                                           0,
//                                           NULL,
//                                           NULL);
//                calclHandleError(err);
//            }

//        }


//        numSubstate = calclmodel3D->host_CA->sizeof_pQb_array;
//        for (int i = 0; i < numSubstate; ++i) {

//            if(calclmodel3DPrev != NULL) {
//                err = clEnqueueWriteBuffer(calclmodel3D->queue,
//                                           calclmodel3D->bufferCurrentByteSubstate,
//                                           CL_TRUE,
//                                           (i*dim)*sizeof(CALbyte),
//                                           sizeof(CALbyte)*sizeBorder,
//                                           calclmodel3DPrev->borderMapper.byteBorder_OUT +(numSubstate*sizeBorder) + i * sizeBorder,
//                                           0,
//                                           NULL,
//                                           NULL);
//                calclHandleError(err);
//            }
//            if(calclmodel3DNext != NULL) {
//                err = clEnqueueWriteBuffer(calclmodel3D->queue,
//                                           calclmodel3D->bufferCurrentByteSubstate,
//                                           CL_TRUE,
//                                           (i * dim + (dim - sizeBorder) )*sizeof(CALbyte),
//                                           sizeof(CALbyte)*sizeBorder,
//                                           calclmodel3DNext->borderMapper.byteBorder_OUT + i * sizeBorder,
//                                           0,
//                                           NULL,
//                                           NULL);
//                calclHandleError(err);
//            }

//        }

//    }


//}

void calclMultiDeviceUpdateHalos3D(struct CALCLMultiDevice3D* multidevice,const CALbyte exchange_full_border) {

    cl_int err;

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

        const int sizeBorder = calclmodel3D->borderSize * calclmodel3D->columns*calclmodel3D->rows;

        int numSubstate = calclmodel3D->host_CA->sizeof_pQr_array;
        for (int i = 0; i < numSubstate; ++i) {

            if (calclmodel3DPrev != NULL && (exchange_full_border || gpu != 0)) {
                err = clEnqueueWriteBuffer(
                            calclmodel3D->queue, calclmodel3D->bufferCurrentRealSubstate,
                            CL_TRUE, (i * dim) * sizeof(CALreal), sizeof(CALreal) * sizeBorder,
                            calclmodel3DPrev->borderMapper.realBorder_OUT +
                            (numSubstate * sizeBorder) + i * sizeBorder,
                            0, NULL, NULL);
                calclHandleError(err);
            }

            if (calclmodel3DNext != NULL &&
                    (exchange_full_border || gpu != multidevice->num_devices - 1)) {
                err = clEnqueueWriteBuffer(
                            calclmodel3D->queue, calclmodel3D->bufferCurrentRealSubstate,
                            CL_TRUE, (i * dim + (dim - sizeBorder)) * sizeof(CALreal),
                            sizeof(CALreal) * sizeBorder,
                            calclmodel3DNext->borderMapper.realBorder_OUT + i * sizeBorder, 0,
                            NULL, NULL);
                calclHandleError(err);
            }
        }

        numSubstate = calclmodel3D->host_CA->sizeof_pQi_array;

        for (int i = 0; i < numSubstate; ++i) {

            if (calclmodel3DPrev != NULL && (exchange_full_border || gpu != 0)) {
                err = clEnqueueWriteBuffer(
                            calclmodel3D->queue, calclmodel3D->bufferCurrentIntSubstate,
                            CL_TRUE, (i * dim) * sizeof(CALint), sizeof(CALint) * sizeBorder,
                            calclmodel3DPrev->borderMapper.intBorder_OUT +
                            (numSubstate * sizeBorder) + i * sizeBorder,
                            0, NULL, NULL);
                calclHandleError(err);
            }
            if (calclmodel3DNext != NULL &&
                    (exchange_full_border || gpu != multidevice->num_devices - 1)) {
                err = clEnqueueWriteBuffer(
                            calclmodel3D->queue, calclmodel3D->bufferCurrentIntSubstate,
                            CL_TRUE, (i * dim + (dim - sizeBorder)) * sizeof(CALint),
                            sizeof(CALint) * sizeBorder,
                            calclmodel3DNext->borderMapper.intBorder_OUT + i * sizeBorder, 0,
                            NULL, NULL);
                calclHandleError(err);
            }
        }

        numSubstate = calclmodel3D->host_CA->sizeof_pQb_array;
        for (int i = 0; i < numSubstate; ++i) {

            if (calclmodel3DPrev != NULL && (exchange_full_border || gpu != 0)) {
                err = clEnqueueWriteBuffer(calclmodel3D->queue,
                                           calclmodel3D->bufferCurrentByteSubstate,
                                           CL_TRUE,
                                           (i * dim) * sizeof(CALbyte),
                                           sizeof(CALbyte) * sizeBorder,
                                           calclmodel3DPrev->borderMapper.byteBorder_OUT +(numSubstate * sizeBorder) + i * sizeBorder,
                                           0,
                                           NULL,
                                           NULL);
                calclHandleError(err);
            }
            if (calclmodel3DNext != NULL &&
                    (exchange_full_border || gpu != multidevice->num_devices - 1)) {
                err = clEnqueueWriteBuffer(calclmodel3D->queue,
                                           calclmodel3D->bufferCurrentByteSubstate,
                                           CL_TRUE,
                                           (i * dim + (dim - sizeBorder)) * sizeof(CALbyte),
                                           sizeof(CALbyte) * sizeBorder,
                                           calclmodel3DNext->borderMapper.byteBorder_OUT + i * sizeBorder,
                                           0,
                                           NULL,
                                           NULL);
                calclHandleError(err);
            }
        }

    }//GPUs
}

void calclMultiDeviceGetHalos(struct CALCLMultiDevice3D* multidevice, int offset, int gpu) {
    struct CALCLModel3D * calclmodel3D = multidevice->device_models[gpu];

    calclCopyGhostb(calclmodel3D->host_CA, calclmodel3D->borderMapper.byteBorder_OUT, offset, multidevice->workloads[gpu], calclmodel3D->borderSize);

    calclCopyGhosti(calclmodel3D->host_CA, calclmodel3D->borderMapper.intBorder_OUT, offset, multidevice->workloads[gpu], calclmodel3D->borderSize);

    calclCopyGhostr(calclmodel3D->host_CA, calclmodel3D->borderMapper.realBorder_OUT, offset, multidevice->workloads[gpu], calclmodel3D->borderSize);
}

void calclMultiDeviceCADef3D(struct CALCLMultiDevice3D* multidevice,
                             struct CALModel3D* host_CA, char* kernel_src,
                             char* kernel_inc, const CALint borderSize,
                             const std::vector<Device>& devices) {

    //assert(host_CA->rows == calclCheckWorkload(multidevice));
    multidevice->context =
            calclCreateContext(multidevice->devices, multidevice->num_devices);



    //manual initialization vector::resize causes crash
    multidevice->singleStepThreadNums = (size_t**)calloc(multidevice->num_devices,sizeof(size_t*));
    //for (int i = 0; i < multidevice->num_devices; ++i)
    //multidevice->singleStepThreadNums[i]=0;

    for (int i = 0; i < multidevice->num_devices; ++i) {
        const cl_uint offset = devices[i].offset;
        multidevice->programs[i] = calclLoadProgram3D(
                    multidevice->context, multidevice->devices[i], kernel_src, kernel_inc);

        multidevice->device_models[i] = calclCADef3D(
                    host_CA, multidevice->context, multidevice->programs[i], multidevice->devices[i],
                    multidevice->workloads[i], offset, devices[i].goffset, borderSize);  // offset

        calclMultiDeviceGetHalos(multidevice, offset, i);

        cl_int err;
        setParametersReduction3D(err, multidevice->device_models[i]);
        //printf("calclMultiDeviceCADef3D finished");
    }
    //considera una barriera qui
    calclMultiDeviceUpdateHalos3D(multidevice, multidevice->exchange_full_border);

    vector_init(&multidevice->kernelsID);



}

void calclMultiDeviceToNode(struct CALCLMultiDevice3D* multidevice) {

    for (int gpu = 0; gpu < multidevice->num_devices; ++gpu) {
        calclGetSubstatesFromDevice3D(multidevice->device_models[gpu],
                                      multidevice->workloads[gpu],
                                      multidevice->device_models[gpu]->offset,
                                      multidevice->device_models[gpu]->borderSize);
        calclMultiDeviceMapperToSubstates3D(multidevice->device_models[gpu]->host_CA,
                                            &multidevice->device_models[gpu]->substateMapper,
                                            multidevice->device_models[gpu]->realSize,
                                            multidevice->device_models[gpu]->offset,
                                            multidevice->device_models[gpu]->borderSize);

    }

}

size_t* computekernelLaunchParams(struct CALCLMultiDevice3D* multidevice, const int gpu,int *dim) {
    size_t* singleStepThreadNum;
    if (multidevice->device_models[0]->opt == CAL_NO_OPT) {
        singleStepThreadNum = (size_t*) malloc(sizeof(size_t) * 3);
        (singleStepThreadNum)[0] = multidevice->device_models[gpu]->rows;
        (singleStepThreadNum)[1] = multidevice->device_models[gpu]->columns;
        (singleStepThreadNum)[2] = multidevice->device_models[gpu]->slices;
        *dim = 3;
    } else {
        singleStepThreadNum = (size_t*) malloc(sizeof(size_t));
        singleStepThreadNum[0] = multidevice->device_models[gpu]->num_active_cells;
        *dim = 1;
    }
    return singleStepThreadNum;
}

void calclMultiDeviceSetWorkGroupSize3D(struct CALCLMultiDevice3D* multidevice, int m, int n, int k) {
    for (int gpu = 0; gpu < multidevice->num_devices; ++gpu) {
        struct CALCLModel3D* calclmodel3D = multidevice->device_models[gpu];
        calclSetWorkGroupDimensions3D(calclmodel3D, m, n, k);

    }
}

void calclMultiDeviceRun3D(struct CALCLMultiDevice3D* multidevice, CALint init_step, CALint final_step) {

    int steps = init_step;

    size_t * threadNumMax = (size_t*) malloc(sizeof(size_t) * 3);
    threadNumMax[0] = multidevice->device_models[0]->rows;
    threadNumMax[1] = multidevice->device_models[0]->columns;
    threadNumMax[2] = multidevice->device_models[0]->slices;
    size_t * singleStepThreadNum;
    int dimNum;

    if (multidevice->device_models[0]->opt == CAL_NO_OPT) {
        singleStepThreadNum = (size_t*) malloc(sizeof(size_t) * 3);
        singleStepThreadNum[0] = threadNumMax[0];
        singleStepThreadNum[1] = threadNumMax[1];
        singleStepThreadNum[2] = threadNumMax[2];
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
                struct CALCLModel3D * calclmodel3D = multidevice->device_models[gpu];
                CALbyte activeCells = calclmodel3D->opt == CAL_OPT_ACTIVE_CELLS_NAIVE;
                cl_int err;

                setParametersReduction3D(err, calclmodel3D);

                if (calclmodel3D->kernelInitSubstates != NULL)
                    calclSetReductionParameters3D(calclmodel3D, &calclmodel3D->kernelInitSubstates);
                if (calclmodel3D->kernelStopCondition != NULL)
                    calclSetReductionParameters3D(calclmodel3D, &calclmodel3D->kernelStopCondition);
                if (calclmodel3D->kernelSteering != NULL)
                    calclSetReductionParameters3D(calclmodel3D, &calclmodel3D->kernelSteering);

                int i = 0;

                for (i = 0; i < calclmodel3D->elementaryProcessesNum; i++) {
                    calclSetReductionParameters3D(calclmodel3D, &calclmodel3D->elementaryProcesses[i]);
                }

                if (activeCells == CAL_TRUE) {
                    for (j = 0; j < calclmodel3D->elementaryProcessesNum; j++) {
                        if(singleStepThreadNum[0] > 0)
                            calclKernelCall3D(calclmodel3D, calclmodel3D->elementaryProcesses[j], dimNum, singleStepThreadNum,
                                              NULL);
                        if(singleStepThreadNum[0] > 0) {
                            calclComputeStreamCompaction3D(calclmodel3D);
                            calclResizeThreadsNum3D(calclmodel3D, singleStepThreadNum);
                        }
                        if(singleStepThreadNum[0] > 0)
                            calclKernelCall3D(calclmodel3D, calclmodel3D->kernelUpdateSubstate, dimNum, singleStepThreadNum, NULL);
                        calclGetBorderFromDeviceToHost3D(calclmodel3D);
                    }

                    calclExecuteReduction3D(calclmodel3D, calclmodel3D->roundedDimensions);

                } else {
                    for (j = 0; j < calclmodel3D->elementaryProcessesNum; j++) {
                        calclKernelCall3D(calclmodel3D, calclmodel3D->elementaryProcesses[j], dimNum, singleStepThreadNum,
                                          calclmodel3D->workGroupDimensions);
                        calclCopySubstatesBuffers3D(calclmodel3D);
                        calclGetBorderFromDeviceToHost3D(calclmodel3D);


                    }
                    calclExecuteReduction3D(calclmodel3D, calclmodel3D->roundedDimensions);
                    //copySubstatesBuffers3D(calclmodel3D);

                }

                //                calclKernelCall3D(calclmodel3D, calclmodel3D->elementaryProcesses[j], dimNum, singleStepThreadNum,
                //                                  NULL, NULL);



            }

            // barrier tutte hanno finito
            for (int gpu = 0; gpu < multidevice->num_devices; ++gpu) {
                clFinish(multidevice->device_models[gpu]->queue);
            }

            for (int gpu = 0; gpu < multidevice->num_devices; ++gpu) {
                calclGetBorderFromDeviceToHost3D(multidevice->device_models[gpu]);
            }

            //scambia bordi
            //calclMultiDeviceUpdateHalos3D(multidevice);

            for (int gpu = 0; gpu < multidevice->num_devices; ++gpu) {
                struct CALCLModel3D * calclmodel3D = multidevice->device_models[gpu];
                CALbyte activeCells = calclmodel3D->opt == CAL_OPT_ACTIVE_CELLS_NAIVE;
                if (activeCells == CAL_TRUE) {
                    if (calclmodel3D->kernelSteering != NULL) {
                        calclKernelCall3D(calclmodel3D, calclmodel3D->kernelSteering, dimNum, singleStepThreadNum, NULL);
                        calclKernelCall3D(calclmodel3D, calclmodel3D->kernelUpdateSubstate, dimNum, singleStepThreadNum, NULL);
                    }
                } else
                {
                    if (calclmodel3D->kernelSteering != NULL) {
                        calclKernelCall3D(calclmodel3D, calclmodel3D->kernelSteering, dimNum, singleStepThreadNum, NULL);
                        calclCopySubstatesBuffers3D(calclmodel3D);
                    }
                }
            }

            for (int gpu = 0; gpu < multidevice->num_devices; ++gpu) {
                clFinish(multidevice->device_models[gpu]->queue);
            }

            for (int gpu = 0; gpu < multidevice->num_devices; ++gpu) {
                calclGetBorderFromDeviceToHost3D(multidevice->device_models[gpu]);
            }

            //scambia bordi
            //calclMultiDeviceUpdateHalos3D(multidevice);


        }//for elementary process

        steps++;

    }// while


    for (int gpu = 0; gpu < multidevice->num_devices; ++gpu) {
        calclGetSubstatesFromDevice3D(multidevice->device_models[gpu],
                                      multidevice->workloads[gpu],
                                      multidevice->device_models[gpu]->offset,
                                      multidevice->device_models[gpu]->borderSize);
        calclMultiDeviceMapperToSubstates3D(multidevice->device_models[gpu]->host_CA,
                                            &multidevice->device_models[gpu]->substateMapper,
                                            multidevice->device_models[gpu]->realSize,
                                            multidevice->device_models[gpu]->offset,
                                            multidevice->device_models[gpu]->borderSize);

    }






}

void calclMultiDeviceSetKernelArg3D(struct CALCLMultiDevice3D * multidevice,const char * kernel, cl_uint arg_index, size_t arg_size, const void *arg_value) {
    int index = vector_search_char3D(&multidevice->kernelsID,kernel);
    // assert(index != -1);

    for (int gpu = 0; gpu < multidevice->num_devices; ++gpu) {
        //CALCLkernel * k = &multidevice->device_models[gpu]->elementaryProcesses[index];
        clSetKernelArg(multidevice->device_models[gpu]->elementaryProcesses[index], MODEL_ARGS_NUM + arg_index, arg_size, arg_value);
    }
}

void calclMultiDeviceStopConditionSetKernelArg3D(struct CALCLMultiDevice3D * multidevice,const char * kernel, cl_uint arg_index, size_t arg_size, const void *arg_value) {
    int index = vector_search_char3D(&multidevice->kernelsID,kernel);
    // assert(index != -1);

    for (int gpu = 0; gpu < multidevice->num_devices; ++gpu) {
        //CALCLkernel * k = &multidevice->device_models[gpu]->elementaryProcesses[index];
        clSetKernelArg(multidevice->device_models[gpu]->kernelStopCondition, MODEL_ARGS_NUM + arg_index, arg_size, arg_value);
    }
}
void calclMultiDeviceSteeringSetKernelArg3D(struct CALCLMultiDevice3D * multidevice,const char * kernel, cl_uint arg_index, size_t arg_size, const void *arg_value) {
    int index = vector_search_char3D(&multidevice->kernelsID,kernel);
    // assert(index != -1);

    for (int gpu = 0; gpu < multidevice->num_devices; ++gpu) {
        //CALCLkernel * k = &multidevice->device_models[gpu]->elementaryProcesses[index];
        clSetKernelArg(multidevice->device_models[gpu]->kernelSteering, MODEL_ARGS_NUM + arg_index, arg_size, arg_value);
    }
}


void calclMultiDeviceAddElementaryProcess3D(struct CALCLMultiDevice3D* multidevice, char * kernelName) {
    struct kernelID * kernel = (kernelID*)malloc(sizeof(struct kernelID));
    kernel->index = vector_total(&multidevice->kernelsID);
    memset(kernel->name,'\0',sizeof(kernel->name));
    strcpy(kernel->name,kernelName);

    VECTOR_ADD(multidevice->kernelsID, kernel);

    for (int i = 0; i < multidevice->num_devices; i++) {

        CALCLprogram p=multidevice->programs[i];
        CALCLkernel kernel = calclGetKernelFromProgram(p,kernelName);
        calclAddElementaryProcess3D(multidevice->device_models[i],&kernel);
    }
}
void calclMultiDeviceAddStopConditionFunc3D(struct CALCLMultiDevice3D* multidevice, char * kernelName) {
    struct kernelID * kernel = (kernelID*)malloc(sizeof(struct kernelID));
    kernel->index = vector_total(&multidevice->kernelsID);
    memset(kernel->name,'\0',sizeof(kernel->name));
    strcpy(kernel->name,kernelName);

    VECTOR_ADD(multidevice->kernelsID, kernel);

    for (int i = 0; i < multidevice->num_devices; i++) {

        CALCLprogram p=multidevice->programs[i];
        CALCLkernel kernel = calclGetKernelFromProgram(p,kernelName);
        calclAddStopConditionFunc3D(multidevice->device_models[i],&kernel);
    }
}
void calclMultiDeviceAddSteeringFunc3D(struct CALCLMultiDevice3D* multidevice, char* kernelName) {


    struct kernelID * kernel = (kernelID*)malloc(sizeof(struct kernelID));
    kernel->index = vector_total(&multidevice->kernelsID);
    memset(kernel->name,'\0',sizeof(kernel->name));
    strcpy(kernel->name,kernelName);

    VECTOR_ADD(multidevice->kernelsID, kernel);

    for (int i = 0; i < multidevice->num_devices; i++) {

        CALCLprogram p=multidevice->programs[i];
        CALCLkernel calclkernel = calclGetKernelFromProgram(p,kernelName);
        calclAddSteeringFunc3D(multidevice->device_models[i],&calclkernel);
    }
}

void calclMultiDeviceFinalize3D(struct CALCLMultiDevice3D* multidevice) {

    free(multidevice->devices);
    free(multidevice->programs);
    // free(multidevice->kernel_events);
    for (int i = 0; i < multidevice->num_devices; ++i) {
        calclFinalize3D(multidevice->device_models[i]);
    }

    vector_free(&multidevice->kernelsID);
    free(multidevice);
}




