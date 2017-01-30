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

#include <OpenCAL-CL/calclMultiGPU2D.h>
#include <OpenCAL/cal2DBuffer.h>

/******************************************************************************
 * 							PRIVATE FUNCTIONS
 ******************************************************************************/


void calclMultiGPUGetSubstateDeviceToHost2D(struct CALCLModel2D* calclmodel2D, const CALint workload, const CALint offset, const CALint borderSize) {

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
                                  CL_TRUE,
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



void calclMultiGPUMapperToSubstates2D(struct CALModel2D * host_CA, CALCLSubstateMapper * mapper,const size_t realSize, const CALint offset, int borderSize) {

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




int vector_search_char(vector *v,const char * search)
{

    for (int i = 0; i < v->total; i++) {
        struct kernelID * tmp = vector_get(v,i);
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


void calclSetKernelArgMultiGPU2D(struct CALCLMultiGPU * multigpu,const char * kernel, cl_uint arg_index, size_t arg_size, const void *arg_value) {
    int index = vector_search_char(&multigpu->kernelsID,kernel);
    assert(index != -1);

    for (int gpu = 0; gpu < multigpu->num_devices; ++gpu) {
        //CALCLkernel * k = &multigpu->device_models[gpu]->elementaryProcesses[index];
        clSetKernelArg(multigpu->device_models[gpu]->elementaryProcesses[index], MODEL_ARGS_NUM + arg_index, arg_size, arg_value);
    }
}



void calclAddSteeringFuncMultiGPU2D(struct CALCLMultiGPU* multigpu, char* kernelName) {


    struct kernelID * kernel = malloc(sizeof(struct kernelID));
    kernel->index = vector_total(&multigpu->kernelsID);
    memset(kernel->name,'\0',sizeof(kernel->name));
    strcpy(kernel->name,kernelName);

    VECTOR_ADD(multigpu->kernelsID, kernel);

    for (int i = 0; i < multigpu->num_devices; i++) {

        CALCLprogram p=multigpu->programs[i];
        CALCLkernel kernel = calclGetKernelFromProgram(p,kernelName);
        calclAddElementaryProcess2D(multigpu->device_models[i],kernel);
    }
}


/******************************************************************************
 * 							PUBLIC FUNCTIONS MULTIGPU
 ******************************************************************************/

void calclSetNumDevice(struct CALCLMultiGPU* multigpu, const CALint _num_devices) {
    multigpu->num_devices = _num_devices;
    multigpu->devices = (CALCLdevice*)malloc(sizeof(CALCLdevice)*multigpu->num_devices);
    multigpu->programs = (CALCLprogram*)malloc(sizeof(CALCLprogram)*multigpu->num_devices);
    multigpu->workloads = (CALint*)malloc(sizeof(CALint)*multigpu->num_devices);
    multigpu->device_models = (struct CALCLModel2D**)malloc(sizeof(struct CALCLModel2D*)*multigpu->num_devices);
    multigpu->pos_device = 0;
}

void calclAddDevice(struct CALCLMultiGPU* multigpu,const CALCLdevice device, const CALint workload) {
    multigpu->devices[multigpu->pos_device] = device;
    multigpu->workloads[multigpu->pos_device] = workload;
    multigpu->pos_device++;
}

int calclCheckWorkload(struct CALCLMultiGPU* multigpu) {
    int tmpsum=0;
    for (int i = 0; i < multigpu->num_devices; ++i) {
        tmpsum +=multigpu->workloads[i];
    }
    return tmpsum;
}


void calclMultiGPUHandleBorders(struct CALCLMultiGPU* multigpu) {


    cl_int err;

    for (int gpu = 0; gpu < multigpu->num_devices; ++gpu) {

        struct CALCLModel2D * calclmodel2D = multigpu->device_models[gpu];
        struct CALCLModel2D * calclmodel2DPrev = NULL;
        const int gpuP = ((gpu-1)+multigpu->num_devices)%multigpu->num_devices;
        const int gpuN = ((gpu+1)+multigpu->num_devices)%multigpu->num_devices;

        if(calclmodel2D->host_CA->T == CAL_SPACE_TOROIDAL || ((gpu-1) >= 0) ) {

            calclmodel2DPrev = multigpu->device_models[gpuP];
        }


        struct CALCLModel2D * calclmodel2DNext = NULL;
        if(calclmodel2D->host_CA->T == CAL_SPACE_TOROIDAL || ((gpu + 1) < multigpu->num_devices) ) {
            calclmodel2DNext = multigpu->device_models[gpuN];
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

void calclMultiGPUHandleBordersMultiNode(struct CALCLMultiGPU* multigpu,const CALbyte exchange_full_border) {


    cl_int err;

    for (int gpu = 0; gpu < multigpu->num_devices; ++gpu) {

        struct CALCLModel2D * calclmodel2D = multigpu->device_models[gpu];
        struct CALCLModel2D * calclmodel2DPrev = NULL;
        const int gpuP = ((gpu-1)+multigpu->num_devices)%multigpu->num_devices;
        const int gpuN = ((gpu+1)+multigpu->num_devices)%multigpu->num_devices;



        if(calclmodel2D->host_CA->T == CAL_SPACE_TOROIDAL || ((gpu-1) >= 0) ) {

            calclmodel2DPrev = multigpu->device_models[gpuP];
        }


        struct CALCLModel2D * calclmodel2DNext = NULL;
        if(calclmodel2D->host_CA->T == CAL_SPACE_TOROIDAL || ((gpu + 1) < multigpu->num_devices) ) {
            calclmodel2DNext = multigpu->device_models[gpuN];
        }



        int dim = calclmodel2D->fullSize;


        const int sizeBorder = calclmodel2D->borderSize*calclmodel2D->columns;

        int numSubstate = calclmodel2D->host_CA->sizeof_pQr_array;
        for (int i = 0; i < numSubstate; ++i) {

            if(calclmodel2DPrev != NULL && (exchange_full_border || gpu != 0 )) {
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

            if(calclmodel2DNext != NULL && (exchange_full_border || gpu != multigpu->num_devices-1)) {
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


            if(calclmodel2DPrev != NULL && (exchange_full_border || gpu != 0 )) {
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
            if(calclmodel2DNext != NULL && (exchange_full_border || gpu != multigpu->num_devices-1)) {
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

            if(calclmodel2DPrev != NULL && (exchange_full_border || gpu != 0 )) {
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
            if(calclmodel2DNext != NULL && (exchange_full_border || gpu != multigpu->num_devices-1)) {
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


void calclMultiGPUGetBorders(struct CALCLMultiGPU* multigpu, int offset, int gpu) {
    struct CALCLModel2D * calclmodel2D = multigpu->device_models[gpu];

    calclCopyGhostb(calclmodel2D->host_CA, calclmodel2D->borderMapper.byteBorder_OUT, offset, multigpu->workloads[gpu], calclmodel2D->borderSize);

    calclCopyGhosti(calclmodel2D->host_CA, calclmodel2D->borderMapper.intBorder_OUT, offset, multigpu->workloads[gpu], calclmodel2D->borderSize);

    calclCopyGhostr(calclmodel2D->host_CA, calclmodel2D->borderMapper.realBorder_OUT, offset, multigpu->workloads[gpu], calclmodel2D->borderSize);
}

void calclMultiGPUDef2D(struct CALCLMultiGPU* multigpu,struct CALModel2D *host_CA ,char* kernel_src,char* kernel_inc, const CALint borderSize, const CALbyte _exchange_full_border) {
    assert(host_CA->rows == calclCheckWorkload(multigpu));
    multigpu->context = calclCreateContext(multigpu->devices,multigpu->num_devices);
    multigpu->exchange_full_border = _exchange_full_border;
    int offset=0;
    for (int i = 0; i < multigpu->num_devices; ++i) {
        multigpu->programs[i] = calclLoadProgram2D(multigpu->context, multigpu->devices[i], kernel_src, kernel_inc);

        multigpu->device_models[i] = calclCADef2D(host_CA,multigpu->context,multigpu->programs[i],multigpu->devices[i],multigpu->workloads[i],offset , borderSize);//offset

        calclMultiGPUGetBorders(multigpu,offset, i);



        offset+=multigpu->workloads[i];
		
		cl_int err;
		setParametersReduction(err, multigpu->device_models[i]);


    }
//considera una barriera qui
    calclMultiGPUHandleBordersMultiNode(multigpu, multigpu->exchange_full_border);

    vector_init(&multigpu->kernelsID);



}

void calcl_executeElementaryProcess(struct CALCLMultiGPU* multigpu,const int el_proc, size_t* singleStepThreadNum,int dimNum)
{

    for (int gpu = 0; gpu < multigpu->num_devices; ++gpu) {
        struct CALCLModel2D * calclmodel2D = multigpu->device_models[gpu];

        cl_int err;

	
        if (calclmodel2D->kernelInitSubstates != NULL)
            calclSetReductionParameters2D(calclmodel2D, calclmodel2D->kernelInitSubstates);
        if (calclmodel2D->kernelStopCondition != NULL)
            calclSetReductionParameters2D(calclmodel2D, calclmodel2D->kernelStopCondition);
        if (calclmodel2D->kernelSteering != NULL)
            calclSetReductionParameters2D(calclmodel2D, calclmodel2D->kernelSteering);

        int i = 0;

        calclSetReductionParameters2D(calclmodel2D, calclmodel2D->elementaryProcesses[el_proc]);



        calclKernelCall2D(calclmodel2D, calclmodel2D->elementaryProcesses[el_proc], dimNum, singleStepThreadNum,
                          NULL, NULL);
        copySubstatesBuffers2D(calclmodel2D);


    }

    // barrier tutte hanno finito
    for (int gpu = 0; gpu < multigpu->num_devices; ++gpu) {
        clFinish(multigpu->device_models[gpu]->queue);
    }

    for (int gpu = 0; gpu < multigpu->num_devices; ++gpu) {
        calclGetBorderFromDeviceToHost2D(multigpu->device_models[gpu]);
    }

    //scambia bordi
    calclMultiGPUHandleBordersMultiNode(multigpu, multigpu->exchange_full_border);

    for (int gpu = 0; gpu < multigpu->num_devices; ++gpu) {
        struct CALCLModel2D * calclmodel2D = multigpu->device_models[gpu];

        if (calclmodel2D->kernelSteering != NULL) {
            calclKernelCall2D(calclmodel2D, calclmodel2D->kernelSteering, dimNum, singleStepThreadNum, NULL, NULL);
            copySubstatesBuffers2D(calclmodel2D);
        }
    }

    for (int gpu = 0; gpu < multigpu->num_devices; ++gpu) {
        clFinish(multigpu->device_models[gpu]->queue);
    }

    for (int gpu = 0; gpu < multigpu->num_devices; ++gpu) {
        calclGetBorderFromDeviceToHost2D(multigpu->device_models[gpu]);
    }

    //scambia bordi
    calclMultiGPUHandleBordersMultiNode(multigpu, multigpu->exchange_full_border);

}

void calclDevicesToNode(struct CALCLMultiGPU* multigpu) {

    for (int gpu = 0; gpu < multigpu->num_devices; ++gpu) {
        calclMultiGPUGetSubstateDeviceToHost2D(multigpu->device_models[gpu],
                                               multigpu->workloads[gpu],
                                               multigpu->device_models[gpu]->offset,
                                               multigpu->device_models[gpu]->borderSize);
        calclMultiGPUMapperToSubstates2D(multigpu->device_models[gpu]->host_CA,
                                         &multigpu->device_models[gpu]->substateMapper,
                                         multigpu->device_models[gpu]->realSize,
                                         multigpu->device_models[gpu]->offset,
                                         multigpu->device_models[gpu]->borderSize);

    }

}

void computekernelLaunchParams(struct CALCLMultiGPU* multigpu, size_t** singleStepThreadNum, int *dim) {
    if (multigpu->device_models[0]->opt == CAL_NO_OPT) {
        *singleStepThreadNum = (size_t*) malloc(sizeof(size_t) * 2);
        (*singleStepThreadNum)[0] = multigpu->device_models[0]->rows;
        (*singleStepThreadNum)[1] = multigpu->device_models[0]->columns;
        *dim = 2;
    } else {
        *singleStepThreadNum = (size_t*) malloc(sizeof(size_t));
        *singleStepThreadNum[0] = multigpu->device_models[0]->host_CA->A->size_current;
        *dim = 1;
    }
}

void calclMultiGPURun2D(struct CALCLMultiGPU* multigpu, CALint init_step, CALint final_step) {

    int steps = init_step;

    size_t * threadNumMax = (size_t*) malloc(sizeof(size_t) * 2);
    threadNumMax[0] = multigpu->device_models[0]->rows;
    threadNumMax[1] = multigpu->device_models[0]->columns;
    size_t * singleStepThreadNum;
    int dimNum;

    if (multigpu->device_models[0]->opt == CAL_NO_OPT) {
        singleStepThreadNum = (size_t*) malloc(sizeof(size_t) * 2);
        singleStepThreadNum[0] = threadNumMax[0];
        singleStepThreadNum[1] = threadNumMax[1];
        dimNum = 2;
    } else {
        singleStepThreadNum = (size_t*) malloc(sizeof(size_t));
        singleStepThreadNum[0] = multigpu->device_models[0]->host_CA->A->size_current;
        dimNum = 1;
    }


    while (steps <= (int) final_step || final_step == CAL_RUN_LOOP) {


        //calcola dimNum e ThreadsNum

        for (int j = 0; j < multigpu->device_models[0]->elementaryProcessesNum; j++) {

            for (int gpu = 0; gpu < multigpu->num_devices; ++gpu) {
                struct CALCLModel2D * calclmodel2D = multigpu->device_models[gpu];
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
            for (int gpu = 0; gpu < multigpu->num_devices; ++gpu) {
                clFinish(multigpu->device_models[gpu]->queue);
            }

            for (int gpu = 0; gpu < multigpu->num_devices; ++gpu) {
                calclGetBorderFromDeviceToHost2D(multigpu->device_models[gpu]);
            }

            //scambia bordi
            calclMultiGPUHandleBorders(multigpu);

            for (int gpu = 0; gpu < multigpu->num_devices; ++gpu) {
                struct CALCLModel2D * calclmodel2D = multigpu->device_models[gpu];
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

            for (int gpu = 0; gpu < multigpu->num_devices; ++gpu) {
                clFinish(multigpu->device_models[gpu]->queue);
            }

            for (int gpu = 0; gpu < multigpu->num_devices; ++gpu) {
                calclGetBorderFromDeviceToHost2D(multigpu->device_models[gpu]);
            }

            //scambia bordi
            calclMultiGPUHandleBorders(multigpu);


        }//for elementary process

        steps++;

    }// while


    for (int gpu = 0; gpu < multigpu->num_devices; ++gpu) {
        calclMultiGPUGetSubstateDeviceToHost2D(multigpu->device_models[gpu],
                                               multigpu->workloads[gpu],
                                               multigpu->device_models[gpu]->offset,
                                               multigpu->device_models[gpu]->borderSize);
        calclMultiGPUMapperToSubstates2D(multigpu->device_models[gpu]->host_CA,
                                         &multigpu->device_models[gpu]->substateMapper,
                                         multigpu->device_models[gpu]->realSize,
                                         multigpu->device_models[gpu]->offset,
                                         multigpu->device_models[gpu]->borderSize);

    }






}


void calclAddElementaryProcessMultiGPU2D(struct CALCLMultiGPU* multigpu, char * kernelName) {
    struct kernelID * kernel = malloc(sizeof(struct kernelID));
    kernel->index = vector_total(&multigpu->kernelsID);
    memset(kernel->name,'\0',sizeof(kernel->name));
    strcpy(kernel->name,kernelName);

    VECTOR_ADD(multigpu->kernelsID, kernel);

    for (int i = 0; i < multigpu->num_devices; i++) {

        CALCLprogram p=multigpu->programs[i];
        CALCLkernel kernel = calclGetKernelFromProgram(p,kernelName);
        calclAddElementaryProcess2D(multigpu->device_models[i],kernel);
    }
}

void calclMultiGPUFinalize(struct CALCLMultiGPU* multigpu) {

    free(multigpu->devices);
    free(multigpu->programs);
    // free(multigpu->kernel_events);
    for (int i = 0; i < multigpu->num_devices; ++i) {
        calclFinalize2D(multigpu->device_models[i]);
    }

    vector_free(&multigpu->kernelsID);
    free(multigpu);
}


