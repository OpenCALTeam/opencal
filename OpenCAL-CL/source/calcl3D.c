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

/*
 * calCL.cpp
 *
 *  Created on: 10/giu/2014
 *      Author: alessio
 */
#include <OpenCAL-CL/calcl3D.h>
#include <OpenCAL-CL/calcl3DReduction.h>
#include <OpenCAL/cal3DBuffer.h>

/******************************************************************************
 * 							PRIVATE FUNCTIONS
 ******************************************************************************/
void calclMapperToSubstates3D(struct CALModel3D *model, CALCLSubstateMapper * mapper) {

    int ssNum_r = model->sizeof_pQr_array;
    int ssNum_i = model->sizeof_pQi_array;
    int ssNum_b = model->sizeof_pQb_array;
    size_t elNum = model->columns * model->rows * model->slices;

    long int outIndex = 0;
    int i;
    unsigned int j;

    for (i = 0; i < ssNum_r; i++) {
        for (j = 0; j < elNum; j++)
            model->pQr_array[i]->current[j] = mapper->realSubstate_current_OUT[outIndex++];
    }

    outIndex = 0;

    for (i = 0; i < ssNum_i; i++) {
        for (j = 0; j < elNum; j++)
            model->pQi_array[i]->current[j] = mapper->intSubstate_current_OUT[outIndex++];
    }

    outIndex = 0;

    for (i = 0; i < ssNum_b; i++) {
        for (j = 0; j < elNum; j++)
            model->pQb_array[i]->current[j] = mapper->byteSubstate_current_OUT[outIndex++];
    }

}

void calclGetBorderFromDeviceToHost3D(struct CALCLModel3D* calclmodel3D) {

    CALCLqueue queue = calclmodel3D->queue;

    cl_int err;
    int dim = calclmodel3D->fullSize;


    int sizeBorder = calclmodel3D->borderSize*calclmodel3D->columns*calclmodel3D->rows;

    // get real borders
    for (int i = 0; i < calclmodel3D->host_CA->sizeof_pQr_array; ++i) {
        err = clEnqueueReadBuffer(queue,
                                  calclmodel3D->bufferCurrentRealSubstate,
                                  CL_TRUE,
                                  (i*dim + sizeBorder)*sizeof(CALreal),
                                  sizeof(CALreal)*sizeBorder,
                                  calclmodel3D->borderMapper.realBorder_OUT + i * sizeBorder,
                                  0,
                                  NULL,
                                  NULL);
        calclHandleError(err);
    }

    // i*dim salto i sottostati
    // dim-(calclmodel3D->columns * calclmodel3D->borderSize) posizione inzio last ghost cells
    // con * 2 ultime borderSize righe dello spazio cellulare su cui viene eseguite la funzione di transizione
    // (i*dim+(dim-calclmodel3D->columns * 2 * calclmodel3D->borderSize))

    int numSubstate = calclmodel3D->host_CA->sizeof_pQr_array;
    for (int i = 0; i < numSubstate; ++i) {
        err = clEnqueueReadBuffer(queue,
                                  calclmodel3D->bufferCurrentRealSubstate,
                                  CL_TRUE,
                                  (i * dim + (dim - sizeBorder * 2))*sizeof(CALreal),
                                  sizeof(CALreal)*sizeBorder,
                                  calclmodel3D->borderMapper.realBorder_OUT + (numSubstate + i) * sizeBorder,
                                  0,
                                  NULL,
                                  NULL);
        calclHandleError(err);
    }





    // get int borders
    numSubstate = calclmodel3D->host_CA->sizeof_pQi_array;
    for (int i = 0; i < numSubstate; ++i) {
        err = clEnqueueReadBuffer(queue,
                                  calclmodel3D->bufferCurrentIntSubstate,
                                  CL_TRUE,
                                  (i*dim + sizeBorder)*sizeof(CALint),
                                  sizeof(CALint)*sizeBorder,
                                  calclmodel3D->borderMapper.intBorder_OUT + i * sizeBorder,
                                  0,
                                  NULL,
                                  NULL);
        calclHandleError(err);
    }

    for (int i = 0; i < calclmodel3D->host_CA->sizeof_pQi_array; ++i) {
        err = clEnqueueReadBuffer(queue,
                                  calclmodel3D->bufferCurrentIntSubstate,
                                  CL_TRUE,
                                  (i * dim + (dim - sizeBorder * 2))*sizeof(CALint),
                                  sizeof(CALint)*sizeBorder,
                                  calclmodel3D->borderMapper.intBorder_OUT+ (numSubstate + i) * sizeBorder,
                                  0,
                                  NULL,
                                  NULL);
        calclHandleError(err);
    }

    // get byte borders
    numSubstate = calclmodel3D->host_CA->sizeof_pQb_array;
    for (int i = 0; i < numSubstate; ++i) {
        err = clEnqueueReadBuffer(queue,
                                  calclmodel3D->bufferCurrentByteSubstate,
                                  CL_TRUE,
                                  (i*dim + sizeBorder)*sizeof(CALbyte),
                                  sizeof(CALbyte)*sizeBorder,
                                  calclmodel3D->borderMapper.byteBorder_OUT + i * sizeBorder,
                                  0,
                                  NULL,
                                  NULL);
        calclHandleError(err);
    }

    for (int i = 0; i < calclmodel3D->host_CA->sizeof_pQb_array; ++i) {
        err = clEnqueueReadBuffer(queue,
                                  calclmodel3D->bufferCurrentByteSubstate,
                                  CL_TRUE,
                                  (i * dim + (dim - sizeBorder * 2))*sizeof(CALbyte),
                                  sizeof(CALbyte)*sizeBorder,
                                  calclmodel3D->borderMapper.byteBorder_OUT+ (numSubstate + i) * sizeBorder,
                                  0,
                                  NULL,
                                  NULL);
        calclHandleError(err);
    }

}



void calclGetSubstatesDeviceToHost3D(struct CALCLModel3D* calclmodel3D) {
    CALCLqueue queue = calclmodel3D->queue;

    cl_int err;
    size_t zero = 0;

    err = clEnqueueReadBuffer(queue, calclmodel3D->bufferCurrentRealSubstate, CL_TRUE, zero, calclmodel3D->substateMapper.bufDIMreal, calclmodel3D->substateMapper.realSubstate_current_OUT, 0, NULL, NULL);
    calclHandleError(err);
    err = clEnqueueReadBuffer(queue, calclmodel3D->bufferCurrentIntSubstate, CL_TRUE, zero, calclmodel3D->substateMapper.bufDIMint, calclmodel3D->substateMapper.intSubstate_current_OUT, 0, NULL, NULL);
    calclHandleError(err);
    err = clEnqueueReadBuffer(queue, calclmodel3D->bufferCurrentByteSubstate, CL_TRUE, zero, calclmodel3D->substateMapper.bufDIMbyte, calclmodel3D->substateMapper.byteSubstate_current_OUT, 0, NULL, NULL);
    calclHandleError(err);

    calclMapperToSubstates3D(calclmodel3D->host_CA, &calclmodel3D->substateMapper);
}

void calclRoundThreadsNum3D(size_t * threadNum, int numDim, size_t multiple) {
    int i;
    for (i = 0; i < numDim; ++i)
        while (threadNum[i] % multiple != 0)
            threadNum[i]++;
}

void calclResizeThreadsNum3D(struct CALCLModel3D * calclmodel3D, size_t * threadNum) {
    cl_int err;
    size_t zero = 0;
    CALCLqueue queue = calclmodel3D->queue;


    err = clEnqueueReadBuffer(queue, calclmodel3D->bufferActiveCellsNum, CL_TRUE, zero, sizeof(int), &calclmodel3D->host_CA->A->size_current, 0, NULL, NULL);
    calclHandleError(err);
    threadNum[0] = calclmodel3D->host_CA->A->size_current;
}

CALCLmem calclGetSubstateBuffer3D(CALCLmem bufferSubstates, cl_buffer_region region) {
    cl_int err;
    CALCLmem sub_buffer = clCreateSubBuffer(bufferSubstates, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, CL_BUFFER_CREATE_TYPE_REGION, &region, &err);
    calclHandleError(err);
    return sub_buffer;
}

void calclCopySubstatesBuffers3D(struct CALCLModel3D * calclmodel3D) {
    CALCLqueue queue = calclmodel3D->queue;

    if (calclmodel3D->host_CA->sizeof_pQr_array > 0)
        clEnqueueCopyBuffer(queue, calclmodel3D->bufferNextRealSubstate, calclmodel3D->bufferCurrentRealSubstate, 0, 0, calclmodel3D->substateMapper.bufDIMreal, 0, NULL, NULL);
    if (calclmodel3D->host_CA->sizeof_pQi_array > 0)
        clEnqueueCopyBuffer(queue, calclmodel3D->bufferNextIntSubstate, calclmodel3D->bufferCurrentIntSubstate, 0, 0, calclmodel3D->substateMapper.bufDIMint, 0, NULL, NULL);
    if (calclmodel3D->host_CA->sizeof_pQb_array > 0)
        clEnqueueCopyBuffer(queue, calclmodel3D->bufferNextByteSubstate, calclmodel3D->bufferCurrentByteSubstate, 0, 0, calclmodel3D->substateMapper.bufDIMbyte, 0, NULL, NULL);
}

CALbyte checkStopCondition3D(struct CALCLModel3D * calclmodel3D, CALint dimNum, size_t * threadsNum) {
    CALCLqueue queue = calclmodel3D->queue;

    calclKernelCall3D(calclmodel3D, calclmodel3D->kernelStopCondition, dimNum, threadsNum, NULL);
    CALbyte stop = CAL_FALSE;
    size_t zero = 0;

    cl_int err = clEnqueueReadBuffer(queue, calclmodel3D->bufferStop, CL_TRUE, zero, sizeof(CALbyte), &stop, 0, NULL, NULL);
    calclHandleError(err);
    return stop;
}

void calclSetKernelStreamCompactionArgs3D(struct CALCLModel3D * calclmodel3D) {
    CALint dim = calclmodel3D->host_CA->rows * calclmodel3D->host_CA->columns * calclmodel3D->host_CA->slices;
    clSetKernelArg(calclmodel3D->kernelComputeCounts, 0, sizeof(CALint), &dim);
    clSetKernelArg(calclmodel3D->kernelComputeCounts, 1, sizeof(CALCLmem), &calclmodel3D->bufferActiveCellsFlags);
    clSetKernelArg(calclmodel3D->kernelComputeCounts, 2, sizeof(CALCLmem), &calclmodel3D->bufferSTCounts);
    clSetKernelArg(calclmodel3D->kernelComputeCounts, 3, sizeof(CALCLmem), &calclmodel3D->bufferSTOffsets1);
    clSetKernelArg(calclmodel3D->kernelComputeCounts, 4, sizeof(CALCLmem), &calclmodel3D->bufferSTCountsDiff);


    int offset = calclmodel3D->streamCompactionThreadsNum / 2;

    clSetKernelArg(calclmodel3D->kernelUpSweep, 0, sizeof(CALCLmem), &calclmodel3D->bufferSTOffsets1);
    clSetKernelArg(calclmodel3D->kernelUpSweep, 1, sizeof(int), &offset);

    clSetKernelArg(calclmodel3D->kernelDownSweep, 0, sizeof(CALCLmem), &calclmodel3D->bufferSTOffsets1);
    clSetKernelArg(calclmodel3D->kernelDownSweep, 1, sizeof(int), &offset);

    clSetKernelArg(calclmodel3D->kernelCompact, 0, sizeof(CALint), &dim);
    clSetKernelArg(calclmodel3D->kernelCompact, 1, sizeof(CALint), &calclmodel3D->host_CA->rows);
    clSetKernelArg(calclmodel3D->kernelCompact, 2, sizeof(CALint), &calclmodel3D->host_CA->columns);
    clSetKernelArg(calclmodel3D->kernelCompact, 3, sizeof(CALCLmem), &calclmodel3D->bufferActiveCellsFlags);
    clSetKernelArg(calclmodel3D->kernelCompact, 4, sizeof(CALCLmem), &calclmodel3D->bufferActiveCellsNum);
    clSetKernelArg(calclmodel3D->kernelCompact, 5, sizeof(CALCLmem), &calclmodel3D->bufferActiveCells);
    clSetKernelArg(calclmodel3D->kernelCompact, 6, sizeof(CALCLmem), &calclmodel3D->bufferSTCounts);
    clSetKernelArg(calclmodel3D->kernelCompact, 7, sizeof(CALCLmem), &calclmodel3D->bufferSTOffsets1);

}

void calclSetKernelsLibArgs3D(struct CALCLModel3D *calclmodel3D) {
    clSetKernelArg(calclmodel3D->kernelUpdateSubstate, 0, sizeof(CALint), &calclmodel3D->host_CA->columns);
    clSetKernelArg(calclmodel3D->kernelUpdateSubstate, 1, sizeof(CALint), &calclmodel3D->host_CA->rows);
    clSetKernelArg(calclmodel3D->kernelUpdateSubstate, 2, sizeof(CALint), &calclmodel3D->host_CA->slices);
    clSetKernelArg(calclmodel3D->kernelUpdateSubstate, 3, sizeof(CALint), &calclmodel3D->host_CA->sizeof_pQb_array);
    clSetKernelArg(calclmodel3D->kernelUpdateSubstate, 4, sizeof(CALint), &calclmodel3D->host_CA->sizeof_pQi_array);
    clSetKernelArg(calclmodel3D->kernelUpdateSubstate, 5, sizeof(CALint), &calclmodel3D->host_CA->sizeof_pQr_array);
    clSetKernelArg(calclmodel3D->kernelUpdateSubstate, 6, sizeof(CALCLmem), &calclmodel3D->bufferCurrentByteSubstate);
    clSetKernelArg(calclmodel3D->kernelUpdateSubstate, 7, sizeof(CALCLmem), &calclmodel3D->bufferCurrentIntSubstate);
    clSetKernelArg(calclmodel3D->kernelUpdateSubstate, 8, sizeof(CALCLmem), &calclmodel3D->bufferCurrentRealSubstate);
    clSetKernelArg(calclmodel3D->kernelUpdateSubstate, 9, sizeof(CALCLmem), &calclmodel3D->bufferNextByteSubstate);
    clSetKernelArg(calclmodel3D->kernelUpdateSubstate, 10, sizeof(CALCLmem), &calclmodel3D->bufferNextIntSubstate);
    clSetKernelArg(calclmodel3D->kernelUpdateSubstate, 11, sizeof(CALCLmem), &calclmodel3D->bufferNextRealSubstate);
    clSetKernelArg(calclmodel3D->kernelUpdateSubstate, 12, sizeof(CALCLmem), &calclmodel3D->bufferActiveCells);
    clSetKernelArg(calclmodel3D->kernelUpdateSubstate, 13, sizeof(CALCLmem), &calclmodel3D->bufferActiveCellsNum);
}

void calclSetModelParameters3D(struct CALCLModel3D * calclmodel3D, CALCLkernel * kernel) {

    clSetKernelArg(*kernel, 0, sizeof(CALCLmem), &calclmodel3D->bufferRows);
    clSetKernelArg(*kernel, 1, sizeof(CALCLmem), &calclmodel3D->bufferColumns);
    clSetKernelArg(*kernel, 2, sizeof(CALCLmem), &calclmodel3D->bufferSlices);
    clSetKernelArg(*kernel, 3, sizeof(CALCLmem), &calclmodel3D->bufferByteSubstateNum);
    clSetKernelArg(*kernel, 4, sizeof(CALCLmem), &calclmodel3D->bufferIntSubstateNum);
    clSetKernelArg(*kernel, 5, sizeof(CALCLmem), &calclmodel3D->bufferRealSubstateNum);
    clSetKernelArg(*kernel, 6, sizeof(CALCLmem), &calclmodel3D->bufferCurrentByteSubstate);
    clSetKernelArg(*kernel, 7, sizeof(CALCLmem), &calclmodel3D->bufferCurrentIntSubstate);
    clSetKernelArg(*kernel, 8, sizeof(CALCLmem), &calclmodel3D->bufferCurrentRealSubstate);
    clSetKernelArg(*kernel, 9, sizeof(CALCLmem), &calclmodel3D->bufferNextByteSubstate);
    clSetKernelArg(*kernel, 10, sizeof(CALCLmem), &calclmodel3D->bufferNextIntSubstate);
    clSetKernelArg(*kernel, 11, sizeof(CALCLmem), &calclmodel3D->bufferNextRealSubstate);
    clSetKernelArg(*kernel, 12, sizeof(CALCLmem), &calclmodel3D->bufferActiveCells);
    clSetKernelArg(*kernel, 13, sizeof(CALCLmem), &calclmodel3D->bufferActiveCellsNum);
    clSetKernelArg(*kernel, 14, sizeof(CALCLmem), &calclmodel3D->bufferActiveCellsFlags);
    clSetKernelArg(*kernel, 15, sizeof(CALCLmem), &calclmodel3D->bufferNeighborhood);
    clSetKernelArg(*kernel, 16, sizeof(CALCLmem), &calclmodel3D->bufferNeighborhoodID);
    clSetKernelArg(*kernel, 17, sizeof(CALCLmem), &calclmodel3D->bufferNeighborhoodSize);
    clSetKernelArg(*kernel, 18, sizeof(CALCLmem), &calclmodel3D->bufferBoundaryCondition);
    clSetKernelArg(*kernel, 19, sizeof(CALCLmem), &calclmodel3D->bufferStop);
    clSetKernelArg(*kernel, 20, sizeof(CALCLmem), &calclmodel3D->bufferSTCountsDiff);
    double chunk_double = ceil((double)(calclmodel3D->host_CA->rows * calclmodel3D->host_CA->columns*calclmodel3D->host_CA->slices)/calclmodel3D->streamCompactionThreadsNum);
    int chunk = (int)chunk_double;
    clSetKernelArg(*kernel, 21, sizeof(int), &chunk);



    clSetKernelArg(*kernel, 52, sizeof(CALCLmem), &calclmodel3D->bufferSingleLayerByteSubstateNum);
    clSetKernelArg(*kernel, 53, sizeof(CALCLmem), &calclmodel3D->bufferSingleLayerIntSubstateNum);
    clSetKernelArg(*kernel, 54, sizeof(CALCLmem), &calclmodel3D->bufferSingleLayerRealSubstateNum);
    clSetKernelArg(*kernel, 55, sizeof(CALCLmem), &calclmodel3D->bufferSingleLayerByteSubstate);
    clSetKernelArg(*kernel, 56, sizeof(CALCLmem), &calclmodel3D->bufferSingleLayerIntSubstate);
    clSetKernelArg(*kernel, 57, sizeof(CALCLmem), &calclmodel3D->bufferSingleLayerRealSubstate);
    clSetKernelArg(*kernel, 58, sizeof(CALint), &calclmodel3D->borderSize);
    clSetKernelArg(*kernel, 59, sizeof(cl_uint), &calclmodel3D->goffset);

}

void calclSetReductionParameters3D(struct CALCLModel3D* calclmodel3D, CALCLkernel * kernel) {

    clSetKernelArg(*kernel, 22, sizeof(CALCLmem), &calclmodel3D->bufferMinimab);
    clSetKernelArg(*kernel, 25, sizeof(CALCLmem), &calclmodel3D->bufferMaximab);
    clSetKernelArg(*kernel, 28, sizeof(CALCLmem), &calclmodel3D->bufferSumb);
    clSetKernelArg(*kernel, 31, sizeof(CALCLmem), &calclmodel3D->bufferLogicalAndsb);
    clSetKernelArg(*kernel, 34, sizeof(CALCLmem), &calclmodel3D->bufferLogicalOrsb);
    clSetKernelArg(*kernel, 37, sizeof(CALCLmem), &calclmodel3D->bufferLogicalXOrsb);
    clSetKernelArg(*kernel, 40, sizeof(CALCLmem), &calclmodel3D->bufferBinaryAndsb);
    clSetKernelArg(*kernel, 43, sizeof(CALCLmem), &calclmodel3D->bufferBinaryOrsb);
    clSetKernelArg(*kernel, 46, sizeof(CALCLmem), &calclmodel3D->bufferBinaryXOrsb);

    clSetKernelArg(*kernel, 23, sizeof(CALCLmem), &calclmodel3D->bufferMinimai);
    clSetKernelArg(*kernel, 26, sizeof(CALCLmem), &calclmodel3D->bufferMaximai);
    clSetKernelArg(*kernel, 29, sizeof(CALCLmem), &calclmodel3D->bufferSumi);
    clSetKernelArg(*kernel, 32, sizeof(CALCLmem), &calclmodel3D->bufferLogicalAndsi);
    clSetKernelArg(*kernel, 35, sizeof(CALCLmem), &calclmodel3D->bufferLogicalOrsi);
    clSetKernelArg(*kernel, 38, sizeof(CALCLmem), &calclmodel3D->bufferLogicalXOrsi);
    clSetKernelArg(*kernel, 41, sizeof(CALCLmem), &calclmodel3D->bufferBinaryAndsi);
    clSetKernelArg(*kernel, 44, sizeof(CALCLmem), &calclmodel3D->bufferBinaryOrsi);
    clSetKernelArg(*kernel, 47, sizeof(CALCLmem), &calclmodel3D->bufferBinaryXOrsi);

    clSetKernelArg(*kernel, 24, sizeof(CALCLmem), &calclmodel3D->bufferMinimar);
    clSetKernelArg(*kernel, 27, sizeof(CALCLmem), &calclmodel3D->bufferMaximar);
    clSetKernelArg(*kernel, 30, sizeof(CALCLmem), &calclmodel3D->bufferSumr);
    clSetKernelArg(*kernel, 33, sizeof(CALCLmem), &calclmodel3D->bufferLogicalAndsr);
    clSetKernelArg(*kernel, 36, sizeof(CALCLmem), &calclmodel3D->bufferLogicalOrsr);
    clSetKernelArg(*kernel, 39, sizeof(CALCLmem), &calclmodel3D->bufferLogicalXOrsr);
    clSetKernelArg(*kernel, 42, sizeof(CALCLmem), &calclmodel3D->bufferBinaryAndsr);
    clSetKernelArg(*kernel, 45, sizeof(CALCLmem), &calclmodel3D->bufferBinaryOrsr);
    clSetKernelArg(*kernel, 48, sizeof(CALCLmem), &calclmodel3D->bufferBinaryXOrsr);

    clSetKernelArg(*kernel, 49, sizeof(CALCLmem), &calclmodel3D->bufferProdb);
    clSetKernelArg(*kernel, 50, sizeof(CALCLmem), &calclmodel3D->bufferProdi);
    clSetKernelArg(*kernel, 51, sizeof(CALCLmem), &calclmodel3D->bufferProdr);

}

void calclRealSubstatesMapper3D(struct CALModel3D *model, CALreal * current, CALreal * next,  const CALint workload, const CALint offset,const CALint borderSize) {
    int ssNum = model->sizeof_pQr_array;
    size_t elNum = model->columns * model->rows * workload;
    int dimLayer =  model->columns * model->rows;
    long int outIndex = borderSize * dimLayer;
    long int outIndex1 = borderSize * dimLayer;
    int i;
    unsigned int j;

    for (i = 0; i < ssNum; i++) {
        for (j = 0; j < elNum; j++)
            current[outIndex++] = model->pQr_array[i]->current[j+offset* dimLayer];
        outIndex=outIndex+2*borderSize* dimLayer;
        for (j = 0; j < elNum; j++)
            next[outIndex1++] = model->pQr_array[i]->next[j+offset* dimLayer];
        outIndex1=outIndex1+2*borderSize* dimLayer;
    }
}
void calclByteSubstatesMapper3D(struct CALModel3D *model, CALbyte * current, CALbyte * next,  const CALint workload, const CALint offset,const CALint borderSize) {
    int ssNum = model->sizeof_pQb_array;
    size_t elNum = model->columns * model->rows * workload;
    int dimLayer =  model->columns * model->rows;
    long int outIndex = borderSize * dimLayer;
    long int outIndex1 = borderSize * dimLayer;
    int i;
    unsigned int j;

    for (i = 0; i < ssNum; i++) {
        for (j = 0; j < elNum; j++)
            current[outIndex++] = model->pQb_array[i]->current[j+offset* dimLayer];
        outIndex=outIndex+2*borderSize* dimLayer;
        for (j = 0; j < elNum; j++)
            next[outIndex1++] = model->pQb_array[i]->next[j+offset* dimLayer];
        outIndex1=outIndex1+2*borderSize* dimLayer;
    }
}
void calclIntSubstatesMapper3D(struct CALModel3D *model, CALint * current, CALint * next,  const CALint workload, const CALint offset,const CALint borderSize) {
    int ssNum = model->sizeof_pQi_array;
    size_t elNum = model->columns * model->rows * workload;
    int dimLayer =  model->columns * model->rows;
    long int outIndex = borderSize * dimLayer;
    long int outIndex1 = borderSize * dimLayer;
    int i;
    unsigned int j;

    for (i = 0; i < ssNum; i++) {
        for (j = 0; j < elNum; j++)
            current[outIndex++] = model->pQi_array[i]->current[j+offset* dimLayer];
        outIndex=outIndex+2*borderSize* dimLayer;
        for (j = 0; j < elNum; j++)
            next[outIndex1++] = model->pQi_array[i]->next[j+offset* dimLayer];
        outIndex1=outIndex1+2*borderSize* dimLayer;
    }
}

void calclSingleLayerRealSubstatesMapper3D(struct CALModel3D * host_CA, CALreal * current) {
    int ssNum = host_CA->sizeof_pQr_single_layer_array;
    size_t elNum = host_CA->columns * host_CA->rows* host_CA->slices;
    long int outIndex = 0;
    int i;
    unsigned int j;

    for (i = 0; i < ssNum; i++)
        for (j = 0; j < elNum; j++)
            current[outIndex++] = host_CA->pQr_single_layer_array[i]->current[j];

}
void calclSingleLayerByteSubstatesMapper3D(struct CALModel3D * host_CA, CALbyte * current) {
    int ssNum = host_CA->sizeof_pQb_single_layer_array;
    size_t elNum = host_CA->columns * host_CA->rows* host_CA->slices;
    long int outIndex = 0;
    int i;
    unsigned int j;

    for (i = 0; i < ssNum; i++)
        for (j = 0; j < elNum; j++)
            current[outIndex++] = host_CA->pQb_single_layer_array[i]->current[j];

}
void calclSingleLayerIntSubstatesMapper3D(struct CALModel3D * host_CA, CALint * current) {
    int ssNum = host_CA->sizeof_pQi_single_layer_array;
    size_t elNum = host_CA->columns * host_CA->rows* host_CA->slices;
    long int outIndex = 0;
    int i;
    unsigned int j;

    for (i = 0; i < ssNum; i++)
        for (j = 0; j < elNum; j++)
            current[outIndex++] = host_CA->pQi_single_layer_array[i]->current[j];

}

void calclSetKernelMergeFlags3D(struct CALCLModel3D* calclmodel3D) {

  clSetKernelArg(calclmodel3D->kernelMergeFlags, 0, sizeof(CALint),
                 &calclmodel3D->rows);
  clSetKernelArg(calclmodel3D->kernelMergeFlags, 1, sizeof(CALint),
                 &calclmodel3D->columns);
  clSetKernelArg(calclmodel3D->kernelMergeFlags, 2, sizeof(CALint),
                 &calclmodel3D->slices);
  clSetKernelArg(calclmodel3D->kernelMergeFlags, 3, sizeof(CALint),
                 &calclmodel3D->borderSize);
  clSetKernelArg(calclmodel3D->kernelMergeFlags, 4, sizeof(CALCLmem),
                 &calclmodel3D->borderMapper.mergeflagsBorder);
  clSetKernelArg(calclmodel3D->kernelMergeFlags, 5, sizeof(CALCLmem),
                 &calclmodel3D->bufferActiveCellsFlagsRealSize);
  //clSetKernelArg(calclmodel3D->kernelMergeFlags, 5, sizeof(CALCLmem),
   //              &calclmodel3D->bufferSTCountsDiff);
 //calclHandleError(err);
  //clSetKernelArg(calclmodel3D->kernelMergeFlags, 6, sizeof(CALint), &calclmodel3D->chunk);
  // clSetKernelArg(calclmodel3D->kernelMergeFlags, 4, sizeof(CALCLmem),
  // &calclmodel3D->bufferActiveCellsFlags);


  clSetKernelArg(calclmodel3D->kernelSetDiffFlags, 0, sizeof(CALint),
                 &calclmodel3D->rows);
  clSetKernelArg(calclmodel3D->kernelSetDiffFlags, 1, sizeof(CALint),
                 &calclmodel3D->columns);
  clSetKernelArg(calclmodel3D->kernelSetDiffFlags, 3, sizeof(CALint),
                 &calclmodel3D->slices);
  clSetKernelArg(calclmodel3D->kernelSetDiffFlags, 4, sizeof(CALint),
                 &calclmodel3D->borderSize);
  clSetKernelArg(calclmodel3D->kernelSetDiffFlags, 5, sizeof(CALint),
                 &calclmodel3D->chunk);
  clSetKernelArg(calclmodel3D->kernelSetDiffFlags, 6, sizeof(CALCLmem),
                 &calclmodel3D->bufferSTCountsDiff);
}

CALCLqueue calclCreateQueue3D(struct CALCLModel3D * calclmodel3D, CALCLcontext context, CALCLdevice device) {
    CALCLqueue queue = calclCreateCommandQueue(context, device);
    size_t cores;
    cl_int err;
    err = clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &cores, NULL);
    calclHandleError(err);

    //TODO choose stream compaction threads num
    calclmodel3D->streamCompactionThreadsNum = cores * 4;

    while (calclmodel3D->host_CA->rows * calclmodel3D->host_CA->columns * calclmodel3D->host_CA->slices <= (int)calclmodel3D->streamCompactionThreadsNum)
        calclmodel3D->streamCompactionThreadsNum /= 2;

    calclmodel3D->bufferSTCounts = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, sizeof(CALint) * calclmodel3D->streamCompactionThreadsNum, NULL, &err);
    calclHandleError(err);
    calclmodel3D->bufferSTOffsets1 = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, sizeof(CALint) * calclmodel3D->streamCompactionThreadsNum, NULL, &err);
    calclHandleError(err);
    CALbyte * diff = (CALbyte*) malloc(sizeof(CALbyte) * calclmodel3D->streamCompactionThreadsNum);
    memset(diff, CAL_TRUE, sizeof(CALbyte) * calclmodel3D->streamCompactionThreadsNum);
    calclmodel3D->bufferSTCountsDiff = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, calclmodel3D->streamCompactionThreadsNum * sizeof(CALbyte), diff, &err);
    calclHandleError(err);
    free(diff);

    CALbyte patter = 0;
    clEnqueueFillBuffer(queue,
                        calclmodel3D->bufferActiveCellsFlags,
                        &patter,
                        sizeof(CALbyte),
                        0,
                        sizeof(CALbyte)*calclmodel3D->fullSize,0,
                        NULL,
                        NULL
      );

    clEnqueueWriteBuffer(
        queue, calclmodel3D->bufferActiveCellsFlags, CL_TRUE,
        sizeof(CALbyte) * calclmodel3D->host_CA->columns * calclmodel3D->borderSize,
        sizeof(CALbyte) * calclmodel3D->realSize,
        calclmodel3D->host_CA->A->flags + (calclmodel3D->offset * calclmodel3D->host_CA->columns), 0, NULL,
        NULL );

    cl_buffer_region region;
    region.origin = sizeof(CALbyte) * calclmodel3D->host_CA->rows* calclmodel3D->host_CA->columns *
                    calclmodel3D->borderSize;
    region.size = sizeof(CALbyte) * calclmodel3D->realSize;

    calclmodel3D->bufferActiveCellsFlagsRealSize = clCreateSubBuffer(
        calclmodel3D->bufferActiveCellsFlags, CL_MEM_READ_WRITE,
        CL_BUFFER_CREATE_TYPE_REGION, &region, &err);
    calclHandleError(err);
    calclSetKernelMergeFlags3D(calclmodel3D);


    calclSetKernelStreamCompactionArgs3D(calclmodel3D);

    return queue;
}

void calclSetKernelCopyArgs3Di(struct CALCLModel3D* calclmodel3D) {
    clSetKernelArg(calclmodel3D->kernelMinCopyi, 0, sizeof(CALCLmem), &calclmodel3D->bufferPartialMini);
    clSetKernelArg(calclmodel3D->kernelMaxCopyi, 0, sizeof(CALCLmem), &calclmodel3D->bufferPartialMaxi);
    clSetKernelArg(calclmodel3D->kernelSumCopyi, 0, sizeof(CALCLmem), &calclmodel3D->bufferPartialSumi);
    clSetKernelArg(calclmodel3D->kernelProdCopyi, 0, sizeof(CALCLmem), &calclmodel3D->bufferPartialProdi);
    clSetKernelArg(calclmodel3D->kernelLogicalAndCopyi, 0, sizeof(CALCLmem), &calclmodel3D->bufferPartialLogicalAndi);
    clSetKernelArg(calclmodel3D->kernelLogicalOrCopyi, 0, sizeof(CALCLmem), &calclmodel3D->bufferPartialLogicalOri);
    clSetKernelArg(calclmodel3D->kernelLogicalXOrCopyi, 0, sizeof(CALCLmem), &calclmodel3D->bufferPartialLogicalXOri);
    clSetKernelArg(calclmodel3D->kernelBinaryAndCopyi, 0, sizeof(CALCLmem), &calclmodel3D->bufferPartialBinaryAndi);
    clSetKernelArg(calclmodel3D->kernelBinaryOrCopyi, 0, sizeof(CALCLmem), &calclmodel3D->bufferPartialBinaryOri);
    clSetKernelArg(calclmodel3D->kernelBinaryXOrCopyi, 0, sizeof(CALCLmem), &calclmodel3D->bufferPartialBinaryXOri);

    clSetKernelArg(calclmodel3D->kernelMinCopyi, 1, sizeof(CALCLmem), &calclmodel3D->bufferCurrentIntSubstate);
    clSetKernelArg(calclmodel3D->kernelMaxCopyi, 1, sizeof(CALCLmem), &calclmodel3D->bufferCurrentIntSubstate);
    clSetKernelArg(calclmodel3D->kernelSumCopyi, 1, sizeof(CALCLmem), &calclmodel3D->bufferCurrentIntSubstate);
    clSetKernelArg(calclmodel3D->kernelProdCopyi, 1, sizeof(CALCLmem), &calclmodel3D->bufferCurrentIntSubstate);
    clSetKernelArg(calclmodel3D->kernelLogicalAndCopyi, 1, sizeof(CALCLmem), &calclmodel3D->bufferCurrentIntSubstate);
    clSetKernelArg(calclmodel3D->kernelLogicalOrCopyi, 1, sizeof(CALCLmem), &calclmodel3D->bufferCurrentIntSubstate);
    clSetKernelArg(calclmodel3D->kernelLogicalXOrCopyi, 1, sizeof(CALCLmem), &calclmodel3D->bufferCurrentIntSubstate);
    clSetKernelArg(calclmodel3D->kernelBinaryAndCopyi, 1, sizeof(CALCLmem), &calclmodel3D->bufferCurrentIntSubstate);
    clSetKernelArg(calclmodel3D->kernelBinaryOrCopyi, 1, sizeof(CALCLmem), &calclmodel3D->bufferCurrentIntSubstate);
    clSetKernelArg(calclmodel3D->kernelBinaryXOrCopyi, 1, sizeof(CALCLmem), &calclmodel3D->bufferCurrentIntSubstate);

    clSetKernelArg(calclmodel3D->kernelMinCopyi, 3, sizeof(CALint), &calclmodel3D->host_CA->rows);
    clSetKernelArg(calclmodel3D->kernelMaxCopyi, 3, sizeof(CALint), &calclmodel3D->host_CA->rows);
    clSetKernelArg(calclmodel3D->kernelSumCopyi, 3, sizeof(CALint), &calclmodel3D->host_CA->rows);
    clSetKernelArg(calclmodel3D->kernelProdCopyi, 3, sizeof(CALint), &calclmodel3D->host_CA->rows);
    clSetKernelArg(calclmodel3D->kernelLogicalAndCopyi, 3, sizeof(CALint), &calclmodel3D->host_CA->rows);
    clSetKernelArg(calclmodel3D->kernelLogicalOrCopyi, 3, sizeof(CALint), &calclmodel3D->host_CA->rows);
    clSetKernelArg(calclmodel3D->kernelLogicalXOrCopyi, 3, sizeof(CALint), &calclmodel3D->host_CA->rows);
    clSetKernelArg(calclmodel3D->kernelBinaryAndCopyi, 3, sizeof(CALint), &calclmodel3D->host_CA->rows);
    clSetKernelArg(calclmodel3D->kernelBinaryOrCopyi, 3, sizeof(CALint), &calclmodel3D->host_CA->rows);
    clSetKernelArg(calclmodel3D->kernelBinaryXOrCopyi, 3, sizeof(CALint), &calclmodel3D->host_CA->rows);

    clSetKernelArg(calclmodel3D->kernelMinCopyi, 4, sizeof(CALint), &calclmodel3D->host_CA->columns);
    clSetKernelArg(calclmodel3D->kernelMaxCopyi, 4, sizeof(CALint), &calclmodel3D->host_CA->columns);
    clSetKernelArg(calclmodel3D->kernelSumCopyi, 4, sizeof(CALint), &calclmodel3D->host_CA->columns);
    clSetKernelArg(calclmodel3D->kernelProdCopyi, 4, sizeof(CALint), &calclmodel3D->host_CA->columns);
    clSetKernelArg(calclmodel3D->kernelLogicalAndCopyi, 4, sizeof(CALint), &calclmodel3D->host_CA->columns);
    clSetKernelArg(calclmodel3D->kernelLogicalOrCopyi, 4, sizeof(CALint), &calclmodel3D->host_CA->columns);
    clSetKernelArg(calclmodel3D->kernelLogicalXOrCopyi, 4, sizeof(CALint), &calclmodel3D->host_CA->columns);
    clSetKernelArg(calclmodel3D->kernelBinaryAndCopyi, 4, sizeof(CALint), &calclmodel3D->host_CA->columns);
    clSetKernelArg(calclmodel3D->kernelBinaryOrCopyi, 4, sizeof(CALint), &calclmodel3D->host_CA->columns);
    clSetKernelArg(calclmodel3D->kernelBinaryXOrCopyi, 4, sizeof(CALint), &calclmodel3D->host_CA->columns);

    clSetKernelArg(calclmodel3D->kernelMinCopyi, 5, sizeof(CALint), &calclmodel3D->host_CA->slices);
    clSetKernelArg(calclmodel3D->kernelMaxCopyi, 5, sizeof(CALint), &calclmodel3D->host_CA->slices);
    clSetKernelArg(calclmodel3D->kernelSumCopyi, 5, sizeof(CALint), &calclmodel3D->host_CA->slices);
    clSetKernelArg(calclmodel3D->kernelProdCopyi, 5, sizeof(CALint), &calclmodel3D->host_CA->slices);
    clSetKernelArg(calclmodel3D->kernelLogicalAndCopyi, 5, sizeof(CALint), &calclmodel3D->host_CA->slices);
    clSetKernelArg(calclmodel3D->kernelLogicalOrCopyi, 5, sizeof(CALint), &calclmodel3D->host_CA->slices);
    clSetKernelArg(calclmodel3D->kernelLogicalXOrCopyi, 5, sizeof(CALint), &calclmodel3D->host_CA->slices);
    clSetKernelArg(calclmodel3D->kernelBinaryAndCopyi, 5, sizeof(CALint), &calclmodel3D->host_CA->slices);
    clSetKernelArg(calclmodel3D->kernelBinaryOrCopyi, 5, sizeof(CALint), &calclmodel3D->host_CA->slices);
    clSetKernelArg(calclmodel3D->kernelBinaryXOrCopyi, 5, sizeof(CALint), &calclmodel3D->host_CA->slices);
}

void calclSetKernelCopyArgs3Db(struct CALCLModel3D* calclmodel3D) {
    clSetKernelArg(calclmodel3D->kernelMinCopyb, 0, sizeof(CALCLmem), &calclmodel3D->bufferPartialMinb);
    clSetKernelArg(calclmodel3D->kernelMaxCopyb, 0, sizeof(CALCLmem), &calclmodel3D->bufferPartialMaxb);
    clSetKernelArg(calclmodel3D->kernelSumCopyb, 0, sizeof(CALCLmem), &calclmodel3D->bufferPartialSumb);
    clSetKernelArg(calclmodel3D->kernelProdCopyb, 0, sizeof(CALCLmem), &calclmodel3D->bufferPartialProdb);
    clSetKernelArg(calclmodel3D->kernelLogicalAndCopyb, 0, sizeof(CALCLmem), &calclmodel3D->bufferPartialLogicalAndb);
    clSetKernelArg(calclmodel3D->kernelLogicalOrCopyb, 0, sizeof(CALCLmem), &calclmodel3D->bufferPartialLogicalOrb);
    clSetKernelArg(calclmodel3D->kernelLogicalXOrCopyb, 0, sizeof(CALCLmem), &calclmodel3D->bufferPartialLogicalXOrb);
    clSetKernelArg(calclmodel3D->kernelBinaryAndCopyb, 0, sizeof(CALCLmem), &calclmodel3D->bufferPartialBinaryAndb);
    clSetKernelArg(calclmodel3D->kernelBinaryOrCopyb, 0, sizeof(CALCLmem), &calclmodel3D->bufferPartialBinaryOrb);
    clSetKernelArg(calclmodel3D->kernelBinaryXOrCopyb, 0, sizeof(CALCLmem), &calclmodel3D->bufferPartialBinaryXOrb);

    clSetKernelArg(calclmodel3D->kernelMinCopyb, 1, sizeof(CALCLmem), &calclmodel3D->bufferCurrentByteSubstate);
    clSetKernelArg(calclmodel3D->kernelMaxCopyb, 1, sizeof(CALCLmem), &calclmodel3D->bufferCurrentByteSubstate);
    clSetKernelArg(calclmodel3D->kernelSumCopyb, 1, sizeof(CALCLmem), &calclmodel3D->bufferCurrentByteSubstate);
    clSetKernelArg(calclmodel3D->kernelProdCopyb, 1, sizeof(CALCLmem), &calclmodel3D->bufferCurrentByteSubstate);
    clSetKernelArg(calclmodel3D->kernelLogicalAndCopyb, 1, sizeof(CALCLmem), &calclmodel3D->bufferCurrentByteSubstate);
    clSetKernelArg(calclmodel3D->kernelLogicalOrCopyb, 1, sizeof(CALCLmem), &calclmodel3D->bufferCurrentByteSubstate);
    clSetKernelArg(calclmodel3D->kernelLogicalXOrCopyb, 1, sizeof(CALCLmem), &calclmodel3D->bufferCurrentByteSubstate);
    clSetKernelArg(calclmodel3D->kernelBinaryAndCopyb, 1, sizeof(CALCLmem), &calclmodel3D->bufferCurrentByteSubstate);
    clSetKernelArg(calclmodel3D->kernelBinaryOrCopyb, 1, sizeof(CALCLmem), &calclmodel3D->bufferCurrentByteSubstate);
    clSetKernelArg(calclmodel3D->kernelBinaryXOrCopyb, 1, sizeof(CALCLmem), &calclmodel3D->bufferCurrentByteSubstate);

    clSetKernelArg(calclmodel3D->kernelMinCopyb, 3, sizeof(CALint), &calclmodel3D->host_CA->rows);
    clSetKernelArg(calclmodel3D->kernelMaxCopyb, 3, sizeof(CALint), &calclmodel3D->host_CA->rows);
    clSetKernelArg(calclmodel3D->kernelSumCopyb, 3, sizeof(CALint), &calclmodel3D->host_CA->rows);
    clSetKernelArg(calclmodel3D->kernelProdCopyb, 3, sizeof(CALint), &calclmodel3D->host_CA->rows);
    clSetKernelArg(calclmodel3D->kernelLogicalAndCopyb, 3, sizeof(CALint), &calclmodel3D->host_CA->rows);
    clSetKernelArg(calclmodel3D->kernelLogicalOrCopyb, 3, sizeof(CALint), &calclmodel3D->host_CA->rows);
    clSetKernelArg(calclmodel3D->kernelLogicalXOrCopyb, 3, sizeof(CALint), &calclmodel3D->host_CA->rows);
    clSetKernelArg(calclmodel3D->kernelBinaryAndCopyb, 3, sizeof(CALint), &calclmodel3D->host_CA->rows);
    clSetKernelArg(calclmodel3D->kernelBinaryOrCopyb, 3, sizeof(CALint), &calclmodel3D->host_CA->rows);
    clSetKernelArg(calclmodel3D->kernelBinaryXOrCopyb, 3, sizeof(CALint), &calclmodel3D->host_CA->rows);

    clSetKernelArg(calclmodel3D->kernelMinCopyb, 4, sizeof(CALint), &calclmodel3D->host_CA->columns);
    clSetKernelArg(calclmodel3D->kernelMaxCopyb, 4, sizeof(CALint), &calclmodel3D->host_CA->columns);
    clSetKernelArg(calclmodel3D->kernelSumCopyb, 4, sizeof(CALint), &calclmodel3D->host_CA->columns);
    clSetKernelArg(calclmodel3D->kernelProdCopyb, 4, sizeof(CALint), &calclmodel3D->host_CA->columns);
    clSetKernelArg(calclmodel3D->kernelLogicalAndCopyb, 4, sizeof(CALint), &calclmodel3D->host_CA->columns);
    clSetKernelArg(calclmodel3D->kernelLogicalOrCopyb, 4, sizeof(CALint), &calclmodel3D->host_CA->columns);
    clSetKernelArg(calclmodel3D->kernelLogicalXOrCopyb, 4, sizeof(CALint), &calclmodel3D->host_CA->columns);
    clSetKernelArg(calclmodel3D->kernelBinaryAndCopyb, 4, sizeof(CALint), &calclmodel3D->host_CA->columns);
    clSetKernelArg(calclmodel3D->kernelBinaryOrCopyb, 4, sizeof(CALint), &calclmodel3D->host_CA->columns);
    clSetKernelArg(calclmodel3D->kernelBinaryXOrCopyb, 4, sizeof(CALint), &calclmodel3D->host_CA->columns);

    clSetKernelArg(calclmodel3D->kernelMinCopyb, 5, sizeof(CALint), &calclmodel3D->host_CA->slices);
    clSetKernelArg(calclmodel3D->kernelMaxCopyb, 5, sizeof(CALint), &calclmodel3D->host_CA->slices);
    clSetKernelArg(calclmodel3D->kernelSumCopyb, 5, sizeof(CALint), &calclmodel3D->host_CA->slices);
    clSetKernelArg(calclmodel3D->kernelProdCopyb, 5, sizeof(CALint), &calclmodel3D->host_CA->slices);
    clSetKernelArg(calclmodel3D->kernelLogicalAndCopyb, 5, sizeof(CALint), &calclmodel3D->host_CA->slices);
    clSetKernelArg(calclmodel3D->kernelLogicalOrCopyb, 5, sizeof(CALint), &calclmodel3D->host_CA->slices);
    clSetKernelArg(calclmodel3D->kernelLogicalXOrCopyb, 5, sizeof(CALint), &calclmodel3D->host_CA->slices);
    clSetKernelArg(calclmodel3D->kernelBinaryAndCopyb, 5, sizeof(CALint), &calclmodel3D->host_CA->slices);
    clSetKernelArg(calclmodel3D->kernelBinaryOrCopyb, 5, sizeof(CALint), &calclmodel3D->host_CA->slices);
    clSetKernelArg(calclmodel3D->kernelBinaryXOrCopyb, 5, sizeof(CALint), &calclmodel3D->host_CA->slices);
}

void calclSetKernelCopyArgs3Dr(struct CALCLModel3D* calclmodel3D) {
    clSetKernelArg(calclmodel3D->kernelMinCopyr, 0, sizeof(CALCLmem), &calclmodel3D->bufferPartialMinr);
    clSetKernelArg(calclmodel3D->kernelMaxCopyr, 0, sizeof(CALCLmem), &calclmodel3D->bufferPartialMaxr);
    clSetKernelArg(calclmodel3D->kernelSumCopyr, 0, sizeof(CALCLmem), &calclmodel3D->bufferPartialSumr);
    clSetKernelArg(calclmodel3D->kernelProdCopyr, 0, sizeof(CALCLmem), &calclmodel3D->bufferPartialProdr);
    clSetKernelArg(calclmodel3D->kernelLogicalAndCopyr, 0, sizeof(CALCLmem), &calclmodel3D->bufferPartialLogicalAndr);
    clSetKernelArg(calclmodel3D->kernelLogicalOrCopyr, 0, sizeof(CALCLmem), &calclmodel3D->bufferPartialLogicalOrr);
    clSetKernelArg(calclmodel3D->kernelLogicalXOrCopyr, 0, sizeof(CALCLmem), &calclmodel3D->bufferPartialLogicalXOrr);
    clSetKernelArg(calclmodel3D->kernelBinaryAndCopyr, 0, sizeof(CALCLmem), &calclmodel3D->bufferPartialBinaryAndr);
    clSetKernelArg(calclmodel3D->kernelBinaryOrCopyr, 0, sizeof(CALCLmem), &calclmodel3D->bufferPartialBinaryOrr);
    clSetKernelArg(calclmodel3D->kernelBinaryXOrCopyr, 0, sizeof(CALCLmem), &calclmodel3D->bufferPartialBinaryXOrr);

    clSetKernelArg(calclmodel3D->kernelMinCopyr, 1, sizeof(CALCLmem), &calclmodel3D->bufferCurrentRealSubstate);
    clSetKernelArg(calclmodel3D->kernelMaxCopyr, 1, sizeof(CALCLmem), &calclmodel3D->bufferCurrentRealSubstate);
    clSetKernelArg(calclmodel3D->kernelSumCopyr, 1, sizeof(CALCLmem), &calclmodel3D->bufferCurrentRealSubstate);
    clSetKernelArg(calclmodel3D->kernelProdCopyr, 1, sizeof(CALCLmem), &calclmodel3D->bufferCurrentRealSubstate);
    clSetKernelArg(calclmodel3D->kernelLogicalAndCopyr, 1, sizeof(CALCLmem), &calclmodel3D->bufferCurrentRealSubstate);
    clSetKernelArg(calclmodel3D->kernelLogicalOrCopyr, 1, sizeof(CALCLmem), &calclmodel3D->bufferCurrentRealSubstate);
    clSetKernelArg(calclmodel3D->kernelLogicalXOrCopyr, 1, sizeof(CALCLmem), &calclmodel3D->bufferCurrentRealSubstate);
    clSetKernelArg(calclmodel3D->kernelBinaryAndCopyr, 1, sizeof(CALCLmem), &calclmodel3D->bufferCurrentRealSubstate);
    clSetKernelArg(calclmodel3D->kernelBinaryOrCopyr, 1, sizeof(CALCLmem), &calclmodel3D->bufferCurrentRealSubstate);
    clSetKernelArg(calclmodel3D->kernelBinaryXOrCopyr, 1, sizeof(CALCLmem), &calclmodel3D->bufferCurrentRealSubstate);

    clSetKernelArg(calclmodel3D->kernelMinCopyr, 3, sizeof(CALint), &calclmodel3D->host_CA->rows);
    clSetKernelArg(calclmodel3D->kernelMaxCopyr, 3, sizeof(CALint), &calclmodel3D->host_CA->rows);
    clSetKernelArg(calclmodel3D->kernelSumCopyr, 3, sizeof(CALint), &calclmodel3D->host_CA->rows);
    clSetKernelArg(calclmodel3D->kernelProdCopyr, 3, sizeof(CALint), &calclmodel3D->host_CA->rows);
    clSetKernelArg(calclmodel3D->kernelLogicalAndCopyr, 3, sizeof(CALint), &calclmodel3D->host_CA->rows);
    clSetKernelArg(calclmodel3D->kernelLogicalOrCopyr, 3, sizeof(CALint), &calclmodel3D->host_CA->rows);
    clSetKernelArg(calclmodel3D->kernelLogicalXOrCopyr, 3, sizeof(CALint), &calclmodel3D->host_CA->rows);
    clSetKernelArg(calclmodel3D->kernelBinaryAndCopyr, 3, sizeof(CALint), &calclmodel3D->host_CA->rows);
    clSetKernelArg(calclmodel3D->kernelBinaryOrCopyr, 3, sizeof(CALint), &calclmodel3D->host_CA->rows);
    clSetKernelArg(calclmodel3D->kernelBinaryXOrCopyr, 3, sizeof(CALint), &calclmodel3D->host_CA->rows);

    clSetKernelArg(calclmodel3D->kernelMinCopyr, 4, sizeof(CALint), &calclmodel3D->host_CA->columns);
    clSetKernelArg(calclmodel3D->kernelMaxCopyr, 4, sizeof(CALint), &calclmodel3D->host_CA->columns);
    clSetKernelArg(calclmodel3D->kernelSumCopyr, 4, sizeof(CALint), &calclmodel3D->host_CA->columns);
    clSetKernelArg(calclmodel3D->kernelProdCopyr, 4, sizeof(CALint), &calclmodel3D->host_CA->columns);
    clSetKernelArg(calclmodel3D->kernelLogicalAndCopyr, 4, sizeof(CALint), &calclmodel3D->host_CA->columns);
    clSetKernelArg(calclmodel3D->kernelLogicalOrCopyr, 4, sizeof(CALint), &calclmodel3D->host_CA->columns);
    clSetKernelArg(calclmodel3D->kernelLogicalXOrCopyr, 4, sizeof(CALint), &calclmodel3D->host_CA->columns);
    clSetKernelArg(calclmodel3D->kernelBinaryAndCopyr, 4, sizeof(CALint), &calclmodel3D->host_CA->columns);
    clSetKernelArg(calclmodel3D->kernelBinaryOrCopyr, 4, sizeof(CALint), &calclmodel3D->host_CA->columns);
    clSetKernelArg(calclmodel3D->kernelBinaryXOrCopyr, 4, sizeof(CALint), &calclmodel3D->host_CA->columns);

    clSetKernelArg(calclmodel3D->kernelMinCopyr, 5, sizeof(CALint), &calclmodel3D->host_CA->slices);
    clSetKernelArg(calclmodel3D->kernelMaxCopyr, 5, sizeof(CALint), &calclmodel3D->host_CA->slices);
    clSetKernelArg(calclmodel3D->kernelSumCopyr, 5, sizeof(CALint), &calclmodel3D->host_CA->slices);
    clSetKernelArg(calclmodel3D->kernelProdCopyr, 5, sizeof(CALint), &calclmodel3D->host_CA->slices);
    clSetKernelArg(calclmodel3D->kernelLogicalAndCopyr, 5, sizeof(CALint), &calclmodel3D->host_CA->slices);
    clSetKernelArg(calclmodel3D->kernelLogicalOrCopyr, 5, sizeof(CALint), &calclmodel3D->host_CA->slices);
    clSetKernelArg(calclmodel3D->kernelLogicalXOrCopyr, 5, sizeof(CALint), &calclmodel3D->host_CA->slices);
    clSetKernelArg(calclmodel3D->kernelBinaryAndCopyr, 5, sizeof(CALint), &calclmodel3D->host_CA->slices);
    clSetKernelArg(calclmodel3D->kernelBinaryOrCopyr, 5, sizeof(CALint), &calclmodel3D->host_CA->slices);
    clSetKernelArg(calclmodel3D->kernelBinaryXOrCopyr, 5, sizeof(CALint), &calclmodel3D->host_CA->slices);
}

int upperPowerOfTwo3D(int n) {
    int power = 1;
    while (power < n)
        power <<= 1;
    return power;
}




/******************************************************************************
 * 							PUBLIC FUNCTIONS
 ******************************************************************************/

struct CALCLModel3D * calclCADef3D(struct CALModel3D *host_CA, CALCLcontext context, CALCLprogram program,CALCLdevice device,
                                   const CALint workload,
                                   const CALint offset,
                                   const CALint goffset,
                                   const CALint _borderSize) {


    printf("goffset = %d \n", goffset);
    struct CALCLModel3D * calclmodel3D = (struct CALCLModel3D*) malloc(sizeof(struct CALCLModel3D));
    //initialize calclmodel3D stuff
    calclmodel3D->host_CA = host_CA;
    calclmodel3D->opt = host_CA->OPTIMIZATION;
    calclmodel3D->cl_update_substates = NULL;
    calclmodel3D->kernelInitSubstates = NULL;
    calclmodel3D->kernelSteering = NULL;
    calclmodel3D->kernelStopCondition = NULL;
    calclmodel3D->elementaryProcessesNum = 0;
    calclmodel3D->steps = 0;

    calclmodel3D->slices = workload;
    calclmodel3D->rows = host_CA->rows;
    calclmodel3D->columns = host_CA->columns;
    calclmodel3D->borderSize = _borderSize;
    calclmodel3D->goffset = goffset;

    printf("calclmodel3D->goffset = %d \n", calclmodel3D->goffset);

    if (host_CA->A == NULL) {
        host_CA->A = malloc( sizeof(struct CALActiveCells3D));
        host_CA->A->cells = NULL;
        host_CA->A->size_current = 0;
        host_CA->A->size_next = 0;
        host_CA->A->flags = (CALbyte*) malloc(sizeof(CALbyte) * host_CA->rows * host_CA->columns * host_CA->slices);
        memset(host_CA->A->flags, CAL_FALSE, sizeof(CALbyte) * host_CA->rows * host_CA->columns * host_CA->slices);
    }

    cl_int err;
    int dimLayers = calclmodel3D->columns *calclmodel3D->rows;
    calclmodel3D->fullSize =
            dimLayers *
            (workload + calclmodel3D->borderSize * 2);
    calclmodel3D->realSize = calclmodel3D->slices * calclmodel3D->columns * calclmodel3D->rows;
    calclmodel3D->offset = offset;
    printf(" calclmodel3D->offset =  %d\n",  calclmodel3D->offset);
    int bufferDim = calclmodel3D->fullSize;
    //int bufferDim = host_CA->columns * host_CA->rows * host_CA->slices;

    calclmodel3D->kernelUpdateSubstate = calclGetKernelFromProgram(program, KER_UPDATESUBSTATES);

    //stream compaction kernels
    calclmodel3D->kernelCompact = calclGetKernelFromProgram(program, KER_STC_COMPACT);
    calclmodel3D->kernelComputeCounts = calclGetKernelFromProgram(program, KER_STC_COMPUTE_COUNTS);
    calclmodel3D->kernelUpSweep = calclGetKernelFromProgram(program, KER_STC_UP_SWEEP);
    calclmodel3D->kernelDownSweep = calclGetKernelFromProgram(program, KER_STC_DOWN_SWEEP);

    calclmodel3D->kernelMergeFlags = calclGetKernelFromProgram(program, KER_MERGE_FLAGS);
    calclmodel3D->kernelSetDiffFlags = calclGetKernelFromProgram(program, KER_SET_DIFF_FLAGS);

    calclmodel3D->kernelMinReductionb = calclGetKernelFromProgram(program, "calclMinReductionKernelb");
    calclmodel3D->kernelMaxReductionb = calclGetKernelFromProgram(program, "calclMaxReductionKernelb");
    calclmodel3D->kernelSumReductionb = calclGetKernelFromProgram(program, "calclSumReductionKernelb");
    calclmodel3D->kernelProdReductionb = calclGetKernelFromProgram(program, "calclProdReductionKernelb");
    calclmodel3D->kernelLogicalAndReductionb = calclGetKernelFromProgram(program, "calclLogicAndReductionKernelb");
    calclmodel3D->kernelLogicalOrReductionb = calclGetKernelFromProgram(program, "calclLogicOrReductionKernelb");
    calclmodel3D->kernelLogicalXOrReductionb = calclGetKernelFromProgram(program, "calclLogicXOrReductionKernelb");
    calclmodel3D->kernelBinaryAndReductionb = calclGetKernelFromProgram(program, "calclBinaryAndReductionKernelb");
    calclmodel3D->kernelBinaryOrReductionb = calclGetKernelFromProgram(program, "calclBinaryOrReductionKernelb");
    calclmodel3D->kernelBinaryXorReductionb = calclGetKernelFromProgram(program, "calclBinaryXOrReductionKernelb");

    calclmodel3D->kernelMinReductioni = calclGetKernelFromProgram(program, "calclMinReductionKerneli");
    calclmodel3D->kernelMaxReductioni = calclGetKernelFromProgram(program, "calclMaxReductionKerneli");
    calclmodel3D->kernelSumReductioni = calclGetKernelFromProgram(program, "calclSumReductionKerneli");
    calclmodel3D->kernelProdReductioni = calclGetKernelFromProgram(program, "calclProdReductionKerneli");
    calclmodel3D->kernelLogicalAndReductioni = calclGetKernelFromProgram(program, "calclLogicAndReductionKerneli");
    calclmodel3D->kernelLogicalOrReductioni = calclGetKernelFromProgram(program, "calclLogicOrReductionKerneli");
    calclmodel3D->kernelLogicalXOrReductioni = calclGetKernelFromProgram(program, "calclLogicXOrReductionKerneli");
    calclmodel3D->kernelBinaryAndReductioni = calclGetKernelFromProgram(program, "calclBinaryAndReductionKerneli");
    calclmodel3D->kernelBinaryOrReductioni = calclGetKernelFromProgram(program, "calclBinaryOrReductionKerneli");
    calclmodel3D->kernelBinaryXorReductioni = calclGetKernelFromProgram(program, "calclBinaryXOrReductionKerneli");

    calclmodel3D->kernelMinReductionr = calclGetKernelFromProgram(program, "calclMinReductionKernelr");
    calclmodel3D->kernelMaxReductionr = calclGetKernelFromProgram(program, "calclMaxReductionKernelr");
    calclmodel3D->kernelSumReductionr = calclGetKernelFromProgram(program, "calclSumReductionKernelr");
    calclmodel3D->kernelProdReductionr = calclGetKernelFromProgram(program, "calclProdReductionKernelr");
    calclmodel3D->kernelLogicalAndReductionr = calclGetKernelFromProgram(program, "calclLogicAndReductionKernelr");
    calclmodel3D->kernelLogicalOrReductionr = calclGetKernelFromProgram(program, "calclLogicOrReductionKernelr");
    calclmodel3D->kernelLogicalXOrReductionr = calclGetKernelFromProgram(program, "calclLogicXOrReductionKernelr");
    calclmodel3D->kernelBinaryAndReductionr = calclGetKernelFromProgram(program, "calclBinaryAndReductionKernelr");
    calclmodel3D->kernelBinaryOrReductionr = calclGetKernelFromProgram(program, "calclBinaryOrReductionKernelr");
    calclmodel3D->kernelBinaryXorReductionr = calclGetKernelFromProgram(program, "calclBinaryXOrReductionKernelr");
    struct CALCell3D * activeCells = (struct CALCell3D*) malloc(sizeof(struct CALCell3D) * calclmodel3D->realSize);
    calclmodel3D->num_active_cells=0;

    if(host_CA->A->size_current !=0){
        int s = 0;
        int i = 0;
        int j = 0;
        printf("Num active cells = %d    %d -- %d \n",calclmodel3D->host_CA->A->size_next,offset,calclmodel3D->slices+offset);
        for (s=offset; s < (calclmodel3D->slices+offset); s++)
            for (i=0; i < calclmodel3D->rows; i++)
                for (j=0; j < calclmodel3D->columns; j++)
                    if (calGetBuffer3DElement(calclmodel3D->host_CA->A->flags,calclmodel3D->host_CA->rows ,calclmodel3D->host_CA->columns, i, j,s))
                    {
                        activeCells[calclmodel3D->num_active_cells].i = i;
                        activeCells[calclmodel3D->num_active_cells].j = j;
                        activeCells[calclmodel3D->num_active_cells].k = s;
                        ++calclmodel3D->num_active_cells;
                    }

        printf("TOTALE N = %d \n",calclmodel3D->num_active_cells);

        //memcpy(activeCells, host_CA->A->cells, sizeof(struct CALCell3D) * host_CA->A->size_current);
    }
    calclmodel3D->bufferActiveCells = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(struct CALCell3D) * calclmodel3D->realSize, activeCells, &err);
    calclHandleError(err);
    free(activeCells);
    calclmodel3D->bufferActiveCellsFlags = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(CALbyte) * calclmodel3D->fullSize, NULL, &err);
    calclHandleError(err);

    calclmodel3D->bufferActiveCellsNum = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(CALint), &calclmodel3D->num_active_cells, &err);
    calclHandleError(err);

    calclmodel3D->bufferByteSubstateNum = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(CALint), &host_CA->sizeof_pQb_array, &err);
    calclHandleError(err);
    calclmodel3D->bufferIntSubstateNum = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(CALint), &host_CA->sizeof_pQi_array, &err);
    calclHandleError(err);
    calclmodel3D->bufferRealSubstateNum = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(CALint), &host_CA->sizeof_pQr_array, &err);
    calclHandleError(err);

    calclmodel3D->bufferColumns = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(CALint), &host_CA->columns, &err);
    calclHandleError(err);
    calclmodel3D->bufferRows = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(CALint), &host_CA->rows, &err);
    calclHandleError(err);
    int slicesplusborder = calclmodel3D->slices+calclmodel3D->borderSize*2;
    calclmodel3D->bufferSlices = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(CALint), &slicesplusborder, &err);
    calclHandleError(err);

    size_t byteSubstatesDim = sizeof(CALbyte) * bufferDim * host_CA->sizeof_pQb_array + 1;
    CALbyte * currentByteSubstates = (CALbyte*) malloc(byteSubstatesDim);
    CALbyte * nextByteSubstates = (CALbyte*) malloc(byteSubstatesDim);
    calclByteSubstatesMapper3D(host_CA, currentByteSubstates, nextByteSubstates,workload, offset, calclmodel3D->borderSize);
    calclmodel3D->bufferCurrentByteSubstate = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, byteSubstatesDim, currentByteSubstates, &err);
    calclHandleError(err);
    calclmodel3D->bufferNextByteSubstate = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, byteSubstatesDim, nextByteSubstates, &err);
    calclHandleError(err);
    free(currentByteSubstates);
    free(nextByteSubstates);

    size_t intSubstatesDim = sizeof(CALint) * bufferDim * host_CA->sizeof_pQi_array + 1;
    CALint * currentIntSubstates = (CALint*) malloc(intSubstatesDim);
    CALint * nextIntSubstates = (CALint*) malloc(intSubstatesDim);
    calclIntSubstatesMapper3D(host_CA, currentIntSubstates, nextIntSubstates,workload, offset, calclmodel3D->borderSize);
    calclmodel3D->bufferCurrentIntSubstate = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, intSubstatesDim, currentIntSubstates, &err);
    calclHandleError(err);
    calclmodel3D->bufferNextIntSubstate = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, intSubstatesDim, nextIntSubstates, &err);
    calclHandleError(err);
    free(currentIntSubstates);
    free(nextIntSubstates);

    size_t realSubstatesDim = sizeof(CALreal) * bufferDim * host_CA->sizeof_pQr_array + 1;
    CALreal * currentRealSubstates = (CALreal*) malloc(realSubstatesDim);
    CALreal * nextRealSubstates = (CALreal*) malloc(realSubstatesDim);
    calclRealSubstatesMapper3D(host_CA, currentRealSubstates, nextRealSubstates,workload, offset, calclmodel3D->borderSize);
    calclmodel3D->bufferCurrentRealSubstate = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, realSubstatesDim, currentRealSubstates, &err);
    calclHandleError(err);
    calclmodel3D->bufferNextRealSubstate = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, realSubstatesDim, nextRealSubstates, &err);
    calclHandleError(err);
    free(currentRealSubstates);
    free(nextRealSubstates);

    calclSetKernelsLibArgs3D(calclmodel3D);

    calclmodel3D->bufferSingleLayerByteSubstateNum = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(CALint), &calclmodel3D->host_CA->sizeof_pQb_single_layer_array, &err);
    calclHandleError(err);
    calclmodel3D->bufferSingleLayerIntSubstateNum = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(CALint), &calclmodel3D->host_CA->sizeof_pQi_single_layer_array, &err);
    calclHandleError(err);
    calclmodel3D->bufferSingleLayerRealSubstateNum = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(CALint), &calclmodel3D->host_CA->sizeof_pQr_single_layer_array, &err);
    calclHandleError(err);

    size_t byteSingleLayerSubstatesDim = sizeof(CALbyte) * bufferDim * calclmodel3D->host_CA->sizeof_pQb_single_layer_array + 1;
    CALbyte * currentSingleLayerByteSubstates = (CALbyte*) malloc(byteSingleLayerSubstatesDim);
    calclSingleLayerByteSubstatesMapper3D(calclmodel3D->host_CA, currentSingleLayerByteSubstates);
    calclmodel3D->bufferSingleLayerByteSubstate = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, byteSingleLayerSubstatesDim, currentSingleLayerByteSubstates, &err);
    calclHandleError(err);
    free(currentSingleLayerByteSubstates);

    size_t intSingleLayerSubstatesDim = sizeof(CALint) * bufferDim * calclmodel3D->host_CA->sizeof_pQi_single_layer_array + 1;
    CALint * currentSingleLayerIntSubstates = (CALint*) malloc(intSingleLayerSubstatesDim);
    calclSingleLayerIntSubstatesMapper3D(calclmodel3D->host_CA, currentSingleLayerIntSubstates);
    calclmodel3D->bufferSingleLayerIntSubstate = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, intSingleLayerSubstatesDim, currentSingleLayerIntSubstates, &err);
    calclHandleError(err);
    free(currentSingleLayerIntSubstates);

    size_t realSingleLayerSubstatesDim = sizeof(CALreal) * bufferDim * calclmodel3D->host_CA->sizeof_pQr_single_layer_array + 1;
    CALreal * currentSingleLayerRealSubstates = (CALreal*) malloc(realSingleLayerSubstatesDim);
    calclSingleLayerRealSubstatesMapper3D(calclmodel3D->host_CA, currentSingleLayerRealSubstates);
    calclmodel3D->bufferSingleLayerRealSubstate = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, realSingleLayerSubstatesDim, currentSingleLayerRealSubstates, &err);
    calclHandleError(err);
    free(currentSingleLayerRealSubstates);





    calclmodel3D->kernelMinCopyi = calclGetKernelFromProgram(program, "copy3Di");
    calclmodel3D->kernelMaxCopyi = calclGetKernelFromProgram(program, "copy3Di");
    calclmodel3D->kernelSumCopyi = calclGetKernelFromProgram(program, "copy3Di");
    calclmodel3D->kernelProdCopyi = calclGetKernelFromProgram(program, "copy3Di");
    calclmodel3D->kernelLogicalAndCopyi = calclGetKernelFromProgram(program, "copy3Di");
    calclmodel3D->kernelLogicalOrCopyi = calclGetKernelFromProgram(program, "copy3Di");
    calclmodel3D->kernelLogicalXOrCopyi = calclGetKernelFromProgram(program, "copy3Di");
    calclmodel3D->kernelBinaryAndCopyi = calclGetKernelFromProgram(program, "copy3Di");
    calclmodel3D->kernelBinaryOrCopyi = calclGetKernelFromProgram(program, "copy3Di");
    calclmodel3D->kernelBinaryXOrCopyi = calclGetKernelFromProgram(program, "copy3Di");

    CALint * partiali = (CALint*) malloc(calclmodel3D->host_CA->rows * calclmodel3D->host_CA->columns* calclmodel3D->host_CA->slices * sizeof(CALint));
    size_t dimCAi =  sizeof(CALint) * calclmodel3D->host_CA->rows * calclmodel3D->host_CA->columns*calclmodel3D->host_CA->slices;
    calclmodel3D->bufferPartialMini = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, dimCAi , partiali, &err);
    calclHandleError(err);
    calclmodel3D->bufferPartialMaxi = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, dimCAi, partiali,
                                                     &err);
    calclHandleError(err);
    calclmodel3D->bufferPartialSumi = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, dimCAi, partiali,
                                                     &err);
    calclHandleError(err);
    calclmodel3D->bufferPartialProdi = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, dimCAi, partiali,
                                                      &err);
    calclHandleError(err);

    calclmodel3D->bufferPartialLogicalAndi = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, dimCAi,
                                                            partiali, &err);
    calclHandleError(err);
    calclmodel3D->bufferPartialLogicalOri = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, dimCAi,
                                                           partiali, &err);
    calclHandleError(err);
    calclmodel3D->bufferPartialLogicalXOri = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, dimCAi,
                                                            partiali, &err);
    calclHandleError(err);

    calclmodel3D->bufferPartialBinaryAndi = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, dimCAi,
                                                           partiali, &err);
    calclHandleError(err);
    calclmodel3D->bufferPartialBinaryOri = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, dimCAi, partiali,
                                                          &err);
    calclHandleError(err);
    calclmodel3D->bufferPartialBinaryXOri = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, dimCAi,
                                                           partiali, &err);
    calclHandleError(err);
    free(partiali);

    calclSetKernelCopyArgs3Di(calclmodel3D);

    calclmodel3D->kernelMinCopyb = calclGetKernelFromProgram(program, "copy3Db");
    calclmodel3D->kernelMaxCopyb = calclGetKernelFromProgram(program, "copy3Db");
    calclmodel3D->kernelSumCopyb = calclGetKernelFromProgram(program, "copy3Db");
    calclmodel3D->kernelProdCopyb = calclGetKernelFromProgram(program, "copy3Db");
    calclmodel3D->kernelLogicalAndCopyb = calclGetKernelFromProgram(program, "copy3Db");
    calclmodel3D->kernelLogicalOrCopyb = calclGetKernelFromProgram(program, "copy3Db");
    calclmodel3D->kernelLogicalXOrCopyb = calclGetKernelFromProgram(program, "copy3Db");
    calclmodel3D->kernelBinaryAndCopyb = calclGetKernelFromProgram(program, "copy3Db");
    calclmodel3D->kernelBinaryOrCopyb = calclGetKernelFromProgram(program, "copy3Db");
    calclmodel3D->kernelBinaryXOrCopyb = calclGetKernelFromProgram(program, "copy3Db");

    CALbyte * partialb = (CALbyte*) malloc(calclmodel3D->host_CA->rows * calclmodel3D->host_CA->columns *calclmodel3D->host_CA->slices * sizeof(CALbyte));
    size_t dimCAb =  sizeof(CALbyte) * calclmodel3D->host_CA->rows * calclmodel3D->host_CA->columns*calclmodel3D->host_CA->slices;
    calclmodel3D->bufferPartialMinb = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,dimCAb, partialb,
                                                     &err);
    calclHandleError(err);
    calclmodel3D->bufferPartialMaxb = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,dimCAb, partialb,
                                                     &err);
    calclHandleError(err);
    calclmodel3D->bufferPartialSumb = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,dimCAb, partialb,
                                                     &err);
    calclHandleError(err);
    calclmodel3D->bufferPartialProdb = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,dimCAb, partialb,
                                                      &err);
    calclHandleError(err);

    calclmodel3D->bufferPartialLogicalAndb = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,dimCAb,
                                                            partialb, &err);
    calclHandleError(err);
    calclmodel3D->bufferPartialLogicalOrb = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,dimCAb,
                                                           partialb, &err);
    calclHandleError(err);
    calclmodel3D->bufferPartialLogicalXOrb = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,dimCAb,
                                                            partialb, &err);
    calclHandleError(err);

    calclmodel3D->bufferPartialBinaryAndb = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,dimCAb,
                                                           partialb, &err);
    calclHandleError(err);
    calclmodel3D->bufferPartialBinaryOrb = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,dimCAb, partialb,
                                                          &err);
    calclHandleError(err);
    calclmodel3D->bufferPartialBinaryXOrb = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,dimCAb,
                                                           partialb, &err);
    calclHandleError(err);
    free(partialb);
    calclSetKernelCopyArgs3Db(calclmodel3D);

    calclmodel3D->kernelMinCopyr = calclGetKernelFromProgram(program, "copy3Dr");
    calclmodel3D->kernelMaxCopyr = calclGetKernelFromProgram(program, "copy3Dr");
    calclmodel3D->kernelSumCopyr = calclGetKernelFromProgram(program, "copy3Dr");
    calclmodel3D->kernelProdCopyr = calclGetKernelFromProgram(program, "copy3Dr");
    calclmodel3D->kernelLogicalAndCopyr = calclGetKernelFromProgram(program, "copy3Dr");
    calclmodel3D->kernelLogicalOrCopyr = calclGetKernelFromProgram(program, "copy3Dr");
    calclmodel3D->kernelLogicalXOrCopyr = calclGetKernelFromProgram(program, "copy3Dr");
    calclmodel3D->kernelBinaryAndCopyr = calclGetKernelFromProgram(program, "copy3Dr");
    calclmodel3D->kernelBinaryOrCopyr = calclGetKernelFromProgram(program, "copy3Dr");
    calclmodel3D->kernelBinaryXOrCopyr = calclGetKernelFromProgram(program, "copy3Dr");

    CALreal * partialr = (CALreal*) malloc(calclmodel3D->host_CA->rows * calclmodel3D->host_CA->columns *calclmodel3D->host_CA->slices* sizeof(CALreal));
    size_t dimCAr =  sizeof(CALreal) * calclmodel3D->host_CA->rows * calclmodel3D->host_CA->columns*calclmodel3D->host_CA->slices;
    calclmodel3D->bufferPartialMinr = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,dimCAr, partialr,
                                                     &err);
    calclHandleError(err);
    calclmodel3D->bufferPartialMaxr = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,dimCAr, partialr,
                                                     &err);
    calclHandleError(err);
    calclmodel3D->bufferPartialSumr = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,dimCAr, partialr,
                                                     &err);
    calclHandleError(err);
    calclmodel3D->bufferPartialProdr = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,dimCAr, partialr,
                                                      &err);
    calclHandleError(err);

    calclmodel3D->bufferPartialLogicalAndr = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,dimCAr,partialr,
                                                            &err);
    calclHandleError(err);
    calclmodel3D->bufferPartialLogicalOrr = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,dimCAr, partialr,
                                                           &err);
    calclHandleError(err);
    calclmodel3D->bufferPartialLogicalXOrr = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,dimCAr,partialr,
                                                            &err);
    calclHandleError(err);

    calclmodel3D->bufferPartialBinaryAndr = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,dimCAr, partialr,
                                                           &err);
    calclHandleError(err);
    calclmodel3D->bufferPartialBinaryOrr = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,dimCAr, partialr,
                                                          &err);
    calclHandleError(err);
    calclmodel3D->bufferPartialBinaryXOrr = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,dimCAr,partialr,
                                                           &err);
    calclHandleError(err);
    free(partialr);
    calclSetKernelCopyArgs3Dr(calclmodel3D);



    //user kernels buffers args

    calclmodel3D->bufferNeighborhood = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(struct CALCell3D) * host_CA->sizeof_X, host_CA->X, &err);
    calclHandleError(err);
    calclmodel3D->bufferNeighborhoodID = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(enum CALNeighborhood3D), &host_CA->X_id, &err);
    calclHandleError(err);
    calclmodel3D->bufferNeighborhoodSize = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(CALint), &host_CA->sizeof_X, &err);
    calclHandleError(err);
    calclmodel3D->bufferBoundaryCondition = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(enum CALSpaceBoundaryCondition), &host_CA->T, &err);
    calclHandleError(err);

    //stop condition buffer
    CALbyte stop = CAL_FALSE;
    calclmodel3D->bufferStop = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(CALbyte), &stop, &err);
    calclHandleError(err);

    //init substates mapper
    calclmodel3D->substateMapper.bufDIMbyte = byteSubstatesDim;
    calclmodel3D->substateMapper.bufDIMreal = realSubstatesDim;
    calclmodel3D->substateMapper.bufDIMint = intSubstatesDim;
    calclmodel3D->substateMapper.byteSubstate_current_OUT = (CALbyte*) malloc(byteSubstatesDim);
    calclmodel3D->substateMapper.realSubstate_current_OUT = (CALreal*) malloc(realSubstatesDim);
    calclmodel3D->substateMapper.intSubstate_current_OUT = (CALint*) malloc(intSubstatesDim);


    calclmodel3D->borderMapper.bufDIMbyte = sizeof(CALbyte)*host_CA->columns*host_CA->rows*host_CA->sizeof_pQb_array*calclmodel3D->borderSize*2;
    calclmodel3D->borderMapper.bufDIMreal = sizeof(CALreal)*host_CA->columns*host_CA->rows*host_CA->sizeof_pQr_array*calclmodel3D->borderSize*2;
    calclmodel3D->borderMapper.bufDIMint = sizeof(CALint)*host_CA->columns*host_CA->rows*host_CA->sizeof_pQi_array*calclmodel3D->borderSize*2;

    calclmodel3D->borderMapper.bufDIMflags = sizeof(CALbyte)*host_CA->columns*host_CA->rows*calclmodel3D->borderSize*2;


    calclmodel3D->borderMapper.byteBorder_OUT = (CALbyte*) malloc(calclmodel3D->borderMapper.bufDIMbyte);
    calclmodel3D->borderMapper.realBorder_OUT = (CALreal*) malloc(calclmodel3D->borderMapper.bufDIMreal);
    calclmodel3D->borderMapper.intBorder_OUT = (CALint*) malloc(calclmodel3D->borderMapper.bufDIMint);

    calclmodel3D->borderMapper.flagsBorder_OUT = (CALbyte*) malloc(calclmodel3D->borderMapper.bufDIMflags);
    memset(calclmodel3D->borderMapper.flagsBorder_OUT, CAL_FALSE, sizeof(CALbyte) * calclmodel3D->borderMapper.bufDIMflags);


    calclmodel3D->borderMapper.mergeflagsBorder =
        clCreateBuffer(context, CL_MEM_READ_ONLY,
                       sizeof(CALbyte) * calclmodel3D->borderMapper.bufDIMflags, NULL, &err);




    calclmodel3D->queue = calclCreateQueue3D(calclmodel3D, context, device);

    CALint dimReductionArrays = calclmodel3D->host_CA->sizeof_pQb_array + calclmodel3D->host_CA->sizeof_pQi_array + calclmodel3D->host_CA->sizeof_pQr_array;

    calclmodel3D->reductionFlagsMinb = (CALbyte*) malloc(sizeof(CALbyte) * (calclmodel3D->host_CA->sizeof_pQb_array));
    calclmodel3D->reductionFlagsMini = (CALbyte*) malloc(sizeof(CALbyte) * (calclmodel3D->host_CA->sizeof_pQi_array));
    calclmodel3D->reductionFlagsMinr = (CALbyte*) malloc(sizeof(CALbyte) * (calclmodel3D->host_CA->sizeof_pQr_array));
    calclmodel3D->minimab = (CALreal*) malloc(sizeof(CALreal) * (calclmodel3D->host_CA->sizeof_pQb_array + 1));
    calclmodel3D->minimai = (CALreal*) malloc(sizeof(CALreal) * (calclmodel3D->host_CA->sizeof_pQi_array + 1));
    calclmodel3D->minimar = (CALreal*) malloc(sizeof(CALreal) * (calclmodel3D->host_CA->sizeof_pQr_array + 1));

    calclmodel3D->reductionFlagsMaxb = (CALbyte*) malloc(sizeof(CALbyte) * (calclmodel3D->host_CA->sizeof_pQb_array));
    calclmodel3D->reductionFlagsMaxi = (CALbyte*) malloc(sizeof(CALbyte) * (calclmodel3D->host_CA->sizeof_pQi_array));
    calclmodel3D->reductionFlagsMaxr = (CALbyte*) malloc(sizeof(CALbyte) * (calclmodel3D->host_CA->sizeof_pQr_array));
    calclmodel3D->maximab = (CALreal*) malloc(sizeof(CALreal) * (calclmodel3D->host_CA->sizeof_pQb_array + 1));
    calclmodel3D->maximai = (CALreal*) malloc(sizeof(CALreal) * (calclmodel3D->host_CA->sizeof_pQi_array + 1));
    calclmodel3D->maximar = (CALreal*) malloc(sizeof(CALreal) * (calclmodel3D->host_CA->sizeof_pQr_array + 1));

    calclmodel3D->reductionFlagsSumb = (CALbyte*) malloc(sizeof(CALbyte) * (calclmodel3D->host_CA->sizeof_pQb_array));
    calclmodel3D->reductionFlagsSumi = (CALbyte*) malloc(sizeof(CALbyte) * (calclmodel3D->host_CA->sizeof_pQi_array));
    calclmodel3D->reductionFlagsSumr = (CALbyte*) malloc(sizeof(CALbyte) * (calclmodel3D->host_CA->sizeof_pQr_array));
    calclmodel3D->sumsb = (CALreal*) malloc(sizeof(CALreal) * (calclmodel3D->host_CA->sizeof_pQb_array + 1));
    calclmodel3D->sumsi = (CALreal*) malloc(sizeof(CALreal) * (calclmodel3D->host_CA->sizeof_pQi_array + 1));
    calclmodel3D->sumsr = (CALreal*) malloc(sizeof(CALreal) * (calclmodel3D->host_CA->sizeof_pQr_array + 1));

    calclmodel3D->reductionFlagsProdb = (CALbyte*) malloc(sizeof(CALbyte) * (calclmodel3D->host_CA->sizeof_pQb_array));
    calclmodel3D->reductionFlagsProdi = (CALbyte*) malloc(sizeof(CALbyte) * (calclmodel3D->host_CA->sizeof_pQi_array));
    calclmodel3D->reductionFlagsProdr = (CALbyte*) malloc(sizeof(CALbyte) * (calclmodel3D->host_CA->sizeof_pQr_array));
    calclmodel3D->prodsb = (CALreal*) malloc(sizeof(CALreal) * (calclmodel3D->host_CA->sizeof_pQb_array + 1));
    calclmodel3D->prodsi = (CALreal*) malloc(sizeof(CALreal) * (calclmodel3D->host_CA->sizeof_pQi_array + 1));
    calclmodel3D->prodsr = (CALreal*) malloc(sizeof(CALreal) * (calclmodel3D->host_CA->sizeof_pQr_array + 1));

    calclmodel3D->reductionFlagsLogicalAndb = (CALbyte*) malloc(sizeof(CALbyte) * (calclmodel3D->host_CA->sizeof_pQb_array));
    calclmodel3D->reductionFlagsLogicalAndi = (CALbyte*) malloc(sizeof(CALbyte) * (calclmodel3D->host_CA->sizeof_pQi_array));
    calclmodel3D->reductionFlagsLogicalAndr = (CALbyte*) malloc(sizeof(CALbyte) * (calclmodel3D->host_CA->sizeof_pQr_array));
    calclmodel3D->logicalAndsb = (CALint*) malloc(sizeof(CALint) * (calclmodel3D->host_CA->sizeof_pQb_array + 1));
    calclmodel3D->logicalAndsi = (CALint*) malloc(sizeof(CALint) * (calclmodel3D->host_CA->sizeof_pQi_array + 1));
    calclmodel3D->logicalAndsr = (CALint*) malloc(sizeof(CALint) * (calclmodel3D->host_CA->sizeof_pQr_array + 1));

    calclmodel3D->reductionFlagsLogicalOrb = (CALbyte*) malloc(sizeof(CALbyte) * (calclmodel3D->host_CA->sizeof_pQb_array));
    calclmodel3D->reductionFlagsLogicalOri = (CALbyte*) malloc(sizeof(CALbyte) * (calclmodel3D->host_CA->sizeof_pQi_array));
    calclmodel3D->reductionFlagsLogicalOrr = (CALbyte*) malloc(sizeof(CALbyte) * (calclmodel3D->host_CA->sizeof_pQr_array));
    calclmodel3D->logicalOrsb = (CALint*) malloc(sizeof(CALint) * (calclmodel3D->host_CA->sizeof_pQb_array + 1));
    calclmodel3D->logicalOrsi = (CALint*) malloc(sizeof(CALint) * (calclmodel3D->host_CA->sizeof_pQi_array + 1));
    calclmodel3D->logicalOrsr = (CALint*) malloc(sizeof(CALint) * (calclmodel3D->host_CA->sizeof_pQr_array + 1));

    calclmodel3D->reductionFlagsLogicalXOrb = (CALbyte*) malloc(sizeof(CALbyte) * (calclmodel3D->host_CA->sizeof_pQb_array));
    calclmodel3D->reductionFlagsLogicalXOri = (CALbyte*) malloc(sizeof(CALbyte) * (calclmodel3D->host_CA->sizeof_pQi_array));
    calclmodel3D->reductionFlagsLogicalXOrr = (CALbyte*) malloc(sizeof(CALbyte) * (calclmodel3D->host_CA->sizeof_pQr_array));
    calclmodel3D->logicalXOrsb = (CALint*) malloc(sizeof(CALint) * (calclmodel3D->host_CA->sizeof_pQb_array + 1));
    calclmodel3D->logicalXOrsi = (CALint*) malloc(sizeof(CALint) * (calclmodel3D->host_CA->sizeof_pQi_array + 1));
    calclmodel3D->logicalXOrsr = (CALint*) malloc(sizeof(CALint) * (calclmodel3D->host_CA->sizeof_pQr_array + 1));

    calclmodel3D->reductionFlagsBinaryAndb = (CALbyte*) malloc(sizeof(CALbyte) * (calclmodel3D->host_CA->sizeof_pQb_array));
    calclmodel3D->reductionFlagsBinaryAndi = (CALbyte*) malloc(sizeof(CALbyte) * (calclmodel3D->host_CA->sizeof_pQi_array));
    calclmodel3D->reductionFlagsBinaryAndr = (CALbyte*) malloc(sizeof(CALbyte) * (calclmodel3D->host_CA->sizeof_pQr_array));
    calclmodel3D->binaryAndsb = (CALint*) malloc(sizeof(CALint) * (calclmodel3D->host_CA->sizeof_pQb_array + 1));
    calclmodel3D->binaryAndsi = (CALint*) malloc(sizeof(CALint) * (calclmodel3D->host_CA->sizeof_pQi_array + 1));
    calclmodel3D->binaryAndsr = (CALint*) malloc(sizeof(CALint) * (calclmodel3D->host_CA->sizeof_pQr_array + 1));

    calclmodel3D->reductionFlagsBinaryOrb = (CALbyte*) malloc(sizeof(CALbyte) * (calclmodel3D->host_CA->sizeof_pQb_array));
    calclmodel3D->reductionFlagsBinaryOri = (CALbyte*) malloc(sizeof(CALbyte) * (calclmodel3D->host_CA->sizeof_pQi_array));
    calclmodel3D->reductionFlagsBinaryOrr = (CALbyte*) malloc(sizeof(CALbyte) * (calclmodel3D->host_CA->sizeof_pQr_array));
    calclmodel3D->binaryOrsb = (CALint*) malloc(sizeof(CALint) * (calclmodel3D->host_CA->sizeof_pQb_array + 1));
    calclmodel3D->binaryOrsi = (CALint*) malloc(sizeof(CALint) * (calclmodel3D->host_CA->sizeof_pQi_array + 1));
    calclmodel3D->binaryOrsr = (CALint*) malloc(sizeof(CALint) * (calclmodel3D->host_CA->sizeof_pQr_array + 1));

    calclmodel3D->reductionFlagsBinaryXOrb = (CALbyte*) malloc(sizeof(CALbyte) * (calclmodel3D->host_CA->sizeof_pQb_array));
    calclmodel3D->reductionFlagsBinaryXOri = (CALbyte*) malloc(sizeof(CALbyte) * (calclmodel3D->host_CA->sizeof_pQi_array));
    calclmodel3D->reductionFlagsBinaryXOrr = (CALbyte*) malloc(sizeof(CALbyte) * (calclmodel3D->host_CA->sizeof_pQr_array));
    calclmodel3D->binaryXOrsb = (CALint*) malloc(sizeof(CALint) * (calclmodel3D->host_CA->sizeof_pQb_array + 1));
    calclmodel3D->binaryXOrsi = (CALint*) malloc(sizeof(CALint) * (calclmodel3D->host_CA->sizeof_pQi_array + 1));
    calclmodel3D->binaryXOrsr = (CALint*) malloc(sizeof(CALint) * (calclmodel3D->host_CA->sizeof_pQr_array + 1));
    int i;
    for (i = 0; i < calclmodel3D->host_CA->sizeof_pQb_array; i++) {
        calclmodel3D->reductionFlagsMinb[i] = CAL_FALSE;
        calclmodel3D->reductionFlagsMaxb[i] = CAL_FALSE;
        calclmodel3D->reductionFlagsSumb[i] = CAL_FALSE;
        calclmodel3D->reductionFlagsProdb[i] = CAL_FALSE;
        calclmodel3D->reductionFlagsLogicalAndb[i] = CAL_FALSE;
        calclmodel3D->reductionFlagsLogicalOrb[i] = CAL_FALSE;
        calclmodel3D->reductionFlagsLogicalXOrb[i] = CAL_FALSE;
        calclmodel3D->reductionFlagsBinaryAndb[i] = CAL_FALSE;
        calclmodel3D->reductionFlagsBinaryOrb[i] = CAL_FALSE;
        calclmodel3D->reductionFlagsBinaryXOrb[i] = CAL_FALSE;
        calclmodel3D->minimab[i] = 0;
        calclmodel3D->maximab[i] = 0;
        calclmodel3D->sumsb[i] = 0;
        calclmodel3D->prodsb[i] = 0;
        calclmodel3D->logicalAndsb[i] = 0;
        calclmodel3D->logicalOrsb[i] = 0;
        calclmodel3D->logicalXOrsb[i] = 0;
        calclmodel3D->binaryAndsb[i] = 0;
        calclmodel3D->binaryOrsb[i] = 0;
        calclmodel3D->binaryXOrsb[i] = 0;
    }
    for ( i = 0; i < calclmodel3D->host_CA->sizeof_pQi_array; i++) {
        calclmodel3D->reductionFlagsMini[i] = CAL_FALSE;
        calclmodel3D->reductionFlagsMaxi[i] = CAL_FALSE;
        calclmodel3D->reductionFlagsSumi[i] = CAL_FALSE;
        calclmodel3D->reductionFlagsProdi[i] = CAL_FALSE;
        calclmodel3D->reductionFlagsLogicalAndi[i] = CAL_FALSE;
        calclmodel3D->reductionFlagsLogicalOri[i] = CAL_FALSE;
        calclmodel3D->reductionFlagsLogicalXOri[i] = CAL_FALSE;
        calclmodel3D->reductionFlagsBinaryAndi[i] = CAL_FALSE;
        calclmodel3D->reductionFlagsBinaryOri[i] = CAL_FALSE;
        calclmodel3D->reductionFlagsBinaryXOri[i] = CAL_FALSE;
        calclmodel3D->minimai[i] = 0;
        calclmodel3D->maximai[i] = 0;
        calclmodel3D->sumsi[i] = 0;
        calclmodel3D->prodsi[i] = 0;
        calclmodel3D->logicalAndsi[i] = 0;
        calclmodel3D->logicalOrsi[i] = 0;
        calclmodel3D->logicalXOrsi[i] = 0;
        calclmodel3D->binaryAndsi[i] = 0;
        calclmodel3D->binaryOrsi[i] = 0;
        calclmodel3D->binaryXOrsi[i] = 0;

    }
    for ( i = 0; i < calclmodel3D->host_CA->sizeof_pQr_array; i++) {
        calclmodel3D->reductionFlagsMinr[i] = CAL_FALSE;
        calclmodel3D->reductionFlagsMaxr[i] = CAL_FALSE;
        calclmodel3D->reductionFlagsSumr[i] = CAL_FALSE;
        calclmodel3D->reductionFlagsProdr[i] = CAL_FALSE;
        calclmodel3D->reductionFlagsLogicalAndr[i] = CAL_FALSE;
        calclmodel3D->reductionFlagsLogicalOrr[i] = CAL_FALSE;
        calclmodel3D->reductionFlagsLogicalXOrr[i] = CAL_FALSE;
        calclmodel3D->reductionFlagsBinaryAndr[i] = CAL_FALSE;
        calclmodel3D->reductionFlagsBinaryOrr[i] = CAL_FALSE;
        calclmodel3D->reductionFlagsBinaryXOrr[i] = CAL_FALSE;
        calclmodel3D->minimar[i] = 0;
        calclmodel3D->maximar[i] = 0;
        calclmodel3D->sumsr[i] = 0;
        calclmodel3D->prodsr[i] = 0;
        calclmodel3D->logicalAndsr[i] = 0;
        calclmodel3D->logicalOrsr[i] = 0;
        calclmodel3D->logicalXOrsr[i] = 0;
        calclmodel3D->binaryAndsr[i] = 0;
        calclmodel3D->binaryOrsr[i] = 0;
        calclmodel3D->binaryXOrsr[i] = 0;

    }

    calclmodel3D->roundedDimensions = upperPowerOfTwo3D(calclmodel3D->host_CA->rows * calclmodel3D->host_CA->columns*calclmodel3D->host_CA->slices);

    calclmodel3D->context = context;
    calclmodel3D->workGroupDimensions=NULL;

    return calclmodel3D;

}
void setParametersReduction3D(cl_int err, struct CALCLModel3D* calclmodel3D){
    int sizeCA = calclmodel3D->host_CA->rows * calclmodel3D->host_CA->columns * calclmodel3D->host_CA->slices;
    //TODO eliminare bufferFlags Urgent

    calclmodel3D->bufferMinimab = clCreateBuffer(calclmodel3D->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(CALreal) * (calclmodel3D->host_CA->sizeof_pQb_array + 1),
                                                 calclmodel3D->minimab, &err);
    calclHandleError(err);
    calclmodel3D->bufferMaximab = clCreateBuffer(calclmodel3D->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(CALreal) * (calclmodel3D->host_CA->sizeof_pQb_array + 1),
                                                 calclmodel3D->maximab, &err);
    calclHandleError(err);
    calclmodel3D->bufferSumb = clCreateBuffer(calclmodel3D->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(CALreal) * (calclmodel3D->host_CA->sizeof_pQb_array + 1),
                                              calclmodel3D->sumsb, &err);
    calclHandleError(err);
    calclmodel3D->bufferProdb = clCreateBuffer(calclmodel3D->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(CALreal) * (calclmodel3D->host_CA->sizeof_pQb_array + 1),
                                               calclmodel3D->prodsb, &err);
    calclHandleError(err);
    calclmodel3D->bufferLogicalAndsb = clCreateBuffer(calclmodel3D->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(CALint) * (calclmodel3D->host_CA->sizeof_pQb_array + 1),
                                                      calclmodel3D->logicalAndsb, &err);
    calclHandleError(err);
    calclmodel3D->bufferLogicalOrsb = clCreateBuffer(calclmodel3D->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(CALint) * (calclmodel3D->host_CA->sizeof_pQb_array + 1),
                                                     calclmodel3D->logicalOrsb, &err);
    calclHandleError(err);
    calclmodel3D->bufferLogicalXOrsb = clCreateBuffer(calclmodel3D->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(CALint) * (calclmodel3D->host_CA->sizeof_pQb_array + 1),
                                                      calclmodel3D->logicalXOrsb, &err);
    calclHandleError(err);
    calclmodel3D->bufferBinaryAndsb = clCreateBuffer(calclmodel3D->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(CALint) * (calclmodel3D->host_CA->sizeof_pQb_array + 1),
                                                     calclmodel3D->binaryAndsb, &err);
    calclHandleError(err);
    calclmodel3D->bufferBinaryOrsb = clCreateBuffer(calclmodel3D->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(CALint) * (calclmodel3D->host_CA->sizeof_pQb_array + 1),
                                                    calclmodel3D->binaryOrsb, &err);
    calclHandleError(err);
    calclmodel3D->bufferBinaryXOrsb = clCreateBuffer(calclmodel3D->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(CALint) * (calclmodel3D->host_CA->sizeof_pQb_array + 1),
                                                     calclmodel3D->binaryXOrsb, &err);
    calclHandleError(err);

    clSetKernelArg(calclmodel3D->kernelMinReductionb, 0, sizeof(CALCLmem), &calclmodel3D->bufferMinimab);
    clSetKernelArg(calclmodel3D->kernelMinReductionb, 2, sizeof(CALCLmem), &calclmodel3D->bufferPartialMinb);
    clSetKernelArg(calclmodel3D->kernelMinReductionb, 4, sizeof(int), &sizeCA);

    clSetKernelArg(calclmodel3D->kernelMaxReductionb, 0, sizeof(CALCLmem), &calclmodel3D->bufferMaximab);
    clSetKernelArg(calclmodel3D->kernelMaxReductionb, 2, sizeof(CALCLmem), &calclmodel3D->bufferPartialMaxb);
    clSetKernelArg(calclmodel3D->kernelMaxReductionb, 4, sizeof(int), &sizeCA);

    clSetKernelArg(calclmodel3D->kernelSumReductionb, 0, sizeof(CALCLmem), &calclmodel3D->bufferSumb);
    clSetKernelArg(calclmodel3D->kernelSumReductionb, 2, sizeof(CALCLmem), &calclmodel3D->bufferPartialSumb);
    clSetKernelArg(calclmodel3D->kernelSumReductionb, 4, sizeof(int), &sizeCA);

    clSetKernelArg(calclmodel3D->kernelProdReductionb, 0, sizeof(CALCLmem), &calclmodel3D->bufferProdb);
    clSetKernelArg(calclmodel3D->kernelProdReductionb, 2, sizeof(CALCLmem), &calclmodel3D->bufferPartialProdb);
    clSetKernelArg(calclmodel3D->kernelProdReductionb, 4, sizeof(int), &sizeCA);

    clSetKernelArg(calclmodel3D->kernelLogicalAndReductionb, 0, sizeof(CALCLmem), &calclmodel3D->bufferLogicalAndsb);
    clSetKernelArg(calclmodel3D->kernelLogicalAndReductionb, 2, sizeof(CALCLmem), &calclmodel3D->bufferPartialLogicalAndb);
    clSetKernelArg(calclmodel3D->kernelLogicalAndReductionb, 4, sizeof(int), &sizeCA);

    clSetKernelArg(calclmodel3D->kernelLogicalOrReductionb, 0, sizeof(CALCLmem), &calclmodel3D->bufferLogicalOrsb);
    clSetKernelArg(calclmodel3D->kernelLogicalOrReductionb, 2, sizeof(CALCLmem), &calclmodel3D->bufferPartialLogicalOrb);
    clSetKernelArg(calclmodel3D->kernelLogicalOrReductionb, 4, sizeof(int), &sizeCA);

    clSetKernelArg(calclmodel3D->kernelLogicalXOrReductionb, 0, sizeof(CALCLmem), &calclmodel3D->bufferLogicalXOrsb);
    clSetKernelArg(calclmodel3D->kernelLogicalXOrReductionb, 2, sizeof(CALCLmem), &calclmodel3D->bufferPartialLogicalXOrb);
    clSetKernelArg(calclmodel3D->kernelLogicalXOrReductionb, 4, sizeof(int), &sizeCA);

    clSetKernelArg(calclmodel3D->kernelBinaryAndReductionb, 0, sizeof(CALCLmem), &calclmodel3D->bufferBinaryAndsb);
    clSetKernelArg(calclmodel3D->kernelBinaryAndReductionb, 2, sizeof(CALCLmem), &calclmodel3D->bufferPartialBinaryAndb);
    clSetKernelArg(calclmodel3D->kernelBinaryAndReductionb, 4, sizeof(int), &sizeCA);

    clSetKernelArg(calclmodel3D->kernelBinaryOrReductionb, 0, sizeof(CALCLmem), &calclmodel3D->bufferBinaryOrsb);
    clSetKernelArg(calclmodel3D->kernelBinaryOrReductionb, 2, sizeof(CALCLmem), &calclmodel3D->bufferBinaryOrsb);
    clSetKernelArg(calclmodel3D->kernelBinaryOrReductionb, 4, sizeof(int), &sizeCA);

    clSetKernelArg(calclmodel3D->kernelBinaryXorReductionb, 0, sizeof(CALCLmem), &calclmodel3D->bufferBinaryXOrsb);
    clSetKernelArg(calclmodel3D->kernelBinaryXorReductionb, 2, sizeof(CALCLmem), &calclmodel3D->bufferBinaryXOrsb);
    clSetKernelArg(calclmodel3D->kernelBinaryXorReductionb, 4, sizeof(int), &sizeCA);

    calclmodel3D->bufferMinimai = clCreateBuffer(calclmodel3D->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(CALreal) * (calclmodel3D->host_CA->sizeof_pQi_array + 1),
                                                 calclmodel3D->minimai, &err);
    calclHandleError(err);
    calclmodel3D->bufferMaximai = clCreateBuffer(calclmodel3D->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(CALreal) * (calclmodel3D->host_CA->sizeof_pQi_array + 1),
                                                 calclmodel3D->maximai, &err);
    calclHandleError(err);
    calclmodel3D->bufferSumi = clCreateBuffer(calclmodel3D->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(CALreal) * (calclmodel3D->host_CA->sizeof_pQi_array + 1), calclmodel3D->sumsi,
                                              &err);
    calclHandleError(err);
    calclmodel3D->bufferProdi = clCreateBuffer(calclmodel3D->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(CALreal) * (calclmodel3D->host_CA->sizeof_pQi_array + 1), calclmodel3D->prodsi,
                                               &err);
    calclHandleError(err);
    calclmodel3D->bufferLogicalAndsi = clCreateBuffer(calclmodel3D->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(CALint) * (calclmodel3D->host_CA->sizeof_pQi_array + 1),
                                                      calclmodel3D->logicalAndsi, &err);
    calclHandleError(err);
    calclmodel3D->bufferLogicalOrsi = clCreateBuffer(calclmodel3D->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(CALint) * (calclmodel3D->host_CA->sizeof_pQi_array + 1),
                                                     calclmodel3D->logicalOrsi, &err);
    calclHandleError(err);
    calclmodel3D->bufferLogicalXOrsi = clCreateBuffer(calclmodel3D->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(CALint) * (calclmodel3D->host_CA->sizeof_pQi_array + 1),
                                                      calclmodel3D->logicalXOrsi, &err);
    calclHandleError(err);
    calclmodel3D->bufferBinaryAndsi = clCreateBuffer(calclmodel3D->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(CALint) * (calclmodel3D->host_CA->sizeof_pQi_array + 1),
                                                     calclmodel3D->binaryAndsi, &err);
    calclHandleError(err);
    calclmodel3D->bufferBinaryOrsi = clCreateBuffer(calclmodel3D->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(CALint) * (calclmodel3D->host_CA->sizeof_pQi_array + 1),
                                                    calclmodel3D->binaryOrsi, &err);
    calclHandleError(err);
    calclmodel3D->bufferBinaryXOrsi = clCreateBuffer(calclmodel3D->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(CALint) * (calclmodel3D->host_CA->sizeof_pQi_array + 1),
                                                     calclmodel3D->binaryXOrsi, &err);
    calclHandleError(err);

    clSetKernelArg(calclmodel3D->kernelMinReductioni, 0, sizeof(CALCLmem), &calclmodel3D->bufferMinimai);
    clSetKernelArg(calclmodel3D->kernelMinReductioni, 2, sizeof(CALCLmem), &calclmodel3D->bufferPartialMini);
    clSetKernelArg(calclmodel3D->kernelMinReductioni, 4, sizeof(int), &sizeCA);

    clSetKernelArg(calclmodel3D->kernelMaxReductioni, 0, sizeof(CALCLmem), &calclmodel3D->bufferMaximai);
    clSetKernelArg(calclmodel3D->kernelMaxReductioni, 2, sizeof(CALCLmem), &calclmodel3D->bufferPartialMaxi);
    clSetKernelArg(calclmodel3D->kernelMaxReductioni, 4, sizeof(int), &sizeCA);

    clSetKernelArg(calclmodel3D->kernelSumReductioni, 0, sizeof(CALCLmem), &calclmodel3D->bufferSumi);
    clSetKernelArg(calclmodel3D->kernelSumReductioni, 2, sizeof(CALCLmem), &calclmodel3D->bufferPartialSumi);
    clSetKernelArg(calclmodel3D->kernelSumReductioni, 4, sizeof(int), &sizeCA);

    clSetKernelArg(calclmodel3D->kernelProdReductioni, 0, sizeof(CALCLmem), &calclmodel3D->bufferProdi);
    clSetKernelArg(calclmodel3D->kernelProdReductioni, 2, sizeof(CALCLmem), &calclmodel3D->bufferPartialProdi);
    clSetKernelArg(calclmodel3D->kernelProdReductioni, 4, sizeof(int), &sizeCA);

    clSetKernelArg(calclmodel3D->kernelLogicalAndReductioni, 0, sizeof(CALCLmem), &calclmodel3D->bufferLogicalAndsi);
    clSetKernelArg(calclmodel3D->kernelLogicalAndReductioni, 2, sizeof(CALCLmem), &calclmodel3D->bufferPartialLogicalAndi);
    clSetKernelArg(calclmodel3D->kernelLogicalAndReductioni, 4, sizeof(int), &sizeCA);

    clSetKernelArg(calclmodel3D->kernelLogicalOrReductioni, 0, sizeof(CALCLmem), &calclmodel3D->bufferLogicalOrsi);
    clSetKernelArg(calclmodel3D->kernelLogicalOrReductioni, 2, sizeof(CALCLmem), &calclmodel3D->bufferPartialLogicalOri);
    clSetKernelArg(calclmodel3D->kernelLogicalOrReductioni, 4, sizeof(int), &sizeCA);

    clSetKernelArg(calclmodel3D->kernelLogicalXOrReductioni, 0, sizeof(CALCLmem), &calclmodel3D->bufferLogicalXOrsi);
    clSetKernelArg(calclmodel3D->kernelLogicalXOrReductioni, 2, sizeof(CALCLmem), &calclmodel3D->bufferPartialLogicalXOri);
    clSetKernelArg(calclmodel3D->kernelLogicalXOrReductioni, 4, sizeof(int), &sizeCA);

    clSetKernelArg(calclmodel3D->kernelBinaryAndReductioni, 0, sizeof(CALCLmem), &calclmodel3D->bufferBinaryAndsi);
    clSetKernelArg(calclmodel3D->kernelBinaryAndReductioni, 2, sizeof(CALCLmem), &calclmodel3D->bufferPartialBinaryAndi);
    clSetKernelArg(calclmodel3D->kernelBinaryAndReductioni, 4, sizeof(int), &sizeCA);

    clSetKernelArg(calclmodel3D->kernelBinaryOrReductioni, 0, sizeof(CALCLmem), &calclmodel3D->bufferBinaryOrsi);
    clSetKernelArg(calclmodel3D->kernelBinaryOrReductioni, 2, sizeof(CALCLmem), &calclmodel3D->bufferPartialBinaryOri);
    clSetKernelArg(calclmodel3D->kernelBinaryOrReductioni, 4, sizeof(int), &sizeCA);

    clSetKernelArg(calclmodel3D->kernelBinaryXorReductioni, 0, sizeof(CALCLmem), &calclmodel3D->bufferBinaryXOrsi);
    clSetKernelArg(calclmodel3D->kernelBinaryXorReductioni, 2, sizeof(CALCLmem), &calclmodel3D->bufferPartialBinaryXOri);
    clSetKernelArg(calclmodel3D->kernelBinaryXorReductioni, 4, sizeof(int), &sizeCA);

    calclmodel3D->bufferMinimar = clCreateBuffer(calclmodel3D->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(CALreal) * (calclmodel3D->host_CA->sizeof_pQr_array + 1),
                                                 calclmodel3D->minimar, &err);
    calclHandleError(err);
    calclmodel3D->bufferMaximar = clCreateBuffer(calclmodel3D->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(CALreal) * (calclmodel3D->host_CA->sizeof_pQr_array + 1),
                                                 calclmodel3D->maximar, &err);
    calclHandleError(err);
    calclmodel3D->bufferSumr = clCreateBuffer(calclmodel3D->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(CALreal) * (calclmodel3D->host_CA->sizeof_pQr_array + 1), calclmodel3D->sumsr,
                                              &err);
    calclHandleError(err);
    calclmodel3D->bufferProdr = clCreateBuffer(calclmodel3D->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(CALreal) * (calclmodel3D->host_CA->sizeof_pQr_array + 1), calclmodel3D->prodsr,
                                               &err);
    calclHandleError(err);
    calclmodel3D->bufferLogicalAndsr = clCreateBuffer(calclmodel3D->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(CALint) * (calclmodel3D->host_CA->sizeof_pQr_array + 1),
                                                      calclmodel3D->logicalAndsr, &err);
    calclHandleError(err);
    calclmodel3D->bufferLogicalOrsr = clCreateBuffer(calclmodel3D->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(CALint) * (calclmodel3D->host_CA->sizeof_pQr_array + 1),
                                                     calclmodel3D->logicalOrsr, &err);
    calclHandleError(err);
    calclmodel3D->bufferLogicalXOrsr = clCreateBuffer(calclmodel3D->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(CALint) * (calclmodel3D->host_CA->sizeof_pQr_array + 1),
                                                      calclmodel3D->logicalXOrsr, &err);
    calclHandleError(err);

    calclmodel3D->bufferBinaryAndsr = clCreateBuffer(calclmodel3D->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(CALint) * (calclmodel3D->host_CA->sizeof_pQr_array + 1),
                                                     calclmodel3D->binaryAndsr, &err);
    calclHandleError(err);

    calclmodel3D->bufferBinaryOrsr = clCreateBuffer(calclmodel3D->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(CALint) * (calclmodel3D->host_CA->sizeof_pQr_array + 1),
                                                    calclmodel3D->binaryOrsr, &err);
    calclHandleError(err);

    calclmodel3D->bufferBinaryXOrsr = clCreateBuffer(calclmodel3D->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(CALint) * (calclmodel3D->host_CA->sizeof_pQr_array + 1),
                                                     calclmodel3D->binaryXOrsr, &err);
    calclHandleError(err);

    clSetKernelArg(calclmodel3D->kernelMinReductionr, 0, sizeof(CALCLmem), &calclmodel3D->bufferMinimar);
    clSetKernelArg(calclmodel3D->kernelMinReductionr, 2, sizeof(CALCLmem), &calclmodel3D->bufferPartialMinr);
    clSetKernelArg(calclmodel3D->kernelMinReductionr, 4, sizeof(int), &sizeCA);

    clSetKernelArg(calclmodel3D->kernelMaxReductionr, 0, sizeof(CALCLmem), &calclmodel3D->bufferMaximar);
    clSetKernelArg(calclmodel3D->kernelMaxReductionr, 2, sizeof(CALCLmem), &calclmodel3D->bufferPartialMaxr);
    clSetKernelArg(calclmodel3D->kernelMaxReductionr, 4, sizeof(int), &sizeCA);

    clSetKernelArg(calclmodel3D->kernelSumReductionr, 0, sizeof(CALCLmem), &calclmodel3D->bufferSumr);
    clSetKernelArg(calclmodel3D->kernelSumReductionr, 2, sizeof(CALCLmem), &calclmodel3D->bufferPartialSumr);
    clSetKernelArg(calclmodel3D->kernelSumReductionr, 4, sizeof(int), &sizeCA);

    clSetKernelArg(calclmodel3D->kernelProdReductionr, 0, sizeof(CALCLmem), &calclmodel3D->bufferProdr);
    clSetKernelArg(calclmodel3D->kernelProdReductionr, 2, sizeof(CALCLmem), &calclmodel3D->bufferPartialProdr);
    clSetKernelArg(calclmodel3D->kernelProdReductionr, 4, sizeof(int), &sizeCA);

    clSetKernelArg(calclmodel3D->kernelLogicalAndReductionr, 0, sizeof(CALCLmem), &calclmodel3D->bufferLogicalAndsr);
    clSetKernelArg(calclmodel3D->kernelLogicalAndReductionr, 2, sizeof(CALCLmem), &calclmodel3D->bufferPartialLogicalAndr);
    clSetKernelArg(calclmodel3D->kernelLogicalAndReductionr, 4, sizeof(int), &sizeCA);

    clSetKernelArg(calclmodel3D->kernelLogicalOrReductionr, 0, sizeof(CALCLmem), &calclmodel3D->bufferLogicalOrsr);
    clSetKernelArg(calclmodel3D->kernelLogicalOrReductionr, 2, sizeof(CALCLmem), &calclmodel3D->bufferPartialLogicalOrr);
    clSetKernelArg(calclmodel3D->kernelLogicalOrReductionr, 4, sizeof(int), &sizeCA);

    clSetKernelArg(calclmodel3D->kernelLogicalXOrReductionr, 0, sizeof(CALCLmem), &calclmodel3D->bufferLogicalXOrsr);
    clSetKernelArg(calclmodel3D->kernelLogicalXOrReductionr, 2, sizeof(CALCLmem), &calclmodel3D->bufferPartialLogicalXOrr);
    clSetKernelArg(calclmodel3D->kernelLogicalXOrReductionr, 4, sizeof(int), &sizeCA);

    clSetKernelArg(calclmodel3D->kernelBinaryAndReductionr, 0, sizeof(CALCLmem), &calclmodel3D->bufferBinaryAndsr);
    clSetKernelArg(calclmodel3D->kernelBinaryAndReductionr, 2, sizeof(CALCLmem), &calclmodel3D->bufferPartialBinaryAndr);
    clSetKernelArg(calclmodel3D->kernelBinaryAndReductionr, 4, sizeof(int), &sizeCA);

    clSetKernelArg(calclmodel3D->kernelBinaryOrReductionr, 0, sizeof(CALCLmem), &calclmodel3D->bufferBinaryOrsr);
    clSetKernelArg(calclmodel3D->kernelBinaryOrReductionr, 2, sizeof(CALCLmem), &calclmodel3D->bufferPartialBinaryOrr);
    clSetKernelArg(calclmodel3D->kernelBinaryOrReductionr, 4, sizeof(int), &sizeCA);

    clSetKernelArg(calclmodel3D->kernelBinaryXorReductionr, 0, sizeof(CALCLmem), &calclmodel3D->bufferBinaryXOrsr);
    clSetKernelArg(calclmodel3D->kernelBinaryXorReductionr, 2, sizeof(CALCLmem), &calclmodel3D->bufferPartialBinaryXOrr);
    clSetKernelArg(calclmodel3D->kernelBinaryXorReductionr, 4, sizeof(int), &sizeCA);

}


void calclRun3D(struct CALCLModel3D* calclmodel3D, unsigned int initialStep, unsigned maxStep) {

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


    //	cl_int err;
    CALbyte stop;
    size_t * threadNumMax = (size_t*) malloc(sizeof(size_t) * 3);
    threadNumMax[0] = calclmodel3D->host_CA->rows;
    threadNumMax[1] = calclmodel3D->host_CA->columns;
    threadNumMax[2] = calclmodel3D->host_CA->slices;
    size_t * singleStepThreadNum;
    int dimNum;

    if (calclmodel3D->opt == CAL_NO_OPT) {
        singleStepThreadNum = (size_t*) malloc(sizeof(size_t) * 3);
        singleStepThreadNum[0] = threadNumMax[0];
        singleStepThreadNum[1] = threadNumMax[1];
        singleStepThreadNum[2] = threadNumMax[2];
        dimNum = 3;
    } else {
        singleStepThreadNum = (size_t*) malloc(sizeof(size_t));
        singleStepThreadNum[0] = calclmodel3D->host_CA->A->size_current;
        dimNum = 1;
    }
    //	calclRoundThreadsNum(singleStepThreadNum, dimNum);

    if (calclmodel3D->kernelInitSubstates != NULL)
        calclKernelCall3D(calclmodel3D, calclmodel3D->kernelInitSubstates, dimNum, threadNumMax, NULL);

    calclmodel3D->steps = initialStep;
    while (calclmodel3D->steps <= (int)maxStep || maxStep == CAL_RUN_LOOP) {
        stop = calclSingleStep3D(calclmodel3D, singleStepThreadNum, dimNum);
        if (stop)
            break;
    }
    calclGetSubstatesDeviceToHost3D(calclmodel3D);

    free(threadNumMax);
    free(singleStepThreadNum);
}

void calclComputeReduction3Di(struct CALCLModel3D * calclmodel3D, int numSubstate, enum REDUCTION_OPERATION operation, int rounded) {

    CALCLqueue queue = calclmodel3D->queue;
    cl_int err;
    int iterations = rounded / 2;
    size_t tmpThreads = iterations;
    int i;

    int count = 0;

    int offset = 1;
    for (i = iterations; i > 0; i /= 2) {
        tmpThreads = i;
        switch (operation) {
        case REDUCTION_MAX:
            clSetKernelArg(calclmodel3D->kernelMaxReductioni, 3, sizeof(CALint), &offset);
            clSetKernelArg(calclmodel3D->kernelMaxReductioni, 5, sizeof(CALint), &count);
            err = clEnqueueNDRangeKernel(queue, calclmodel3D->kernelMaxReductioni, 1,
                                         NULL, &tmpThreads, NULL, 0, NULL, NULL);
            calclHandleError(err);
            break;
        case REDUCTION_MIN:
            clSetKernelArg(calclmodel3D->kernelMinReductioni, 3, sizeof(CALint), &offset);
            clSetKernelArg(calclmodel3D->kernelMinReductioni, 5, sizeof(CALint), &count);
            err = clEnqueueNDRangeKernel(queue, calclmodel3D->kernelMinReductioni, 1, NULL, &tmpThreads, NULL, 0, NULL, NULL);
            calclHandleError(err);
            break;
        case REDUCTION_SUM:
            clSetKernelArg(calclmodel3D->kernelSumReductioni, 3, sizeof(CALint), &offset);
            clSetKernelArg(calclmodel3D->kernelSumReductioni, 5, sizeof(CALint), &count);
            err = clEnqueueNDRangeKernel(queue, calclmodel3D->kernelSumReductioni, 1,
                                         NULL, &tmpThreads, NULL, 0, NULL, NULL);
            calclHandleError(err);
            break;
        case REDUCTION_PROD:
            clSetKernelArg(calclmodel3D->kernelProdReductioni, 3, sizeof(CALint), &offset);
            clSetKernelArg(calclmodel3D->kernelProdReductioni, 5, sizeof(CALint), &count);
            err = clEnqueueNDRangeKernel(queue, calclmodel3D->kernelProdReductioni, 1,
                                         NULL, &tmpThreads, NULL, 0, NULL, NULL);
            calclHandleError(err);
            break;
        case REDUCTION_LOGICAL_AND:
            clSetKernelArg(calclmodel3D->kernelLogicalAndReductioni, 3, sizeof(CALint), &offset);
            clSetKernelArg(calclmodel3D->kernelLogicalAndReductioni, 5, sizeof(CALint), &count);
            err = clEnqueueNDRangeKernel(queue, calclmodel3D->kernelLogicalAndReductioni, 1,
                                         NULL, &tmpThreads, NULL, 0, NULL, NULL);
            calclHandleError(err);
            break;
        case REDUCTION_BINARY_AND:
            clSetKernelArg(calclmodel3D->kernelBinaryAndReductioni, 3, sizeof(CALint), &offset);
            clSetKernelArg(calclmodel3D->kernelBinaryAndReductioni, 5, sizeof(CALint), &count);
            err = clEnqueueNDRangeKernel(queue, calclmodel3D->kernelBinaryAndReductioni, 1,
                                         NULL, &tmpThreads, NULL, 0, NULL, NULL);
            calclHandleError(err);
            break;
        case REDUCTION_LOGICAL_OR:
            clSetKernelArg(calclmodel3D->kernelLogicalOrReductioni, 3, sizeof(CALint), &offset);
            clSetKernelArg(calclmodel3D->kernelLogicalOrReductioni, 5, sizeof(CALint), &count);
            err = clEnqueueNDRangeKernel(queue, calclmodel3D->kernelLogicalOrReductioni, 1,
                                         NULL, &tmpThreads, NULL, 0, NULL, NULL);
            calclHandleError(err);
            break;
        case REDUCTION_BINARY_OR:
            clSetKernelArg(calclmodel3D->kernelBinaryOrReductioni, 3, sizeof(CALint), &offset);
            clSetKernelArg(calclmodel3D->kernelBinaryOrReductioni, 5, sizeof(CALint), &count);
            err = clEnqueueNDRangeKernel(queue, calclmodel3D->kernelBinaryOrReductioni, 1,
                                         NULL, &tmpThreads, NULL, 0, NULL, NULL);
            calclHandleError(err);
            break;
        case REDUCTION_LOGICAL_XOR:
            clSetKernelArg(calclmodel3D->kernelLogicalXOrReductioni, 3, sizeof(CALint), &offset);
            clSetKernelArg(calclmodel3D->kernelLogicalXOrReductioni, 5, sizeof(CALint), &count);
            err = clEnqueueNDRangeKernel(queue, calclmodel3D->kernelLogicalXOrReductioni, 1,
                                         NULL, &tmpThreads, NULL, 0, NULL, NULL);
            calclHandleError(err);
            break;
        case REDUCTION_BINARY_XOR:
            clSetKernelArg(calclmodel3D->kernelBinaryXorReductioni, 3, sizeof(CALint), &offset);
            clSetKernelArg(calclmodel3D->kernelBinaryXorReductioni, 5, sizeof(CALint), &count);
            err = clEnqueueNDRangeKernel(queue, calclmodel3D->kernelBinaryXorReductioni, 1,
                                         NULL, &tmpThreads, NULL, 0, NULL, NULL);
            calclHandleError(err);
            break;
        default:
            break;
        }
        offset <<= 1;
        count = count * 2 + 1;

    }

}


void calclComputeReduction3Db(struct CALCLModel3D * calclmodel3D, int numSubstate, enum REDUCTION_OPERATION operation, int rounded) {

    CALCLqueue queue = calclmodel3D->queue;
    cl_int err;
    int iterations = rounded / 2;
    size_t tmpThreads = iterations;
    int i;

    int count = 0;

    int offset = 1;
    for (i = iterations; i > 0; i /= 2) {
        tmpThreads = i;
        switch (operation) {
        case REDUCTION_MAX:
            clSetKernelArg(calclmodel3D->kernelMaxReductionb, 3, sizeof(CALint), &offset);
            clSetKernelArg(calclmodel3D->kernelMaxReductionb, 5, sizeof(CALint), &count);
            err = clEnqueueNDRangeKernel(queue, calclmodel3D->kernelMaxReductionb, 1,
                                         NULL, &tmpThreads, NULL, 0, NULL, NULL);
            calclHandleError(err);
            break;
        case REDUCTION_MIN:
            clSetKernelArg(calclmodel3D->kernelMinReductionb, 3, sizeof(CALint), &offset);
            clSetKernelArg(calclmodel3D->kernelMinReductionb, 5, sizeof(CALint), &count);
            err = clEnqueueNDRangeKernel(queue, calclmodel3D->kernelMinReductionb, 1, NULL, &tmpThreads, NULL, 0, NULL, NULL);
            calclHandleError(err);
            break;
        case REDUCTION_SUM:
            clSetKernelArg(calclmodel3D->kernelSumReductionb, 3, sizeof(CALint), &offset);
            clSetKernelArg(calclmodel3D->kernelSumReductionb, 5, sizeof(CALint), &count);
            err = clEnqueueNDRangeKernel(queue, calclmodel3D->kernelSumReductionb, 1,
                                         NULL, &tmpThreads, NULL, 0, NULL, NULL);
            calclHandleError(err);
            break;
        case REDUCTION_PROD:
            clSetKernelArg(calclmodel3D->kernelProdReductionb, 3, sizeof(CALint), &offset);
            clSetKernelArg(calclmodel3D->kernelProdReductionb, 5, sizeof(CALint), &count);
            err = clEnqueueNDRangeKernel(queue, calclmodel3D->kernelProdReductionb, 1,
                                         NULL, &tmpThreads, NULL, 0, NULL, NULL);
            calclHandleError(err);
            break;
        case REDUCTION_LOGICAL_AND:
            clSetKernelArg(calclmodel3D->kernelLogicalAndReductionb, 3, sizeof(CALint), &offset);
            clSetKernelArg(calclmodel3D->kernelLogicalAndReductionb, 5, sizeof(CALint), &count);
            err = clEnqueueNDRangeKernel(queue, calclmodel3D->kernelLogicalAndReductionb, 1,
                                         NULL, &tmpThreads, NULL, 0, NULL, NULL);
            calclHandleError(err);
            break;
        case REDUCTION_BINARY_AND:
            clSetKernelArg(calclmodel3D->kernelBinaryAndReductionb, 3, sizeof(CALint), &offset);
            clSetKernelArg(calclmodel3D->kernelBinaryAndReductionb, 5, sizeof(CALint), &count);
            err = clEnqueueNDRangeKernel(queue, calclmodel3D->kernelBinaryAndReductionb, 1,
                                         NULL, &tmpThreads, NULL, 0, NULL, NULL);
            calclHandleError(err);
            break;
        case REDUCTION_LOGICAL_OR:
            clSetKernelArg(calclmodel3D->kernelLogicalOrReductionb, 3, sizeof(CALint), &offset);
            clSetKernelArg(calclmodel3D->kernelLogicalOrReductionb, 5, sizeof(CALint), &count);
            err = clEnqueueNDRangeKernel(queue, calclmodel3D->kernelLogicalOrReductionb, 1,
                                         NULL, &tmpThreads, NULL, 0, NULL, NULL);
            calclHandleError(err);
            break;
        case REDUCTION_BINARY_OR:
            clSetKernelArg(calclmodel3D->kernelBinaryOrReductionb, 3, sizeof(CALint), &offset);
            clSetKernelArg(calclmodel3D->kernelBinaryOrReductionb, 5, sizeof(CALint), &count);
            err = clEnqueueNDRangeKernel(queue, calclmodel3D->kernelBinaryOrReductionb, 1,
                                         NULL, &tmpThreads, NULL, 0, NULL, NULL);
            calclHandleError(err);
            break;
        case REDUCTION_LOGICAL_XOR:
            clSetKernelArg(calclmodel3D->kernelLogicalXOrReductionb, 3, sizeof(CALint), &offset);
            clSetKernelArg(calclmodel3D->kernelLogicalXOrReductionb, 5, sizeof(CALint), &count);
            err = clEnqueueNDRangeKernel(queue, calclmodel3D->kernelLogicalXOrReductionb, 1,
                                         NULL, &tmpThreads, NULL, 0, NULL, NULL);
            calclHandleError(err);
            break;
        case REDUCTION_BINARY_XOR:
            clSetKernelArg(calclmodel3D->kernelBinaryXorReductionb, 3, sizeof(CALint), &offset);
            clSetKernelArg(calclmodel3D->kernelBinaryXorReductionb, 5, sizeof(CALint), &count);
            err = clEnqueueNDRangeKernel(queue, calclmodel3D->kernelBinaryXorReductionb, 1,
                                         NULL, &tmpThreads, NULL, 0, NULL, NULL);
            calclHandleError(err);
            break;
        default:
            break;
        }
        offset <<= 1;
        count = count * 2 + 1;

    }

}

void calclComputeReduction3Dr(struct CALCLModel3D * calclmodel3D, int numSubstate, enum REDUCTION_OPERATION operation, int rounded) {

    CALCLqueue queue = calclmodel3D->queue;
    cl_int err;
    int iterations = rounded / 2;
    size_t tmpThreads = iterations;
    int i;

    int count = 0;

    int offset = 1;
    for (i = iterations; i > 0; i /= 2) {
        tmpThreads = i;

        switch (operation) {
        case REDUCTION_MAX:
            clSetKernelArg(calclmodel3D->kernelMaxReductionr, 3, sizeof(CALint), &offset);
            clSetKernelArg(calclmodel3D->kernelMaxReductionr, 5, sizeof(CALint), &count);
            err = clEnqueueNDRangeKernel(queue, calclmodel3D->kernelMaxReductionr, 1,
                                         NULL, &tmpThreads, NULL, 0, NULL, NULL);
            calclHandleError(err);
            break;
        case REDUCTION_MIN:
            clSetKernelArg(calclmodel3D->kernelMinReductionr, 3, sizeof(CALint), &offset);
            clSetKernelArg(calclmodel3D->kernelMinReductionr, 5, sizeof(CALint), &count);
            err = clEnqueueNDRangeKernel(queue, calclmodel3D->kernelMinReductionr, 1, NULL, &tmpThreads, NULL, 0, NULL, NULL);
            calclHandleError(err);
            break;
        case REDUCTION_SUM:
            clSetKernelArg(calclmodel3D->kernelSumReductionr, 3, sizeof(CALint), &offset);
            clSetKernelArg(calclmodel3D->kernelSumReductionr, 5, sizeof(CALint), &count);
            err = clEnqueueNDRangeKernel(queue, calclmodel3D->kernelSumReductionr, 1,
                                         NULL, &tmpThreads, NULL, 0, NULL, NULL);
            calclHandleError(err);
            break;
        case REDUCTION_PROD:
            clSetKernelArg(calclmodel3D->kernelProdReductionr, 3, sizeof(CALint), &offset);
            clSetKernelArg(calclmodel3D->kernelProdReductionr, 5, sizeof(CALint), &count);
            err = clEnqueueNDRangeKernel(queue, calclmodel3D->kernelProdReductionr, 1,
                                         NULL, &tmpThreads, NULL, 0, NULL, NULL);
            calclHandleError(err);
            break;
        case REDUCTION_LOGICAL_AND:
            clSetKernelArg(calclmodel3D->kernelLogicalAndReductionr, 3, sizeof(CALint), &offset);
            clSetKernelArg(calclmodel3D->kernelLogicalAndReductionr, 5, sizeof(CALint), &count);
            err = clEnqueueNDRangeKernel(queue, calclmodel3D->kernelLogicalAndReductionr, 1,
                                         NULL, &tmpThreads, NULL, 0, NULL, NULL);
            calclHandleError(err);
            break;
        case REDUCTION_BINARY_AND:
            clSetKernelArg(calclmodel3D->kernelBinaryAndReductionr, 3, sizeof(CALint), &offset);
            clSetKernelArg(calclmodel3D->kernelBinaryAndReductionr, 5, sizeof(CALint), &count);
            err = clEnqueueNDRangeKernel(queue, calclmodel3D->kernelBinaryAndReductionr, 1,
                                         NULL, &tmpThreads, NULL, 0, NULL, NULL);
            calclHandleError(err);
            break;
        case REDUCTION_LOGICAL_OR:
            clSetKernelArg(calclmodel3D->kernelLogicalOrReductionr, 3, sizeof(CALint), &offset);
            clSetKernelArg(calclmodel3D->kernelLogicalOrReductionr, 5, sizeof(CALint), &count);
            err = clEnqueueNDRangeKernel(queue, calclmodel3D->kernelLogicalOrReductionr, 1,
                                         NULL, &tmpThreads, NULL, 0, NULL, NULL);
            calclHandleError(err);
            break;
        case REDUCTION_BINARY_OR:
            clSetKernelArg(calclmodel3D->kernelBinaryOrReductionr, 3, sizeof(CALint), &offset);
            clSetKernelArg(calclmodel3D->kernelBinaryOrReductionr, 5, sizeof(CALint), &count);
            err = clEnqueueNDRangeKernel(queue, calclmodel3D->kernelBinaryOrReductionr, 1,
                                         NULL, &tmpThreads, NULL, 0, NULL, NULL);
            calclHandleError(err);
            break;
        case REDUCTION_LOGICAL_XOR:
            clSetKernelArg(calclmodel3D->kernelLogicalXOrReductionr, 3, sizeof(CALint), &offset);
            clSetKernelArg(calclmodel3D->kernelLogicalXOrReductionr, 5, sizeof(CALint), &count);
            err = clEnqueueNDRangeKernel(queue, calclmodel3D->kernelLogicalXOrReductionr, 1,
                                         NULL, &tmpThreads, NULL, 0, NULL, NULL);
            calclHandleError(err);
            break;
        case REDUCTION_BINARY_XOR:
            clSetKernelArg(calclmodel3D->kernelBinaryXorReductionr, 3, sizeof(CALint), &offset);
            clSetKernelArg(calclmodel3D->kernelBinaryXorReductionr, 5, sizeof(CALint), &count);
            err = clEnqueueNDRangeKernel(queue, calclmodel3D->kernelBinaryXorReductionr, 1,
                                         NULL, &tmpThreads, NULL, 0, NULL, NULL);
            calclHandleError(err);
            break;
        default:
            break;
        }
        offset <<= 1;
        count = count * 2 + 1;

    }

}


void calclExecuteReduction3D(struct CALCLModel3D* calclmodel3D, int rounded) {

    int i = 0;
    cl_int err;
    size_t tmp = calclmodel3D->host_CA->rows * calclmodel3D->host_CA->columns * calclmodel3D->host_CA->slices;

    for (i = 0; i < calclmodel3D->host_CA->sizeof_pQb_array; i++) {
        if (calclmodel3D->reductionFlagsMinb[i]) {
            clSetKernelArg(calclmodel3D->kernelMinReductionb, 1, sizeof(CALint), &i);
            clSetKernelArg(calclmodel3D->kernelMinCopyb, 2, sizeof(CALint), &i);
            err = clEnqueueNDRangeKernel(calclmodel3D->queue, calclmodel3D->kernelMinCopyb, 1, NULL, &tmp, NULL, 0, NULL, NULL);
            calclHandleError(err);
            calclComputeReduction3Db(calclmodel3D, i, REDUCTION_MIN, rounded);
        }
        if (calclmodel3D->reductionFlagsMaxb[i]) {
            clSetKernelArg(calclmodel3D->kernelMaxReductionb, 1, sizeof(CALint), &i);
            clSetKernelArg(calclmodel3D->kernelMaxCopyb, 2, sizeof(CALint), &i);
            err = clEnqueueNDRangeKernel(calclmodel3D->queue, calclmodel3D->kernelMaxCopyb, 1, NULL, &tmp, NULL, 0, NULL, NULL);
            calclHandleError(err);
            calclComputeReduction3Db(calclmodel3D, i, REDUCTION_MAX, rounded);
        }
        if (calclmodel3D->reductionFlagsSumb[i]) {
            clSetKernelArg(calclmodel3D->kernelSumReductionb, 1, sizeof(CALint), &i);
            clSetKernelArg(calclmodel3D->kernelSumCopyb, 2, sizeof(CALint), &i);
            err = clEnqueueNDRangeKernel(calclmodel3D->queue, calclmodel3D->kernelSumCopyb, 1, NULL, &tmp, NULL, 0, NULL, NULL);
            calclHandleError(err);
            calclComputeReduction3Db(calclmodel3D, i, REDUCTION_SUM, rounded);
        }
        if (calclmodel3D->reductionFlagsProdb[i]) {
            clSetKernelArg(calclmodel3D->kernelProdReductionb, 1, sizeof(CALint), &i);
            clSetKernelArg(calclmodel3D->kernelProdCopyb, 2, sizeof(CALint), &i);
            err = clEnqueueNDRangeKernel(calclmodel3D->queue, calclmodel3D->kernelProdCopyb, 1, NULL, &tmp, NULL, 0, NULL, NULL);
            calclHandleError(err);
            calclComputeReduction3Dr(calclmodel3D, i, REDUCTION_PROD, rounded);
        }
        if (calclmodel3D->reductionFlagsLogicalAndb[i]) {
            clSetKernelArg(calclmodel3D->kernelLogicalAndReductionb, 1, sizeof(CALint), &i);
            clSetKernelArg(calclmodel3D->kernelLogicalAndCopyb, 2, sizeof(CALint), &i);
            err = clEnqueueNDRangeKernel(calclmodel3D->queue, calclmodel3D->kernelLogicalAndCopyb, 1, NULL, &tmp, NULL, 0, NULL, NULL);
            calclHandleError(err);
            calclComputeReduction3Db(calclmodel3D, i, REDUCTION_LOGICAL_AND, rounded);
        }
        if (calclmodel3D->reductionFlagsLogicalOrb[i]) {
            clSetKernelArg(calclmodel3D->kernelLogicalOrReductionb, 1, sizeof(CALint), &i);
            clSetKernelArg(calclmodel3D->kernelLogicalOrCopyb, 2, sizeof(CALint), &i);
            err = clEnqueueNDRangeKernel(calclmodel3D->queue, calclmodel3D->kernelLogicalOrCopyb, 1, NULL, &tmp, NULL, 0, NULL, NULL);
            calclHandleError(err);
            calclComputeReduction3Db(calclmodel3D, i, REDUCTION_LOGICAL_OR, rounded);
        }
        if (calclmodel3D->reductionFlagsLogicalXOrb[i]) {
            clSetKernelArg(calclmodel3D->kernelLogicalXOrReductionb, 1, sizeof(CALint), &i);
            clSetKernelArg(calclmodel3D->kernelLogicalXOrCopyb, 2, sizeof(CALint), &i);
            err = clEnqueueNDRangeKernel(calclmodel3D->queue, calclmodel3D->kernelLogicalXOrCopyb, 1, NULL, &tmp, NULL, 0, NULL, NULL);
            calclHandleError(err);
            calclComputeReduction3Db(calclmodel3D, i, REDUCTION_LOGICAL_XOR, rounded);
        }
        if (calclmodel3D->reductionFlagsBinaryAndb[i]) {
            clSetKernelArg(calclmodel3D->kernelBinaryAndReductionb, 1, sizeof(CALint), &i);
            clSetKernelArg(calclmodel3D->kernelBinaryAndCopyb, 2, sizeof(CALint), &i);
            err = clEnqueueNDRangeKernel(calclmodel3D->queue, calclmodel3D->kernelBinaryAndCopyb, 1, NULL, &tmp, NULL, 0, NULL, NULL);
            calclHandleError(err);
            calclComputeReduction3Db(calclmodel3D, i, REDUCTION_BINARY_AND, rounded);
        }
        if (calclmodel3D->reductionFlagsBinaryOrb[i]) {
            clSetKernelArg(calclmodel3D->kernelBinaryOrReductionb, 1, sizeof(CALint), &i);
            clSetKernelArg(calclmodel3D->kernelBinaryOrCopyb, 2, sizeof(CALint), &i);
            err = clEnqueueNDRangeKernel(calclmodel3D->queue, calclmodel3D->kernelBinaryOrCopyb, 1, NULL, &tmp, NULL, 0, NULL, NULL);
            calclHandleError(err);
            calclComputeReduction3Db(calclmodel3D, i, REDUCTION_BINARY_OR, rounded);
        }
        if (calclmodel3D->reductionFlagsBinaryXOrb[i]) {
            clSetKernelArg(calclmodel3D->kernelBinaryXorReductionb, 1, sizeof(CALint), &i);
            clSetKernelArg(calclmodel3D->kernelBinaryXOrCopyb, 2, sizeof(CALint), &i);
            err = clEnqueueNDRangeKernel(calclmodel3D->queue, calclmodel3D->kernelBinaryXOrCopyb, 1, NULL, &tmp, NULL, 0, NULL, NULL);
            calclHandleError(err);
            calclComputeReduction3Db(calclmodel3D, i, REDUCTION_BINARY_XOR, rounded);
        }
    }


    for (i = 0; i < calclmodel3D->host_CA->sizeof_pQi_array; i++) {
        if (calclmodel3D->reductionFlagsMini[i]) {
            clSetKernelArg(calclmodel3D->kernelMinReductioni, 1, sizeof(CALint), &i);
            clSetKernelArg(calclmodel3D->kernelMinCopyi, 2, sizeof(CALint), &i);
            err = clEnqueueNDRangeKernel(calclmodel3D->queue, calclmodel3D->kernelMinCopyi, 1, NULL, &tmp, NULL, 0, NULL, NULL);
            calclHandleError(err);
            calclComputeReduction3Di(calclmodel3D, i, REDUCTION_MIN, rounded);
        }
        if (calclmodel3D->reductionFlagsMaxi[i]) {
            clSetKernelArg(calclmodel3D->kernelMaxReductioni, 1, sizeof(CALint), &i);
            clSetKernelArg(calclmodel3D->kernelMaxCopyi, 2, sizeof(CALint), &i);
            err = clEnqueueNDRangeKernel(calclmodel3D->queue, calclmodel3D->kernelMaxCopyi, 1, NULL, &tmp, NULL, 0, NULL, NULL);
            calclHandleError(err);
            calclComputeReduction3Di(calclmodel3D, i, REDUCTION_MAX, rounded);
        }
        if (calclmodel3D->reductionFlagsSumi[i]) {
            clSetKernelArg(calclmodel3D->kernelSumReductioni, 1, sizeof(CALint), &i);
            clSetKernelArg(calclmodel3D->kernelSumCopyi, 2, sizeof(CALint), &i);
            err = clEnqueueNDRangeKernel(calclmodel3D->queue, calclmodel3D->kernelSumCopyi, 1, NULL, &tmp, NULL, 0, NULL, NULL);
            calclHandleError(err);
            calclComputeReduction3Di(calclmodel3D, i, REDUCTION_SUM, rounded);
        }
        if (calclmodel3D->reductionFlagsProdi[i]) {
            clSetKernelArg(calclmodel3D->kernelProdReductioni, 1, sizeof(CALint), &i);
            clSetKernelArg(calclmodel3D->kernelProdCopyi, 2, sizeof(CALint), &i);
            err = clEnqueueNDRangeKernel(calclmodel3D->queue, calclmodel3D->kernelProdCopyi, 1, NULL, &tmp, NULL, 0, NULL, NULL);
            calclHandleError(err);
            calclComputeReduction3Dr(calclmodel3D, i, REDUCTION_PROD, rounded);
        }
        if (calclmodel3D->reductionFlagsLogicalAndi[i]) {
            clSetKernelArg(calclmodel3D->kernelLogicalAndReductioni, 1, sizeof(CALint), &i);
            clSetKernelArg(calclmodel3D->kernelLogicalAndCopyi, 2, sizeof(CALint), &i);
            err = clEnqueueNDRangeKernel(calclmodel3D->queue, calclmodel3D->kernelLogicalAndCopyi, 1, NULL, &tmp, NULL, 0, NULL, NULL);
            calclHandleError(err);
            calclComputeReduction3Di(calclmodel3D, i, REDUCTION_LOGICAL_AND, rounded);
        }
        if (calclmodel3D->reductionFlagsLogicalOri[i]) {
            clSetKernelArg(calclmodel3D->kernelLogicalOrReductioni, 1, sizeof(CALint), &i);
            clSetKernelArg(calclmodel3D->kernelLogicalOrCopyi, 2, sizeof(CALint), &i);
            err = clEnqueueNDRangeKernel(calclmodel3D->queue, calclmodel3D->kernelLogicalOrCopyi, 1, NULL, &tmp, NULL, 0, NULL, NULL);
            calclHandleError(err);
            calclComputeReduction3Di(calclmodel3D, i, REDUCTION_LOGICAL_OR, rounded);
        }
        if (calclmodel3D->reductionFlagsLogicalXOri[i]) {
            clSetKernelArg(calclmodel3D->kernelLogicalXOrReductioni, 1, sizeof(CALint), &i);
            clSetKernelArg(calclmodel3D->kernelLogicalXOrCopyi, 2, sizeof(CALint), &i);
            err = clEnqueueNDRangeKernel(calclmodel3D->queue, calclmodel3D->kernelLogicalXOrCopyi, 1, NULL, &tmp, NULL, 0, NULL, NULL);
            calclHandleError(err);
            calclComputeReduction3Di(calclmodel3D, i, REDUCTION_LOGICAL_XOR, rounded);
        }
        if (calclmodel3D->reductionFlagsBinaryAndi[i]) {
            clSetKernelArg(calclmodel3D->kernelBinaryAndReductioni, 1, sizeof(CALint), &i);
            clSetKernelArg(calclmodel3D->kernelBinaryAndCopyi, 2, sizeof(CALint), &i);
            err = clEnqueueNDRangeKernel(calclmodel3D->queue, calclmodel3D->kernelBinaryAndCopyi, 1, NULL, &tmp, NULL, 0, NULL, NULL);
            calclHandleError(err);
            calclComputeReduction3Di(calclmodel3D, i, REDUCTION_BINARY_AND, rounded);
        }
        if (calclmodel3D->reductionFlagsBinaryOri[i]) {
            clSetKernelArg(calclmodel3D->kernelBinaryOrReductioni, 1, sizeof(CALint), &i);
            clSetKernelArg(calclmodel3D->kernelBinaryOrCopyi, 2, sizeof(CALint), &i);
            err = clEnqueueNDRangeKernel(calclmodel3D->queue, calclmodel3D->kernelBinaryOrCopyi, 1, NULL, &tmp, NULL, 0, NULL, NULL);
            calclHandleError(err);
            calclComputeReduction3Di(calclmodel3D, i, REDUCTION_BINARY_OR, rounded);
        }
        if (calclmodel3D->reductionFlagsBinaryXOri[i]) {
            clSetKernelArg(calclmodel3D->kernelBinaryXorReductioni, 1, sizeof(CALint), &i);
            clSetKernelArg(calclmodel3D->kernelBinaryXOrCopyi, 2, sizeof(CALint), &i);
            err = clEnqueueNDRangeKernel(calclmodel3D->queue, calclmodel3D->kernelBinaryXOrCopyi, 1, NULL, &tmp, NULL, 0, NULL, NULL);
            calclHandleError(err);
            calclComputeReduction3Di(calclmodel3D, i, REDUCTION_BINARY_XOR, rounded);
        }
    }

    for (i = 0; i < calclmodel3D->host_CA->sizeof_pQr_array; i++) {

        if (calclmodel3D->reductionFlagsMinr[i]) {
            clSetKernelArg(calclmodel3D->kernelMinReductionr, 1, sizeof(CALint), &i);
            clSetKernelArg(calclmodel3D->kernelMinCopyr, 2, sizeof(CALint), &i);
            err = clEnqueueNDRangeKernel(calclmodel3D->queue, calclmodel3D->kernelMinCopyr, 1, NULL, &tmp, NULL, 0, NULL, NULL);
            calclHandleError(err);
            calclComputeReduction3Dr(calclmodel3D, i, REDUCTION_MIN, rounded);
        }
        if (calclmodel3D->reductionFlagsMaxr[i]) {
            clSetKernelArg(calclmodel3D->kernelMaxReductionr, 1, sizeof(CALint), &i);
            clSetKernelArg(calclmodel3D->kernelMaxCopyr, 2, sizeof(CALint), &i);
            err = clEnqueueNDRangeKernel(calclmodel3D->queue, calclmodel3D->kernelMaxCopyr, 1, NULL, &tmp, NULL, 0, NULL, NULL);
            calclHandleError(err);
            calclComputeReduction3Dr(calclmodel3D, i, REDUCTION_MAX, rounded);
        }
        if (calclmodel3D->reductionFlagsSumr[i]) {
            clSetKernelArg(calclmodel3D->kernelSumReductionr, 1, sizeof(CALint), &i);
            clSetKernelArg(calclmodel3D->kernelSumCopyr, 2, sizeof(CALint), &i);
            err = clEnqueueNDRangeKernel(calclmodel3D->queue, calclmodel3D->kernelSumCopyr, 1, NULL, &tmp, NULL, 0, NULL, NULL);
            calclHandleError(err);
            calclComputeReduction3Dr(calclmodel3D, i, REDUCTION_SUM, rounded);
        }
        if (calclmodel3D->reductionFlagsProdr[i]) {
            clSetKernelArg(calclmodel3D->kernelProdReductionr, 1, sizeof(CALint), &i);
            clSetKernelArg(calclmodel3D->kernelProdCopyr, 2, sizeof(CALint), &i);
            err = clEnqueueNDRangeKernel(calclmodel3D->queue, calclmodel3D->kernelProdCopyr, 1, NULL, &tmp, NULL, 0, NULL, NULL);
            calclHandleError(err);
            calclComputeReduction3Dr(calclmodel3D, i, REDUCTION_PROD, rounded);
        }
        if (calclmodel3D->reductionFlagsLogicalAndr[i]) {
            clSetKernelArg(calclmodel3D->kernelLogicalAndReductionr, 1, sizeof(CALint), &i);
            clSetKernelArg(calclmodel3D->kernelLogicalAndCopyr, 2, sizeof(CALint), &i);
            err = clEnqueueNDRangeKernel(calclmodel3D->queue, calclmodel3D->kernelLogicalAndCopyr, 1, NULL, &tmp, NULL, 0, NULL, NULL);
            calclHandleError(err);
            calclComputeReduction3Dr(calclmodel3D, i, REDUCTION_LOGICAL_AND, rounded);
        }
        if (calclmodel3D->reductionFlagsLogicalOrr[i]) {
            clSetKernelArg(calclmodel3D->kernelLogicalOrReductionr, 1, sizeof(CALint), &i);
            clSetKernelArg(calclmodel3D->kernelLogicalOrCopyr, 2, sizeof(CALint), &i);
            err = clEnqueueNDRangeKernel(calclmodel3D->queue, calclmodel3D->kernelLogicalOrCopyr, 1, NULL, &tmp, NULL, 0, NULL, NULL);
            calclHandleError(err);
            calclComputeReduction3Dr(calclmodel3D, i, REDUCTION_LOGICAL_OR, rounded);
        }
        if (calclmodel3D->reductionFlagsLogicalXOrr[i]) {
            clSetKernelArg(calclmodel3D->kernelLogicalXOrReductionr, 1, sizeof(CALint), &i);
            clSetKernelArg(calclmodel3D->kernelLogicalXOrCopyr, 2, sizeof(CALint), &i);
            err = clEnqueueNDRangeKernel(calclmodel3D->queue, calclmodel3D->kernelLogicalXOrCopyr, 1, NULL, &tmp, NULL, 0, NULL, NULL);
            calclHandleError(err);
            calclComputeReduction3Dr(calclmodel3D, i, REDUCTION_LOGICAL_XOR, rounded);
        }
        if (calclmodel3D->reductionFlagsBinaryAndr[i]) {
            clSetKernelArg(calclmodel3D->kernelBinaryAndReductionr, 1, sizeof(CALint), &i);
            clSetKernelArg(calclmodel3D->kernelBinaryAndCopyr, 2, sizeof(CALint), &i);
            err = clEnqueueNDRangeKernel(calclmodel3D->queue, calclmodel3D->kernelBinaryAndCopyr, 1, NULL, &tmp, NULL, 0, NULL, NULL);
            calclHandleError(err);
            calclComputeReduction3Dr(calclmodel3D, i, REDUCTION_BINARY_AND, rounded);
        }
        if (calclmodel3D->reductionFlagsBinaryOrr[i]) {
            clSetKernelArg(calclmodel3D->kernelBinaryOrReductionr, 1, sizeof(CALint), &i);
            clSetKernelArg(calclmodel3D->kernelBinaryOrCopyr, 2, sizeof(CALint), &i);
            err = clEnqueueNDRangeKernel(calclmodel3D->queue, calclmodel3D->kernelBinaryOrCopyr, 1, NULL, &tmp, NULL, 0, NULL, NULL);
            calclHandleError(err);
            calclComputeReduction3Dr(calclmodel3D, i, REDUCTION_BINARY_OR, rounded);
        }
        if (calclmodel3D->reductionFlagsBinaryXOrr[i]) {
            clSetKernelArg(calclmodel3D->kernelBinaryXorReductionr, 1, sizeof(CALint), &i);
            clSetKernelArg(calclmodel3D->kernelBinaryXOrCopyr, 2, sizeof(CALint), &i);
            err = clEnqueueNDRangeKernel(calclmodel3D->queue, calclmodel3D->kernelBinaryXOrCopyr, 1, NULL, &tmp, NULL, 0, NULL, NULL);
            calclHandleError(err);
            calclComputeReduction3Dr(calclmodel3D, i, REDUCTION_BINARY_XOR, rounded);
        }
    }

}


CALbyte calclSingleStep3D(struct CALCLModel3D* calclmodel3D, size_t * threadsNum, int dimNum) {

    CALbyte activeCells = calclmodel3D->opt == CAL_OPT_ACTIVE_CELLS_NAIVE;
    int j;
    if (activeCells) {
        for (j = 0; j < calclmodel3D->elementaryProcessesNum; j++) {

            calclKernelCall3D(calclmodel3D, calclmodel3D->elementaryProcesses[j] , dimNum, threadsNum, NULL);
            calclComputeStreamCompaction3D(calclmodel3D);
            calclResizeThreadsNum3D(calclmodel3D, threadsNum);
            calclKernelCall3D(calclmodel3D, calclmodel3D->kernelUpdateSubstate, dimNum, threadsNum, NULL);
        }

        calclExecuteReduction3D(calclmodel3D, calclmodel3D->roundedDimensions);

        if (calclmodel3D->kernelSteering != NULL) {
            calclKernelCall3D(calclmodel3D, calclmodel3D->kernelSteering, dimNum, threadsNum, NULL);
            calclKernelCall3D(calclmodel3D, calclmodel3D->kernelUpdateSubstate, dimNum, threadsNum, NULL);
        }

    } else {
        for (j = 0; j < calclmodel3D->elementaryProcessesNum; j++) {
            calclKernelCall3D(calclmodel3D, calclmodel3D->elementaryProcesses[j], dimNum, threadsNum, NULL);
            calclCopySubstatesBuffers3D(calclmodel3D);

        }

        calclExecuteReduction3D(calclmodel3D, calclmodel3D->roundedDimensions);

        if (calclmodel3D->kernelSteering != NULL) {
            calclKernelCall3D(calclmodel3D, calclmodel3D->kernelSteering, dimNum, threadsNum, NULL);
            calclCopySubstatesBuffers3D(calclmodel3D);
        }

    }

    if (calclmodel3D->cl_update_substates != NULL && calclmodel3D->steps % calclmodel3D->callbackSteps == 0) {
        calclGetSubstatesDeviceToHost3D(calclmodel3D);
        calclmodel3D->cl_update_substates(calclmodel3D->host_CA);
    }

    calclmodel3D->steps++;

    if (calclmodel3D->kernelStopCondition != NULL) {
        return checkStopCondition3D(calclmodel3D, dimNum, threadsNum);
    }

    return CAL_FALSE;

}

void calclKernelCall3D(struct CALCLModel3D * calclmodel3D, CALCLkernel ker, int numDim, size_t * dimSize, size_t * localDimSize) {
    CALCLqueue queue = calclmodel3D->queue;

    //	cl_event timing_event;
    //	cl_ulong time_start, time_end, read_time;
    cl_int err;
    CALCLdevice device;
    size_t multiple;
    err = clGetCommandQueueInfo(queue, CL_QUEUE_DEVICE, sizeof(CALCLdevice), &device, NULL);
    calclHandleError(err);
    err = clGetKernelWorkGroupInfo(ker, device, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(size_t), &multiple, NULL);
    calclHandleError(err);

    calclRoundThreadsNum3D(dimSize, numDim, multiple);
    err = clEnqueueNDRangeKernel(queue, ker, numDim, NULL, dimSize, localDimSize, 0, NULL, NULL);
    calclHandleError(err);
    //	err = clEnqueueNDRangeKernel(queue, ker, numDim, NULL, dimSize, localDimSize, 0, NULL, &timing_event);
    //
    //	err = clFinish(queue);
    //	calclHandleError(err);
    //
    //	clGetEventProfilingInfo(timing_event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
    //	clGetEventProfilingInfo(timing_event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
    //	read_time = time_end - time_start;
    //	char kernel_name[40];
    //	clGetKernelInfo(ker, CL_KERNEL_FUNCTION_NAME, sizeof(kernel_name), kernel_name, NULL);
    //
    //	out.open(kernel_name, ios_base::app);
    //	out << read_time << "\n";
    //	out.close();
    //
    //	clReleaseEvent(timing_event);
    //	printf("kernel %s %lu\n", kernel_name, read_time);

    //err = clFinish(queue);
    //calclHandleError(err);

}

void calclComputeStreamCompaction3D(struct CALCLModel3D * calclmodel3D) {
    CALCLqueue queue = calclmodel3D->queue;

    calclKernelCall3D(calclmodel3D, calclmodel3D->kernelComputeCounts, 1, &calclmodel3D->streamCompactionThreadsNum, NULL);
    int iterations = calclmodel3D->streamCompactionThreadsNum;
    size_t tmpThreads = iterations;
    cl_int err;

    int i;

    for (i = iterations / 2; i > 0; i /= 2) {
        tmpThreads = i;
        err = clEnqueueNDRangeKernel(queue, calclmodel3D->kernelUpSweep, 1, NULL, &tmpThreads, NULL, 0, NULL, NULL);
        calclHandleError(err);
    }

    iterations = calclmodel3D->streamCompactionThreadsNum;

    for (i = 1; i < iterations; i *= 2) {
        tmpThreads = i;
        err = clEnqueueNDRangeKernel(queue, calclmodel3D->kernelDownSweep, 1, NULL, &tmpThreads, NULL, 0, NULL, NULL);
        calclHandleError(err);
    }

    calclKernelCall3D(calclmodel3D, calclmodel3D->kernelCompact, 1, &calclmodel3D->streamCompactionThreadsNum, NULL);
}



void calclSetKernelArgs3D(CALCLkernel * kernel, CALCLmem * args, cl_uint numArgs) {
    unsigned int i;
    for (i = 0; i < numArgs; i++)
        clSetKernelArg(*kernel, MODEL_ARGS_NUM + i, sizeof(CALCLmem), &args[i]);
}

void calclAddStopConditionFunc3D(struct CALCLModel3D * calclmodel3D, CALCLkernel * kernel) {
    calclmodel3D->kernelStopCondition = *kernel;
    calclSetModelParameters3D(calclmodel3D, kernel);
}

void calclAddInitFunc3D(struct CALCLModel3D * calclmodel3D, CALCLkernel * kernel) {
    calclmodel3D->kernelInitSubstates = *kernel;
    calclSetModelParameters3D(calclmodel3D, kernel);
}

void calclAddSteeringFunc3D(struct CALCLModel3D * calclmodel3D, CALCLkernel * kernel) {
    calclmodel3D->kernelSteering = *kernel;
    calclSetModelParameters3D(calclmodel3D, kernel);
}

void calclBackToHostFunc3D(struct CALCLModel3D* calclmodel3D, void (*cl_update_substates)(struct CALModel3D*), int callbackSteps) {
    calclmodel3D->cl_update_substates = cl_update_substates;
    calclmodel3D->callbackSteps = callbackSteps;
}

void calclAddElementaryProcess3D(struct CALCLModel3D* calclmodel3D, CALCLkernel * kernel) {

    cl_uint size = calclmodel3D->elementaryProcessesNum;

    CALCLkernel * ep = calclmodel3D->elementaryProcesses;
    CALCLkernel * ep_new = (CALCLkernel*) malloc(sizeof(CALCLkernel) * (size + 1));

    unsigned int i;

    for (i = 0; i < size; i++)
        ep_new[i] = ep[i];

    ep_new[size] = *kernel;

    if (size > 0)
        free(ep);

    calclmodel3D->elementaryProcessesNum++;
    calclmodel3D->elementaryProcesses = ep_new;

    calclSetModelParameters3D(calclmodel3D, kernel);
}



void calclFinalize3D(struct CALCLModel3D * calclmodel3D) {
    int i;

    clReleaseKernel(calclmodel3D->kernelCompact);
    clReleaseKernel(calclmodel3D->kernelComputeCounts);
    clReleaseKernel(calclmodel3D->kernelDownSweep);
    clReleaseKernel(calclmodel3D->kernelInitSubstates);
    clReleaseKernel(calclmodel3D->kernelSteering);
    clReleaseKernel(calclmodel3D->kernelUpSweep);
    clReleaseKernel(calclmodel3D->kernelUpdateSubstate);
    clReleaseKernel(calclmodel3D->kernelStopCondition);

    clReleaseKernel(calclmodel3D->kernelMergeFlags);
    clReleaseKernel(calclmodel3D->kernelSetDiffFlags);

    for (i = 0; i < calclmodel3D->elementaryProcessesNum; ++i)
        clReleaseKernel(calclmodel3D->elementaryProcesses[i]);


    clReleaseMemObject(calclmodel3D->bufferActiveCells);
    clReleaseMemObject(calclmodel3D->bufferActiveCellsFlags);
    clReleaseMemObject(calclmodel3D->bufferActiveCellsNum);
    clReleaseMemObject(calclmodel3D->bufferBoundaryCondition);
    clReleaseMemObject(calclmodel3D->bufferByteSubstateNum);
    clReleaseMemObject(calclmodel3D->bufferColumns);
    clReleaseMemObject(calclmodel3D->bufferSlices);
    clReleaseMemObject(calclmodel3D->bufferCurrentByteSubstate);
    clReleaseMemObject(calclmodel3D->bufferCurrentIntSubstate);
    clReleaseMemObject(calclmodel3D->bufferCurrentRealSubstate);
    clReleaseMemObject(calclmodel3D->bufferIntSubstateNum);
    clReleaseMemObject(calclmodel3D->bufferNeighborhood);
    clReleaseMemObject(calclmodel3D->bufferNeighborhoodID);
    clReleaseMemObject(calclmodel3D->bufferNeighborhoodSize);
    clReleaseMemObject(calclmodel3D->bufferNextByteSubstate);
    clReleaseMemObject(calclmodel3D->bufferNextIntSubstate);
    clReleaseMemObject(calclmodel3D->bufferNextRealSubstate);
    clReleaseMemObject(calclmodel3D->bufferRealSubstateNum);
    clReleaseMemObject(calclmodel3D->bufferRows);
    clReleaseMemObject(calclmodel3D->bufferSTCounts);
    clReleaseMemObject(calclmodel3D->bufferSTOffsets1);
    clReleaseMemObject(calclmodel3D->bufferStop);
    clReleaseMemObject(calclmodel3D->bufferSTCountsDiff);

    clReleaseKernel(calclmodel3D->kernelMinReductionb);
    clReleaseKernel(calclmodel3D->kernelMinReductioni);
    clReleaseKernel(calclmodel3D->kernelMinReductionr);
    clReleaseKernel(calclmodel3D->kernelMaxReductionb);
    clReleaseKernel(calclmodel3D->kernelMaxReductioni);
    clReleaseKernel(calclmodel3D->kernelMaxReductionr);
    clReleaseKernel(calclmodel3D->kernelSumReductionb);
    clReleaseKernel(calclmodel3D->kernelSumReductioni);
    clReleaseKernel(calclmodel3D->kernelSumReductionr);
    clReleaseKernel(calclmodel3D->kernelProdReductionb);
    clReleaseKernel(calclmodel3D->kernelProdReductioni);
    clReleaseKernel(calclmodel3D->kernelProdReductionr);
    clReleaseKernel(calclmodel3D->kernelLogicalAndReductionb);
    clReleaseKernel(calclmodel3D->kernelLogicalAndReductioni);
    clReleaseKernel(calclmodel3D->kernelLogicalAndReductionr);
    clReleaseKernel(calclmodel3D->kernelLogicalOrReductionb);
    clReleaseKernel(calclmodel3D->kernelLogicalOrReductioni);
    clReleaseKernel(calclmodel3D->kernelLogicalOrReductionr);
    clReleaseKernel(calclmodel3D->kernelLogicalXOrReductionb);
    clReleaseKernel(calclmodel3D->kernelLogicalXOrReductioni);
    clReleaseKernel(calclmodel3D->kernelLogicalXOrReductionr);
    clReleaseKernel(calclmodel3D->kernelBinaryAndReductionb);
    clReleaseKernel(calclmodel3D->kernelBinaryAndReductioni);
    clReleaseKernel(calclmodel3D->kernelBinaryAndReductionr);
    clReleaseKernel(calclmodel3D->kernelBinaryOrReductionb);
    clReleaseKernel(calclmodel3D->kernelBinaryOrReductioni);
    clReleaseKernel(calclmodel3D->kernelBinaryOrReductionr);
    clReleaseKernel(calclmodel3D->kernelBinaryXorReductionb);
    clReleaseKernel(calclmodel3D->kernelBinaryXorReductioni);
    clReleaseKernel(calclmodel3D->kernelBinaryXorReductionr);

    clReleaseKernel(calclmodel3D->kernelMinCopyb);
    clReleaseKernel(calclmodel3D->kernelMinCopyi);
    clReleaseKernel(calclmodel3D->kernelMinCopyr);
    clReleaseKernel(calclmodel3D->kernelMaxCopyb);
    clReleaseKernel(calclmodel3D->kernelMaxCopyi);
    clReleaseKernel(calclmodel3D->kernelMaxCopyr);
    clReleaseKernel(calclmodel3D->kernelSumCopyb);
    clReleaseKernel(calclmodel3D->kernelSumCopyi);
    clReleaseKernel(calclmodel3D->kernelSumCopyr);
    clReleaseKernel(calclmodel3D->kernelProdCopyb);
    clReleaseKernel(calclmodel3D->kernelProdCopyi);
    clReleaseKernel(calclmodel3D->kernelProdCopyr);
    clReleaseKernel(calclmodel3D->kernelLogicalAndCopyb);
    clReleaseKernel(calclmodel3D->kernelLogicalAndCopyi);
    clReleaseKernel(calclmodel3D->kernelLogicalAndCopyr);
    clReleaseKernel(calclmodel3D->kernelLogicalOrCopyb);
    clReleaseKernel(calclmodel3D->kernelLogicalOrCopyi);
    clReleaseKernel(calclmodel3D->kernelLogicalOrCopyr);
    clReleaseKernel(calclmodel3D->kernelLogicalXOrCopyb);
    clReleaseKernel(calclmodel3D->kernelLogicalXOrCopyi);
    clReleaseKernel(calclmodel3D->kernelLogicalXOrCopyr);
    clReleaseKernel(calclmodel3D->kernelBinaryAndCopyb);
    clReleaseKernel(calclmodel3D->kernelBinaryAndCopyi);
    clReleaseKernel(calclmodel3D->kernelBinaryAndCopyr);
    clReleaseKernel(calclmodel3D->kernelBinaryOrCopyb);
    clReleaseKernel(calclmodel3D->kernelBinaryOrCopyi);
    clReleaseKernel(calclmodel3D->kernelBinaryOrCopyr);
    clReleaseKernel(calclmodel3D->kernelBinaryXOrCopyb);
    clReleaseKernel(calclmodel3D->kernelBinaryXOrCopyi);
    clReleaseKernel(calclmodel3D->kernelBinaryXOrCopyr);

    clReleaseMemObject(calclmodel3D->bufferBinaryAndsb);
    clReleaseMemObject(calclmodel3D->bufferBinaryAndsi);
    clReleaseMemObject(calclmodel3D->bufferBinaryAndsr);
    clReleaseMemObject(calclmodel3D->bufferBinaryOrsb);
    clReleaseMemObject(calclmodel3D->bufferBinaryOrsi);
    clReleaseMemObject(calclmodel3D->bufferBinaryOrsr);
    clReleaseMemObject(calclmodel3D->bufferBinaryXOrsb);
    clReleaseMemObject(calclmodel3D->bufferBinaryXOrsi);
    clReleaseMemObject(calclmodel3D->bufferBinaryXOrsr);

    clReleaseMemObject(calclmodel3D->bufferLogicalAndsb);
    clReleaseMemObject(calclmodel3D->bufferLogicalAndsi);
    clReleaseMemObject(calclmodel3D->bufferLogicalAndsr);
    clReleaseMemObject(calclmodel3D->bufferLogicalOrsb);
    clReleaseMemObject(calclmodel3D->bufferLogicalOrsi);
    clReleaseMemObject(calclmodel3D->bufferLogicalOrsr);
    clReleaseMemObject(calclmodel3D->bufferLogicalXOrsb);
    clReleaseMemObject(calclmodel3D->bufferLogicalXOrsi);
    clReleaseMemObject(calclmodel3D->bufferLogicalXOrsr);

    clReleaseMemObject(calclmodel3D->bufferMinimab);
    clReleaseMemObject(calclmodel3D->bufferMinimai);
    clReleaseMemObject(calclmodel3D->bufferMinimar);
    clReleaseMemObject(calclmodel3D->bufferPartialMaxb);
    clReleaseMemObject(calclmodel3D->bufferPartialMaxi);
    clReleaseMemObject(calclmodel3D->bufferPartialMaxr);
    clReleaseMemObject(calclmodel3D->bufferPartialSumb);
    clReleaseMemObject(calclmodel3D->bufferPartialSumi);
    clReleaseMemObject(calclmodel3D->bufferPartialSumr);

    clReleaseMemObject(calclmodel3D->bufferPartialLogicalAndb);
    clReleaseMemObject(calclmodel3D->bufferPartialLogicalAndi);
    clReleaseMemObject(calclmodel3D->bufferPartialLogicalAndr);
    clReleaseMemObject(calclmodel3D->bufferPartialLogicalOrb);
    clReleaseMemObject(calclmodel3D->bufferPartialLogicalOri);
    clReleaseMemObject(calclmodel3D->bufferPartialLogicalOrr);
    clReleaseMemObject(calclmodel3D->bufferPartialLogicalXOrb);
    clReleaseMemObject(calclmodel3D->bufferPartialLogicalXOri);
    clReleaseMemObject(calclmodel3D->bufferPartialLogicalXOrr);

    clReleaseMemObject(calclmodel3D->bufferPartialBinaryAndb);
    clReleaseMemObject(calclmodel3D->bufferPartialBinaryAndi);
    clReleaseMemObject(calclmodel3D->bufferPartialBinaryAndr);
    clReleaseMemObject(calclmodel3D->bufferPartialBinaryOrb);
    clReleaseMemObject(calclmodel3D->bufferPartialBinaryOri);
    clReleaseMemObject(calclmodel3D->bufferPartialBinaryOrr);
    clReleaseMemObject(calclmodel3D->bufferPartialBinaryXOrb);
    clReleaseMemObject(calclmodel3D->bufferPartialBinaryXOri);
    clReleaseMemObject(calclmodel3D->bufferPartialBinaryXOrr);



    clReleaseCommandQueue(calclmodel3D->queue);

    free(calclmodel3D->borderMapper.realBorder_OUT);
    free(calclmodel3D->borderMapper.intBorder_OUT);
    free(calclmodel3D->borderMapper.byteBorder_OUT);


    free(calclmodel3D->substateMapper.byteSubstate_current_OUT);
    free(calclmodel3D->substateMapper.intSubstate_current_OUT);
    free(calclmodel3D->substateMapper.realSubstate_current_OUT);

    free(calclmodel3D->elementaryProcesses);
    free(calclmodel3D);

}

CALCLprogram calclLoadProgram3D(CALCLcontext context, CALCLdevice device, char* path_user_kernel, char* path_user_include) {
    char* u = " -cl-denorms-are-zero -cl-finite-math-only ";
    char* pathOpenCALCL= getenv("OPENCALCL_KERNEL_PATH");
    if (pathOpenCALCL == NULL) {
        perror("please configure environment variable OPENCALCL_KERNEL_PATH");
        exit(1);
    }
    char* tmp;
    if (path_user_include == NULL) {
        tmp = (char*) malloc(sizeof(char) * (strlen(pathOpenCALCL) + strlen(KERNEL_INCLUDE_DIR) + strlen(" -I ") + strlen(u) + 1));
        strcpy(tmp, " -I ");
    } else {
        tmp = (char*) malloc(sizeof(char) * (strlen(path_user_include) + strlen(pathOpenCALCL) + strlen(KERNEL_INCLUDE_DIR) + strlen(" -I ") * 2 + strlen(u) + 1));
        strcpy(tmp, " -I ");
        strcat(tmp, path_user_include);
        strcat(tmp, " -I ");
    }
    strcat(tmp, pathOpenCALCL);
    strcat(tmp, KERNEL_INCLUDE_DIR);
    strcat(tmp, u);

    int num_files;
    char** filesNames;
    char** paths = (char**) malloc(sizeof(char*) * 2);
    char* tmp2 = (char*) malloc(sizeof(char) * (strlen(pathOpenCALCL) + strlen(KERNEL_SOURCE_DIR))+1);
    strcpy(tmp2,pathOpenCALCL );
    strcat(tmp2,KERNEL_SOURCE_DIR );

    paths[0] = path_user_kernel;
    paths[1] = tmp2;

    calclGetDirFiles(paths, 2, &filesNames, &num_files);

    CALCLprogram program = calclGetProgramFromFiles(filesNames, num_files, tmp, context, &device, 1);
    int i;
    for (i = 0; i < num_files; i++) {
        free(filesNames[i]);
    }
    free(filesNames);
    free(tmp);
    return program;
}

int calclSetKernelArg3D(CALCLkernel kernel, cl_uint arg_index,size_t arg_size,const void *arg_value){
    return  clSetKernelArg(kernel,MODEL_ARGS_NUM + arg_index, arg_size,arg_value);
}

void calclSetWorkGroupDimensions3D(struct CALCLModel3D * calclmodel3D, int m, int n)
{
    calclmodel3D->workGroupDimensions = (size_t*) malloc(sizeof(size_t)*2);
    calclmodel3D->workGroupDimensions[0]=m;
    calclmodel3D->workGroupDimensions[1]=n;
}
