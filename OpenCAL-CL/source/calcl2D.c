// (C) Copyright University of Calabria and others.
// All rights reserved. This program and the accompanying materials
// are made available under the terms of the GNU Lesser General Public License
// (LGPL) version 2.1 which accompanies this distribution, and is available at
// http://www.gnu.org/licenses/lgpl-2.1.html
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
// Lesser General Public License for more details.

/*
 * calCL.cpp
 *
 *  Created on: 10/giu/2014
 *      Author: alessio
 */
#include <OpenCAL-CL/calcl2D.h>

/******************************************************************************
 * 							PRIVATE FUNCTIONS
 ******************************************************************************/
void calclMapperToSubstates2D(struct CALModel2D *model, CALCLSubstateMapper * mapper) {

	int ssNum_r = model->sizeof_pQr_array;
	int ssNum_i = model->sizeof_pQi_array;
	int ssNum_b = model->sizeof_pQb_array;
	size_t elNum = model->columns * model->rows;

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

void calclGetSubstateKernel2D(CALCLModel2D* calclmodel2D, struct CALModel2D * model) {

	CALCLqueue queue = calclmodel2D->queue;

	cl_int err;
	size_t zero = 0;

	err = clEnqueueReadBuffer(queue, calclmodel2D->bufferCurrentRealSubstate, CL_TRUE, zero, calclmodel2D->substateMapper.bufDIMreal, calclmodel2D->substateMapper.realSubstate_current_OUT, 0, NULL, NULL);
	calclHandleError(err);
	err = clEnqueueReadBuffer(queue, calclmodel2D->bufferCurrentIntSubstate, CL_TRUE, zero, calclmodel2D->substateMapper.bufDIMint, calclmodel2D->substateMapper.intSubstate_current_OUT, 0, NULL, NULL);
	calclHandleError(err);
	err = clEnqueueReadBuffer(queue, calclmodel2D->bufferCurrentByteSubstate, CL_TRUE, zero, calclmodel2D->substateMapper.bufDIMbyte, calclmodel2D->substateMapper.byteSubstate_current_OUT, 0, NULL, NULL);
	calclHandleError(err);

	calclMapperToSubstates2D(model, &calclmodel2D->substateMapper);
}

void calclRoundThreadsNum2D(size_t * threadNum, int numDim, size_t multiple) {
	int i;
	for (i = 0; i < numDim; ++i)
		while (threadNum[i] % multiple != 0)
			threadNum[i]++;
}

void calclResizeThreadsNum2D(CALCLModel2D * calclmodel2D, struct CALModel2D *model, size_t * threadNum) {
	CALCLqueue queue = calclmodel2D->queue;

	cl_int err;
	size_t zero = 0;

	err = clEnqueueReadBuffer(queue, calclmodel2D->bufferActiveCellsNum, CL_TRUE, zero, sizeof(int), &model->A.size_current, 0, NULL, NULL);
	calclHandleError(err);
	threadNum[0] = model->A.size_current;
}

CALCLmem calclGetSubstateBuffer2D(CALCLmem bufferSubstates, cl_buffer_region region) {
	cl_int err;
	CALCLmem sub_buffer = clCreateSubBuffer(bufferSubstates, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, CL_BUFFER_CREATE_TYPE_REGION, &region, &err);
	calclHandleError(err);
	return sub_buffer;
}

void copySubstatesBuffers2D(struct CALModel2D * model, CALCLModel2D * calclmodel2D) {
	CALCLqueue queue = calclmodel2D->queue;

	if (model->sizeof_pQr_array > 0)
		clEnqueueCopyBuffer(queue, calclmodel2D->bufferNextRealSubstate, calclmodel2D->bufferCurrentRealSubstate, 0, 0, calclmodel2D->substateMapper.bufDIMreal, 0, NULL, NULL);
	if (model->sizeof_pQi_array > 0)
		clEnqueueCopyBuffer(queue, calclmodel2D->bufferNextIntSubstate, calclmodel2D->bufferCurrentIntSubstate, 0, 0, calclmodel2D->substateMapper.bufDIMint, 0, NULL, NULL);
	if (model->sizeof_pQb_array > 0)
		clEnqueueCopyBuffer(queue, calclmodel2D->bufferNextByteSubstate, calclmodel2D->bufferCurrentByteSubstate, 0, 0, calclmodel2D->substateMapper.bufDIMbyte, 0, NULL, NULL);
}

CALbyte checkStopCondition2D(CALCLModel2D * calclmodel2D, CALint dimNum, size_t * threadsNum) {
	CALCLqueue queue = calclmodel2D->queue;

	calclKernelCall2D(calclmodel2D, calclmodel2D->kernelStopCondition, dimNum, threadsNum, NULL);
	CALbyte stop = CAL_FALSE;
	size_t zero = 0;

	cl_int err = clEnqueueReadBuffer(queue, calclmodel2D->bufferStop, CL_TRUE, zero, sizeof(CALbyte), &stop, 0, NULL, NULL);
	calclHandleError(err);
	return stop;
}

void calclSetKernelStreamCompactionArgs2D(CALCLModel2D * calclmodel2D, struct CALModel2D * model) {
	CALint dim = model->rows * model->columns;
	clSetKernelArg(calclmodel2D->kernelComputeCounts, 0, sizeof(CALint), &dim);
	clSetKernelArg(calclmodel2D->kernelComputeCounts, 1, sizeof(CALCLmem), &calclmodel2D->bufferActiveCellsFlags);
	clSetKernelArg(calclmodel2D->kernelComputeCounts, 2, sizeof(CALCLmem), &calclmodel2D->bufferSTCounts);
	clSetKernelArg(calclmodel2D->kernelComputeCounts, 3, sizeof(CALCLmem), &calclmodel2D->bufferSTOffsets1);
	clSetKernelArg(calclmodel2D->kernelComputeCounts, 4, sizeof(CALCLmem), &calclmodel2D->bufferSTCountsDiff);

	int offset = calclmodel2D->streamCompactionThreadsNum / 2;

	clSetKernelArg(calclmodel2D->kernelUpSweep, 0, sizeof(CALCLmem), &calclmodel2D->bufferSTOffsets1);
	clSetKernelArg(calclmodel2D->kernelUpSweep, 1, sizeof(int), &offset);

	clSetKernelArg(calclmodel2D->kernelDownSweep, 0, sizeof(CALCLmem), &calclmodel2D->bufferSTOffsets1);
	clSetKernelArg(calclmodel2D->kernelDownSweep, 1, sizeof(int), &offset);

	clSetKernelArg(calclmodel2D->kernelCompact, 0, sizeof(CALint), &dim);
	clSetKernelArg(calclmodel2D->kernelCompact, 1, sizeof(CALint), &model->columns);
	clSetKernelArg(calclmodel2D->kernelCompact, 2, sizeof(CALCLmem), &calclmodel2D->bufferActiveCellsFlags);
	clSetKernelArg(calclmodel2D->kernelCompact, 3, sizeof(CALCLmem), &calclmodel2D->bufferActiveCellsNum);
	clSetKernelArg(calclmodel2D->kernelCompact, 4, sizeof(CALCLmem), &calclmodel2D->bufferActiveCells);
	clSetKernelArg(calclmodel2D->kernelCompact, 5, sizeof(CALCLmem), &calclmodel2D->bufferSTCounts);
	clSetKernelArg(calclmodel2D->kernelCompact, 6, sizeof(CALCLmem), &calclmodel2D->bufferSTOffsets1);

}

void calclSetKernelsLibArgs2D(CALCLModel2D *calclmodel2D, struct CALModel2D * model) {
	clSetKernelArg(calclmodel2D->kernelUpdateSubstate, 0, sizeof(CALint), &model->columns);
	clSetKernelArg(calclmodel2D->kernelUpdateSubstate, 1, sizeof(CALint), &model->rows);
	clSetKernelArg(calclmodel2D->kernelUpdateSubstate, 2, sizeof(CALint), &model->sizeof_pQb_array);
	clSetKernelArg(calclmodel2D->kernelUpdateSubstate, 3, sizeof(CALint), &model->sizeof_pQi_array);
	clSetKernelArg(calclmodel2D->kernelUpdateSubstate, 4, sizeof(CALint), &model->sizeof_pQr_array);
	clSetKernelArg(calclmodel2D->kernelUpdateSubstate, 5, sizeof(CALCLmem), &calclmodel2D->bufferCurrentByteSubstate);
	clSetKernelArg(calclmodel2D->kernelUpdateSubstate, 6, sizeof(CALCLmem), &calclmodel2D->bufferCurrentIntSubstate);
	clSetKernelArg(calclmodel2D->kernelUpdateSubstate, 7, sizeof(CALCLmem), &calclmodel2D->bufferCurrentRealSubstate);
	clSetKernelArg(calclmodel2D->kernelUpdateSubstate, 8, sizeof(CALCLmem), &calclmodel2D->bufferNextByteSubstate);
	clSetKernelArg(calclmodel2D->kernelUpdateSubstate, 9, sizeof(CALCLmem), &calclmodel2D->bufferNextIntSubstate);
	clSetKernelArg(calclmodel2D->kernelUpdateSubstate, 10, sizeof(CALCLmem), &calclmodel2D->bufferNextRealSubstate);
	clSetKernelArg(calclmodel2D->kernelUpdateSubstate, 11, sizeof(CALCLmem), &calclmodel2D->bufferActiveCells);
	clSetKernelArg(calclmodel2D->kernelUpdateSubstate, 12, sizeof(CALCLmem), &calclmodel2D->bufferActiveCellsNum);

}

void calclSetModelParameters2D(CALCLModel2D* calclmodel2D, struct CALModel2D * model, CALCLkernel * kernel) {

	clSetKernelArg(*kernel, 0, sizeof(CALCLmem), &calclmodel2D->bufferRows);
	clSetKernelArg(*kernel, 1, sizeof(CALCLmem), &calclmodel2D->bufferColumns);
	clSetKernelArg(*kernel, 2, sizeof(CALCLmem), &calclmodel2D->bufferByteSubstateNum);
	clSetKernelArg(*kernel, 3, sizeof(CALCLmem), &calclmodel2D->bufferIntSubstateNum);
	clSetKernelArg(*kernel, 4, sizeof(CALCLmem), &calclmodel2D->bufferRealSubstateNum);
	clSetKernelArg(*kernel, 5, sizeof(CALCLmem), &calclmodel2D->bufferCurrentByteSubstate);
	clSetKernelArg(*kernel, 6, sizeof(CALCLmem), &calclmodel2D->bufferCurrentIntSubstate);
	clSetKernelArg(*kernel, 7, sizeof(CALCLmem), &calclmodel2D->bufferCurrentRealSubstate);
	clSetKernelArg(*kernel, 8, sizeof(CALCLmem), &calclmodel2D->bufferNextByteSubstate);
	clSetKernelArg(*kernel, 9, sizeof(CALCLmem), &calclmodel2D->bufferNextIntSubstate);
	clSetKernelArg(*kernel, 10, sizeof(CALCLmem), &calclmodel2D->bufferNextRealSubstate);
	clSetKernelArg(*kernel, 11, sizeof(CALCLmem), &calclmodel2D->bufferActiveCells);
	clSetKernelArg(*kernel, 12, sizeof(CALCLmem), &calclmodel2D->bufferActiveCellsNum);
	clSetKernelArg(*kernel, 13, sizeof(CALCLmem), &calclmodel2D->bufferActiveCellsFlags);
	clSetKernelArg(*kernel, 14, sizeof(CALCLmem), &calclmodel2D->bufferNeighborhood);
	clSetKernelArg(*kernel, 15, sizeof(CALCLmem), &calclmodel2D->bufferNeighborhoodID);
	clSetKernelArg(*kernel, 16, sizeof(CALCLmem), &calclmodel2D->bufferNeighborhoodSize);
	clSetKernelArg(*kernel, 17, sizeof(CALCLmem), &calclmodel2D->bufferBoundaryCondition);
	clSetKernelArg(*kernel, 18, sizeof(CALCLmem), &calclmodel2D->bufferStop);
	clSetKernelArg(*kernel, 19, sizeof(CALCLmem), &calclmodel2D->bufferSTCountsDiff);
	double chunk_double = ceil((double)(model->rows * model->columns)/calclmodel2D->streamCompactionThreadsNum);
	int chunk = (int)chunk_double;
	clSetKernelArg(*kernel, 20, sizeof(int), &chunk);

}

void calclRealSubstatesMapper2D(struct CALModel2D *model, CALreal * current, CALreal * next) {
	int ssNum = model->sizeof_pQr_array;
	size_t elNum = model->columns * model->rows;
	long int outIndex = 0;
	long int outIndex1 = 0;
	int i;
	unsigned int j;

	for (i = 0; i < ssNum; i++) {
		for (j = 0; j < elNum; j++)
			current[outIndex++] = model->pQr_array[i]->current[j];
		for (j = 0; j < elNum; j++)
			next[outIndex1++] = model->pQr_array[i]->next[j];
	}
}
void calclByteSubstatesMapper2D(struct CALModel2D *model, CALbyte * current, CALbyte * next) {
	int ssNum = model->sizeof_pQb_array;
	size_t elNum = model->columns * model->rows;
	long int outIndex = 0;
	long int outIndex1 = 0;
	int i;
	unsigned int j;

	for (i = 0; i < ssNum; i++) {
		for (j = 0; j < elNum; j++)
			current[outIndex++] = model->pQb_array[i]->current[j];
		for (j = 0; j < elNum; j++)
			next[outIndex1++] = model->pQb_array[i]->next[j];
	}
}
void calclIntSubstatesMapper2D(struct CALModel2D *model, CALint * current, CALint * next) {
	int ssNum = model->sizeof_pQi_array;
	size_t elNum = model->columns * model->rows;
	long int outIndex = 0;
	long int outIndex1 = 0;
	int i;
	unsigned int j;

	for (i = 0; i < ssNum; i++) {
		for (j = 0; j < elNum; j++)
			current[outIndex++] = model->pQi_array[i]->current[j];
		for (j = 0; j < elNum; j++)
			next[outIndex1++] = model->pQi_array[i]->next[j];
	}
}

CALCLqueue calclCreateQueue2D(CALCLModel2D * calclmodel2D, struct CALModel2D * model, CALCLcontext context, CALCLdevice device) {
	CALCLqueue queue = calclCreateCommandQueue(context, device);
	size_t cores;
	cl_int err;
	err = clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &cores, NULL);
	calclHandleError(err);

	//TODO choose stream compaction threads num
	calclmodel2D->streamCompactionThreadsNum = cores * 4;

	while (model->rows * model->columns <= (int)calclmodel2D->streamCompactionThreadsNum)
		calclmodel2D->streamCompactionThreadsNum /= 2;

	calclmodel2D->bufferSTCounts = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, sizeof(CALint) * calclmodel2D->streamCompactionThreadsNum, NULL, &err);
	calclHandleError(err);
	calclmodel2D->bufferSTOffsets1 = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, sizeof(CALint) * calclmodel2D->streamCompactionThreadsNum, NULL, &err);
	calclHandleError(err);
	CALbyte * diff = (CALbyte*) malloc(sizeof(CALbyte) * calclmodel2D->streamCompactionThreadsNum);
	memset(diff, CAL_TRUE, sizeof(CALbyte) * calclmodel2D->streamCompactionThreadsNum);
	calclmodel2D->bufferSTCountsDiff = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, calclmodel2D->streamCompactionThreadsNum * sizeof(CALbyte), diff, &err);
	calclHandleError(err);
	free(diff);
	calclSetKernelStreamCompactionArgs2D(calclmodel2D, model);

	return queue;
}

/******************************************************************************
 * 							PUBLIC FUNCTIONS
 ******************************************************************************/

CALCLModel2D * calclCADef2D(struct CALModel2D *model, CALCLcontext context, CALCLprogram program, CALCLdevice device) {

	CALCLModel2D * calclmodel2D = (CALCLModel2D*) malloc(sizeof(CALCLModel2D));
	calclmodel2D->opt = model->OPTIMIZATION;
	calclmodel2D->cl_update_substates = NULL;
	calclmodel2D->kernelInitSubstates = NULL;
	calclmodel2D->kernelSteering = NULL;
	calclmodel2D->kernelStopCondition = NULL;
	calclmodel2D->elementaryProcessesNum = 0;
	calclmodel2D->steps = 0;

	if (model->A.flags == NULL) {
		model->A.flags = (CALbyte*) malloc(sizeof(CALbyte) * model->rows * model->columns);
		memset(model->A.flags, CAL_FALSE, sizeof(CALbyte) * model->rows * model->columns);
	}

	cl_int err;
	int bufferDim = model->columns * model->rows;

	calclmodel2D->kernelUpdateSubstate = calclGetKernelFromProgram(&program, KER_UPDATESUBSTATES);

	//stream compaction kernels
	calclmodel2D->kernelCompact = calclGetKernelFromProgram(&program, KER_STC_COMPACT);
	calclmodel2D->kernelComputeCounts = calclGetKernelFromProgram(&program, KER_STC_COMPUTE_COUNTS);
	calclmodel2D->kernelUpSweep = calclGetKernelFromProgram(&program, KER_STC_UP_SWEEP);
	calclmodel2D->kernelDownSweep = calclGetKernelFromProgram(&program, KER_STC_DOWN_SWEEP);

	struct CALCell2D * activeCells = (struct CALCell2D*) malloc(sizeof(struct CALCell2D) * bufferDim);
	memcpy(activeCells, model->A.cells, sizeof(struct CALCell2D) * model->A.size_current);

	calclmodel2D->bufferActiveCells = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(struct CALCell2D) * bufferDim, activeCells, &err);
	calclHandleError(err);
	free(activeCells);
	calclmodel2D->bufferActiveCellsFlags = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(CALbyte) * bufferDim, model->A.flags, &err);
	calclHandleError(err);

	calclmodel2D->bufferActiveCellsNum = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(CALint), &model->A.size_current, &err);
	calclHandleError(err);

	calclmodel2D->bufferByteSubstateNum = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(CALint), &model->sizeof_pQb_array, &err);
	calclHandleError(err);
	calclmodel2D->bufferIntSubstateNum = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(CALint), &model->sizeof_pQi_array, &err);
	calclHandleError(err);
	calclmodel2D->bufferRealSubstateNum = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(CALint), &model->sizeof_pQr_array, &err);
	calclHandleError(err);

	calclmodel2D->bufferColumns = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(CALint), &model->columns, &err);
	calclHandleError(err);
	calclmodel2D->bufferRows = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(CALint), &model->rows, &err);
	calclHandleError(err);

	size_t byteSubstatesDim = sizeof(CALbyte) * bufferDim * model->sizeof_pQb_array + 1;
	CALbyte * currentByteSubstates = (CALbyte*) malloc(byteSubstatesDim);
	CALbyte * nextByteSubstates = (CALbyte*) malloc(byteSubstatesDim);
	calclByteSubstatesMapper2D(model, currentByteSubstates, nextByteSubstates);
	calclmodel2D->bufferCurrentByteSubstate = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, byteSubstatesDim, currentByteSubstates, &err);
	calclHandleError(err);
	calclmodel2D->bufferNextByteSubstate = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, byteSubstatesDim, nextByteSubstates, &err);
	calclHandleError(err);
	free(currentByteSubstates);
	free(nextByteSubstates);

	size_t intSubstatesDim = sizeof(CALint) * bufferDim * model->sizeof_pQi_array + 1;
	CALint * currentIntSubstates = (CALint*) malloc(intSubstatesDim);
	CALint * nextIntSubstates = (CALint*) malloc(intSubstatesDim);
	calclIntSubstatesMapper2D(model, currentIntSubstates, nextIntSubstates);
	calclmodel2D->bufferCurrentIntSubstate = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, intSubstatesDim, currentIntSubstates, &err);
	calclHandleError(err);
	calclmodel2D->bufferNextIntSubstate = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, intSubstatesDim, nextIntSubstates, &err);
	calclHandleError(err);
	free(currentIntSubstates);
	free(nextIntSubstates);

	size_t realSubstatesDim = sizeof(CALreal) * bufferDim * model->sizeof_pQr_array + 1;
	CALreal * currentRealSubstates = (CALreal*) malloc(realSubstatesDim);
	CALreal * nextRealSubstates = (CALreal*) malloc(realSubstatesDim);
	calclRealSubstatesMapper2D(model, currentRealSubstates, nextRealSubstates);
	calclmodel2D->bufferCurrentRealSubstate = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, realSubstatesDim, currentRealSubstates, &err);
	calclHandleError(err);
	calclmodel2D->bufferNextRealSubstate = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, realSubstatesDim, nextRealSubstates, &err);
	calclHandleError(err);
	free(currentRealSubstates);
	free(nextRealSubstates);

	calclSetKernelsLibArgs2D(calclmodel2D, model);

	//user kernels buffers args

	calclmodel2D->bufferNeighborhood = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(struct CALCell2D) * model->sizeof_X, model->X, &err);
	calclHandleError(err);
	calclmodel2D->bufferNeighborhoodID = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(enum CALNeighborhood2D), &model->X_id, &err);
	calclHandleError(err);
	calclmodel2D->bufferNeighborhoodSize = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(CALint), &model->sizeof_X, &err);
	calclHandleError(err);
	calclmodel2D->bufferBoundaryCondition = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(enum CALSpaceBoundaryCondition), &model->T, &err);
	calclHandleError(err);

	//stop condition buffer
	CALbyte stop = CAL_FALSE;
	calclmodel2D->bufferStop = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(CALbyte), &stop, &err);
	calclHandleError(err);

	//init substates mapper
	calclmodel2D->substateMapper.bufDIMbyte = byteSubstatesDim;
	calclmodel2D->substateMapper.bufDIMreal = realSubstatesDim;
	calclmodel2D->substateMapper.bufDIMint = intSubstatesDim;
	calclmodel2D->substateMapper.byteSubstate_current_OUT = (CALbyte*) malloc(byteSubstatesDim);
	calclmodel2D->substateMapper.realSubstate_current_OUT = (CALreal*) malloc(realSubstatesDim);
	calclmodel2D->substateMapper.intSubstate_current_OUT = (CALint*) malloc(intSubstatesDim);

	calclmodel2D->queue = calclCreateQueue2D(calclmodel2D, model, context, device);

	return calclmodel2D;

}

void calclRun2D(CALCLModel2D* calclmodel2D, struct CALModel2D * model, unsigned int initialStep, unsigned maxStep) {
//	cl_int err;
	CALbyte stop;
	size_t * threadNumMax = (size_t*) malloc(sizeof(size_t) * 2);
	threadNumMax[0] = model->rows;
	threadNumMax[1] = model->columns;
	size_t * singleStepThreadNum;
	int dimNum;

	if (calclmodel2D->opt == CAL_NO_OPT) {
		singleStepThreadNum = (size_t*) malloc(sizeof(size_t) * 2);
		singleStepThreadNum[0] = threadNumMax[0];
		singleStepThreadNum[1] = threadNumMax[1];
		dimNum = 2;
	} else {
		singleStepThreadNum = (size_t*) malloc(sizeof(size_t));
		singleStepThreadNum[0] = model->A.size_current;
		dimNum = 1;
	}

	if (calclmodel2D->kernelInitSubstates != NULL)
		calclKernelCall2D(calclmodel2D, calclmodel2D->kernelInitSubstates, 1, threadNumMax, NULL);

	//TODO call update


	calclmodel2D->steps = initialStep;
	while (calclmodel2D->steps <= (int)maxStep || maxStep == CAL_RUN_LOOP) {
		stop = calclSingleStep2D(calclmodel2D, model, singleStepThreadNum, dimNum);
		if (stop == CAL_TRUE)
			break;
	}
	calclGetSubstateKernel2D(calclmodel2D, model);
	free(threadNumMax);
	free(singleStepThreadNum);
}

CALbyte calclSingleStep2D(CALCLModel2D* calclmodel2D, struct CALModel2D * model, size_t * threadsNum, int dimNum) {

	CALbyte activeCells = calclmodel2D->opt == CAL_OPT_ACTIVE_CELLS;
	int j;


	if (activeCells == CAL_TRUE) {
		for (j = 0; j < calclmodel2D->elementaryProcessesNum; j++) {

			calclKernelCall2D(calclmodel2D, calclmodel2D->elementaryProcesses[j],  dimNum, threadsNum, NULL);
			calclComputeStreamCompaction2D(calclmodel2D);
			calclResizeThreadsNum2D(calclmodel2D, model, threadsNum);
			calclKernelCall2D(calclmodel2D, calclmodel2D->kernelUpdateSubstate, dimNum, threadsNum, NULL);

		}
		if (calclmodel2D->kernelSteering != NULL) {
			calclKernelCall2D(calclmodel2D, calclmodel2D->kernelSteering, dimNum, threadsNum, NULL);
			calclKernelCall2D(calclmodel2D, calclmodel2D->kernelUpdateSubstate, dimNum, threadsNum, NULL);
		}

	} else {
		for (j = 0; j < calclmodel2D->elementaryProcessesNum; j++) {

			calclKernelCall2D(calclmodel2D, calclmodel2D->elementaryProcesses[j], dimNum, threadsNum, NULL);
			copySubstatesBuffers2D(model, calclmodel2D);

		}
		if (calclmodel2D->kernelSteering != NULL) {
			calclKernelCall2D(calclmodel2D, calclmodel2D->kernelSteering, dimNum, threadsNum, NULL);
			copySubstatesBuffers2D(model, calclmodel2D);
		}

	}

	if (calclmodel2D->cl_update_substates != NULL && calclmodel2D->steps % calclmodel2D->callbackSteps == 0) {
		calclGetSubstateKernel2D(calclmodel2D, model);
		calclmodel2D->cl_update_substates(model);
	}

	calclmodel2D->steps++;

	if (calclmodel2D->kernelStopCondition != NULL) {
		return checkStopCondition2D(calclmodel2D, dimNum, threadsNum);
	}

	return CAL_FALSE;

}

FILE * file;
void calclKernelCall2D(CALCLModel2D* calclmodel2D, CALCLkernel ker, int numDim, size_t * dimSize, size_t * localDimSize) {

//	cl_event timing_event;
//	cl_ulong time_start, cl_ulong time_end, read_time;
	cl_int err;
	CALCLdevice device;
	size_t multiple;
	CALCLqueue queue = calclmodel2D->queue;
	err = clGetCommandQueueInfo(queue, CL_QUEUE_DEVICE, sizeof(CALCLdevice), &device, NULL);
	calclHandleError(err);
	err = clGetKernelWorkGroupInfo(ker, device, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(size_t), &multiple, NULL);
	calclHandleError(err);

	calclRoundThreadsNum2D(dimSize, numDim, multiple);
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
//	file = fopen(kernel_name, "a");
//	fprintf(file, "%lu\n", read_time);
//	clReleaseEvent(timing_event);
//	fclose(file);

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

void calclComputeStreamCompaction2D(CALCLModel2D * calclmodel2D) {
	CALCLqueue queue = calclmodel2D->queue;
	calclKernelCall2D(calclmodel2D, calclmodel2D->kernelComputeCounts, 1, &calclmodel2D->streamCompactionThreadsNum, NULL);
	cl_int err;
	int iterations = calclmodel2D->streamCompactionThreadsNum;
	size_t tmpThreads = iterations;
	int i;

	for (i = iterations / 2; i > 0; i /= 2) {
		tmpThreads = i;
		err = clEnqueueNDRangeKernel(queue, calclmodel2D->kernelUpSweep, 1, NULL, &tmpThreads, NULL, 0, NULL, NULL);
		calclHandleError(err);
	}

	iterations = calclmodel2D->streamCompactionThreadsNum;

	for (i = 1; i < iterations; i *= 2) {
		tmpThreads = i;
		err = clEnqueueNDRangeKernel(queue, calclmodel2D->kernelDownSweep, 1, NULL, &tmpThreads, NULL, 0, NULL, NULL);
		calclHandleError(err);
	}

	calclKernelCall2D(calclmodel2D, calclmodel2D->kernelCompact, 1, &calclmodel2D->streamCompactionThreadsNum, NULL);
}

void calclSetKernelArgs2D(CALCLkernel * kernel, CALCLmem * args, cl_uint numArgs) {
	unsigned int i;
	for (i = 0; i < numArgs; i++)
		clSetKernelArg(*kernel, MODEL_ARGS_NUM + i, sizeof(CALCLmem), &args[i]);
}

void calclAddStopConditionFunc2D(CALCLModel2D * calclmodel2D,struct CALModel2D * model, CALCLkernel * kernel) {
	calclmodel2D->kernelStopCondition = *kernel;
	calclSetModelParameters2D(calclmodel2D,model, kernel);
}

void calclAddInitFunc2D(CALCLModel2D* calclmodel2D,struct CALModel2D * model, CALCLkernel * kernel) {
	calclmodel2D->kernelInitSubstates = *kernel;
	calclSetModelParameters2D(calclmodel2D,model, kernel);
}

void calclAddSteeringFunc2D(CALCLModel2D* calclmodel2D,struct CALModel2D * model, CALCLkernel * kernel) {
	calclmodel2D->kernelSteering = *kernel;
	calclSetModelParameters2D(calclmodel2D,model, kernel);
}

void calclBackToHostFunc2D(CALCLModel2D* calclmodel2D, void (*cl_update_substates)(struct CALModel2D*), int callbackSteps) {
	calclmodel2D->cl_update_substates = cl_update_substates;
	calclmodel2D->callbackSteps = callbackSteps;
}

void calclAddElementaryProcess2D(CALCLModel2D* calclmodel2D,struct CALModel2D * model, CALCLkernel * kernel) {

	cl_uint size = calclmodel2D->elementaryProcessesNum;

	CALCLkernel * ep = calclmodel2D->elementaryProcesses;
	CALCLkernel * ep_new = (CALCLkernel*) malloc(sizeof(CALCLkernel) * (size + 1));

	unsigned int i;
	for (i = 0; i < size; i++)
		ep_new[i] = ep[i];

	ep_new[size] = *kernel;

	if (size > 0)
		free(ep);

	calclmodel2D->elementaryProcessesNum++;
	calclmodel2D->elementaryProcesses = ep_new;

	calclSetModelParameters2D(calclmodel2D,model, kernel);
}

void calclFinalizeToolkit2D(CALCLModel2D * calclmodel2D) {
	int i;

	clReleaseKernel(calclmodel2D->kernelCompact);
	clReleaseKernel(calclmodel2D->kernelComputeCounts);
	clReleaseKernel(calclmodel2D->kernelDownSweep);
	clReleaseKernel(calclmodel2D->kernelInitSubstates);
	clReleaseKernel(calclmodel2D->kernelSteering);
	clReleaseKernel(calclmodel2D->kernelUpSweep);
	clReleaseKernel(calclmodel2D->kernelUpdateSubstate);
	clReleaseKernel(calclmodel2D->kernelStopCondition);

	for (i = 0; i < calclmodel2D->elementaryProcessesNum; ++i)
		clReleaseKernel(calclmodel2D->elementaryProcesses[i]);

	clReleaseMemObject(calclmodel2D->bufferActiveCells);
	clReleaseMemObject(calclmodel2D->bufferActiveCellsFlags);
	clReleaseMemObject(calclmodel2D->bufferActiveCellsNum);
	clReleaseMemObject(calclmodel2D->bufferBoundaryCondition);
	clReleaseMemObject(calclmodel2D->bufferByteSubstateNum);
	clReleaseMemObject(calclmodel2D->bufferColumns);
	clReleaseMemObject(calclmodel2D->bufferCurrentByteSubstate);
	clReleaseMemObject(calclmodel2D->bufferCurrentIntSubstate);
	clReleaseMemObject(calclmodel2D->bufferCurrentRealSubstate);
	clReleaseMemObject(calclmodel2D->bufferIntSubstateNum);
	clReleaseMemObject(calclmodel2D->bufferNeighborhood);
	clReleaseMemObject(calclmodel2D->bufferNeighborhoodID);
	clReleaseMemObject(calclmodel2D->bufferNeighborhoodSize);
	clReleaseMemObject(calclmodel2D->bufferNextByteSubstate);
	clReleaseMemObject(calclmodel2D->bufferNextIntSubstate);
	clReleaseMemObject(calclmodel2D->bufferNextRealSubstate);
	clReleaseMemObject(calclmodel2D->bufferRealSubstateNum);
	clReleaseMemObject(calclmodel2D->bufferRows);
	clReleaseMemObject(calclmodel2D->bufferSTCounts);
	clReleaseMemObject(calclmodel2D->bufferSTOffsets1);
	clReleaseMemObject(calclmodel2D->bufferStop);
	clReleaseMemObject(calclmodel2D->bufferSTCountsDiff);
	clReleaseCommandQueue(calclmodel2D->queue);

	free(calclmodel2D->substateMapper.byteSubstate_current_OUT);
	free(calclmodel2D->substateMapper.intSubstate_current_OUT);
	free(calclmodel2D->substateMapper.realSubstate_current_OUT);

	free(calclmodel2D->elementaryProcesses);
	free(calclmodel2D);


}

CALCLprogram calclLoadProgram2D(CALCLcontext context, CALCLdevice device, char* path_user_kernel, char* path_user_include) {
	char* u = " -cl-denorms-are-zero -cl-finite-math-only ";
	char* pathOpenCALCL= getenv("OPENCALCL_PATH");
	if (pathOpenCALCL == NULL) {
		perror("please configure environment variable OPENCALCL_PATH");
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
	//printf("include %s \n", tmp);
	int num_files;
	char** filesNames;
	char** paths = (char**) malloc(sizeof(char*) * 2);
	char* tmp2 = (char*) malloc(sizeof(char) * (strlen(pathOpenCALCL) + strlen(KERNEL_SOURCE_DIR)));
	strcpy(tmp2,pathOpenCALCL );
	strcat(tmp2,KERNEL_SOURCE_DIR );
	//printf("source %s \n", tmp2);

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

int calclSetKernelArg2D(CALCLkernel* kernel, cl_uint arg_index,size_t arg_size,const void *arg_value){
	return  clSetKernelArg(*kernel,MODEL_ARGS_NUM + arg_index, arg_size,arg_value);
}
