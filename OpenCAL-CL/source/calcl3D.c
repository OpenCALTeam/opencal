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
#include <OpenCAL-CL/calcl3D.h>

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


}

void calclRealSubstatesMapper3D(struct CALModel3D *model, CALreal * current, CALreal * next) {
	int ssNum = model->sizeof_pQr_array;
	size_t elNum = model->columns * model->rows * model->slices;
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
void calclByteSubstatesMapper3D(struct CALModel3D *model, CALbyte * current, CALbyte * next) {
	int ssNum = model->sizeof_pQb_array;
	size_t elNum = model->columns * model->rows * model->slices;
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
void calclIntSubstatesMapper3D(struct CALModel3D *model, CALint * current, CALint * next) {
	int ssNum = model->sizeof_pQi_array;
	size_t elNum = model->columns * model->rows * model->slices;
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
	calclSetKernelStreamCompactionArgs3D(calclmodel3D);

	return queue;
}

/******************************************************************************
 * 							PUBLIC FUNCTIONS
 ******************************************************************************/

struct CALCLModel3D * calclCADef3D(struct CALModel3D *host_CA, CALCLcontext context, CALCLprogram program,CALCLdevice device) {

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


	if (host_CA->A->flags == NULL) {
		host_CA->A->flags = (CALbyte*) malloc(sizeof(CALbyte) * host_CA->rows * host_CA->columns * host_CA->slices);
		memset(host_CA->A->flags, CAL_FALSE, sizeof(CALbyte) * host_CA->rows * host_CA->columns * host_CA->slices);
	}

	cl_int err;
	int bufferDim = host_CA->columns * host_CA->rows * host_CA->slices;

	calclmodel3D->kernelUpdateSubstate = calclGetKernelFromProgram(&program, KER_UPDATESUBSTATES);

	//stream compaction kernels
	calclmodel3D->kernelCompact = calclGetKernelFromProgram(&program, KER_STC_COMPACT);
	calclmodel3D->kernelComputeCounts = calclGetKernelFromProgram(&program, KER_STC_COMPUTE_COUNTS);
	calclmodel3D->kernelUpSweep = calclGetKernelFromProgram(&program, KER_STC_UP_SWEEP);
	calclmodel3D->kernelDownSweep = calclGetKernelFromProgram(&program, KER_STC_DOWN_SWEEP);

	struct CALCell3D * activeCells = (struct CALCell3D*) malloc(sizeof(struct CALCell3D) * bufferDim);
	memcpy(activeCells, host_CA->A->cells, sizeof(struct CALCell3D) * host_CA->A->size_current);

	calclmodel3D->bufferActiveCells = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(struct CALCell3D) * bufferDim, activeCells, &err);
	calclHandleError(err);
	free(activeCells);
	calclmodel3D->bufferActiveCellsFlags = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(CALbyte) * bufferDim, host_CA->A->flags, &err);
	calclHandleError(err);

	calclmodel3D->bufferActiveCellsNum = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(CALint), &host_CA->A->size_current, &err);
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
	calclmodel3D->bufferSlices = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(CALint), &host_CA->slices, &err);
	calclHandleError(err);

	size_t byteSubstatesDim = sizeof(CALbyte) * bufferDim * host_CA->sizeof_pQb_array + 1;
	CALbyte * currentByteSubstates = (CALbyte*) malloc(byteSubstatesDim);
	CALbyte * nextByteSubstates = (CALbyte*) malloc(byteSubstatesDim);
	calclByteSubstatesMapper3D(host_CA, currentByteSubstates, nextByteSubstates);
	calclmodel3D->bufferCurrentByteSubstate = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, byteSubstatesDim, currentByteSubstates, &err);
	calclHandleError(err);
	calclmodel3D->bufferNextByteSubstate = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, byteSubstatesDim, nextByteSubstates, &err);
	calclHandleError(err);
	free(currentByteSubstates);
	free(nextByteSubstates);

	size_t intSubstatesDim = sizeof(CALint) * bufferDim * host_CA->sizeof_pQi_array + 1;
	CALint * currentIntSubstates = (CALint*) malloc(intSubstatesDim);
	CALint * nextIntSubstates = (CALint*) malloc(intSubstatesDim);
	calclIntSubstatesMapper3D(host_CA, currentIntSubstates, nextIntSubstates);
	calclmodel3D->bufferCurrentIntSubstate = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, intSubstatesDim, currentIntSubstates, &err);
	calclHandleError(err);
	calclmodel3D->bufferNextIntSubstate = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, intSubstatesDim, nextIntSubstates, &err);
	calclHandleError(err);
	free(currentIntSubstates);
	free(nextIntSubstates);

	size_t realSubstatesDim = sizeof(CALreal) * bufferDim * host_CA->sizeof_pQr_array + 1;
	CALreal * currentRealSubstates = (CALreal*) malloc(realSubstatesDim);
	CALreal * nextRealSubstates = (CALreal*) malloc(realSubstatesDim);
	calclRealSubstatesMapper3D(host_CA, currentRealSubstates, nextRealSubstates);
	calclmodel3D->bufferCurrentRealSubstate = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, realSubstatesDim, currentRealSubstates, &err);
	calclHandleError(err);
	calclmodel3D->bufferNextRealSubstate = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, realSubstatesDim, nextRealSubstates, &err);
	calclHandleError(err);
	free(currentRealSubstates);
	free(nextRealSubstates);

	calclSetKernelsLibArgs3D(calclmodel3D);

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

	calclmodel3D->queue = calclCreateQueue3D(calclmodel3D, context, device);

	return calclmodel3D;

}

void calclRun3D(struct CALCLModel3D* calclmodel3D, unsigned int initialStep, unsigned maxStep) {

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

CALbyte calclSingleStep3D(struct CALCLModel3D* calclmodel3D, size_t * threadsNum, int dimNum) {

	CALbyte activeCells = calclmodel3D->opt == CAL_OPT_ACTIVE_CELLS;
	int j;

	if (activeCells) {
		for (j = 0; j < calclmodel3D->elementaryProcessesNum; j++) {

			calclKernelCall3D(calclmodel3D, calclmodel3D->elementaryProcesses[j] , dimNum, threadsNum, NULL);
			calclComputeStreamCompaction3D(calclmodel3D);
			calclResizeThreadsNum3D(calclmodel3D, threadsNum);
			calclKernelCall3D(calclmodel3D, calclmodel3D->kernelUpdateSubstate, dimNum, threadsNum, NULL);
		}
		if (calclmodel3D->kernelSteering != NULL) {
			calclKernelCall3D(calclmodel3D, calclmodel3D->kernelSteering, dimNum, threadsNum, NULL);
			calclKernelCall3D(calclmodel3D, calclmodel3D->kernelUpdateSubstate, dimNum, threadsNum, NULL);
		}

	} else {
		for (j = 0; j < calclmodel3D->elementaryProcessesNum; j++) {
			calclKernelCall3D(calclmodel3D, calclmodel3D->elementaryProcesses[j], dimNum, threadsNum, NULL);
			calclCopySubstatesBuffers3D(calclmodel3D);

		}
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
	clReleaseCommandQueue(calclmodel3D->queue);

	free(calclmodel3D->substateMapper.byteSubstate_current_OUT);
	free(calclmodel3D->substateMapper.intSubstate_current_OUT);
	free(calclmodel3D->substateMapper.realSubstate_current_OUT);

	free(calclmodel3D->elementaryProcesses);
	free(calclmodel3D);

}

CALCLprogram calclLoadProgram3D(CALCLcontext context, CALCLdevice device, char* path_user_kernel, char* path_user_include) {
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

	int num_files;
	char** filesNames;
	char** paths = (char**) malloc(sizeof(char*) * 2);
	char* tmp2 = (char*) malloc(sizeof(char) * (strlen(pathOpenCALCL) + strlen(KERNEL_SOURCE_DIR)));
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
