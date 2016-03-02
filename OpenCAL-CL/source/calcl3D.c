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

void calclGetSubstateKernel3D(CALCLToolkit3D* toolkit3d, struct CALModel3D * model) {

	CALCLqueue queue = toolkit3d->queue;

	cl_int err;
	size_t zero = 0;

	err = clEnqueueReadBuffer(queue, toolkit3d->bufferCurrentRealSubstate, CL_TRUE, zero, toolkit3d->substateMapper.bufDIMreal, toolkit3d->substateMapper.realSubstate_current_OUT, 0, NULL, NULL);
	calclHandleError(err);
	err = clEnqueueReadBuffer(queue, toolkit3d->bufferCurrentIntSubstate, CL_TRUE, zero, toolkit3d->substateMapper.bufDIMint, toolkit3d->substateMapper.intSubstate_current_OUT, 0, NULL, NULL);
	calclHandleError(err);
	err = clEnqueueReadBuffer(queue, toolkit3d->bufferCurrentByteSubstate, CL_TRUE, zero, toolkit3d->substateMapper.bufDIMbyte, toolkit3d->substateMapper.byteSubstate_current_OUT, 0, NULL, NULL);
	calclHandleError(err);

	calclMapperToSubstates3D(model, &toolkit3d->substateMapper);
}

void calclRoundThreadsNum3D(size_t * threadNum, int numDim, size_t multiple) {
	int i;
	for (i = 0; i < numDim; ++i)
		while (threadNum[i] % multiple != 0)
			threadNum[i]++;
}

void calclResizeThreadsNum3D(CALCLToolkit3D * toolkit3d, struct CALModel3D *model, size_t * threadNum) {
	cl_int err;
	size_t zero = 0;
	CALCLqueue queue = toolkit3d->queue;


	err = clEnqueueReadBuffer(queue, toolkit3d->bufferActiveCellsNum, CL_TRUE, zero, sizeof(int), &model->A.size_current, 0, NULL, NULL);
	calclHandleError(err);
	threadNum[0] = model->A.size_current;
}

CALCLmem calclGetSubstateBuffer3D(CALCLmem bufferSubstates, cl_buffer_region region) {
	cl_int err;
	CALCLmem sub_buffer = clCreateSubBuffer(bufferSubstates, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, CL_BUFFER_CREATE_TYPE_REGION, &region, &err);
	calclHandleError(err);
	return sub_buffer;
}

void calclCopySubstatesBuffers3D(CALCLToolkit3D * toolkit3d, struct CALModel3D * model) {
	CALCLqueue queue = toolkit3d->queue;

	if (model->sizeof_pQr_array > 0)
		clEnqueueCopyBuffer(queue, toolkit3d->bufferNextRealSubstate, toolkit3d->bufferCurrentRealSubstate, 0, 0, toolkit3d->substateMapper.bufDIMreal, 0, NULL, NULL);
	if (model->sizeof_pQi_array > 0)
		clEnqueueCopyBuffer(queue, toolkit3d->bufferNextIntSubstate, toolkit3d->bufferCurrentIntSubstate, 0, 0, toolkit3d->substateMapper.bufDIMint, 0, NULL, NULL);
	if (model->sizeof_pQb_array > 0)
		clEnqueueCopyBuffer(queue, toolkit3d->bufferNextByteSubstate, toolkit3d->bufferCurrentByteSubstate, 0, 0, toolkit3d->substateMapper.bufDIMbyte, 0, NULL, NULL);
}

CALbyte checkStopCondition3D(CALCLToolkit3D * toolkit3d, CALint dimNum, size_t * threadsNum) {
	CALCLqueue queue = toolkit3d->queue;

	calclKernelCall3D(toolkit3d, toolkit3d->kernelStopCondition, dimNum, threadsNum, NULL);
	CALbyte stop = CAL_FALSE;
	size_t zero = 0;

	cl_int err = clEnqueueReadBuffer(queue, toolkit3d->bufferStop, CL_TRUE, zero, sizeof(CALbyte), &stop, 0, NULL, NULL);
	calclHandleError(err);
	return stop;
}

void calclSetKernelStreamCompactionArgs3D(CALCLToolkit3D * toolkit, struct CALModel3D * model) {
	CALint dim = model->rows * model->columns * model->slices;
	clSetKernelArg(toolkit->kernelComputeCounts, 0, sizeof(CALint), &dim);
	clSetKernelArg(toolkit->kernelComputeCounts, 1, sizeof(CALCLmem), &toolkit->bufferActiveCellsFlags);
	clSetKernelArg(toolkit->kernelComputeCounts, 2, sizeof(CALCLmem), &toolkit->bufferSTCounts);
	clSetKernelArg(toolkit->kernelComputeCounts, 3, sizeof(CALCLmem), &toolkit->bufferSTOffsets1);
	clSetKernelArg(toolkit->kernelComputeCounts, 4, sizeof(CALCLmem), &toolkit->bufferSTCountsDiff);


	int offset = toolkit->streamCompactionThreadsNum / 2;

	clSetKernelArg(toolkit->kernelUpSweep, 0, sizeof(CALCLmem), &toolkit->bufferSTOffsets1);
	clSetKernelArg(toolkit->kernelUpSweep, 1, sizeof(int), &offset);

	clSetKernelArg(toolkit->kernelDownSweep, 0, sizeof(CALCLmem), &toolkit->bufferSTOffsets1);
	clSetKernelArg(toolkit->kernelDownSweep, 1, sizeof(int), &offset);

	clSetKernelArg(toolkit->kernelCompact, 0, sizeof(CALint), &dim);
	clSetKernelArg(toolkit->kernelCompact, 1, sizeof(CALint), &model->rows);
	clSetKernelArg(toolkit->kernelCompact, 2, sizeof(CALint), &model->columns);
	clSetKernelArg(toolkit->kernelCompact, 3, sizeof(CALCLmem), &toolkit->bufferActiveCellsFlags);
	clSetKernelArg(toolkit->kernelCompact, 4, sizeof(CALCLmem), &toolkit->bufferActiveCellsNum);
	clSetKernelArg(toolkit->kernelCompact, 5, sizeof(CALCLmem), &toolkit->bufferActiveCells);
	clSetKernelArg(toolkit->kernelCompact, 6, sizeof(CALCLmem), &toolkit->bufferSTCounts);
	clSetKernelArg(toolkit->kernelCompact, 7, sizeof(CALCLmem), &toolkit->bufferSTOffsets1);

}

void calclSetKernelsLibArgs3D(CALCLToolkit3D *toolkit, struct CALModel3D * model) {
	clSetKernelArg(toolkit->kernelUpdateSubstate, 0, sizeof(CALint), &model->columns);
	clSetKernelArg(toolkit->kernelUpdateSubstate, 1, sizeof(CALint), &model->rows);
	clSetKernelArg(toolkit->kernelUpdateSubstate, 2, sizeof(CALint), &model->slices);
	clSetKernelArg(toolkit->kernelUpdateSubstate, 3, sizeof(CALint), &model->sizeof_pQb_array);
	clSetKernelArg(toolkit->kernelUpdateSubstate, 4, sizeof(CALint), &model->sizeof_pQi_array);
	clSetKernelArg(toolkit->kernelUpdateSubstate, 5, sizeof(CALint), &model->sizeof_pQr_array);
	clSetKernelArg(toolkit->kernelUpdateSubstate, 6, sizeof(CALCLmem), &toolkit->bufferCurrentByteSubstate);
	clSetKernelArg(toolkit->kernelUpdateSubstate, 7, sizeof(CALCLmem), &toolkit->bufferCurrentIntSubstate);
	clSetKernelArg(toolkit->kernelUpdateSubstate, 8, sizeof(CALCLmem), &toolkit->bufferCurrentRealSubstate);
	clSetKernelArg(toolkit->kernelUpdateSubstate, 9, sizeof(CALCLmem), &toolkit->bufferNextByteSubstate);
	clSetKernelArg(toolkit->kernelUpdateSubstate, 10, sizeof(CALCLmem), &toolkit->bufferNextIntSubstate);
	clSetKernelArg(toolkit->kernelUpdateSubstate, 11, sizeof(CALCLmem), &toolkit->bufferNextRealSubstate);
	clSetKernelArg(toolkit->kernelUpdateSubstate, 12, sizeof(CALCLmem), &toolkit->bufferActiveCells);
	clSetKernelArg(toolkit->kernelUpdateSubstate, 13, sizeof(CALCLmem), &toolkit->bufferActiveCellsNum);
}

void calclSetModelParameters3D(CALCLToolkit3D * toolkit3d,struct CALModel3D * model, CALCLkernel * kernel) {

	clSetKernelArg(*kernel, 0, sizeof(CALCLmem), &toolkit3d->bufferRows);
	clSetKernelArg(*kernel, 1, sizeof(CALCLmem), &toolkit3d->bufferColumns);
	clSetKernelArg(*kernel, 2, sizeof(CALCLmem), &toolkit3d->bufferSlices);
	clSetKernelArg(*kernel, 3, sizeof(CALCLmem), &toolkit3d->bufferByteSubstateNum);
	clSetKernelArg(*kernel, 4, sizeof(CALCLmem), &toolkit3d->bufferIntSubstateNum);
	clSetKernelArg(*kernel, 5, sizeof(CALCLmem), &toolkit3d->bufferRealSubstateNum);
	clSetKernelArg(*kernel, 6, sizeof(CALCLmem), &toolkit3d->bufferCurrentByteSubstate);
	clSetKernelArg(*kernel, 7, sizeof(CALCLmem), &toolkit3d->bufferCurrentIntSubstate);
	clSetKernelArg(*kernel, 8, sizeof(CALCLmem), &toolkit3d->bufferCurrentRealSubstate);
	clSetKernelArg(*kernel, 9, sizeof(CALCLmem), &toolkit3d->bufferNextByteSubstate);
	clSetKernelArg(*kernel, 10, sizeof(CALCLmem), &toolkit3d->bufferNextIntSubstate);
	clSetKernelArg(*kernel, 11, sizeof(CALCLmem), &toolkit3d->bufferNextRealSubstate);
	clSetKernelArg(*kernel, 12, sizeof(CALCLmem), &toolkit3d->bufferActiveCells);
	clSetKernelArg(*kernel, 13, sizeof(CALCLmem), &toolkit3d->bufferActiveCellsNum);
	clSetKernelArg(*kernel, 14, sizeof(CALCLmem), &toolkit3d->bufferActiveCellsFlags);
	clSetKernelArg(*kernel, 15, sizeof(CALCLmem), &toolkit3d->bufferNeighborhood);
	clSetKernelArg(*kernel, 16, sizeof(CALCLmem), &toolkit3d->bufferNeighborhoodID);
	clSetKernelArg(*kernel, 17, sizeof(CALCLmem), &toolkit3d->bufferNeighborhoodSize);
	clSetKernelArg(*kernel, 18, sizeof(CALCLmem), &toolkit3d->bufferBoundaryCondition);
	clSetKernelArg(*kernel, 19, sizeof(CALCLmem), &toolkit3d->bufferStop);
	clSetKernelArg(*kernel, 20, sizeof(CALCLmem), &toolkit3d->bufferSTCountsDiff);
	double chunk_double = ceil((double)(model->rows * model->columns*model->slices)/toolkit3d->streamCompactionThreadsNum);
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
CALCLqueue calclCreateQueue3D(CALCLToolkit3D * toolkit, struct CALModel3D * model, CALCLcontext context, CALCLdevice device) {
	CALCLqueue queue = calclCreateCommandQueue(context, device);
	size_t cores;
	cl_int err;
	err = clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &cores, NULL);
	calclHandleError(err);

	//TODO choose stream compaction threads num
	toolkit->streamCompactionThreadsNum = cores * 4;

	while (model->rows * model->columns * model->slices <= (int)toolkit->streamCompactionThreadsNum)
		toolkit->streamCompactionThreadsNum /= 2;

	toolkit->bufferSTCounts = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, sizeof(CALint) * toolkit->streamCompactionThreadsNum, NULL, &err);
	calclHandleError(err);
	toolkit->bufferSTOffsets1 = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, sizeof(CALint) * toolkit->streamCompactionThreadsNum, NULL, &err);
	calclHandleError(err);
	CALbyte * diff = (CALbyte*) malloc(sizeof(CALbyte) * toolkit->streamCompactionThreadsNum);
	memset(diff, CAL_TRUE, sizeof(CALbyte) * toolkit->streamCompactionThreadsNum);
	toolkit->bufferSTCountsDiff = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, toolkit->streamCompactionThreadsNum * sizeof(CALbyte), diff, &err);
	calclHandleError(err);
	free(diff);
	calclSetKernelStreamCompactionArgs3D(toolkit, model);

	return queue;
}

/******************************************************************************
 * 							PUBLIC FUNCTIONS
 ******************************************************************************/

CALCLToolkit3D * calclCreateToolkit3D(struct CALModel3D *model, CALCLcontext context, CALCLprogram program,CALCLdevice device) {

	CALCLToolkit3D * toolkit = (CALCLToolkit3D*) malloc(sizeof(CALCLToolkit3D));
	//initialize toolkit stuff
	toolkit->opt = model->OPTIMIZATION;
	toolkit->cl_update_substates = NULL;
	toolkit->kernelInitSubstates = NULL;
	toolkit->kernelSteering = NULL;
	toolkit->kernelStopCondition = NULL;
	toolkit->elementaryProcessesNum = 0;
	toolkit->steps = 0;


	if (model->A.flags == NULL) {
		model->A.flags = (CALbyte*) malloc(sizeof(CALbyte) * model->rows * model->columns * model->slices);
		memset(model->A.flags, CAL_FALSE, sizeof(CALbyte) * model->rows * model->columns * model->slices);
	}

	cl_int err;
	int bufferDim = model->columns * model->rows * model->slices;

	toolkit->kernelUpdateSubstate = calclGetKernelFromProgram(&program, KER_UPDATESUBSTATES);

	//stream compaction kernels
	toolkit->kernelCompact = calclGetKernelFromProgram(&program, KER_STC_COMPACT);
	toolkit->kernelComputeCounts = calclGetKernelFromProgram(&program, KER_STC_COMPUTE_COUNTS);
	toolkit->kernelUpSweep = calclGetKernelFromProgram(&program, KER_STC_UP_SWEEP);
	toolkit->kernelDownSweep = calclGetKernelFromProgram(&program, KER_STC_DOWN_SWEEP);

	struct CALCell3D * activeCells = (struct CALCell3D*) malloc(sizeof(struct CALCell3D) * bufferDim);
	memcpy(activeCells, model->A.cells, sizeof(struct CALCell3D) * model->A.size_current);

	toolkit->bufferActiveCells = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(struct CALCell3D) * bufferDim, activeCells, &err);
	calclHandleError(err);
	free(activeCells);
	toolkit->bufferActiveCellsFlags = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(CALbyte) * bufferDim, model->A.flags, &err);
	calclHandleError(err);

	toolkit->bufferActiveCellsNum = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(CALint), &model->A.size_current, &err);
	calclHandleError(err);

	toolkit->bufferByteSubstateNum = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(CALint), &model->sizeof_pQb_array, &err);
	calclHandleError(err);
	toolkit->bufferIntSubstateNum = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(CALint), &model->sizeof_pQi_array, &err);
	calclHandleError(err);
	toolkit->bufferRealSubstateNum = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(CALint), &model->sizeof_pQr_array, &err);
	calclHandleError(err);

	toolkit->bufferColumns = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(CALint), &model->columns, &err);
	calclHandleError(err);
	toolkit->bufferRows = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(CALint), &model->rows, &err);
	calclHandleError(err);
	toolkit->bufferSlices = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(CALint), &model->slices, &err);
	calclHandleError(err);

	size_t byteSubstatesDim = sizeof(CALbyte) * bufferDim * model->sizeof_pQb_array + 1;
	CALbyte * currentByteSubstates = (CALbyte*) malloc(byteSubstatesDim);
	CALbyte * nextByteSubstates = (CALbyte*) malloc(byteSubstatesDim);
	calclByteSubstatesMapper3D(model, currentByteSubstates, nextByteSubstates);
	toolkit->bufferCurrentByteSubstate = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, byteSubstatesDim, currentByteSubstates, &err);
	calclHandleError(err);
	toolkit->bufferNextByteSubstate = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, byteSubstatesDim, nextByteSubstates, &err);
	calclHandleError(err);
	free(currentByteSubstates);
	free(nextByteSubstates);

	size_t intSubstatesDim = sizeof(CALint) * bufferDim * model->sizeof_pQi_array + 1;
	CALint * currentIntSubstates = (CALint*) malloc(intSubstatesDim);
	CALint * nextIntSubstates = (CALint*) malloc(intSubstatesDim);
	calclIntSubstatesMapper3D(model, currentIntSubstates, nextIntSubstates);
	toolkit->bufferCurrentIntSubstate = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, intSubstatesDim, currentIntSubstates, &err);
	calclHandleError(err);
	toolkit->bufferNextIntSubstate = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, intSubstatesDim, nextIntSubstates, &err);
	calclHandleError(err);
	free(currentIntSubstates);
	free(nextIntSubstates);

	size_t realSubstatesDim = sizeof(CALreal) * bufferDim * model->sizeof_pQr_array + 1;
	CALreal * currentRealSubstates = (CALreal*) malloc(realSubstatesDim);
	CALreal * nextRealSubstates = (CALreal*) malloc(realSubstatesDim);
	calclRealSubstatesMapper3D(model, currentRealSubstates, nextRealSubstates);
	toolkit->bufferCurrentRealSubstate = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, realSubstatesDim, currentRealSubstates, &err);
	calclHandleError(err);
	toolkit->bufferNextRealSubstate = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, realSubstatesDim, nextRealSubstates, &err);
	calclHandleError(err);
	free(currentRealSubstates);
	free(nextRealSubstates);

	calclSetKernelsLibArgs3D(toolkit, model);

	//user kernels buffers args

	toolkit->bufferNeighborhood = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(struct CALCell3D) * model->sizeof_X, model->X, &err);
	calclHandleError(err);
	toolkit->bufferNeighborhoodID = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(enum CALNeighborhood3D), &model->X_id, &err);
	calclHandleError(err);
	toolkit->bufferNeighborhoodSize = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(CALint), &model->sizeof_X, &err);
	calclHandleError(err);
	toolkit->bufferBoundaryCondition = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(enum CALSpaceBoundaryCondition), &model->T, &err);
	calclHandleError(err);

	//stop condition buffer
	CALbyte stop = CAL_FALSE;
	toolkit->bufferStop = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(CALbyte), &stop, &err);
	calclHandleError(err);

	//init substates mapper
	toolkit->substateMapper.bufDIMbyte = byteSubstatesDim;
	toolkit->substateMapper.bufDIMreal = realSubstatesDim;
	toolkit->substateMapper.bufDIMint = intSubstatesDim;
	toolkit->substateMapper.byteSubstate_current_OUT = (CALbyte*) malloc(byteSubstatesDim);
	toolkit->substateMapper.realSubstate_current_OUT = (CALreal*) malloc(realSubstatesDim);
	toolkit->substateMapper.intSubstate_current_OUT = (CALint*) malloc(intSubstatesDim);

	toolkit->queue = calclCreateQueue3D(toolkit, model, context, device);

	return toolkit;

}

void calclRun3D(CALCLToolkit3D* toolkit3d, struct CALModel3D * model, unsigned int initialStep, unsigned maxStep) {

//	cl_int err;
	CALbyte stop;
	size_t * threadNumMax = (size_t*) malloc(sizeof(size_t) * 3);
	threadNumMax[0] = model->rows;
	threadNumMax[1] = model->columns;
	threadNumMax[2] = model->slices;
	size_t * singleStepThreadNum;
	int dimNum;

	if (toolkit3d->opt == CAL_NO_OPT) {
		singleStepThreadNum = (size_t*) malloc(sizeof(size_t) * 3);
		singleStepThreadNum[0] = threadNumMax[0];
		singleStepThreadNum[1] = threadNumMax[1];
		singleStepThreadNum[2] = threadNumMax[2];
		dimNum = 3;
	} else {
		singleStepThreadNum = (size_t*) malloc(sizeof(size_t));
		singleStepThreadNum[0] = model->A.size_current;
		dimNum = 1;
	}
//	calclRoundThreadsNum(singleStepThreadNum, dimNum);

	if (toolkit3d->kernelInitSubstates != NULL)
		calclKernelCall3D(toolkit3d, toolkit3d->kernelInitSubstates, dimNum, threadNumMax, NULL);

	toolkit3d->steps = initialStep;
	while (toolkit3d->steps <= (int)maxStep || maxStep == CAL_RUN_LOOP) {
		stop = calclSingleStep3D(toolkit3d, model, singleStepThreadNum, dimNum);
		if (stop)
			break;
	}

	calclGetSubstateKernel3D(toolkit3d, model);
	free(threadNumMax);
	free(singleStepThreadNum);
}

CALbyte calclSingleStep3D(CALCLToolkit3D* toolkit3d, struct CALModel3D * model, size_t * threadsNum, int dimNum) {

	CALbyte activeCells = toolkit3d->opt == CAL_OPT_ACTIVE_CELLS;
	int j;

	if (activeCells) {
		for (j = 0; j < toolkit3d->elementaryProcessesNum; j++) {

			calclKernelCall3D(toolkit3d, toolkit3d->elementaryProcesses[j] , dimNum, threadsNum, NULL);
			calclComputeStreamCompaction3D(toolkit3d);
			calclResizeThreadsNum3D(toolkit3d, model, threadsNum);
			calclKernelCall3D(toolkit3d, toolkit3d->kernelUpdateSubstate, dimNum, threadsNum, NULL);
		}
		if (toolkit3d->kernelSteering != NULL) {
			calclKernelCall3D(toolkit3d, toolkit3d->kernelSteering, dimNum, threadsNum, NULL);
			calclKernelCall3D(toolkit3d, toolkit3d->kernelUpdateSubstate, dimNum, threadsNum, NULL);
		}

	} else {
		for (j = 0; j < toolkit3d->elementaryProcessesNum; j++) {
			calclKernelCall3D(toolkit3d, toolkit3d->elementaryProcesses[j], dimNum, threadsNum, NULL);
			calclCopySubstatesBuffers3D(toolkit3d, model);

		}
		if (toolkit3d->kernelSteering != NULL) {
			calclKernelCall3D(toolkit3d, toolkit3d->kernelSteering, dimNum, threadsNum, NULL);
			calclCopySubstatesBuffers3D(toolkit3d, model);
		}

	}

	if (toolkit3d->cl_update_substates != NULL && toolkit3d->steps % toolkit3d->callbackSteps == 0) {
		calclGetSubstateKernel3D(toolkit3d, model);
		toolkit3d->cl_update_substates(model);
	}

	toolkit3d->steps++;

	if (toolkit3d->kernelStopCondition != NULL) {
		return checkStopCondition3D(toolkit3d, dimNum, threadsNum);
	}

	return CAL_FALSE;

}

void calclKernelCall3D(CALCLToolkit3D * toolkit3d, CALCLkernel ker, int numDim, size_t * dimSize, size_t * localDimSize) {
	CALCLqueue queue = toolkit3d->queue;

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

void calclComputeStreamCompaction3D(CALCLToolkit3D * toolkit) {
	CALCLqueue queue = toolkit->queue;

	calclKernelCall3D(toolkit, toolkit->kernelComputeCounts, 1, &toolkit->streamCompactionThreadsNum, NULL);
	int iterations = toolkit->streamCompactionThreadsNum;
	size_t tmpThreads = iterations;
	cl_int err;

	int i;

	for (i = iterations / 2; i > 0; i /= 2) {
		tmpThreads = i;
		err = clEnqueueNDRangeKernel(queue, toolkit->kernelUpSweep, 1, NULL, &tmpThreads, NULL, 0, NULL, NULL);
		calclHandleError(err);
	}

	iterations = toolkit->streamCompactionThreadsNum;

	for (i = 1; i < iterations; i *= 2) {
		tmpThreads = i;
		err = clEnqueueNDRangeKernel(queue, toolkit->kernelDownSweep, 1, NULL, &tmpThreads, NULL, 0, NULL, NULL);
		calclHandleError(err);
	}

	calclKernelCall3D(toolkit, toolkit->kernelCompact, 1, &toolkit->streamCompactionThreadsNum, NULL);
}



void calclSetKernelArgs3D(CALCLkernel * kernel, CALCLmem * args, cl_uint numArgs) {
	unsigned int i;
	for (i = 0; i < numArgs; i++)
		clSetKernelArg(*kernel, MODEL_ARGS_NUM + i, sizeof(CALCLmem), &args[i]);
}

void calclSetStopConditionKernel3D(CALCLToolkit3D * toolkit3d,struct CALModel3D * model, CALCLkernel * kernel) {
	toolkit3d->kernelStopCondition = *kernel;
	calclSetModelParameters3D(toolkit3d,model, kernel);
}

void calclSetInitSubstatesKernel3D(CALCLToolkit3D * toolkit3d,struct CALModel3D * model, CALCLkernel * kernel) {
	toolkit3d->kernelInitSubstates = *kernel;
	calclSetModelParameters3D(toolkit3d,model, kernel);
}

void calclSetSteeringKernel3D(CALCLToolkit3D * toolkit3d,struct CALModel3D * model, CALCLkernel * kernel) {
	toolkit3d->kernelSteering = *kernel;
	calclSetModelParameters3D(toolkit3d, model,kernel);
}

void calclSetUpdateSubstatesFunction3D(CALCLToolkit3D* toolkit3d, void (*cl_update_substates)(struct CALModel3D*), int callbackSteps) {
	toolkit3d->cl_update_substates = cl_update_substates;
	toolkit3d->callbackSteps = callbackSteps;
}

void calclAddElementaryProcessKernel3D(CALCLToolkit3D* toolkit3d,struct CALModel3D * model, CALCLkernel * kernel) {

	cl_uint size = toolkit3d->elementaryProcessesNum;

	CALCLkernel * ep = toolkit3d->elementaryProcesses;
	CALCLkernel * ep_new = (CALCLkernel*) malloc(sizeof(CALCLkernel) * (size + 1));

	unsigned int i;

	for (i = 0; i < size; i++)
		ep_new[i] = ep[i];

	ep_new[size] = *kernel;

	if (size > 0)
		free(ep);

	toolkit3d->elementaryProcessesNum++;
	toolkit3d->elementaryProcesses = ep_new;

	calclSetModelParameters3D(toolkit3d,model, kernel);
}



void calclFinalizeToolkit3D(CALCLToolkit3D * toolkit) {
	int i;

	clReleaseKernel(toolkit->kernelCompact);
	clReleaseKernel(toolkit->kernelComputeCounts);
	clReleaseKernel(toolkit->kernelDownSweep);
	clReleaseKernel(toolkit->kernelInitSubstates);
	clReleaseKernel(toolkit->kernelSteering);
	clReleaseKernel(toolkit->kernelUpSweep);
	clReleaseKernel(toolkit->kernelUpdateSubstate);
	clReleaseKernel(toolkit->kernelStopCondition);

	for (i = 0; i < toolkit->elementaryProcessesNum; ++i)
		clReleaseKernel(toolkit->elementaryProcesses[i]);

	clReleaseMemObject(toolkit->bufferActiveCells);
	clReleaseMemObject(toolkit->bufferActiveCellsFlags);
	clReleaseMemObject(toolkit->bufferActiveCellsNum);
	clReleaseMemObject(toolkit->bufferBoundaryCondition);
	clReleaseMemObject(toolkit->bufferByteSubstateNum);
	clReleaseMemObject(toolkit->bufferColumns);
	clReleaseMemObject(toolkit->bufferSlices);
	clReleaseMemObject(toolkit->bufferCurrentByteSubstate);
	clReleaseMemObject(toolkit->bufferCurrentIntSubstate);
	clReleaseMemObject(toolkit->bufferCurrentRealSubstate);
	clReleaseMemObject(toolkit->bufferIntSubstateNum);
	clReleaseMemObject(toolkit->bufferNeighborhood);
	clReleaseMemObject(toolkit->bufferNeighborhoodID);
	clReleaseMemObject(toolkit->bufferNeighborhoodSize);
	clReleaseMemObject(toolkit->bufferNextByteSubstate);
	clReleaseMemObject(toolkit->bufferNextIntSubstate);
	clReleaseMemObject(toolkit->bufferNextRealSubstate);
	clReleaseMemObject(toolkit->bufferRealSubstateNum);
	clReleaseMemObject(toolkit->bufferRows);
	clReleaseMemObject(toolkit->bufferSTCounts);
	clReleaseMemObject(toolkit->bufferSTOffsets1);
	clReleaseMemObject(toolkit->bufferStop);
	clReleaseMemObject(toolkit->bufferSTCountsDiff);
	clReleaseCommandQueue(toolkit->queue);

	free(toolkit->substateMapper.byteSubstate_current_OUT);
	free(toolkit->substateMapper.intSubstate_current_OUT);
	free(toolkit->substateMapper.realSubstate_current_OUT);

	free(toolkit->elementaryProcesses);
	free(toolkit);

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
		tmp = (char*) malloc(sizeof(char) * (strlen(KERNEL_INCLUDE_DIR) + strlen(" -I ") + strlen(u) + 1));
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
