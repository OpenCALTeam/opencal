/*
 * calCL.cpp
 *
 *  Created on: 10/giu/2014
 *      Author: alessio
 */
#include "calcl2D.h"

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

void calclGetSubstateKernel2D(CALCLToolkit2D* toolkit2d, struct CALModel2D * model) {

	CALCLqueue queue = toolkit2d->queue;

	cl_int err;
	size_t zero = 0;

	err = clEnqueueReadBuffer(queue, toolkit2d->bufferCurrentRealSubstate, CL_TRUE, zero, toolkit2d->substateMapper.bufDIMreal, toolkit2d->substateMapper.realSubstate_current_OUT, 0, NULL, NULL);
	calclHandleError(err);
	err = clEnqueueReadBuffer(queue, toolkit2d->bufferCurrentIntSubstate, CL_TRUE, zero, toolkit2d->substateMapper.bufDIMint, toolkit2d->substateMapper.intSubstate_current_OUT, 0, NULL, NULL);
	calclHandleError(err);
	err = clEnqueueReadBuffer(queue, toolkit2d->bufferCurrentByteSubstate, CL_TRUE, zero, toolkit2d->substateMapper.bufDIMbyte, toolkit2d->substateMapper.byteSubstate_current_OUT, 0, NULL, NULL);
	calclHandleError(err);

	calclMapperToSubstates2D(model, &toolkit2d->substateMapper);
}

void calclRoundThreadsNum2D(size_t * threadNum, int numDim, size_t multiple) {
	int i;
	for (i = 0; i < numDim; ++i)
		while (threadNum[i] % multiple != 0)
			threadNum[i]++;
}

void calclResizeThreadsNum2D(CALCLToolkit2D * toolkit2d, struct CALModel2D *model, size_t * threadNum) {
	CALCLqueue queue = toolkit2d->queue;

	cl_int err;
	size_t zero = 0;

	err = clEnqueueReadBuffer(queue, toolkit2d->bufferActiveCellsNum, CL_TRUE, zero, sizeof(int), &model->A.size_current, 0, NULL, NULL);
	calclHandleError(err);
	threadNum[0] = model->A.size_current;
}

CALCLmem calclGetSubstateBuffer2D(CALCLmem bufferSubstates, cl_buffer_region region) {
	cl_int err;
	CALCLmem sub_buffer = clCreateSubBuffer(bufferSubstates, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, CL_BUFFER_CREATE_TYPE_REGION, &region, &err);
	calclHandleError(err);
	return sub_buffer;
}

void copySubstatesBuffers2D(struct CALModel2D * model, CALCLToolkit2D * toolkit2d) {
	CALCLqueue queue = toolkit2d->queue;

	if (model->sizeof_pQr_array > 0)
		clEnqueueCopyBuffer(queue, toolkit2d->bufferNextRealSubstate, toolkit2d->bufferCurrentRealSubstate, 0, 0, toolkit2d->substateMapper.bufDIMreal, 0, NULL, NULL);
	if (model->sizeof_pQi_array > 0)
		clEnqueueCopyBuffer(queue, toolkit2d->bufferNextIntSubstate, toolkit2d->bufferCurrentIntSubstate, 0, 0, toolkit2d->substateMapper.bufDIMint, 0, NULL, NULL);
	if (model->sizeof_pQb_array > 0)
		clEnqueueCopyBuffer(queue, toolkit2d->bufferNextByteSubstate, toolkit2d->bufferCurrentByteSubstate, 0, 0, toolkit2d->substateMapper.bufDIMbyte, 0, NULL, NULL);
}

CALbyte checkStopCondition2D(CALCLToolkit2D * toolkit2d, CALint dimNum, size_t * threadsNum) {
	CALCLqueue queue = toolkit2d->queue;

	calclKernelCall2D(toolkit2d, toolkit2d->kernelStopCondition, dimNum, threadsNum, NULL);
	CALbyte stop = CAL_FALSE;
	size_t zero = 0;

	cl_int err = clEnqueueReadBuffer(queue, toolkit2d->bufferStop, CL_TRUE, zero, sizeof(CALbyte), &stop, 0, NULL, NULL);
	calclHandleError(err);
	return stop;
}

void calclSetKernelStreamCompactionArgs2D(CALCLToolkit2D * toolkit, struct CALModel2D * model) {
	CALint dim = model->rows * model->columns;
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
	clSetKernelArg(toolkit->kernelCompact, 1, sizeof(CALint), &model->columns);
	clSetKernelArg(toolkit->kernelCompact, 2, sizeof(CALCLmem), &toolkit->bufferActiveCellsFlags);
	clSetKernelArg(toolkit->kernelCompact, 3, sizeof(CALCLmem), &toolkit->bufferActiveCellsNum);
	clSetKernelArg(toolkit->kernelCompact, 4, sizeof(CALCLmem), &toolkit->bufferActiveCells);
	clSetKernelArg(toolkit->kernelCompact, 5, sizeof(CALCLmem), &toolkit->bufferSTCounts);
	clSetKernelArg(toolkit->kernelCompact, 6, sizeof(CALCLmem), &toolkit->bufferSTOffsets1);

}

void calclSetKernelsLibArgs2D(CALCLToolkit2D *toolkit, struct CALModel2D * model) {
	clSetKernelArg(toolkit->kernelUpdateSubstate, 0, sizeof(CALint), &model->columns);
	clSetKernelArg(toolkit->kernelUpdateSubstate, 1, sizeof(CALint), &model->rows);
	clSetKernelArg(toolkit->kernelUpdateSubstate, 2, sizeof(CALint), &model->sizeof_pQb_array);
	clSetKernelArg(toolkit->kernelUpdateSubstate, 3, sizeof(CALint), &model->sizeof_pQi_array);
	clSetKernelArg(toolkit->kernelUpdateSubstate, 4, sizeof(CALint), &model->sizeof_pQr_array);
	clSetKernelArg(toolkit->kernelUpdateSubstate, 5, sizeof(CALCLmem), &toolkit->bufferCurrentByteSubstate);
	clSetKernelArg(toolkit->kernelUpdateSubstate, 6, sizeof(CALCLmem), &toolkit->bufferCurrentIntSubstate);
	clSetKernelArg(toolkit->kernelUpdateSubstate, 7, sizeof(CALCLmem), &toolkit->bufferCurrentRealSubstate);
	clSetKernelArg(toolkit->kernelUpdateSubstate, 8, sizeof(CALCLmem), &toolkit->bufferNextByteSubstate);
	clSetKernelArg(toolkit->kernelUpdateSubstate, 9, sizeof(CALCLmem), &toolkit->bufferNextIntSubstate);
	clSetKernelArg(toolkit->kernelUpdateSubstate, 10, sizeof(CALCLmem), &toolkit->bufferNextRealSubstate);
	clSetKernelArg(toolkit->kernelUpdateSubstate, 11, sizeof(CALCLmem), &toolkit->bufferActiveCells);
	clSetKernelArg(toolkit->kernelUpdateSubstate, 12, sizeof(CALCLmem), &toolkit->bufferActiveCellsNum);

}

void calclSetModelParameters2D(CALCLToolkit2D* toolkit2d, struct CALModel2D * model, CALCLkernel * kernel) {

	clSetKernelArg(*kernel, 0, sizeof(CALCLmem), &toolkit2d->bufferRows);
	clSetKernelArg(*kernel, 1, sizeof(CALCLmem), &toolkit2d->bufferColumns);
	clSetKernelArg(*kernel, 2, sizeof(CALCLmem), &toolkit2d->bufferByteSubstateNum);
	clSetKernelArg(*kernel, 3, sizeof(CALCLmem), &toolkit2d->bufferIntSubstateNum);
	clSetKernelArg(*kernel, 4, sizeof(CALCLmem), &toolkit2d->bufferRealSubstateNum);
	clSetKernelArg(*kernel, 5, sizeof(CALCLmem), &toolkit2d->bufferCurrentByteSubstate);
	clSetKernelArg(*kernel, 6, sizeof(CALCLmem), &toolkit2d->bufferCurrentIntSubstate);
	clSetKernelArg(*kernel, 7, sizeof(CALCLmem), &toolkit2d->bufferCurrentRealSubstate);
	clSetKernelArg(*kernel, 8, sizeof(CALCLmem), &toolkit2d->bufferNextByteSubstate);
	clSetKernelArg(*kernel, 9, sizeof(CALCLmem), &toolkit2d->bufferNextIntSubstate);
	clSetKernelArg(*kernel, 10, sizeof(CALCLmem), &toolkit2d->bufferNextRealSubstate);
	clSetKernelArg(*kernel, 11, sizeof(CALCLmem), &toolkit2d->bufferActiveCells);
	clSetKernelArg(*kernel, 12, sizeof(CALCLmem), &toolkit2d->bufferActiveCellsNum);
	clSetKernelArg(*kernel, 13, sizeof(CALCLmem), &toolkit2d->bufferActiveCellsFlags);
	clSetKernelArg(*kernel, 14, sizeof(CALCLmem), &toolkit2d->bufferNeighborhood);
	clSetKernelArg(*kernel, 15, sizeof(CALCLmem), &toolkit2d->bufferNeighborhoodID);
	clSetKernelArg(*kernel, 16, sizeof(CALCLmem), &toolkit2d->bufferNeighborhoodSize);
	clSetKernelArg(*kernel, 17, sizeof(CALCLmem), &toolkit2d->bufferBoundaryCondition);
	clSetKernelArg(*kernel, 18, sizeof(CALCLmem), &toolkit2d->bufferStop);
	clSetKernelArg(*kernel, 19, sizeof(CALCLmem), &toolkit2d->bufferSTCountsDiff);
	double chunk_double = ceil((double)(model->rows * model->columns)/toolkit2d->streamCompactionThreadsNum);
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

CALCLqueue calclCreateQueue2D(CALCLToolkit2D * toolkit, struct CALModel2D * model, CALCLcontext context, CALCLdevice device) {
	CALCLqueue queue = calclCreateCommandQueue(context, device);
	size_t cores;
	cl_int err;
	err = clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &cores, NULL);
	calclHandleError(err);

	//TODO choose stream compaction threads num
	toolkit->streamCompactionThreadsNum = cores * 4;

	while (model->rows * model->columns <= (int)toolkit->streamCompactionThreadsNum)
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
	calclSetKernelStreamCompactionArgs2D(toolkit, model);

	return queue;
}

/******************************************************************************
 * 							PUBLIC FUNCTIONS
 ******************************************************************************/

CALCLToolkit2D * calclCreateToolkit2D(struct CALModel2D *model, CALCLcontext context, CALCLprogram program, CALCLdevice device, enum CALOptimization opt) {

	CALCLToolkit2D * toolkit = (CALCLToolkit2D*) malloc(sizeof(CALCLToolkit2D));
	toolkit->opt = opt;
	toolkit->cl_update_substates = NULL;
	toolkit->kernelInitSubstates = NULL;
	toolkit->kernelSteering = NULL;
	toolkit->kernelStopCondition = NULL;
	toolkit->elementaryProcessesNum = 0;
	toolkit->steps = 0;

	if (model->A.flags == NULL) {
		model->A.flags = (CALbyte*) malloc(sizeof(CALbyte) * model->rows * model->columns);
		memset(model->A.flags, CAL_FALSE, sizeof(CALbyte) * model->rows * model->columns);
	}

	cl_int err;
	int bufferDim = model->columns * model->rows;

	toolkit->kernelUpdateSubstate = calclGetKernelFromProgram(&program, KER_UPDATESUBSTATES);

	//stream compaction kernels
	toolkit->kernelCompact = calclGetKernelFromProgram(&program, KER_STC_COMPACT);
	toolkit->kernelComputeCounts = calclGetKernelFromProgram(&program, KER_STC_COMPUTE_COUNTS);
	toolkit->kernelUpSweep = calclGetKernelFromProgram(&program, KER_STC_UP_SWEEP);
	toolkit->kernelDownSweep = calclGetKernelFromProgram(&program, KER_STC_DOWN_SWEEP);

	struct CALCell2D * activeCells = (struct CALCell2D*) malloc(sizeof(struct CALCell2D) * bufferDim);
	memcpy(activeCells, model->A.cells, sizeof(struct CALCell2D) * model->A.size_current);

	toolkit->bufferActiveCells = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(struct CALCell2D) * bufferDim, activeCells, &err);
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

	size_t byteSubstatesDim = sizeof(CALbyte) * bufferDim * model->sizeof_pQb_array + 1;
	CALbyte * currentByteSubstates = (CALbyte*) malloc(byteSubstatesDim);
	CALbyte * nextByteSubstates = (CALbyte*) malloc(byteSubstatesDim);
	calclByteSubstatesMapper2D(model, currentByteSubstates, nextByteSubstates);
	toolkit->bufferCurrentByteSubstate = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, byteSubstatesDim, currentByteSubstates, &err);
	calclHandleError(err);
	toolkit->bufferNextByteSubstate = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, byteSubstatesDim, nextByteSubstates, &err);
	calclHandleError(err);
	free(currentByteSubstates);
	free(nextByteSubstates);

	size_t intSubstatesDim = sizeof(CALint) * bufferDim * model->sizeof_pQi_array + 1;
	CALint * currentIntSubstates = (CALint*) malloc(intSubstatesDim);
	CALint * nextIntSubstates = (CALint*) malloc(intSubstatesDim);
	calclIntSubstatesMapper2D(model, currentIntSubstates, nextIntSubstates);
	toolkit->bufferCurrentIntSubstate = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, intSubstatesDim, currentIntSubstates, &err);
	calclHandleError(err);
	toolkit->bufferNextIntSubstate = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, intSubstatesDim, nextIntSubstates, &err);
	calclHandleError(err);
	free(currentIntSubstates);
	free(nextIntSubstates);

	size_t realSubstatesDim = sizeof(CALreal) * bufferDim * model->sizeof_pQr_array + 1;
	CALreal * currentRealSubstates = (CALreal*) malloc(realSubstatesDim);
	CALreal * nextRealSubstates = (CALreal*) malloc(realSubstatesDim);
	calclRealSubstatesMapper2D(model, currentRealSubstates, nextRealSubstates);
	toolkit->bufferCurrentRealSubstate = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, realSubstatesDim, currentRealSubstates, &err);
	calclHandleError(err);
	toolkit->bufferNextRealSubstate = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, realSubstatesDim, nextRealSubstates, &err);
	calclHandleError(err);
	free(currentRealSubstates);
	free(nextRealSubstates);

	calclSetKernelsLibArgs2D(toolkit, model);

	//user kernels buffers args

	toolkit->bufferNeighborhood = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(struct CALCell2D) * model->sizeof_X, model->X, &err);
	calclHandleError(err);
	toolkit->bufferNeighborhoodID = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(enum CALNeighborhood2D), &model->X_id, &err);
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

	toolkit->queue = calclCreateQueue2D(toolkit, model, context, device);

	return toolkit;

}

void calclRun2D(CALCLToolkit2D* toolkit2d, struct CALModel2D * model, unsigned maxStep) {
//	cl_int err;
	CALbyte stop;
	size_t * threadNumMax = (size_t*) malloc(sizeof(size_t) * 2);
	threadNumMax[0] = model->rows;
	threadNumMax[1] = model->columns;
	size_t * singleStepThreadNum;
	int dimNum;

	if (toolkit2d->opt == CAL_NO_OPT) {
		singleStepThreadNum = (size_t*) malloc(sizeof(size_t) * 2);
		singleStepThreadNum[0] = threadNumMax[0];
		singleStepThreadNum[1] = threadNumMax[1];
		dimNum = 2;
	} else {
		singleStepThreadNum = (size_t*) malloc(sizeof(size_t));
		singleStepThreadNum[0] = model->A.size_current;
		dimNum = 1;
	}

	if (toolkit2d->kernelInitSubstates != NULL)
		calclKernelCall2D(toolkit2d, toolkit2d->kernelInitSubstates, 1, threadNumMax, NULL);

	toolkit2d->steps = 0;
	while (toolkit2d->steps < (int)maxStep || maxStep == CAL_RUN_LOOP) {
		stop = calclSingleStep2D(toolkit2d, model, singleStepThreadNum, dimNum);
		if (stop == CAL_TRUE)
			break;
	}
	calclGetSubstateKernel2D(toolkit2d, model);
	free(threadNumMax);
	free(singleStepThreadNum);
}

CALbyte calclSingleStep2D(CALCLToolkit2D* toolkit2d, struct CALModel2D * model, size_t * threadsNum, int dimNum) {

	CALbyte activeCells = toolkit2d->opt == CAL_OPT_ACTIVE_CELLS;
	int j;


	if (activeCells == CAL_TRUE) {
		for (j = 0; j < toolkit2d->elementaryProcessesNum; j++) {

			calclKernelCall2D(toolkit2d, toolkit2d->elementaryProcesses[j],  dimNum, threadsNum, NULL);
			calclComputeStreamCompaction2D(toolkit2d);
			calclResizeThreadsNum2D(toolkit2d, model, threadsNum);
			calclKernelCall2D(toolkit2d, toolkit2d->kernelUpdateSubstate, dimNum, threadsNum, NULL);

		}
		if (toolkit2d->kernelSteering != NULL) {
			calclKernelCall2D(toolkit2d, toolkit2d->kernelSteering, dimNum, threadsNum, NULL);
			calclKernelCall2D(toolkit2d, toolkit2d->kernelUpdateSubstate, dimNum, threadsNum, NULL);
		}

	} else {
		for (j = 0; j < toolkit2d->elementaryProcessesNum; j++) {

			calclKernelCall2D(toolkit2d, toolkit2d->elementaryProcesses[j], dimNum, threadsNum, NULL);
			copySubstatesBuffers2D(model, toolkit2d);

		}
		if (toolkit2d->kernelSteering != NULL) {
			calclKernelCall2D(toolkit2d, toolkit2d->kernelSteering, dimNum, threadsNum, NULL);
			copySubstatesBuffers2D(model, toolkit2d);
		}

	}

	if (toolkit2d->cl_update_substates != NULL && toolkit2d->steps % toolkit2d->callbackSteps == 0) {
		calclGetSubstateKernel2D(toolkit2d, model);
		toolkit2d->cl_update_substates(model);
	}

	toolkit2d->steps++;

	if (toolkit2d->kernelStopCondition != NULL) {
		return checkStopCondition2D(toolkit2d, dimNum, threadsNum);
	}

	return CAL_FALSE;

}

/**

 //#include <fstream>
 //ofstream out;
 */
FILE * file;
void calclKernelCall2D(CALCLToolkit2D* toolkit2d, CALCLkernel ker, int numDim, size_t * dimSize, size_t * localDimSize) {

//	cl_event timing_event;
//	cl_ulong time_start, cl_ulong time_end, read_time;
	cl_int err;
	CALCLdevice device;
	size_t multiple;
	CALCLqueue queue = toolkit2d->queue;
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

void calclComputeStreamCompaction2D(CALCLToolkit2D * toolkit) {
	CALCLqueue queue = toolkit->queue;
	calclKernelCall2D(toolkit, toolkit->kernelComputeCounts, 1, &toolkit->streamCompactionThreadsNum, NULL);
	cl_int err;
	int iterations = toolkit->streamCompactionThreadsNum;
	size_t tmpThreads = iterations;
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

	calclKernelCall2D(toolkit, toolkit->kernelCompact, 1, &toolkit->streamCompactionThreadsNum, NULL);
}

void calclSetCALKernelArgs2D(CALCLkernel * kernel, CALCLmem * args, cl_uint numArgs) {
	unsigned int i;
	for (i = 0; i < numArgs; i++)
		clSetKernelArg(*kernel, MODEL_ARGS_NUM + i, sizeof(CALCLmem), &args[i]);
}

void calclSetStopConditionKernel2D(CALCLToolkit2D * toolkit2d,struct CALModel2D * model, CALCLkernel * kernel) {
	toolkit2d->kernelStopCondition = *kernel;
	calclSetModelParameters2D(toolkit2d,model, kernel);
}

void calclSetInitSubstatesKernel2D(CALCLToolkit2D* toolkit2d,struct CALModel2D * model, CALCLkernel * kernel) {
	toolkit2d->kernelInitSubstates = *kernel;
	calclSetModelParameters2D(toolkit2d,model, kernel);
}

void calclSetSteeringKernel2D(CALCLToolkit2D* toolkit2d,struct CALModel2D * model, CALCLkernel * kernel) {
	toolkit2d->kernelSteering = *kernel;
	calclSetModelParameters2D(toolkit2d,model, kernel);
}

void calclSetUpdateSubstatesFunction2D(CALCLToolkit2D* toolkit2d, void (*cl_update_substates)(struct CALModel2D*), int callbackSteps) {
	toolkit2d->cl_update_substates = cl_update_substates;
	toolkit2d->callbackSteps = callbackSteps;
}

void calclAddElementaryProcessKernel2D(CALCLToolkit2D* toolkit2d,struct CALModel2D * model, CALCLkernel * kernel) {

	cl_uint size = toolkit2d->elementaryProcessesNum;

	CALCLkernel * ep = toolkit2d->elementaryProcesses;
	CALCLkernel * ep_new = (CALCLkernel*) malloc(sizeof(CALCLkernel) * (size + 1));

	unsigned int i;
	for (i = 0; i < size; i++)
		ep_new[i] = ep[i];

	ep_new[size] = *kernel;

	if (size > 0)
		free(ep);

	toolkit2d->elementaryProcessesNum++;
	toolkit2d->elementaryProcesses = ep_new;

	calclSetModelParameters2D(toolkit2d,model, kernel);
}

void calclFinalizeToolkit2D(CALCLToolkit2D * toolkit) {
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

CALCLprogram calclLoadProgramLib2D(CALCLcontext context, CALCLdevice device, char* path_user_kernel, char* path_user_include) {
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

