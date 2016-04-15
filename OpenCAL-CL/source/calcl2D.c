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
void calclMapperToSubstates2D(struct CALModel2D * host_CA, CALCLSubstateMapper * mapper) {

	int ssNum_r = host_CA->sizeof_pQr_array;
	int ssNum_i = host_CA->sizeof_pQi_array;
	int ssNum_b = host_CA->sizeof_pQb_array;
	size_t elNum = host_CA->columns * host_CA->rows;

	long int outIndex = 0;

	int i;
	unsigned int j;

	for (i = 0; i < ssNum_r; i++) {
		for (j = 0; j < elNum; j++)
			host_CA->pQr_array[i]->current[j] = mapper->realSubstate_current_OUT[outIndex++];
	}

	outIndex = 0;

	for (i = 0; i < ssNum_i; i++) {
		for (j = 0; j < elNum; j++)
			host_CA->pQi_array[i]->current[j] = mapper->intSubstate_current_OUT[outIndex++];
	}

	outIndex = 0;

	for (i = 0; i < ssNum_b; i++) {
		for (j = 0; j < elNum; j++)
			host_CA->pQb_array[i]->current[j] = mapper->byteSubstate_current_OUT[outIndex++];
	}

}

void calclGetSubstatesDeviceToHost2D(struct CALCLModel2D* calclmodel2D) {

	CALCLqueue queue = calclmodel2D->queue;

	cl_int err;
	size_t zero = 0;
	err = clEnqueueReadBuffer(queue, calclmodel2D->bufferCurrentRealSubstate, CL_TRUE, zero, calclmodel2D->substateMapper.bufDIMreal, calclmodel2D->substateMapper.realSubstate_current_OUT, 0, NULL,
	NULL);
	calclHandleError(err);
	err = clEnqueueReadBuffer(queue, calclmodel2D->bufferCurrentIntSubstate, CL_TRUE, zero, calclmodel2D->substateMapper.bufDIMint, calclmodel2D->substateMapper.intSubstate_current_OUT, 0, NULL,
	NULL);
	calclHandleError(err);
	err = clEnqueueReadBuffer(queue, calclmodel2D->bufferCurrentByteSubstate, CL_TRUE, zero, calclmodel2D->substateMapper.bufDIMbyte, calclmodel2D->substateMapper.byteSubstate_current_OUT, 0, NULL,
	NULL);
	calclHandleError(err);

	calclMapperToSubstates2D(calclmodel2D->host_CA, &calclmodel2D->substateMapper);
}

void calclRoundThreadsNum2D(size_t * threadNum, int numDim, size_t multiple) {
	int i;
	for (i = 0; i < numDim; ++i)
		while (threadNum[i] % multiple != 0)
			threadNum[i]++;
}

void calclResizeThreadsNum2D(struct CALCLModel2D * calclmodel2D, size_t * threadNum) {
	CALCLqueue queue = calclmodel2D->queue;

	cl_int err;
	size_t zero = 0;

	err = clEnqueueReadBuffer(queue, calclmodel2D->bufferActiveCellsNum, CL_TRUE, zero, sizeof(int), &calclmodel2D->host_CA->A.size_current, 0, NULL, NULL);
	calclHandleError(err);
	threadNum[0] = calclmodel2D->host_CA->A.size_current;
}

CALCLmem calclGetSubstateBuffer2D(CALCLmem bufferSubstates, cl_buffer_region region) {
	cl_int err;
	CALCLmem sub_buffer = clCreateSubBuffer(bufferSubstates, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, CL_BUFFER_CREATE_TYPE_REGION, &region, &err);
	calclHandleError(err);
	return sub_buffer;
}

void copySubstatesBuffers2D(struct CALCLModel2D * calclmodel2D) {
	CALCLqueue queue = calclmodel2D->queue;

	if (calclmodel2D->host_CA->sizeof_pQr_array > 0)
		clEnqueueCopyBuffer(queue, calclmodel2D->bufferNextRealSubstate, calclmodel2D->bufferCurrentRealSubstate, 0, 0, calclmodel2D->substateMapper.bufDIMreal, 0, NULL, NULL);
	if (calclmodel2D->host_CA->sizeof_pQi_array > 0)
		clEnqueueCopyBuffer(queue, calclmodel2D->bufferNextIntSubstate, calclmodel2D->bufferCurrentIntSubstate, 0, 0, calclmodel2D->substateMapper.bufDIMint, 0, NULL, NULL);
	if (calclmodel2D->host_CA->sizeof_pQb_array > 0)
		clEnqueueCopyBuffer(queue, calclmodel2D->bufferNextByteSubstate, calclmodel2D->bufferCurrentByteSubstate, 0, 0, calclmodel2D->substateMapper.bufDIMbyte, 0, NULL, NULL);
}

CALbyte checkStopCondition2D(struct CALCLModel2D * calclmodel2D, CALint dimNum, size_t * threadsNum) {
	CALCLqueue queue = calclmodel2D->queue;

	calclKernelCall2D(calclmodel2D, calclmodel2D->kernelStopCondition, dimNum, threadsNum, NULL);
	CALbyte stop = CAL_FALSE;
	size_t zero = 0;

	cl_int err = clEnqueueReadBuffer(queue, calclmodel2D->bufferStop, CL_TRUE, zero, sizeof(CALbyte), &stop, 0, NULL, NULL);
	calclHandleError(err);
	return stop;
}

void calclSetKernelStreamCompactionArgs2D(struct CALCLModel2D * calclmodel2D) {
	CALint dim = calclmodel2D->host_CA->rows * calclmodel2D->host_CA->columns;
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
	clSetKernelArg(calclmodel2D->kernelCompact, 1, sizeof(CALint), &calclmodel2D->host_CA->columns);
	clSetKernelArg(calclmodel2D->kernelCompact, 2, sizeof(CALCLmem), &calclmodel2D->bufferActiveCellsFlags);
	clSetKernelArg(calclmodel2D->kernelCompact, 3, sizeof(CALCLmem), &calclmodel2D->bufferActiveCellsNum);
	clSetKernelArg(calclmodel2D->kernelCompact, 4, sizeof(CALCLmem), &calclmodel2D->bufferActiveCells);
	clSetKernelArg(calclmodel2D->kernelCompact, 5, sizeof(CALCLmem), &calclmodel2D->bufferSTCounts);
	clSetKernelArg(calclmodel2D->kernelCompact, 6, sizeof(CALCLmem), &calclmodel2D->bufferSTOffsets1);

}

void calclSetKernelsLibArgs2D(struct CALCLModel2D *calclmodel2D) {
	clSetKernelArg(calclmodel2D->kernelUpdateSubstate, 0, sizeof(CALint), &calclmodel2D->host_CA->columns);
	clSetKernelArg(calclmodel2D->kernelUpdateSubstate, 1, sizeof(CALint), &calclmodel2D->host_CA->rows);
	clSetKernelArg(calclmodel2D->kernelUpdateSubstate, 2, sizeof(CALint), &calclmodel2D->host_CA->sizeof_pQb_array);
	clSetKernelArg(calclmodel2D->kernelUpdateSubstate, 3, sizeof(CALint), &calclmodel2D->host_CA->sizeof_pQi_array);
	clSetKernelArg(calclmodel2D->kernelUpdateSubstate, 4, sizeof(CALint), &calclmodel2D->host_CA->sizeof_pQr_array);
	clSetKernelArg(calclmodel2D->kernelUpdateSubstate, 5, sizeof(CALCLmem), &calclmodel2D->bufferCurrentByteSubstate);
	clSetKernelArg(calclmodel2D->kernelUpdateSubstate, 6, sizeof(CALCLmem), &calclmodel2D->bufferCurrentIntSubstate);
	clSetKernelArg(calclmodel2D->kernelUpdateSubstate, 7, sizeof(CALCLmem), &calclmodel2D->bufferCurrentRealSubstate);
	clSetKernelArg(calclmodel2D->kernelUpdateSubstate, 8, sizeof(CALCLmem), &calclmodel2D->bufferNextByteSubstate);
	clSetKernelArg(calclmodel2D->kernelUpdateSubstate, 9, sizeof(CALCLmem), &calclmodel2D->bufferNextIntSubstate);
	clSetKernelArg(calclmodel2D->kernelUpdateSubstate, 10, sizeof(CALCLmem), &calclmodel2D->bufferNextRealSubstate);
	clSetKernelArg(calclmodel2D->kernelUpdateSubstate, 11, sizeof(CALCLmem), &calclmodel2D->bufferActiveCells);
	clSetKernelArg(calclmodel2D->kernelUpdateSubstate, 12, sizeof(CALCLmem), &calclmodel2D->bufferActiveCellsNum);

}

void calclSetModelParameters2D(struct CALCLModel2D* calclmodel2D, CALCLkernel * kernel) {

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
	double chunk_double = ceil((double) (calclmodel2D->host_CA->rows * calclmodel2D->host_CA->columns) / calclmodel2D->streamCompactionThreadsNum);
	int chunk = (int) chunk_double;
	clSetKernelArg(*kernel, 20, sizeof(int), &chunk);

}
void calclSetReductionParameters2D(struct CALCLModel2D* calclmodel2D, CALCLkernel * kernel) {

	clSetKernelArg(*kernel, 21, sizeof(CALCLmem), &calclmodel2D->bufferMinimab);
	clSetKernelArg(*kernel, 24, sizeof(CALCLmem), &calclmodel2D->bufferMiximab);
	clSetKernelArg(*kernel, 27, sizeof(CALCLmem), &calclmodel2D->bufferSumb);
	clSetKernelArg(*kernel, 30, sizeof(CALCLmem), &calclmodel2D->bufferLogicalAndsb);
	clSetKernelArg(*kernel, 33, sizeof(CALCLmem), &calclmodel2D->bufferLogicalOrsb);
	clSetKernelArg(*kernel, 36, sizeof(CALCLmem), &calclmodel2D->bufferLogicalXOrsb);
	clSetKernelArg(*kernel, 39, sizeof(CALCLmem), &calclmodel2D->bufferBinaryAndsb);
	clSetKernelArg(*kernel, 42, sizeof(CALCLmem), &calclmodel2D->bufferBinaryOrsb);
	clSetKernelArg(*kernel, 45, sizeof(CALCLmem), &calclmodel2D->bufferBinaryXOrsb);

	clSetKernelArg(*kernel, 22, sizeof(CALCLmem), &calclmodel2D->bufferMinimai);
	clSetKernelArg(*kernel, 25, sizeof(CALCLmem), &calclmodel2D->bufferMiximai);
	clSetKernelArg(*kernel, 28, sizeof(CALCLmem), &calclmodel2D->bufferSumi);
	clSetKernelArg(*kernel, 31, sizeof(CALCLmem), &calclmodel2D->bufferLogicalAndsi);
	clSetKernelArg(*kernel, 34, sizeof(CALCLmem), &calclmodel2D->bufferLogicalOrsi);
	clSetKernelArg(*kernel, 37, sizeof(CALCLmem), &calclmodel2D->bufferLogicalXOrsi);
	clSetKernelArg(*kernel, 40, sizeof(CALCLmem), &calclmodel2D->bufferBinaryAndsi);
	clSetKernelArg(*kernel, 43, sizeof(CALCLmem), &calclmodel2D->bufferBinaryOrsi);
	clSetKernelArg(*kernel, 46, sizeof(CALCLmem), &calclmodel2D->bufferBinaryXOrsi);

	clSetKernelArg(*kernel, 23, sizeof(CALCLmem), &calclmodel2D->bufferMinimar);
	clSetKernelArg(*kernel, 26, sizeof(CALCLmem), &calclmodel2D->bufferMiximar);
	clSetKernelArg(*kernel, 29, sizeof(CALCLmem), &calclmodel2D->bufferSumr);
	clSetKernelArg(*kernel, 32, sizeof(CALCLmem), &calclmodel2D->bufferLogicalAndsr);
	clSetKernelArg(*kernel, 35, sizeof(CALCLmem), &calclmodel2D->bufferLogicalOrsr);
	clSetKernelArg(*kernel, 38, sizeof(CALCLmem), &calclmodel2D->bufferLogicalXOrsr);
	clSetKernelArg(*kernel, 41, sizeof(CALCLmem), &calclmodel2D->bufferBinaryAndsr);
	clSetKernelArg(*kernel, 44, sizeof(CALCLmem), &calclmodel2D->bufferBinaryOrsr);
	clSetKernelArg(*kernel, 47, sizeof(CALCLmem), &calclmodel2D->bufferBinaryXOrsr);

}

void calclSetKernelCopyArgsi(struct CALCLModel2D* calclmodel2D) {
	clSetKernelArg(calclmodel2D->kernelMinCopyi, 0, sizeof(CALCLmem), &calclmodel2D->bufferPartialMini);
	clSetKernelArg(calclmodel2D->kernelMaxCopyi, 0, sizeof(CALCLmem), &calclmodel2D->bufferPartialMaxi);
	clSetKernelArg(calclmodel2D->kernelSumCopyi, 0, sizeof(CALCLmem), &calclmodel2D->bufferPartialSumi);
	clSetKernelArg(calclmodel2D->kernelProdCopyi, 0, sizeof(CALCLmem), &calclmodel2D->bufferPartialProdi);
	clSetKernelArg(calclmodel2D->kernelLogicalAndCopyi, 0, sizeof(CALCLmem), &calclmodel2D->bufferPartialLogicalAndi);
	clSetKernelArg(calclmodel2D->kernelLogicalOrCopyi, 0, sizeof(CALCLmem), &calclmodel2D->bufferPartialLogicalOri);
	clSetKernelArg(calclmodel2D->kernelLogicalXOrCopyi, 0, sizeof(CALCLmem), &calclmodel2D->bufferPartialLogicalXOri);
	clSetKernelArg(calclmodel2D->kernelBinaryAndCopyi, 0, sizeof(CALCLmem), &calclmodel2D->bufferPartialBinaryAndi);
	clSetKernelArg(calclmodel2D->kernelBinaryOrCopyi, 0, sizeof(CALCLmem), &calclmodel2D->bufferPartialBinaryOri);
	clSetKernelArg(calclmodel2D->kernelBinaryXOrCopyi, 0, sizeof(CALCLmem), &calclmodel2D->bufferPartialBinaryXOri);

	clSetKernelArg(calclmodel2D->kernelMinCopyi, 1, sizeof(CALCLmem), &calclmodel2D->bufferCurrentIntSubstate);
	clSetKernelArg(calclmodel2D->kernelMaxCopyi, 1, sizeof(CALCLmem), &calclmodel2D->bufferCurrentIntSubstate);
	clSetKernelArg(calclmodel2D->kernelSumCopyi, 1, sizeof(CALCLmem), &calclmodel2D->bufferCurrentIntSubstate);
	clSetKernelArg(calclmodel2D->kernelProdCopyi, 1, sizeof(CALCLmem), &calclmodel2D->bufferCurrentIntSubstate);
	clSetKernelArg(calclmodel2D->kernelLogicalAndCopyi, 1, sizeof(CALCLmem), &calclmodel2D->bufferCurrentIntSubstate);
	clSetKernelArg(calclmodel2D->kernelLogicalOrCopyi, 1, sizeof(CALCLmem), &calclmodel2D->bufferCurrentIntSubstate);
	clSetKernelArg(calclmodel2D->kernelLogicalXOrCopyi, 1, sizeof(CALCLmem), &calclmodel2D->bufferCurrentIntSubstate);
	clSetKernelArg(calclmodel2D->kernelBinaryAndCopyi, 1, sizeof(CALCLmem), &calclmodel2D->bufferCurrentIntSubstate);
	clSetKernelArg(calclmodel2D->kernelBinaryOrCopyi, 1, sizeof(CALCLmem), &calclmodel2D->bufferCurrentIntSubstate);
	clSetKernelArg(calclmodel2D->kernelBinaryXOrCopyi, 1, sizeof(CALCLmem), &calclmodel2D->bufferCurrentIntSubstate);

	clSetKernelArg(calclmodel2D->kernelMinCopyi, 3, sizeof(CALint), &calclmodel2D->host_CA->rows);
	clSetKernelArg(calclmodel2D->kernelMaxCopyi, 3, sizeof(CALint), &calclmodel2D->host_CA->rows);
	clSetKernelArg(calclmodel2D->kernelSumCopyi, 3, sizeof(CALint), &calclmodel2D->host_CA->rows);
	clSetKernelArg(calclmodel2D->kernelProdCopyi, 3, sizeof(CALint), &calclmodel2D->host_CA->rows);
	clSetKernelArg(calclmodel2D->kernelLogicalAndCopyi, 3, sizeof(CALint), &calclmodel2D->host_CA->rows);
	clSetKernelArg(calclmodel2D->kernelLogicalOrCopyi, 3, sizeof(CALint), &calclmodel2D->host_CA->rows);
	clSetKernelArg(calclmodel2D->kernelLogicalXOrCopyi, 3, sizeof(CALint), &calclmodel2D->host_CA->rows);
	clSetKernelArg(calclmodel2D->kernelBinaryAndCopyi, 3, sizeof(CALint), &calclmodel2D->host_CA->rows);
	clSetKernelArg(calclmodel2D->kernelBinaryOrCopyi, 3, sizeof(CALint), &calclmodel2D->host_CA->rows);
	clSetKernelArg(calclmodel2D->kernelBinaryXOrCopyi, 3, sizeof(CALint), &calclmodel2D->host_CA->rows);

	clSetKernelArg(calclmodel2D->kernelMinCopyi, 4, sizeof(CALint), &calclmodel2D->host_CA->columns);
	clSetKernelArg(calclmodel2D->kernelMaxCopyi, 4, sizeof(CALint), &calclmodel2D->host_CA->columns);
	clSetKernelArg(calclmodel2D->kernelSumCopyi, 4, sizeof(CALint), &calclmodel2D->host_CA->columns);
	clSetKernelArg(calclmodel2D->kernelProdCopyi, 4, sizeof(CALint), &calclmodel2D->host_CA->columns);
	clSetKernelArg(calclmodel2D->kernelLogicalAndCopyi, 4, sizeof(CALint), &calclmodel2D->host_CA->columns);
	clSetKernelArg(calclmodel2D->kernelLogicalOrCopyi, 4, sizeof(CALint), &calclmodel2D->host_CA->columns);
	clSetKernelArg(calclmodel2D->kernelLogicalXOrCopyi, 4, sizeof(CALint), &calclmodel2D->host_CA->columns);
	clSetKernelArg(calclmodel2D->kernelBinaryAndCopyi, 4, sizeof(CALint), &calclmodel2D->host_CA->columns);
	clSetKernelArg(calclmodel2D->kernelBinaryOrCopyi, 4, sizeof(CALint), &calclmodel2D->host_CA->columns);
	clSetKernelArg(calclmodel2D->kernelBinaryXOrCopyi, 4, sizeof(CALint), &calclmodel2D->host_CA->columns);
}

void calclSetKernelCopyArgsb(struct CALCLModel2D* calclmodel2D) {
	clSetKernelArg(calclmodel2D->kernelMinCopyb, 0, sizeof(CALCLmem), &calclmodel2D->bufferPartialMinb);
	clSetKernelArg(calclmodel2D->kernelMaxCopyb, 0, sizeof(CALCLmem), &calclmodel2D->bufferPartialMaxb);
	clSetKernelArg(calclmodel2D->kernelSumCopyb, 0, sizeof(CALCLmem), &calclmodel2D->bufferPartialSumb);
	clSetKernelArg(calclmodel2D->kernelProdCopyb, 0, sizeof(CALCLmem), &calclmodel2D->bufferPartialProdb);
	clSetKernelArg(calclmodel2D->kernelLogicalAndCopyb, 0, sizeof(CALCLmem), &calclmodel2D->bufferPartialLogicalAndb);
	clSetKernelArg(calclmodel2D->kernelLogicalOrCopyb, 0, sizeof(CALCLmem), &calclmodel2D->bufferPartialLogicalOrb);
	clSetKernelArg(calclmodel2D->kernelLogicalXOrCopyb, 0, sizeof(CALCLmem), &calclmodel2D->bufferPartialLogicalXOrb);
	clSetKernelArg(calclmodel2D->kernelBinaryAndCopyb, 0, sizeof(CALCLmem), &calclmodel2D->bufferPartialBinaryAndb);
	clSetKernelArg(calclmodel2D->kernelBinaryOrCopyb, 0, sizeof(CALCLmem), &calclmodel2D->bufferPartialBinaryOrb);
	clSetKernelArg(calclmodel2D->kernelBinaryXOrCopyb, 0, sizeof(CALCLmem), &calclmodel2D->bufferPartialBinaryXOrb);

	clSetKernelArg(calclmodel2D->kernelMinCopyb, 1, sizeof(CALCLmem), &calclmodel2D->bufferCurrentByteSubstate);
	clSetKernelArg(calclmodel2D->kernelMaxCopyb, 1, sizeof(CALCLmem), &calclmodel2D->bufferCurrentByteSubstate);
	clSetKernelArg(calclmodel2D->kernelSumCopyb, 1, sizeof(CALCLmem), &calclmodel2D->bufferCurrentByteSubstate);
	clSetKernelArg(calclmodel2D->kernelProdCopyb, 1, sizeof(CALCLmem), &calclmodel2D->bufferCurrentByteSubstate);
	clSetKernelArg(calclmodel2D->kernelLogicalAndCopyb, 1, sizeof(CALCLmem), &calclmodel2D->bufferCurrentByteSubstate);
	clSetKernelArg(calclmodel2D->kernelLogicalOrCopyb, 1, sizeof(CALCLmem), &calclmodel2D->bufferCurrentByteSubstate);
	clSetKernelArg(calclmodel2D->kernelLogicalXOrCopyb, 1, sizeof(CALCLmem), &calclmodel2D->bufferCurrentByteSubstate);
	clSetKernelArg(calclmodel2D->kernelBinaryAndCopyb, 1, sizeof(CALCLmem), &calclmodel2D->bufferCurrentByteSubstate);
	clSetKernelArg(calclmodel2D->kernelBinaryOrCopyb, 1, sizeof(CALCLmem), &calclmodel2D->bufferCurrentByteSubstate);
	clSetKernelArg(calclmodel2D->kernelBinaryXOrCopyb, 1, sizeof(CALCLmem), &calclmodel2D->bufferCurrentByteSubstate);

	clSetKernelArg(calclmodel2D->kernelMinCopyb, 3, sizeof(CALint), &calclmodel2D->host_CA->rows);
	clSetKernelArg(calclmodel2D->kernelMaxCopyb, 3, sizeof(CALint), &calclmodel2D->host_CA->rows);
	clSetKernelArg(calclmodel2D->kernelSumCopyb, 3, sizeof(CALint), &calclmodel2D->host_CA->rows);
	clSetKernelArg(calclmodel2D->kernelProdCopyb, 3, sizeof(CALint), &calclmodel2D->host_CA->rows);
	clSetKernelArg(calclmodel2D->kernelLogicalAndCopyb, 3, sizeof(CALint), &calclmodel2D->host_CA->rows);
	clSetKernelArg(calclmodel2D->kernelLogicalOrCopyb, 3, sizeof(CALint), &calclmodel2D->host_CA->rows);
	clSetKernelArg(calclmodel2D->kernelLogicalXOrCopyb, 3, sizeof(CALint), &calclmodel2D->host_CA->rows);
	clSetKernelArg(calclmodel2D->kernelBinaryAndCopyb, 3, sizeof(CALint), &calclmodel2D->host_CA->rows);
	clSetKernelArg(calclmodel2D->kernelBinaryOrCopyb, 3, sizeof(CALint), &calclmodel2D->host_CA->rows);
	clSetKernelArg(calclmodel2D->kernelBinaryXOrCopyb, 3, sizeof(CALint), &calclmodel2D->host_CA->rows);

	clSetKernelArg(calclmodel2D->kernelMinCopyb, 4, sizeof(CALint), &calclmodel2D->host_CA->columns);
	clSetKernelArg(calclmodel2D->kernelMaxCopyb, 4, sizeof(CALint), &calclmodel2D->host_CA->columns);
	clSetKernelArg(calclmodel2D->kernelSumCopyb, 4, sizeof(CALint), &calclmodel2D->host_CA->columns);
	clSetKernelArg(calclmodel2D->kernelProdCopyb, 4, sizeof(CALint), &calclmodel2D->host_CA->columns);
	clSetKernelArg(calclmodel2D->kernelLogicalAndCopyb, 4, sizeof(CALint), &calclmodel2D->host_CA->columns);
	clSetKernelArg(calclmodel2D->kernelLogicalOrCopyb, 4, sizeof(CALint), &calclmodel2D->host_CA->columns);
	clSetKernelArg(calclmodel2D->kernelLogicalXOrCopyb, 4, sizeof(CALint), &calclmodel2D->host_CA->columns);
	clSetKernelArg(calclmodel2D->kernelBinaryAndCopyb, 4, sizeof(CALint), &calclmodel2D->host_CA->columns);
	clSetKernelArg(calclmodel2D->kernelBinaryOrCopyb, 4, sizeof(CALint), &calclmodel2D->host_CA->columns);
	clSetKernelArg(calclmodel2D->kernelBinaryXOrCopyb, 4, sizeof(CALint), &calclmodel2D->host_CA->columns);
}

void calclSetKernelCopyArgsr(struct CALCLModel2D* calclmodel2D) {
	clSetKernelArg(calclmodel2D->kernelMinCopyr, 0, sizeof(CALCLmem), &calclmodel2D->bufferPartialMinr);
	clSetKernelArg(calclmodel2D->kernelMaxCopyr, 0, sizeof(CALCLmem), &calclmodel2D->bufferPartialMaxr);
	clSetKernelArg(calclmodel2D->kernelSumCopyr, 0, sizeof(CALCLmem), &calclmodel2D->bufferPartialSumr);
	clSetKernelArg(calclmodel2D->kernelProdCopyr, 0, sizeof(CALCLmem), &calclmodel2D->bufferPartialProdr);
	clSetKernelArg(calclmodel2D->kernelLogicalAndCopyr, 0, sizeof(CALCLmem), &calclmodel2D->bufferPartialLogicalAndr);
	clSetKernelArg(calclmodel2D->kernelLogicalOrCopyr, 0, sizeof(CALCLmem), &calclmodel2D->bufferPartialLogicalOrr);
	clSetKernelArg(calclmodel2D->kernelLogicalXOrCopyr, 0, sizeof(CALCLmem), &calclmodel2D->bufferPartialLogicalXOrr);
	clSetKernelArg(calclmodel2D->kernelBinaryAndCopyr, 0, sizeof(CALCLmem), &calclmodel2D->bufferPartialBinaryAndr);
	clSetKernelArg(calclmodel2D->kernelBinaryOrCopyr, 0, sizeof(CALCLmem), &calclmodel2D->bufferPartialBinaryOrr);
	clSetKernelArg(calclmodel2D->kernelBinaryXOrCopyr, 0, sizeof(CALCLmem), &calclmodel2D->bufferPartialBinaryXOrr);

	clSetKernelArg(calclmodel2D->kernelMinCopyr, 1, sizeof(CALCLmem), &calclmodel2D->bufferCurrentRealSubstate);
	clSetKernelArg(calclmodel2D->kernelMaxCopyr, 1, sizeof(CALCLmem), &calclmodel2D->bufferCurrentRealSubstate);
	clSetKernelArg(calclmodel2D->kernelSumCopyr, 1, sizeof(CALCLmem), &calclmodel2D->bufferCurrentRealSubstate);
	clSetKernelArg(calclmodel2D->kernelProdCopyr, 1, sizeof(CALCLmem), &calclmodel2D->bufferCurrentRealSubstate);
	clSetKernelArg(calclmodel2D->kernelLogicalAndCopyr, 1, sizeof(CALCLmem), &calclmodel2D->bufferCurrentRealSubstate);
	clSetKernelArg(calclmodel2D->kernelLogicalOrCopyr, 1, sizeof(CALCLmem), &calclmodel2D->bufferCurrentRealSubstate);
	clSetKernelArg(calclmodel2D->kernelLogicalXOrCopyr, 1, sizeof(CALCLmem), &calclmodel2D->bufferCurrentRealSubstate);
	clSetKernelArg(calclmodel2D->kernelBinaryAndCopyr, 1, sizeof(CALCLmem), &calclmodel2D->bufferCurrentRealSubstate);
	clSetKernelArg(calclmodel2D->kernelBinaryOrCopyr, 1, sizeof(CALCLmem), &calclmodel2D->bufferCurrentRealSubstate);
	clSetKernelArg(calclmodel2D->kernelBinaryXOrCopyr, 1, sizeof(CALCLmem), &calclmodel2D->bufferCurrentRealSubstate);

	clSetKernelArg(calclmodel2D->kernelMinCopyr, 3, sizeof(CALint), &calclmodel2D->host_CA->rows);
	clSetKernelArg(calclmodel2D->kernelMaxCopyr, 3, sizeof(CALint), &calclmodel2D->host_CA->rows);
	clSetKernelArg(calclmodel2D->kernelSumCopyr, 3, sizeof(CALint), &calclmodel2D->host_CA->rows);
	clSetKernelArg(calclmodel2D->kernelProdCopyr, 3, sizeof(CALint), &calclmodel2D->host_CA->rows);
	clSetKernelArg(calclmodel2D->kernelLogicalAndCopyr, 3, sizeof(CALint), &calclmodel2D->host_CA->rows);
	clSetKernelArg(calclmodel2D->kernelLogicalOrCopyr, 3, sizeof(CALint), &calclmodel2D->host_CA->rows);
	clSetKernelArg(calclmodel2D->kernelLogicalXOrCopyr, 3, sizeof(CALint), &calclmodel2D->host_CA->rows);
	clSetKernelArg(calclmodel2D->kernelBinaryAndCopyr, 3, sizeof(CALint), &calclmodel2D->host_CA->rows);
	clSetKernelArg(calclmodel2D->kernelBinaryOrCopyr, 3, sizeof(CALint), &calclmodel2D->host_CA->rows);
	clSetKernelArg(calclmodel2D->kernelBinaryXOrCopyr, 3, sizeof(CALint), &calclmodel2D->host_CA->rows);

	clSetKernelArg(calclmodel2D->kernelMinCopyr, 4, sizeof(CALint), &calclmodel2D->host_CA->columns);
	clSetKernelArg(calclmodel2D->kernelMaxCopyr, 4, sizeof(CALint), &calclmodel2D->host_CA->columns);
	clSetKernelArg(calclmodel2D->kernelSumCopyr, 4, sizeof(CALint), &calclmodel2D->host_CA->columns);
	clSetKernelArg(calclmodel2D->kernelProdCopyr, 4, sizeof(CALint), &calclmodel2D->host_CA->columns);
	clSetKernelArg(calclmodel2D->kernelLogicalAndCopyr, 4, sizeof(CALint), &calclmodel2D->host_CA->columns);
	clSetKernelArg(calclmodel2D->kernelLogicalOrCopyr, 4, sizeof(CALint), &calclmodel2D->host_CA->columns);
	clSetKernelArg(calclmodel2D->kernelLogicalXOrCopyr, 4, sizeof(CALint), &calclmodel2D->host_CA->columns);
	clSetKernelArg(calclmodel2D->kernelBinaryAndCopyr, 4, sizeof(CALint), &calclmodel2D->host_CA->columns);
	clSetKernelArg(calclmodel2D->kernelBinaryOrCopyr, 4, sizeof(CALint), &calclmodel2D->host_CA->columns);
	clSetKernelArg(calclmodel2D->kernelBinaryXOrCopyr, 4, sizeof(CALint), &calclmodel2D->host_CA->columns);
}

void calclRealSubstatesMapper2D(struct CALModel2D * host_CA, CALreal * current, CALreal * next) {
	int ssNum = host_CA->sizeof_pQr_array;
	size_t elNum = host_CA->columns * host_CA->rows;
	long int outIndex = 0;
	long int outIndex1 = 0;
	int i;
	unsigned int j;

	for (i = 0; i < ssNum; i++) {
		for (j = 0; j < elNum; j++)
			current[outIndex++] = host_CA->pQr_array[i]->current[j];
		for (j = 0; j < elNum; j++)
			next[outIndex1++] = host_CA->pQr_array[i]->next[j];
	}
}
void calclByteSubstatesMapper2D(struct CALModel2D * host_CA, CALbyte * current, CALbyte * next) {
	int ssNum = host_CA->sizeof_pQb_array;
	size_t elNum = host_CA->columns * host_CA->rows;
	long int outIndex = 0;
	long int outIndex1 = 0;
	int i;
	unsigned int j;

	for (i = 0; i < ssNum; i++) {
		for (j = 0; j < elNum; j++)
			current[outIndex++] = host_CA->pQb_array[i]->current[j];
		for (j = 0; j < elNum; j++)
			next[outIndex1++] = host_CA->pQb_array[i]->next[j];
	}
}
void calclIntSubstatesMapper2D(struct CALModel2D * host_CA, CALint * current, CALint * next) {
	int ssNum = host_CA->sizeof_pQi_array;
	size_t elNum = host_CA->columns * host_CA->rows;
	long int outIndex = 0;
	long int outIndex1 = 0;
	int i;
	unsigned int j;

	for (i = 0; i < ssNum; i++) {
		for (j = 0; j < elNum; j++)
			current[outIndex++] = host_CA->pQi_array[i]->current[j];
		for (j = 0; j < elNum; j++)
			next[outIndex1++] = host_CA->pQi_array[i]->next[j];
	}
}

CALCLqueue calclCreateQueue2D(struct CALCLModel2D * calclmodel2D, CALCLcontext context, CALCLdevice device) {
	CALCLqueue queue = calclCreateCommandQueue(context, device);
	size_t cores;
	cl_int err;
	err = clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &cores, NULL);
	calclHandleError(err);

	//printf("cores = %d ", cores);
	//TODO choose stream compaction threads num
	calclmodel2D->streamCompactionThreadsNum = cores * 4;

	while (calclmodel2D->host_CA->rows * calclmodel2D->host_CA->columns <= (int) calclmodel2D->streamCompactionThreadsNum)
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
	calclSetKernelStreamCompactionArgs2D(calclmodel2D);

	return queue;
}

CALbyte existOneTrue2D(struct CALCLModel2D* calclmodel2D, CALbyte * tmp, CALint dim) {

	for (int i = 0; i < dim; i++) {
		if (tmp[i])
			return 1;
	}
	return 0;
}

/******************************************************************************
 * 							PUBLIC FUNCTIONS
 ******************************************************************************/

struct CALCLModel2D * calclCADef2D(struct CALModel2D *host_CA, CALCLcontext context, CALCLprogram program, CALCLdevice device) {

	struct CALCLModel2D * calclmodel2D = (struct CALCLModel2D*) malloc(sizeof(struct CALCLModel2D));
	calclmodel2D->host_CA = host_CA;
	calclmodel2D->opt = host_CA->OPTIMIZATION;
	calclmodel2D->cl_update_substates = NULL;
	calclmodel2D->kernelInitSubstates = NULL;
	calclmodel2D->kernelSteering = NULL;
	calclmodel2D->kernelStopCondition = NULL;
	calclmodel2D->elementaryProcessesNum = 0;
	calclmodel2D->steps = 0;

	if (calclmodel2D->host_CA->A.flags == NULL) {
		calclmodel2D->host_CA->A.flags = (CALbyte*) malloc(sizeof(CALbyte) * calclmodel2D->host_CA->rows * calclmodel2D->host_CA->columns);
		memset(calclmodel2D->host_CA->A.flags, CAL_FALSE, sizeof(CALbyte) * calclmodel2D->host_CA->rows * calclmodel2D->host_CA->columns);
	}

	cl_int err;
	int bufferDim = calclmodel2D->host_CA->columns * calclmodel2D->host_CA->rows;

	calclmodel2D->kernelUpdateSubstate = calclGetKernelFromProgram(&program, KER_UPDATESUBSTATES);

	//stream compaction kernels
	calclmodel2D->kernelCompact = calclGetKernelFromProgram(&program, KER_STC_COMPACT);
	calclmodel2D->kernelComputeCounts = calclGetKernelFromProgram(&program, KER_STC_COMPUTE_COUNTS);
	calclmodel2D->kernelUpSweep = calclGetKernelFromProgram(&program, KER_STC_UP_SWEEP);
	calclmodel2D->kernelDownSweep = calclGetKernelFromProgram(&program, KER_STC_DOWN_SWEEP);

	calclmodel2D->kernelMinReductionb = calclGetKernelFromProgram(&program, "calclMinReductionKernel2Db");
	calclmodel2D->kernelMaxReductionb = calclGetKernelFromProgram(&program, "calclMaxReductionKernel2Db");
	calclmodel2D->kernelSumReductionb = calclGetKernelFromProgram(&program, "calclSumReductionKernel2Db");
	calclmodel2D->kernelProdReductionb = calclGetKernelFromProgram(&program, "calclProdReductionKernel2Db");
	calclmodel2D->kernelLogicalAndReductionb = calclGetKernelFromProgram(&program, "calclLogicAndReductionKernel2Db");
	calclmodel2D->kernelLogicalOrReductionb = calclGetKernelFromProgram(&program, "calclLogicOrReductionKernel2Db");
	calclmodel2D->kernelLogicalXOrReductionb = calclGetKernelFromProgram(&program, "calclLogicXOrReductionKernel2Db");
	calclmodel2D->kernelBinaryAndReductionb = calclGetKernelFromProgram(&program, "calclBinaryAndReductionKernel2Db");
	calclmodel2D->kernelBinaryOrReductionb = calclGetKernelFromProgram(&program, "calclBinaryOrReductionKernel2Db");
	calclmodel2D->kernelBinaryXorReductionb = calclGetKernelFromProgram(&program, "calclBinaryXOrReductionKernel2Db");

	calclmodel2D->kernelMinReductioni = calclGetKernelFromProgram(&program, "calclMinReductionKernel2Di");
	calclmodel2D->kernelMaxReductioni = calclGetKernelFromProgram(&program, "calclMaxReductionKernel2Di");
	calclmodel2D->kernelSumReductioni = calclGetKernelFromProgram(&program, "calclSumReductionKernel2Di");
	calclmodel2D->kernelProdReductioni = calclGetKernelFromProgram(&program, "calclProdReductionKernel2Di");
	calclmodel2D->kernelLogicalAndReductioni = calclGetKernelFromProgram(&program, "calclLogicAndReductionKernel2Di");
	calclmodel2D->kernelLogicalOrReductioni = calclGetKernelFromProgram(&program, "calclLogicOrReductionKernel2Di");
	calclmodel2D->kernelLogicalXOrReductioni = calclGetKernelFromProgram(&program, "calclLogicXOrReductionKernel2Di");
	calclmodel2D->kernelBinaryAndReductioni = calclGetKernelFromProgram(&program, "calclBinaryAndReductionKernel2Di");
	calclmodel2D->kernelBinaryOrReductioni = calclGetKernelFromProgram(&program, "calclBinaryOrReductionKernel2Di");
	calclmodel2D->kernelBinaryXorReductioni = calclGetKernelFromProgram(&program, "calclBinaryXOrReductionKernel2Di");

	calclmodel2D->kernelMinReductionr = calclGetKernelFromProgram(&program, "calclMinReductionKernel2Dr");
	calclmodel2D->kernelMaxReductionr = calclGetKernelFromProgram(&program, "calclMaxReductionKernel2Dr");
	calclmodel2D->kernelSumReductionr = calclGetKernelFromProgram(&program, "calclSumReductionKernel2Dr");
	calclmodel2D->kernelProdReductionr = calclGetKernelFromProgram(&program, "calclProdReductionKernel2Dr");
	calclmodel2D->kernelLogicalAndReductionr = calclGetKernelFromProgram(&program, "calclLogicAndReductionKernel2Dr");
	calclmodel2D->kernelLogicalOrReductionr = calclGetKernelFromProgram(&program, "calclLogicOrReductionKernel2Dr");
	calclmodel2D->kernelLogicalXOrReductionr = calclGetKernelFromProgram(&program, "calclLogicXOrReductionKernel2Dr");
	calclmodel2D->kernelBinaryAndReductionr = calclGetKernelFromProgram(&program, "calclBinaryAndReductionKernel2Dr");
	calclmodel2D->kernelBinaryOrReductionr = calclGetKernelFromProgram(&program, "calclBinaryOrReductionKernel2Dr");
	calclmodel2D->kernelBinaryXorReductionr = calclGetKernelFromProgram(&program, "calclBinaryXOrReductionKernel2Dr");

	struct CALCell2D * activeCells = (struct CALCell2D*) malloc(sizeof(struct CALCell2D) * bufferDim);
	memcpy(activeCells, calclmodel2D->host_CA->A.cells, sizeof(struct CALCell2D) * calclmodel2D->host_CA->A.size_current);

	calclmodel2D->bufferActiveCells = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(struct CALCell2D) * bufferDim, activeCells, &err);
	calclHandleError(err);
	free(activeCells);
	calclmodel2D->bufferActiveCellsFlags = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(CALbyte) * bufferDim, calclmodel2D->host_CA->A.flags, &err);
	calclHandleError(err);

	calclmodel2D->bufferActiveCellsNum = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(CALint), &calclmodel2D->host_CA->A.size_current, &err);
	calclHandleError(err);

	calclmodel2D->bufferByteSubstateNum = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(CALint), &calclmodel2D->host_CA->sizeof_pQb_array, &err);
	calclHandleError(err);
	calclmodel2D->bufferIntSubstateNum = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(CALint), &calclmodel2D->host_CA->sizeof_pQi_array, &err);
	calclHandleError(err);
	calclmodel2D->bufferRealSubstateNum = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(CALint), &calclmodel2D->host_CA->sizeof_pQr_array, &err);
	calclHandleError(err);

	calclmodel2D->bufferColumns = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(CALint), &calclmodel2D->host_CA->columns, &err);
	calclHandleError(err);
	calclmodel2D->bufferRows = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(CALint), &calclmodel2D->host_CA->rows, &err);
	calclHandleError(err);

	size_t byteSubstatesDim = sizeof(CALbyte) * bufferDim * calclmodel2D->host_CA->sizeof_pQb_array + 1;
	CALbyte * currentByteSubstates = (CALbyte*) malloc(byteSubstatesDim);
	CALbyte * nextByteSubstates = (CALbyte*) malloc(byteSubstatesDim);
	calclByteSubstatesMapper2D(calclmodel2D->host_CA, currentByteSubstates, nextByteSubstates);
	calclmodel2D->bufferCurrentByteSubstate = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, byteSubstatesDim, currentByteSubstates, &err);
	calclHandleError(err);
	calclmodel2D->bufferNextByteSubstate = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, byteSubstatesDim, nextByteSubstates, &err);
	calclHandleError(err);
	free(currentByteSubstates);
	free(nextByteSubstates);

	size_t intSubstatesDim = sizeof(CALint) * bufferDim * calclmodel2D->host_CA->sizeof_pQi_array + 1;
	CALint * currentIntSubstates = (CALint*) malloc(intSubstatesDim);
	CALint * nextIntSubstates = (CALint*) malloc(intSubstatesDim);
	calclIntSubstatesMapper2D(calclmodel2D->host_CA, currentIntSubstates, nextIntSubstates);
	calclmodel2D->bufferCurrentIntSubstate = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, intSubstatesDim, currentIntSubstates, &err);
	calclHandleError(err);
	calclmodel2D->bufferNextIntSubstate = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, intSubstatesDim, nextIntSubstates, &err);
	calclHandleError(err);
	free(currentIntSubstates);
	free(nextIntSubstates);

	size_t realSubstatesDim = sizeof(CALreal) * bufferDim * calclmodel2D->host_CA->sizeof_pQr_array + 1;
	CALreal * currentRealSubstates = (CALreal*) malloc(realSubstatesDim);
	CALreal * nextRealSubstates = (CALreal*) malloc(realSubstatesDim);
	calclRealSubstatesMapper2D(calclmodel2D->host_CA, currentRealSubstates, nextRealSubstates);
	calclmodel2D->bufferCurrentRealSubstate = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, realSubstatesDim, currentRealSubstates, &err);
	calclHandleError(err);
	calclmodel2D->bufferNextRealSubstate = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, realSubstatesDim, nextRealSubstates, &err);
	calclHandleError(err);
	free(currentRealSubstates);
	free(nextRealSubstates);

	calclSetKernelsLibArgs2D(calclmodel2D);

	calclmodel2D->kernelMinCopyi = calclGetKernelFromProgram(&program, "copyi");
	calclmodel2D->kernelMaxCopyi = calclGetKernelFromProgram(&program, "copyi");
	calclmodel2D->kernelSumCopyi = calclGetKernelFromProgram(&program, "copyi");
	calclmodel2D->kernelProdCopyi = calclGetKernelFromProgram(&program, "copyi");
	calclmodel2D->kernelLogicalAndCopyi = calclGetKernelFromProgram(&program, "copyi");
	calclmodel2D->kernelLogicalOrCopyi = calclGetKernelFromProgram(&program, "copyi");
	calclmodel2D->kernelLogicalXOrCopyi = calclGetKernelFromProgram(&program, "copyi");
	calclmodel2D->kernelBinaryAndCopyi = calclGetKernelFromProgram(&program, "copyi");
	calclmodel2D->kernelBinaryOrCopyi = calclGetKernelFromProgram(&program, "copyi");
	calclmodel2D->kernelBinaryXOrCopyi = calclGetKernelFromProgram(&program, "copyi");

	CALint * partialMini = (CALint*) malloc(calclmodel2D->host_CA->rows * calclmodel2D->host_CA->columns * sizeof(CALint));
	calclmodel2D->bufferPartialMini = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(CALint) * calclmodel2D->host_CA->rows * calclmodel2D->host_CA->columns, partialMini,
			&err);
	calclHandleError(err);
	calclmodel2D->bufferPartialMaxi = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(CALint) * calclmodel2D->host_CA->rows * calclmodel2D->host_CA->columns, partialMini,
			&err);
	calclHandleError(err);
	calclmodel2D->bufferPartialSumi = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(CALint) * calclmodel2D->host_CA->rows * calclmodel2D->host_CA->columns, partialMini,
			&err);
	calclHandleError(err);
	calclmodel2D->bufferPartialProdi = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(CALint) * calclmodel2D->host_CA->rows * calclmodel2D->host_CA->columns, partialMini,
			&err);
	calclHandleError(err);

	calclmodel2D->bufferPartialLogicalAndi = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(CALint) * calclmodel2D->host_CA->rows * calclmodel2D->host_CA->columns,
			partialMini, &err);
	calclHandleError(err);
	calclmodel2D->bufferPartialLogicalOri = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(CALint) * calclmodel2D->host_CA->rows * calclmodel2D->host_CA->columns,
			partialMini, &err);
	calclHandleError(err);
	calclmodel2D->bufferPartialLogicalXOri = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(CALint) * calclmodel2D->host_CA->rows * calclmodel2D->host_CA->columns,
			partialMini, &err);
	calclHandleError(err);

	calclmodel2D->bufferPartialBinaryAndi = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(CALint) * calclmodel2D->host_CA->rows * calclmodel2D->host_CA->columns,
			partialMini, &err);
	calclHandleError(err);
	calclmodel2D->bufferPartialBinaryOri = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(CALint) * calclmodel2D->host_CA->rows * calclmodel2D->host_CA->columns, partialMini,
			&err);
	calclHandleError(err);
	calclmodel2D->bufferPartialBinaryXOri = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(CALint) * calclmodel2D->host_CA->rows * calclmodel2D->host_CA->columns,
			partialMini, &err);
	calclHandleError(err);
	free(partialMini);

	calclSetKernelCopyArgsi(calclmodel2D);

	calclmodel2D->kernelMinCopyb = calclGetKernelFromProgram(&program, "copyb");
	calclmodel2D->kernelMaxCopyb = calclGetKernelFromProgram(&program, "copyb");
	calclmodel2D->kernelSumCopyb = calclGetKernelFromProgram(&program, "copyb");
	calclmodel2D->kernelProdCopyb = calclGetKernelFromProgram(&program, "copyb");
	calclmodel2D->kernelLogicalAndCopyb = calclGetKernelFromProgram(&program, "copyb");
	calclmodel2D->kernelLogicalOrCopyb = calclGetKernelFromProgram(&program, "copyb");
	calclmodel2D->kernelLogicalXOrCopyb = calclGetKernelFromProgram(&program, "copyb");
	calclmodel2D->kernelBinaryAndCopyb = calclGetKernelFromProgram(&program, "copyb");
	calclmodel2D->kernelBinaryOrCopyb = calclGetKernelFromProgram(&program, "copyb");
	calclmodel2D->kernelBinaryXOrCopyb = calclGetKernelFromProgram(&program, "copyb");

	CALbyte * partialMinb = (CALbyte*) malloc(calclmodel2D->host_CA->rows * calclmodel2D->host_CA->columns * sizeof(CALbyte));
	calclmodel2D->bufferPartialMinb = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(CALbyte) * calclmodel2D->host_CA->rows * calclmodel2D->host_CA->columns, partialMinb,
			&err);
	calclHandleError(err);
	calclmodel2D->bufferPartialMaxb = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(CALbyte) * calclmodel2D->host_CA->rows * calclmodel2D->host_CA->columns, partialMinb,
			&err);
	calclHandleError(err);
	calclmodel2D->bufferPartialSumb = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(CALbyte) * calclmodel2D->host_CA->rows * calclmodel2D->host_CA->columns, partialMinb,
			&err);
	calclHandleError(err);
	calclmodel2D->bufferPartialProdb = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(CALbyte) * calclmodel2D->host_CA->rows * calclmodel2D->host_CA->columns, partialMinb,
			&err);
	calclHandleError(err);

	calclmodel2D->bufferPartialLogicalAndb = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(CALbyte) * calclmodel2D->host_CA->rows * calclmodel2D->host_CA->columns,
			partialMinb, &err);
	calclHandleError(err);
	calclmodel2D->bufferPartialLogicalOrb = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(CALbyte) * calclmodel2D->host_CA->rows * calclmodel2D->host_CA->columns,
			partialMinb, &err);
	calclHandleError(err);
	calclmodel2D->bufferPartialLogicalXOrb = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(CALbyte) * calclmodel2D->host_CA->rows * calclmodel2D->host_CA->columns,
			partialMinb, &err);
	calclHandleError(err);

	calclmodel2D->bufferPartialBinaryAndb = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(CALbyte) * calclmodel2D->host_CA->rows * calclmodel2D->host_CA->columns,
			partialMinb, &err);
	calclHandleError(err);
	calclmodel2D->bufferPartialBinaryOrb = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(CALbyte) * calclmodel2D->host_CA->rows * calclmodel2D->host_CA->columns, partialMinb,
			&err);
	calclHandleError(err);
	calclmodel2D->bufferPartialBinaryXOrb = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(CALbyte) * calclmodel2D->host_CA->rows * calclmodel2D->host_CA->columns,
			partialMinb, &err);
	calclHandleError(err);
	free(partialMinb);
	calclSetKernelCopyArgsb(calclmodel2D);

	calclmodel2D->kernelMinCopyr = calclGetKernelFromProgram(&program, "copyr");
	calclmodel2D->kernelMaxCopyr = calclGetKernelFromProgram(&program, "copyr");
	calclmodel2D->kernelSumCopyr = calclGetKernelFromProgram(&program, "copyr");
	calclmodel2D->kernelProdCopyr = calclGetKernelFromProgram(&program, "copyr");
	calclmodel2D->kernelLogicalAndCopyr = calclGetKernelFromProgram(&program, "copyr");
	calclmodel2D->kernelLogicalOrCopyr = calclGetKernelFromProgram(&program, "copyr");
	calclmodel2D->kernelLogicalXOrCopyr = calclGetKernelFromProgram(&program, "copyr");
	calclmodel2D->kernelBinaryAndCopyr = calclGetKernelFromProgram(&program, "copyr");
	calclmodel2D->kernelBinaryOrCopyr = calclGetKernelFromProgram(&program, "copyr");
	calclmodel2D->kernelBinaryXOrCopyr = calclGetKernelFromProgram(&program, "copyr");

	CALreal * partialr = (CALreal*) malloc(calclmodel2D->host_CA->rows * calclmodel2D->host_CA->columns * sizeof(CALreal));
	calclmodel2D->bufferPartialMinr = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(CALreal) * calclmodel2D->host_CA->rows * calclmodel2D->host_CA->columns, partialr,
			&err);
	calclHandleError(err);
	calclmodel2D->bufferPartialMaxr = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(CALreal) * calclmodel2D->host_CA->rows * calclmodel2D->host_CA->columns, partialr,
			&err);
	calclHandleError(err);
	calclmodel2D->bufferPartialSumr = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(CALreal) * calclmodel2D->host_CA->rows * calclmodel2D->host_CA->columns, partialr,
			&err);
	calclHandleError(err);
	calclmodel2D->bufferPartialProdr = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(CALreal) * calclmodel2D->host_CA->rows * calclmodel2D->host_CA->columns, partialr,
			&err);
	calclHandleError(err);

	calclmodel2D->bufferPartialLogicalAndr = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(CALreal) * calclmodel2D->host_CA->rows * calclmodel2D->host_CA->columns,partialr,
			&err);
	calclHandleError(err);
	calclmodel2D->bufferPartialLogicalOrr = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(CALreal) * calclmodel2D->host_CA->rows * calclmodel2D->host_CA->columns, partialr,
			&err);
	calclHandleError(err);
	calclmodel2D->bufferPartialLogicalXOrr = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(CALreal) * calclmodel2D->host_CA->rows * calclmodel2D->host_CA->columns,partialr,
			&err);
	calclHandleError(err);

	calclmodel2D->bufferPartialBinaryAndr = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(CALreal) * calclmodel2D->host_CA->rows * calclmodel2D->host_CA->columns, partialr,
			&err);
	calclHandleError(err);
	calclmodel2D->bufferPartialBinaryOrr = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(CALreal) * calclmodel2D->host_CA->rows * calclmodel2D->host_CA->columns, partialr,
			&err);
	calclHandleError(err);
	calclmodel2D->bufferPartialBinaryXOrr = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(CALreal) * calclmodel2D->host_CA->rows * calclmodel2D->host_CA->columns,partialr,
			&err);
	calclHandleError(err);
	free(partialr);
	calclSetKernelCopyArgsr(calclmodel2D);

	//user kernels buffers args

	calclmodel2D->bufferNeighborhood = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(struct CALCell2D) * calclmodel2D->host_CA->sizeof_X, calclmodel2D->host_CA->X, &err);
	calclHandleError(err);
	calclmodel2D->bufferNeighborhoodID = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(enum CALNeighborhood2D), &calclmodel2D->host_CA->X_id, &err);
	calclHandleError(err);
	calclmodel2D->bufferNeighborhoodSize = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(CALint), &calclmodel2D->host_CA->sizeof_X, &err);
	calclHandleError(err);
	calclmodel2D->bufferBoundaryCondition = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(enum CALSpaceBoundaryCondition), &calclmodel2D->host_CA->T, &err);
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

	calclmodel2D->queue = calclCreateQueue2D(calclmodel2D, context, device);

	//TODO Reduction
	CALint dimReductionArrays = calclmodel2D->host_CA->sizeof_pQb_array + calclmodel2D->host_CA->sizeof_pQi_array + calclmodel2D->host_CA->sizeof_pQr_array;

	calclmodel2D->reductionFlagsMinb = (CALbyte*) malloc(sizeof(CALbyte) * (calclmodel2D->host_CA->sizeof_pQb_array));
	calclmodel2D->reductionFlagsMini = (CALbyte*) malloc(sizeof(CALbyte) * (calclmodel2D->host_CA->sizeof_pQi_array));
	calclmodel2D->reductionFlagsMinr = (CALbyte*) malloc(sizeof(CALbyte) * (calclmodel2D->host_CA->sizeof_pQr_array));
	calclmodel2D->minimab = (CALreal*) malloc(sizeof(CALreal) * (calclmodel2D->host_CA->sizeof_pQb_array + 1));
	calclmodel2D->minimai = (CALreal*) malloc(sizeof(CALreal) * (calclmodel2D->host_CA->sizeof_pQi_array + 1));
	calclmodel2D->minimar = (CALreal*) malloc(sizeof(CALreal) * (calclmodel2D->host_CA->sizeof_pQr_array + 1));

	calclmodel2D->reductionFlagsMaxb = (CALbyte*) malloc(sizeof(CALbyte) * (calclmodel2D->host_CA->sizeof_pQb_array));
	calclmodel2D->reductionFlagsMaxi = (CALbyte*) malloc(sizeof(CALbyte) * (calclmodel2D->host_CA->sizeof_pQi_array));
	calclmodel2D->reductionFlagsMaxr = (CALbyte*) malloc(sizeof(CALbyte) * (calclmodel2D->host_CA->sizeof_pQr_array));
	calclmodel2D->maximab = (CALreal*) malloc(sizeof(CALreal) * (calclmodel2D->host_CA->sizeof_pQb_array + 1));
	calclmodel2D->maximai = (CALreal*) malloc(sizeof(CALreal) * (calclmodel2D->host_CA->sizeof_pQi_array + 1));
	calclmodel2D->maximar = (CALreal*) malloc(sizeof(CALreal) * (calclmodel2D->host_CA->sizeof_pQr_array + 1));

	calclmodel2D->reductionFlagsSumb = (CALbyte*) malloc(sizeof(CALbyte) * (calclmodel2D->host_CA->sizeof_pQb_array));
	calclmodel2D->reductionFlagsSumi = (CALbyte*) malloc(sizeof(CALbyte) * (calclmodel2D->host_CA->sizeof_pQi_array));
	calclmodel2D->reductionFlagsSumr = (CALbyte*) malloc(sizeof(CALbyte) * (calclmodel2D->host_CA->sizeof_pQr_array));
	calclmodel2D->sumsb = (CALreal*) malloc(sizeof(CALreal) * (calclmodel2D->host_CA->sizeof_pQb_array + 1));
	calclmodel2D->sumsi = (CALreal*) malloc(sizeof(CALreal) * (calclmodel2D->host_CA->sizeof_pQi_array + 1));
	calclmodel2D->sumsr = (CALreal*) malloc(sizeof(CALreal) * (calclmodel2D->host_CA->sizeof_pQr_array + 1));

	calclmodel2D->reductionFlagsLogicalAndb = (CALbyte*) malloc(sizeof(CALbyte) * (calclmodel2D->host_CA->sizeof_pQb_array));
	calclmodel2D->reductionFlagsLogicalAndi = (CALbyte*) malloc(sizeof(CALbyte) * (calclmodel2D->host_CA->sizeof_pQi_array));
	calclmodel2D->reductionFlagsLogicalAndr = (CALbyte*) malloc(sizeof(CALbyte) * (calclmodel2D->host_CA->sizeof_pQr_array));
	calclmodel2D->logicalAndsb = (CALint*) malloc(sizeof(CALint) * (calclmodel2D->host_CA->sizeof_pQb_array + 1));
	calclmodel2D->logicalAndsi = (CALint*) malloc(sizeof(CALint) * (calclmodel2D->host_CA->sizeof_pQi_array + 1));
	calclmodel2D->logicalAndsr = (CALint*) malloc(sizeof(CALint) * (calclmodel2D->host_CA->sizeof_pQr_array + 1));

	calclmodel2D->reductionFlagsLogicalOrb = (CALbyte*) malloc(sizeof(CALbyte) * (calclmodel2D->host_CA->sizeof_pQb_array));
	calclmodel2D->reductionFlagsLogicalOri = (CALbyte*) malloc(sizeof(CALbyte) * (calclmodel2D->host_CA->sizeof_pQi_array));
	calclmodel2D->reductionFlagsLogicalOrr = (CALbyte*) malloc(sizeof(CALbyte) * (calclmodel2D->host_CA->sizeof_pQr_array));
	calclmodel2D->logicalOrsb = (CALint*) malloc(sizeof(CALint) * (calclmodel2D->host_CA->sizeof_pQb_array + 1));
	calclmodel2D->logicalOrsi = (CALint*) malloc(sizeof(CALint) * (calclmodel2D->host_CA->sizeof_pQi_array + 1));
	calclmodel2D->logicalOrsr = (CALint*) malloc(sizeof(CALint) * (calclmodel2D->host_CA->sizeof_pQr_array + 1));

	calclmodel2D->reductionFlagsLogicalXOrb = (CALbyte*) malloc(sizeof(CALbyte) * (calclmodel2D->host_CA->sizeof_pQb_array));
	calclmodel2D->reductionFlagsLogicalXOri = (CALbyte*) malloc(sizeof(CALbyte) * (calclmodel2D->host_CA->sizeof_pQi_array));
	calclmodel2D->reductionFlagsLogicalXOrr = (CALbyte*) malloc(sizeof(CALbyte) * (calclmodel2D->host_CA->sizeof_pQr_array));
	calclmodel2D->logicalXOrsb = (CALint*) malloc(sizeof(CALint) * (calclmodel2D->host_CA->sizeof_pQb_array + 1));
	calclmodel2D->logicalXOrsi = (CALint*) malloc(sizeof(CALint) * (calclmodel2D->host_CA->sizeof_pQi_array + 1));
	calclmodel2D->logicalXOrsr = (CALint*) malloc(sizeof(CALint) * (calclmodel2D->host_CA->sizeof_pQr_array + 1));

	calclmodel2D->reductionFlagsBinaryAndb = (CALbyte*) malloc(sizeof(CALbyte) * (calclmodel2D->host_CA->sizeof_pQb_array));
	calclmodel2D->reductionFlagsBinaryAndi = (CALbyte*) malloc(sizeof(CALbyte) * (calclmodel2D->host_CA->sizeof_pQi_array));
	calclmodel2D->reductionFlagsBinaryAndr = (CALbyte*) malloc(sizeof(CALbyte) * (calclmodel2D->host_CA->sizeof_pQr_array));
	calclmodel2D->binaryAndsb = (CALint*) malloc(sizeof(CALint) * (calclmodel2D->host_CA->sizeof_pQb_array + 1));
	calclmodel2D->binaryAndsi = (CALint*) malloc(sizeof(CALint) * (calclmodel2D->host_CA->sizeof_pQi_array + 1));
	calclmodel2D->binaryAndsr = (CALint*) malloc(sizeof(CALint) * (calclmodel2D->host_CA->sizeof_pQr_array + 1));

	calclmodel2D->reductionFlagsBinaryOrb = (CALbyte*) malloc(sizeof(CALbyte) * (calclmodel2D->host_CA->sizeof_pQb_array));
	calclmodel2D->reductionFlagsBinaryOri = (CALbyte*) malloc(sizeof(CALbyte) * (calclmodel2D->host_CA->sizeof_pQi_array));
	calclmodel2D->reductionFlagsBinaryOrr = (CALbyte*) malloc(sizeof(CALbyte) * (calclmodel2D->host_CA->sizeof_pQr_array));
	calclmodel2D->binaryOrsb = (CALint*) malloc(sizeof(CALint) * (calclmodel2D->host_CA->sizeof_pQb_array + 1));
	calclmodel2D->binaryOrsi = (CALint*) malloc(sizeof(CALint) * (calclmodel2D->host_CA->sizeof_pQi_array + 1));
	calclmodel2D->binaryOrsr = (CALint*) malloc(sizeof(CALint) * (calclmodel2D->host_CA->sizeof_pQr_array + 1));

	calclmodel2D->reductionFlagsBinaryXOrb = (CALbyte*) malloc(sizeof(CALbyte) * (calclmodel2D->host_CA->sizeof_pQb_array));
	calclmodel2D->reductionFlagsBinaryXOri = (CALbyte*) malloc(sizeof(CALbyte) * (calclmodel2D->host_CA->sizeof_pQi_array));
	calclmodel2D->reductionFlagsBinaryXOrr = (CALbyte*) malloc(sizeof(CALbyte) * (calclmodel2D->host_CA->sizeof_pQr_array));
	calclmodel2D->binaryXOrsb = (CALint*) malloc(sizeof(CALint) * (calclmodel2D->host_CA->sizeof_pQb_array + 1));
	calclmodel2D->binaryXOrsi = (CALint*) malloc(sizeof(CALint) * (calclmodel2D->host_CA->sizeof_pQi_array + 1));
	calclmodel2D->binaryXOrsr = (CALint*) malloc(sizeof(CALint) * (calclmodel2D->host_CA->sizeof_pQr_array + 1));

	for (int i = 0; i < calclmodel2D->host_CA->sizeof_pQb_array; i++) {
		calclmodel2D->reductionFlagsMinb[i] = CAL_FALSE;
		calclmodel2D->reductionFlagsMaxb[i] = CAL_FALSE;
		calclmodel2D->reductionFlagsSumb[i] = CAL_FALSE;
		calclmodel2D->reductionFlagsLogicalAndb[i] = CAL_FALSE;
		calclmodel2D->reductionFlagsLogicalOrb[i] = CAL_FALSE;
		calclmodel2D->reductionFlagsLogicalXOrb[i] = CAL_FALSE;
		calclmodel2D->reductionFlagsBinaryAndb[i] = CAL_FALSE;
		calclmodel2D->reductionFlagsBinaryOrb[i] = CAL_FALSE;
		calclmodel2D->reductionFlagsBinaryXOrb[i] = CAL_FALSE;
		calclmodel2D->minimab[i] = 0;
		calclmodel2D->maximab[i] = 0;
		calclmodel2D->sumsb[i] = 0;
		calclmodel2D->logicalAndsb[i] = 0;
		calclmodel2D->logicalOrsb[i] = 0;
		calclmodel2D->logicalXOrsb[i] = 0;
		calclmodel2D->binaryAndsb[i] = 0;
		calclmodel2D->binaryOrsb[i] = 0;
		calclmodel2D->binaryXOrsb[i] = 0;
	}
	for (int i = 0; i < calclmodel2D->host_CA->sizeof_pQi_array; i++) {
		calclmodel2D->reductionFlagsMini[i] = CAL_FALSE;
		calclmodel2D->reductionFlagsMaxi[i] = CAL_FALSE;
		calclmodel2D->reductionFlagsSumi[i] = CAL_FALSE;
		calclmodel2D->reductionFlagsLogicalAndi[i] = CAL_FALSE;
		calclmodel2D->reductionFlagsLogicalOri[i] = CAL_FALSE;
		calclmodel2D->reductionFlagsLogicalXOri[i] = CAL_FALSE;
		calclmodel2D->reductionFlagsBinaryAndi[i] = CAL_FALSE;
		calclmodel2D->reductionFlagsBinaryOri[i] = CAL_FALSE;
		calclmodel2D->reductionFlagsBinaryXOri[i] = CAL_FALSE;
		calclmodel2D->minimai[i] = 0;
		calclmodel2D->maximai[i] = 0;
		calclmodel2D->sumsi[i] = 0;
		calclmodel2D->logicalAndsi[i] = 0;
		calclmodel2D->logicalOrsi[i] = 0;
		calclmodel2D->logicalXOrsi[i] = 0;
		calclmodel2D->binaryAndsi[i] = 0;
		calclmodel2D->binaryOrsi[i] = 0;
		calclmodel2D->binaryXOrsi[i] = 0;

	}
	for (int i = 0; i < calclmodel2D->host_CA->sizeof_pQr_array; i++) {
		calclmodel2D->reductionFlagsMinr[i] = CAL_FALSE;
		calclmodel2D->reductionFlagsMaxr[i] = CAL_FALSE;
		calclmodel2D->reductionFlagsSumr[i] = CAL_FALSE;
		calclmodel2D->reductionFlagsLogicalAndr[i] = CAL_FALSE;
		calclmodel2D->reductionFlagsLogicalOrr[i] = CAL_FALSE;
		calclmodel2D->reductionFlagsLogicalXOrr[i] = CAL_FALSE;
		calclmodel2D->reductionFlagsBinaryAndr[i] = CAL_FALSE;
		calclmodel2D->reductionFlagsBinaryOrr[i] = CAL_FALSE;
		calclmodel2D->reductionFlagsBinaryXOrr[i] = CAL_FALSE;
		calclmodel2D->minimar[i] = 0;
		calclmodel2D->maximar[i] = 0;
		calclmodel2D->sumsr[i] = 0;
		calclmodel2D->logicalAndsr[i] = 0;
		calclmodel2D->logicalOrsr[i] = 0;
		calclmodel2D->logicalXOrsr[i] = 0;
		calclmodel2D->binaryAndsr[i] = 0;
		calclmodel2D->binaryOrsr[i] = 0;
		calclmodel2D->binaryXOrsr[i] = 0;

	}

	calclmodel2D->roundedDimensions = upperPowerOfTwo(calclmodel2D->host_CA->rows * calclmodel2D->host_CA->columns);

	calclmodel2D->context = context;
	return calclmodel2D;

}

int upperPowerOfTwo(int n) {
	int power = 1;
	while (power < n)
		power <<= 1;
	return power;
}

void calclRun2D(struct CALCLModel2D* calclmodel2D, unsigned int initialStep, unsigned maxStep) {

	//TODO Reduction
	cl_int err;

	//CALint dimReductionArrays = calclmodel2D->host_CA->sizeof_pQb_array + calclmodel2D->host_CA->sizeof_pQi_array + calclmodel2D->host_CA->sizeof_pQr_array;
	int offset = calclmodel2D->streamCompactionThreadsNum;
	int sizeCA = calclmodel2D->host_CA->rows * calclmodel2D->host_CA->columns;
	//TODO eliminare bufferFlags Urgent

	calclmodel2D->bufferMinimab = clCreateBuffer(calclmodel2D->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(CALreal) * (calclmodel2D->host_CA->sizeof_pQb_array + 1),
			calclmodel2D->minimab, &err);
	calclHandleError(err);
	calclmodel2D->bufferMiximab = clCreateBuffer(calclmodel2D->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(CALreal) * (calclmodel2D->host_CA->sizeof_pQb_array + 1),
			calclmodel2D->maximab, &err);
	calclHandleError(err);
	calclmodel2D->bufferSumb = clCreateBuffer(calclmodel2D->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(CALreal) * (calclmodel2D->host_CA->sizeof_pQb_array + 1), calclmodel2D->sumsb,
			&err);
	calclHandleError(err);
	calclmodel2D->bufferLogicalAndsb = clCreateBuffer(calclmodel2D->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(CALint) * (calclmodel2D->host_CA->sizeof_pQb_array + 1),
			calclmodel2D->logicalAndsb, &err);
	calclHandleError(err);
	calclmodel2D->bufferLogicalOrsb = clCreateBuffer(calclmodel2D->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(CALint) * (calclmodel2D->host_CA->sizeof_pQb_array + 1),
			calclmodel2D->logicalOrsb, &err);
	calclHandleError(err);
	calclmodel2D->bufferLogicalXOrsb = clCreateBuffer(calclmodel2D->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(CALint) * (calclmodel2D->host_CA->sizeof_pQb_array + 1),
			calclmodel2D->logicalXOrsb, &err);
	calclHandleError(err);
	calclmodel2D->bufferBinaryAndsb = clCreateBuffer(calclmodel2D->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(CALint) * (calclmodel2D->host_CA->sizeof_pQb_array + 1),
			calclmodel2D->binaryAndsb, &err);
	calclHandleError(err);
	calclmodel2D->bufferBinaryOrsb = clCreateBuffer(calclmodel2D->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(CALint) * (calclmodel2D->host_CA->sizeof_pQb_array + 1),
			calclmodel2D->binaryOrsb, &err);
	calclHandleError(err);
	calclmodel2D->bufferBinaryXOrsb = clCreateBuffer(calclmodel2D->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(CALint) * (calclmodel2D->host_CA->sizeof_pQb_array + 1),
			calclmodel2D->binaryXOrsb, &err);
	calclHandleError(err);

	clSetKernelArg(calclmodel2D->kernelMinReductionb, 0, sizeof(CALCLmem), &calclmodel2D->bufferMinimab);
	clSetKernelArg(calclmodel2D->kernelMinReductionb, 2, sizeof(CALCLmem), &calclmodel2D->bufferPartialMinb);
	clSetKernelArg(calclmodel2D->kernelMinReductionb, 4, sizeof(int), &sizeCA);

	clSetKernelArg(calclmodel2D->kernelMaxReductionb, 0, sizeof(CALCLmem), &calclmodel2D->bufferMiximab);
	clSetKernelArg(calclmodel2D->kernelMaxReductionb, 2, sizeof(CALCLmem), &calclmodel2D->bufferPartialMaxr);
	clSetKernelArg(calclmodel2D->kernelMaxReductionb, 4, sizeof(int), &sizeCA);

	clSetKernelArg(calclmodel2D->kernelSumReductionb, 0, sizeof(CALCLmem), &calclmodel2D->bufferSumb);
	clSetKernelArg(calclmodel2D->kernelSumReductionb, 2, sizeof(CALCLmem), &calclmodel2D->bufferPartialSumb);
	clSetKernelArg(calclmodel2D->kernelSumReductionb, 4, sizeof(int), &sizeCA);

	clSetKernelArg(calclmodel2D->kernelLogicalAndReductionb, 0, sizeof(CALCLmem), &calclmodel2D->bufferLogicalAndsb);
	clSetKernelArg(calclmodel2D->kernelLogicalAndReductionb, 2, sizeof(CALCLmem), &calclmodel2D->bufferPartialLogicalAndb);
	clSetKernelArg(calclmodel2D->kernelLogicalAndReductionb, 4, sizeof(int), &sizeCA);

	clSetKernelArg(calclmodel2D->kernelLogicalOrReductionb, 0, sizeof(CALCLmem), &calclmodel2D->bufferLogicalOrsb);
	clSetKernelArg(calclmodel2D->kernelLogicalOrReductionb, 2, sizeof(CALCLmem), &calclmodel2D->bufferPartialLogicalOrb);
	clSetKernelArg(calclmodel2D->kernelLogicalOrReductionb, 4, sizeof(int), &sizeCA);

	clSetKernelArg(calclmodel2D->kernelLogicalXOrReductionb, 0, sizeof(CALCLmem), &calclmodel2D->bufferLogicalXOrsb);
	clSetKernelArg(calclmodel2D->kernelLogicalXOrReductionb, 2, sizeof(CALCLmem), &calclmodel2D->bufferPartialLogicalXOrb);
	clSetKernelArg(calclmodel2D->kernelLogicalXOrReductionb, 4, sizeof(int), &sizeCA);

	clSetKernelArg(calclmodel2D->kernelBinaryAndReductionb, 0, sizeof(CALCLmem), &calclmodel2D->bufferBinaryAndsb);
	clSetKernelArg(calclmodel2D->kernelBinaryAndReductionb, 2, sizeof(CALCLmem), &calclmodel2D->bufferPartialBinaryAndb);
	clSetKernelArg(calclmodel2D->kernelBinaryAndReductionb, 4, sizeof(int), &sizeCA);

	clSetKernelArg(calclmodel2D->kernelBinaryOrReductionb, 0, sizeof(CALCLmem), &calclmodel2D->bufferBinaryOrsb);
	clSetKernelArg(calclmodel2D->kernelBinaryOrReductionb, 2, sizeof(CALCLmem), &calclmodel2D->bufferBinaryOrsb);
	clSetKernelArg(calclmodel2D->kernelBinaryOrReductionb, 4, sizeof(int), &sizeCA);

	clSetKernelArg(calclmodel2D->kernelBinaryXorReductionb, 0, sizeof(CALCLmem), &calclmodel2D->bufferBinaryXOrsb);
	clSetKernelArg(calclmodel2D->kernelBinaryXorReductionb, 2, sizeof(CALCLmem), &calclmodel2D->bufferBinaryXOrsb);
	clSetKernelArg(calclmodel2D->kernelBinaryXorReductionb, 4, sizeof(int), &sizeCA);

	calclmodel2D->bufferMinimai = clCreateBuffer(calclmodel2D->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(CALreal) * (calclmodel2D->host_CA->sizeof_pQi_array + 1),
			calclmodel2D->minimai, &err);
	calclHandleError(err);
	calclmodel2D->bufferMiximai = clCreateBuffer(calclmodel2D->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(CALreal) * (calclmodel2D->host_CA->sizeof_pQi_array + 1),
			calclmodel2D->maximai, &err);
	calclHandleError(err);
	calclmodel2D->bufferSumi = clCreateBuffer(calclmodel2D->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(CALreal) * (calclmodel2D->host_CA->sizeof_pQi_array + 1), calclmodel2D->sumsi,
			&err);
	calclHandleError(err);
	calclmodel2D->bufferLogicalAndsi = clCreateBuffer(calclmodel2D->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(CALint) * (calclmodel2D->host_CA->sizeof_pQi_array + 1),
			calclmodel2D->logicalAndsi, &err);
	calclHandleError(err);
	calclmodel2D->bufferLogicalOrsi = clCreateBuffer(calclmodel2D->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(CALint) * (calclmodel2D->host_CA->sizeof_pQi_array + 1),
			calclmodel2D->logicalOrsi, &err);
	calclHandleError(err);
	calclmodel2D->bufferLogicalXOrsi = clCreateBuffer(calclmodel2D->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(CALint) * (calclmodel2D->host_CA->sizeof_pQi_array + 1),
			calclmodel2D->logicalXOrsi, &err);
	calclHandleError(err);
	calclmodel2D->bufferBinaryAndsi = clCreateBuffer(calclmodel2D->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(CALint) * (calclmodel2D->host_CA->sizeof_pQi_array + 1),
			calclmodel2D->binaryAndsi, &err);
	calclHandleError(err);
	calclmodel2D->bufferBinaryOrsi = clCreateBuffer(calclmodel2D->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(CALint) * (calclmodel2D->host_CA->sizeof_pQi_array + 1),
			calclmodel2D->binaryOrsi, &err);
	calclHandleError(err);
	calclmodel2D->bufferBinaryXOrsi = clCreateBuffer(calclmodel2D->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(CALint) * (calclmodel2D->host_CA->sizeof_pQi_array + 1),
			calclmodel2D->binaryXOrsi, &err);
	calclHandleError(err);

	clSetKernelArg(calclmodel2D->kernelMinReductioni, 0, sizeof(CALCLmem), &calclmodel2D->bufferMinimai);
	clSetKernelArg(calclmodel2D->kernelMinReductioni, 2, sizeof(CALCLmem), &calclmodel2D->bufferPartialMini);
	clSetKernelArg(calclmodel2D->kernelMinReductioni, 4, sizeof(int), &sizeCA);

	clSetKernelArg(calclmodel2D->kernelMaxReductioni, 0, sizeof(CALCLmem), &calclmodel2D->bufferMiximai);
	clSetKernelArg(calclmodel2D->kernelMaxReductioni, 2, sizeof(CALCLmem), &calclmodel2D->bufferPartialMaxi);
	clSetKernelArg(calclmodel2D->kernelMaxReductioni, 4, sizeof(int), &sizeCA);

	clSetKernelArg(calclmodel2D->kernelSumReductioni, 0, sizeof(CALCLmem), &calclmodel2D->bufferSumi);
	clSetKernelArg(calclmodel2D->kernelSumReductioni, 2, sizeof(CALCLmem), &calclmodel2D->bufferPartialSumi);
	clSetKernelArg(calclmodel2D->kernelSumReductioni, 4, sizeof(int), &sizeCA);

	clSetKernelArg(calclmodel2D->kernelLogicalAndReductioni, 0, sizeof(CALCLmem), &calclmodel2D->bufferLogicalAndsi);
	clSetKernelArg(calclmodel2D->kernelLogicalAndReductioni, 2, sizeof(CALCLmem), &calclmodel2D->bufferPartialLogicalAndi);
	clSetKernelArg(calclmodel2D->kernelLogicalAndReductioni, 4, sizeof(int), &sizeCA);

	clSetKernelArg(calclmodel2D->kernelLogicalOrReductioni, 0, sizeof(CALCLmem), &calclmodel2D->bufferLogicalOrsi);
	clSetKernelArg(calclmodel2D->kernelLogicalOrReductioni, 2, sizeof(CALCLmem), &calclmodel2D->bufferPartialLogicalOri);
	clSetKernelArg(calclmodel2D->kernelLogicalOrReductioni, 4, sizeof(int), &sizeCA);

	clSetKernelArg(calclmodel2D->kernelLogicalXOrReductioni, 0, sizeof(CALCLmem), &calclmodel2D->bufferLogicalXOrsi);
	clSetKernelArg(calclmodel2D->kernelLogicalXOrReductioni, 2, sizeof(CALCLmem), &calclmodel2D->bufferPartialLogicalXOri);
	clSetKernelArg(calclmodel2D->kernelLogicalXOrReductioni, 4, sizeof(int), &sizeCA);

	clSetKernelArg(calclmodel2D->kernelBinaryAndReductioni, 0, sizeof(CALCLmem), &calclmodel2D->bufferBinaryAndsi);
	clSetKernelArg(calclmodel2D->kernelBinaryAndReductioni, 2, sizeof(CALCLmem), &calclmodel2D->bufferPartialBinaryAndi);
	clSetKernelArg(calclmodel2D->kernelBinaryAndReductioni, 4, sizeof(int), &sizeCA);

	clSetKernelArg(calclmodel2D->kernelBinaryOrReductioni, 0, sizeof(CALCLmem), &calclmodel2D->bufferBinaryOrsi);
	clSetKernelArg(calclmodel2D->kernelBinaryOrReductioni, 2, sizeof(CALCLmem), &calclmodel2D->bufferPartialBinaryOri);
	clSetKernelArg(calclmodel2D->kernelBinaryOrReductioni, 4, sizeof(int), &sizeCA);

	clSetKernelArg(calclmodel2D->kernelBinaryXorReductioni, 0, sizeof(CALCLmem), &calclmodel2D->bufferBinaryXOrsi);
	clSetKernelArg(calclmodel2D->kernelBinaryXorReductioni, 2, sizeof(CALCLmem), &calclmodel2D->bufferPartialBinaryXOri);
	clSetKernelArg(calclmodel2D->kernelBinaryXorReductioni, 4, sizeof(int), &sizeCA);

	calclmodel2D->bufferMinimar = clCreateBuffer(calclmodel2D->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(CALreal) * (calclmodel2D->host_CA->sizeof_pQr_array + 1),
			calclmodel2D->minimar, &err);
	calclHandleError(err);
	calclmodel2D->bufferMiximar = clCreateBuffer(calclmodel2D->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(CALreal) * (calclmodel2D->host_CA->sizeof_pQr_array + 1),
			calclmodel2D->maximar, &err);
	calclHandleError(err);
	calclmodel2D->bufferSumr = clCreateBuffer(calclmodel2D->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(CALreal) * (calclmodel2D->host_CA->sizeof_pQr_array + 1), calclmodel2D->sumsr,
			&err);
	calclHandleError(err);
	calclmodel2D->bufferLogicalAndsr = clCreateBuffer(calclmodel2D->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(CALint) * (calclmodel2D->host_CA->sizeof_pQr_array + 1),
			calclmodel2D->logicalAndsr, &err);
	calclHandleError(err);
	calclmodel2D->bufferLogicalOrsr = clCreateBuffer(calclmodel2D->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(CALint) * (calclmodel2D->host_CA->sizeof_pQr_array + 1),
			calclmodel2D->logicalOrsr, &err);
	calclHandleError(err);
	calclmodel2D->bufferLogicalXOrsr = clCreateBuffer(calclmodel2D->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(CALint) * (calclmodel2D->host_CA->sizeof_pQr_array + 1),
			calclmodel2D->logicalXOrsr, &err);
	calclHandleError(err);

	calclmodel2D->bufferBinaryAndsr = clCreateBuffer(calclmodel2D->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(CALint) * (calclmodel2D->host_CA->sizeof_pQr_array + 1),
			calclmodel2D->binaryAndsr, &err);
	calclHandleError(err);

	calclmodel2D->bufferBinaryOrsr = clCreateBuffer(calclmodel2D->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(CALint) * (calclmodel2D->host_CA->sizeof_pQr_array + 1),
			calclmodel2D->binaryOrsr, &err);
	calclHandleError(err);

	calclmodel2D->bufferBinaryXOrsr = clCreateBuffer(calclmodel2D->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(CALint) * (calclmodel2D->host_CA->sizeof_pQr_array + 1),
			calclmodel2D->binaryXOrsr, &err);
	calclHandleError(err);

	clSetKernelArg(calclmodel2D->kernelMinReductionr, 0, sizeof(CALCLmem), &calclmodel2D->bufferMinimar);
	clSetKernelArg(calclmodel2D->kernelMinReductionr, 2, sizeof(CALCLmem), &calclmodel2D->bufferPartialMinr);
	clSetKernelArg(calclmodel2D->kernelMinReductionr, 4, sizeof(int), &sizeCA);

	clSetKernelArg(calclmodel2D->kernelMaxReductionr, 0, sizeof(CALCLmem), &calclmodel2D->bufferMiximar);
	clSetKernelArg(calclmodel2D->kernelMaxReductionr, 2, sizeof(CALCLmem), &calclmodel2D->bufferPartialMaxr);
	clSetKernelArg(calclmodel2D->kernelMaxReductionr, 4, sizeof(int), &sizeCA);

	clSetKernelArg(calclmodel2D->kernelSumReductionr, 0, sizeof(CALCLmem), &calclmodel2D->bufferSumr);
	clSetKernelArg(calclmodel2D->kernelSumReductionr, 2, sizeof(CALCLmem), &calclmodel2D->bufferPartialSumr);
	clSetKernelArg(calclmodel2D->kernelSumReductionr, 4, sizeof(int), &sizeCA);

	clSetKernelArg(calclmodel2D->kernelLogicalAndReductionr, 0, sizeof(CALCLmem), &calclmodel2D->bufferLogicalAndsr);
	clSetKernelArg(calclmodel2D->kernelLogicalAndReductionr, 2, sizeof(CALCLmem), &calclmodel2D->bufferPartialLogicalAndr);
	clSetKernelArg(calclmodel2D->kernelLogicalAndReductionr, 4, sizeof(int), &sizeCA);

	clSetKernelArg(calclmodel2D->kernelLogicalOrReductionr, 0, sizeof(CALCLmem), &calclmodel2D->bufferLogicalOrsr);
	clSetKernelArg(calclmodel2D->kernelLogicalOrReductionr, 2, sizeof(CALCLmem), &calclmodel2D->bufferPartialLogicalOrr);
	clSetKernelArg(calclmodel2D->kernelLogicalOrReductionr, 4, sizeof(int), &sizeCA);

	clSetKernelArg(calclmodel2D->kernelLogicalXOrReductionr, 0, sizeof(CALCLmem), &calclmodel2D->bufferLogicalXOrsr);
	clSetKernelArg(calclmodel2D->kernelLogicalXOrReductionr, 2, sizeof(CALCLmem), &calclmodel2D->bufferPartialLogicalXOrr);
	clSetKernelArg(calclmodel2D->kernelLogicalXOrReductionr, 4, sizeof(int), &sizeCA);

	clSetKernelArg(calclmodel2D->kernelBinaryAndReductionr, 0, sizeof(CALCLmem), &calclmodel2D->bufferBinaryAndsr);
	clSetKernelArg(calclmodel2D->kernelBinaryAndReductionr, 2, sizeof(CALCLmem), &calclmodel2D->bufferPartialBinaryAndr);
	clSetKernelArg(calclmodel2D->kernelBinaryAndReductionr, 4, sizeof(int), &sizeCA);

	clSetKernelArg(calclmodel2D->kernelBinaryOrReductionr, 0, sizeof(CALCLmem), &calclmodel2D->bufferBinaryOrsr);
	clSetKernelArg(calclmodel2D->kernelBinaryOrReductionr, 2, sizeof(CALCLmem), &calclmodel2D->bufferPartialBinaryOrr);
	clSetKernelArg(calclmodel2D->kernelBinaryOrReductionr, 4, sizeof(int), &sizeCA);

	clSetKernelArg(calclmodel2D->kernelBinaryXorReductionr, 0, sizeof(CALCLmem), &calclmodel2D->bufferBinaryXOrsr);
	clSetKernelArg(calclmodel2D->kernelBinaryXorReductionr, 2, sizeof(CALCLmem), &calclmodel2D->bufferPartialBinaryXOrr);
	clSetKernelArg(calclmodel2D->kernelBinaryXorReductionr, 4, sizeof(int), &sizeCA);

	if (calclmodel2D->kernelInitSubstates != NULL)
		calclSetReductionParameters2D(calclmodel2D, &calclmodel2D->kernelInitSubstates);
	if (calclmodel2D->kernelStopCondition != NULL)
		calclSetReductionParameters2D(calclmodel2D, &calclmodel2D->kernelStopCondition);
	if (calclmodel2D->kernelSteering != NULL)
		calclSetReductionParameters2D(calclmodel2D, &calclmodel2D->kernelSteering);

	int i = 0;

	for (i = 0; i < calclmodel2D->elementaryProcessesNum; i++) {
		calclSetReductionParameters2D(calclmodel2D, &calclmodel2D->elementaryProcesses[i]);
	}

	CALbyte stop;
	size_t * threadNumMax = (size_t*) malloc(sizeof(size_t) * 2);
	threadNumMax[0] = calclmodel2D->host_CA->rows;
	threadNumMax[1] = calclmodel2D->host_CA->columns;
	size_t * singleStepThreadNum;
	int dimNum;

	if (calclmodel2D->opt == CAL_NO_OPT) {
		singleStepThreadNum = (size_t*) malloc(sizeof(size_t) * 2);
		singleStepThreadNum[0] = threadNumMax[0];
		singleStepThreadNum[1] = threadNumMax[1];
		dimNum = 2;
	} else {
		singleStepThreadNum = (size_t*) malloc(sizeof(size_t));
		singleStepThreadNum[0] = calclmodel2D->host_CA->A.size_current;
		dimNum = 1;
	}

	if (calclmodel2D->kernelInitSubstates != NULL)
		calclKernelCall2D(calclmodel2D, calclmodel2D->kernelInitSubstates, 1, threadNumMax, NULL);

	//TODO call update

	calclmodel2D->steps = initialStep;
	while (calclmodel2D->steps <= (int) maxStep || maxStep == CAL_RUN_LOOP) {
		stop = calclSingleStep2D(calclmodel2D, singleStepThreadNum, dimNum);
		if (stop == CAL_TRUE)
			break;
	}
	calclGetSubstatesDeviceToHost2D(calclmodel2D);
	free(threadNumMax);

	free(singleStepThreadNum);
}

void calclComputeReduction2Di(struct CALCLModel2D * calclmodel2D, int numSubstate, enum REDUCTION_OPERATION operation, int rounded) {

	CALCLqueue queue = calclmodel2D->queue;
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
			clSetKernelArg(calclmodel2D->kernelMaxReductioni, 3, sizeof(CALint), &offset);
			clSetKernelArg(calclmodel2D->kernelMaxReductioni, 5, sizeof(CALint), &count);
			err = clEnqueueNDRangeKernel(queue, calclmodel2D->kernelMaxReductioni, 1,
			NULL, &tmpThreads, NULL, 0, NULL, NULL);
			calclHandleError(err);
			break;
		case REDUCTION_MIN:
			clSetKernelArg(calclmodel2D->kernelMinReductioni, 3, sizeof(CALint), &offset);
			clSetKernelArg(calclmodel2D->kernelMinReductioni, 5, sizeof(CALint), &count);
			err = clEnqueueNDRangeKernel(queue, calclmodel2D->kernelMinReductioni, 1, NULL, &tmpThreads, NULL, 0, NULL, NULL);
			calclHandleError(err);
			break;
		case REDUCTION_SUM:
			clSetKernelArg(calclmodel2D->kernelSumReductioni, 3, sizeof(CALint), &offset);
			clSetKernelArg(calclmodel2D->kernelSumReductioni, 5, sizeof(CALint), &count);
			err = clEnqueueNDRangeKernel(queue, calclmodel2D->kernelSumReductioni, 1,
			NULL, &tmpThreads, NULL, 0, NULL, NULL);
			calclHandleError(err);
			break;
		case REDUCTION_PROD:
			clSetKernelArg(calclmodel2D->kernelProdReductioni, 3, sizeof(CALint), &offset);
			clSetKernelArg(calclmodel2D->kernelProdReductioni, 5, sizeof(CALint), &count);
			err = clEnqueueNDRangeKernel(queue, calclmodel2D->kernelProdReductioni, 1,
			NULL, &tmpThreads, NULL, 0, NULL, NULL);
			calclHandleError(err);
			break;
		case REDUCTION_LOGICAL_AND:
			clSetKernelArg(calclmodel2D->kernelLogicalAndReductioni, 3, sizeof(CALint), &offset);
			clSetKernelArg(calclmodel2D->kernelLogicalAndReductioni, 5, sizeof(CALint), &count);
			err = clEnqueueNDRangeKernel(queue, calclmodel2D->kernelLogicalAndReductioni, 1,
			NULL, &tmpThreads, NULL, 0, NULL, NULL);
			calclHandleError(err);
			break;
		case REDUCTION_BINARY_AND:
			clSetKernelArg(calclmodel2D->kernelBinaryAndReductioni, 3, sizeof(CALint), &offset);
			clSetKernelArg(calclmodel2D->kernelBinaryAndReductioni, 5, sizeof(CALint), &count);
			err = clEnqueueNDRangeKernel(queue, calclmodel2D->kernelBinaryAndReductioni, 1,
			NULL, &tmpThreads, NULL, 0, NULL, NULL);
			calclHandleError(err);
			break;
		case REDUCTION_LOGICAL_OR:
			clSetKernelArg(calclmodel2D->kernelLogicalOrReductioni, 3, sizeof(CALint), &offset);
			clSetKernelArg(calclmodel2D->kernelLogicalOrReductioni, 5, sizeof(CALint), &count);
			err = clEnqueueNDRangeKernel(queue, calclmodel2D->kernelLogicalOrReductioni, 1,
			NULL, &tmpThreads, NULL, 0, NULL, NULL);
			calclHandleError(err);
			break;
		case REDUCTION_BINARY_OR:
			clSetKernelArg(calclmodel2D->kernelBinaryOrReductioni, 3, sizeof(CALint), &offset);
			clSetKernelArg(calclmodel2D->kernelBinaryOrReductioni, 5, sizeof(CALint), &count);
			err = clEnqueueNDRangeKernel(queue, calclmodel2D->kernelBinaryOrReductioni, 1,
			NULL, &tmpThreads, NULL, 0, NULL, NULL);
			calclHandleError(err);
			break;
		case REDUCTION_LOGICAL_XOR:
			clSetKernelArg(calclmodel2D->kernelLogicalXOrReductioni, 3, sizeof(CALint), &offset);
			clSetKernelArg(calclmodel2D->kernelLogicalXOrReductioni, 5, sizeof(CALint), &count);
			err = clEnqueueNDRangeKernel(queue, calclmodel2D->kernelLogicalXOrReductioni, 1,
			NULL, &tmpThreads, NULL, 0, NULL, NULL);
			calclHandleError(err);
			break;
		case REDUCTION_BINARY_XOR:
			clSetKernelArg(calclmodel2D->kernelBinaryXorReductioni, 3, sizeof(CALint), &offset);
			clSetKernelArg(calclmodel2D->kernelBinaryXorReductioni, 5, sizeof(CALint), &count);
			err = clEnqueueNDRangeKernel(queue, calclmodel2D->kernelBinaryXorReductioni, 1,
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

void calclComputeReduction2Db(struct CALCLModel2D * calclmodel2D, int numSubstate, enum REDUCTION_OPERATION operation, int rounded) {

	CALCLqueue queue = calclmodel2D->queue;
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
			clSetKernelArg(calclmodel2D->kernelMaxReductionb, 3, sizeof(CALint), &offset);
			clSetKernelArg(calclmodel2D->kernelMaxReductionb, 5, sizeof(CALint), &count);
			err = clEnqueueNDRangeKernel(queue, calclmodel2D->kernelMaxReductionb, 1,
			NULL, &tmpThreads, NULL, 0, NULL, NULL);
			calclHandleError(err);
			break;
		case REDUCTION_MIN:
			clSetKernelArg(calclmodel2D->kernelMinReductionb, 3, sizeof(CALint), &offset);
			clSetKernelArg(calclmodel2D->kernelMinReductionb, 5, sizeof(CALint), &count);
			err = clEnqueueNDRangeKernel(queue, calclmodel2D->kernelMinReductionb, 1, NULL, &tmpThreads, NULL, 0, NULL, NULL);
			calclHandleError(err);
			break;
		case REDUCTION_SUM:
			clSetKernelArg(calclmodel2D->kernelSumReductionb, 3, sizeof(CALint), &offset);
			clSetKernelArg(calclmodel2D->kernelSumReductionb, 5, sizeof(CALint), &count);
			err = clEnqueueNDRangeKernel(queue, calclmodel2D->kernelSumReductionb, 1,
			NULL, &tmpThreads, NULL, 0, NULL, NULL);
			calclHandleError(err);
			break;
		case REDUCTION_PROD:
			clSetKernelArg(calclmodel2D->kernelProdReductionb, 3, sizeof(CALint), &offset);
			clSetKernelArg(calclmodel2D->kernelProdReductionb, 5, sizeof(CALint), &count);
			err = clEnqueueNDRangeKernel(queue, calclmodel2D->kernelProdReductionb, 1,
			NULL, &tmpThreads, NULL, 0, NULL, NULL);
			calclHandleError(err);
			break;
		case REDUCTION_LOGICAL_AND:
			clSetKernelArg(calclmodel2D->kernelLogicalAndReductionb, 3, sizeof(CALint), &offset);
			clSetKernelArg(calclmodel2D->kernelLogicalAndReductionb, 5, sizeof(CALint), &count);
			err = clEnqueueNDRangeKernel(queue, calclmodel2D->kernelLogicalAndReductionb, 1,
			NULL, &tmpThreads, NULL, 0, NULL, NULL);
			calclHandleError(err);
			break;
		case REDUCTION_BINARY_AND:
			clSetKernelArg(calclmodel2D->kernelBinaryAndReductionb, 3, sizeof(CALint), &offset);
			clSetKernelArg(calclmodel2D->kernelBinaryAndReductionb, 5, sizeof(CALint), &count);
			err = clEnqueueNDRangeKernel(queue, calclmodel2D->kernelBinaryAndReductionb, 1,
			NULL, &tmpThreads, NULL, 0, NULL, NULL);
			calclHandleError(err);
			break;
		case REDUCTION_LOGICAL_OR:
			clSetKernelArg(calclmodel2D->kernelLogicalOrReductionb, 3, sizeof(CALint), &offset);
			clSetKernelArg(calclmodel2D->kernelLogicalOrReductionb, 5, sizeof(CALint), &count);
			err = clEnqueueNDRangeKernel(queue, calclmodel2D->kernelLogicalOrReductionb, 1,
			NULL, &tmpThreads, NULL, 0, NULL, NULL);
			calclHandleError(err);
			break;
		case REDUCTION_BINARY_OR:
			clSetKernelArg(calclmodel2D->kernelBinaryOrReductionb, 3, sizeof(CALint), &offset);
			clSetKernelArg(calclmodel2D->kernelBinaryOrReductionb, 5, sizeof(CALint), &count);
			err = clEnqueueNDRangeKernel(queue, calclmodel2D->kernelBinaryOrReductionb, 1,
			NULL, &tmpThreads, NULL, 0, NULL, NULL);
			calclHandleError(err);
			break;
		case REDUCTION_LOGICAL_XOR:
			clSetKernelArg(calclmodel2D->kernelLogicalXOrReductionb, 3, sizeof(CALint), &offset);
			clSetKernelArg(calclmodel2D->kernelLogicalXOrReductionb, 5, sizeof(CALint), &count);
			err = clEnqueueNDRangeKernel(queue, calclmodel2D->kernelLogicalXOrReductionb, 1,
			NULL, &tmpThreads, NULL, 0, NULL, NULL);
			calclHandleError(err);
			break;
		case REDUCTION_BINARY_XOR:
			clSetKernelArg(calclmodel2D->kernelBinaryXorReductionb, 3, sizeof(CALint), &offset);
			clSetKernelArg(calclmodel2D->kernelBinaryXorReductionb, 5, sizeof(CALint), &count);
			err = clEnqueueNDRangeKernel(queue, calclmodel2D->kernelBinaryXorReductionb, 1,
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

void calclComputeReduction2Dr(struct CALCLModel2D * calclmodel2D, int numSubstate, enum REDUCTION_OPERATION operation, int rounded) {

	CALCLqueue queue = calclmodel2D->queue;
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
			clSetKernelArg(calclmodel2D->kernelMaxReductionr, 3, sizeof(CALint), &offset);
			clSetKernelArg(calclmodel2D->kernelMaxReductionr, 5, sizeof(CALint), &count);
			err = clEnqueueNDRangeKernel(queue, calclmodel2D->kernelMaxReductionr, 1,
			NULL, &tmpThreads, NULL, 0, NULL, NULL);
			calclHandleError(err);
			break;
		case REDUCTION_MIN:
			clSetKernelArg(calclmodel2D->kernelMinReductionr, 3, sizeof(CALint), &offset);
			clSetKernelArg(calclmodel2D->kernelMinReductionr, 5, sizeof(CALint), &count);
			err = clEnqueueNDRangeKernel(queue, calclmodel2D->kernelMinReductionr, 1, NULL, &tmpThreads, NULL, 0, NULL, NULL);
			calclHandleError(err);
			break;
		case REDUCTION_SUM:
			clSetKernelArg(calclmodel2D->kernelSumReductionr, 3, sizeof(CALint), &offset);
			clSetKernelArg(calclmodel2D->kernelSumReductionr, 5, sizeof(CALint), &count);
			err = clEnqueueNDRangeKernel(queue, calclmodel2D->kernelSumReductionr, 1,
			NULL, &tmpThreads, NULL, 0, NULL, NULL);
			calclHandleError(err);
			break;
		case REDUCTION_PROD:
			clSetKernelArg(calclmodel2D->kernelProdReductionr, 3, sizeof(CALint), &offset);
			clSetKernelArg(calclmodel2D->kernelProdReductionr, 5, sizeof(CALint), &count);
			err = clEnqueueNDRangeKernel(queue, calclmodel2D->kernelProdReductionr, 1,
			NULL, &tmpThreads, NULL, 0, NULL, NULL);
			calclHandleError(err);
			break;
		case REDUCTION_LOGICAL_AND:
			clSetKernelArg(calclmodel2D->kernelLogicalAndReductionr, 3, sizeof(CALint), &offset);
			clSetKernelArg(calclmodel2D->kernelLogicalAndReductionr, 5, sizeof(CALint), &count);
			err = clEnqueueNDRangeKernel(queue, calclmodel2D->kernelLogicalAndReductionr, 1,
			NULL, &tmpThreads, NULL, 0, NULL, NULL);
			calclHandleError(err);
			break;
		case REDUCTION_BINARY_AND:
			clSetKernelArg(calclmodel2D->kernelBinaryAndReductionr, 3, sizeof(CALint), &offset);
			clSetKernelArg(calclmodel2D->kernelBinaryAndReductionr, 5, sizeof(CALint), &count);
			err = clEnqueueNDRangeKernel(queue, calclmodel2D->kernelBinaryAndReductionr, 1,
			NULL, &tmpThreads, NULL, 0, NULL, NULL);
			calclHandleError(err);
			break;
		case REDUCTION_LOGICAL_OR:
			clSetKernelArg(calclmodel2D->kernelLogicalOrReductionr, 3, sizeof(CALint), &offset);
			clSetKernelArg(calclmodel2D->kernelLogicalOrReductionr, 5, sizeof(CALint), &count);
			err = clEnqueueNDRangeKernel(queue, calclmodel2D->kernelLogicalOrReductionr, 1,
			NULL, &tmpThreads, NULL, 0, NULL, NULL);
			calclHandleError(err);
			break;
		case REDUCTION_BINARY_OR:
			clSetKernelArg(calclmodel2D->kernelBinaryOrReductionr, 3, sizeof(CALint), &offset);
			clSetKernelArg(calclmodel2D->kernelBinaryOrReductionr, 5, sizeof(CALint), &count);
			err = clEnqueueNDRangeKernel(queue, calclmodel2D->kernelBinaryOrReductionr, 1,
			NULL, &tmpThreads, NULL, 0, NULL, NULL);
			calclHandleError(err);
			break;
		case REDUCTION_LOGICAL_XOR:
			clSetKernelArg(calclmodel2D->kernelLogicalXOrReductionr, 3, sizeof(CALint), &offset);
			clSetKernelArg(calclmodel2D->kernelLogicalXOrReductionr, 5, sizeof(CALint), &count);
			err = clEnqueueNDRangeKernel(queue, calclmodel2D->kernelLogicalXOrReductionr, 1,
			NULL, &tmpThreads, NULL, 0, NULL, NULL);
			calclHandleError(err);
			break;
		case REDUCTION_BINARY_XOR:
			clSetKernelArg(calclmodel2D->kernelBinaryXorReductionr, 3, sizeof(CALint), &offset);
			clSetKernelArg(calclmodel2D->kernelBinaryXorReductionr, 5, sizeof(CALint), &count);
			err = clEnqueueNDRangeKernel(queue, calclmodel2D->kernelBinaryXorReductionr, 1,
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

void calclExecuteReduction2D(struct CALCLModel2D* calclmodel2D, int rounded) {

	int i = 0;
	CALint dimReductionArrays = calclmodel2D->host_CA->sizeof_pQb_array + calclmodel2D->host_CA->sizeof_pQi_array + calclmodel2D->host_CA->sizeof_pQr_array;
	cl_int err;
	size_t tmp = calclmodel2D->host_CA->rows * calclmodel2D->host_CA->columns;

	for (i = 0; i < calclmodel2D->host_CA->sizeof_pQb_array; i++) {
		printf("OK byte \n");

		if (calclmodel2D->reductionFlagsMinb[i]) {
			clSetKernelArg(calclmodel2D->kernelMinReductionb, 1, sizeof(CALint), &i);
			clSetKernelArg(calclmodel2D->kernelMinCopyb, 2, sizeof(CALint), &i);
			err = clEnqueueNDRangeKernel(calclmodel2D->queue, calclmodel2D->kernelMinCopyb, 1, NULL, &tmp, NULL, 0, NULL, NULL);
			calclHandleError(err);
			calclComputeReduction2Db(calclmodel2D, i, REDUCTION_MIN, rounded);
		}
		if (calclmodel2D->reductionFlagsMaxb[i]) {
			clSetKernelArg(calclmodel2D->kernelMaxReductionb, 1, sizeof(CALint), &i);
			clSetKernelArg(calclmodel2D->kernelMaxCopyb, 2, sizeof(CALint), &i);
			err = clEnqueueNDRangeKernel(calclmodel2D->queue, calclmodel2D->kernelMaxCopyb, 1, NULL, &tmp, NULL, 0, NULL, NULL);
			calclHandleError(err);
			calclComputeReduction2Db(calclmodel2D, i, REDUCTION_MAX, rounded);
		}
		if (calclmodel2D->reductionFlagsSumb[i]) {
			clSetKernelArg(calclmodel2D->kernelSumReductionb, 1, sizeof(CALint), &i);
			clSetKernelArg(calclmodel2D->kernelSumCopyb, 2, sizeof(CALint), &i);
			err = clEnqueueNDRangeKernel(calclmodel2D->queue, calclmodel2D->kernelSumCopyb, 1, NULL, &tmp, NULL, 0, NULL, NULL);
			calclHandleError(err);
			calclComputeReduction2Db(calclmodel2D, i, REDUCTION_SUM, rounded);
		}
		if (calclmodel2D->reductionFlagsLogicalAndb[i]) {
			clSetKernelArg(calclmodel2D->kernelLogicalAndReductionb, 1, sizeof(CALint), &i);
			clSetKernelArg(calclmodel2D->kernelLogicalAndCopyb, 2, sizeof(CALint), &i);
			err = clEnqueueNDRangeKernel(calclmodel2D->queue, calclmodel2D->kernelLogicalAndCopyb, 1, NULL, &tmp, NULL, 0, NULL, NULL);
			calclHandleError(err);
			calclComputeReduction2Db(calclmodel2D, i, REDUCTION_LOGICAL_AND, rounded);
		}
		if (calclmodel2D->reductionFlagsLogicalOrb[i]) {
			clSetKernelArg(calclmodel2D->kernelLogicalOrReductionb, 1, sizeof(CALint), &i);
			clSetKernelArg(calclmodel2D->kernelLogicalOrCopyb, 2, sizeof(CALint), &i);
			err = clEnqueueNDRangeKernel(calclmodel2D->queue, calclmodel2D->kernelLogicalOrCopyb, 1, NULL, &tmp, NULL, 0, NULL, NULL);
			calclHandleError(err);
			calclComputeReduction2Db(calclmodel2D, i, REDUCTION_LOGICAL_OR, rounded);
		}
		if (calclmodel2D->reductionFlagsLogicalXOrb[i]) {
			clSetKernelArg(calclmodel2D->kernelLogicalXOrReductionb, 1, sizeof(CALint), &i);
			clSetKernelArg(calclmodel2D->kernelLogicalXOrCopyb, 2, sizeof(CALint), &i);
			err = clEnqueueNDRangeKernel(calclmodel2D->queue, calclmodel2D->kernelLogicalXOrCopyb, 1, NULL, &tmp, NULL, 0, NULL, NULL);
			calclHandleError(err);
			calclComputeReduction2Db(calclmodel2D, i, REDUCTION_LOGICAL_XOR, rounded);
		}
		if (calclmodel2D->reductionFlagsBinaryAndb[i]) {
			clSetKernelArg(calclmodel2D->kernelBinaryAndReductionb, 1, sizeof(CALint), &i);
			clSetKernelArg(calclmodel2D->kernelBinaryAndCopyb, 2, sizeof(CALint), &i);
			err = clEnqueueNDRangeKernel(calclmodel2D->queue, calclmodel2D->kernelBinaryAndCopyb, 1, NULL, &tmp, NULL, 0, NULL, NULL);
			calclHandleError(err);
			calclComputeReduction2Db(calclmodel2D, i, REDUCTION_BINARY_AND, rounded);
		}
		if (calclmodel2D->reductionFlagsBinaryOrb[i]) {
			clSetKernelArg(calclmodel2D->kernelBinaryOrReductionb, 1, sizeof(CALint), &i);
			clSetKernelArg(calclmodel2D->kernelBinaryOrCopyb, 2, sizeof(CALint), &i);
			err = clEnqueueNDRangeKernel(calclmodel2D->queue, calclmodel2D->kernelBinaryOrCopyb, 1, NULL, &tmp, NULL, 0, NULL, NULL);
			calclHandleError(err);
			calclComputeReduction2Db(calclmodel2D, i, REDUCTION_BINARY_OR, rounded);
		}
		if (calclmodel2D->reductionFlagsBinaryXOrb[i]) {
			clSetKernelArg(calclmodel2D->kernelBinaryXorReductionb, 1, sizeof(CALint), &i);
			clSetKernelArg(calclmodel2D->kernelBinaryXOrCopyb, 2, sizeof(CALint), &i);
			err = clEnqueueNDRangeKernel(calclmodel2D->queue, calclmodel2D->kernelBinaryXOrCopyb, 1, NULL, &tmp, NULL, 0, NULL, NULL);
			calclHandleError(err);
			calclComputeReduction2Db(calclmodel2D, i, REDUCTION_BINARY_XOR, rounded);
		}
	}


	for (i = 0; i < calclmodel2D->host_CA->sizeof_pQi_array; i++) {
		printf("OK int \n");
		if (calclmodel2D->reductionFlagsMini[i]) {
			clSetKernelArg(calclmodel2D->kernelMinReductioni, 1, sizeof(CALint), &i);
			clSetKernelArg(calclmodel2D->kernelMinCopyi, 2, sizeof(CALint), &i);
			err = clEnqueueNDRangeKernel(calclmodel2D->queue, calclmodel2D->kernelMinCopyi, 1, NULL, &tmp, NULL, 0, NULL, NULL);
			calclHandleError(err);
			calclComputeReduction2Di(calclmodel2D, i, REDUCTION_MIN, rounded);
		}
		if (calclmodel2D->reductionFlagsMaxi[i]) {
			clSetKernelArg(calclmodel2D->kernelMaxReductioni, 1, sizeof(CALint), &i);
			clSetKernelArg(calclmodel2D->kernelMaxCopyi, 2, sizeof(CALint), &i);
			err = clEnqueueNDRangeKernel(calclmodel2D->queue, calclmodel2D->kernelMaxCopyi, 1, NULL, &tmp, NULL, 0, NULL, NULL);
			calclHandleError(err);
			calclComputeReduction2Di(calclmodel2D, i, REDUCTION_MAX, rounded);
		}
		if (calclmodel2D->reductionFlagsSumi[i]) {
			clSetKernelArg(calclmodel2D->kernelSumReductioni, 1, sizeof(CALint), &i);
			clSetKernelArg(calclmodel2D->kernelSumCopyi, 2, sizeof(CALint), &i);
			err = clEnqueueNDRangeKernel(calclmodel2D->queue, calclmodel2D->kernelSumCopyi, 1, NULL, &tmp, NULL, 0, NULL, NULL);
			calclHandleError(err);
			calclComputeReduction2Di(calclmodel2D, i, REDUCTION_SUM, rounded);
		}
		if (calclmodel2D->reductionFlagsLogicalAndi[i]) {
			clSetKernelArg(calclmodel2D->kernelLogicalAndReductioni, 1, sizeof(CALint), &i);
			clSetKernelArg(calclmodel2D->kernelLogicalAndCopyi, 2, sizeof(CALint), &i);
			err = clEnqueueNDRangeKernel(calclmodel2D->queue, calclmodel2D->kernelLogicalAndCopyi, 1, NULL, &tmp, NULL, 0, NULL, NULL);
			calclHandleError(err);
			calclComputeReduction2Di(calclmodel2D, i, REDUCTION_LOGICAL_AND, rounded);
		}
		if (calclmodel2D->reductionFlagsLogicalOri[i]) {
			clSetKernelArg(calclmodel2D->kernelLogicalOrReductioni, 1, sizeof(CALint), &i);
			clSetKernelArg(calclmodel2D->kernelLogicalOrCopyi, 2, sizeof(CALint), &i);
			err = clEnqueueNDRangeKernel(calclmodel2D->queue, calclmodel2D->kernelLogicalOrCopyi, 1, NULL, &tmp, NULL, 0, NULL, NULL);
			calclHandleError(err);
			calclComputeReduction2Di(calclmodel2D, i, REDUCTION_LOGICAL_OR, rounded);
		}
		if (calclmodel2D->reductionFlagsLogicalXOri[i]) {
			clSetKernelArg(calclmodel2D->kernelLogicalXOrReductioni, 1, sizeof(CALint), &i);
			clSetKernelArg(calclmodel2D->kernelLogicalXOrCopyi, 2, sizeof(CALint), &i);
			err = clEnqueueNDRangeKernel(calclmodel2D->queue, calclmodel2D->kernelLogicalXOrCopyi, 1, NULL, &tmp, NULL, 0, NULL, NULL);
			calclHandleError(err);
			calclComputeReduction2Di(calclmodel2D, i, REDUCTION_LOGICAL_XOR, rounded);
		}
		if (calclmodel2D->reductionFlagsBinaryAndi[i]) {
			clSetKernelArg(calclmodel2D->kernelBinaryAndReductioni, 1, sizeof(CALint), &i);
			clSetKernelArg(calclmodel2D->kernelBinaryAndCopyi, 2, sizeof(CALint), &i);
			err = clEnqueueNDRangeKernel(calclmodel2D->queue, calclmodel2D->kernelBinaryAndCopyi, 1, NULL, &tmp, NULL, 0, NULL, NULL);
			calclHandleError(err);
			calclComputeReduction2Di(calclmodel2D, i, REDUCTION_BINARY_AND, rounded);
		}
		if (calclmodel2D->reductionFlagsBinaryOri[i]) {
			clSetKernelArg(calclmodel2D->kernelBinaryOrReductioni, 1, sizeof(CALint), &i);
			clSetKernelArg(calclmodel2D->kernelBinaryOrCopyi, 2, sizeof(CALint), &i);
			err = clEnqueueNDRangeKernel(calclmodel2D->queue, calclmodel2D->kernelBinaryOrCopyi, 1, NULL, &tmp, NULL, 0, NULL, NULL);
			calclHandleError(err);
			calclComputeReduction2Di(calclmodel2D, i, REDUCTION_BINARY_OR, rounded);
		}
		if (calclmodel2D->reductionFlagsBinaryXOri[i]) {
			clSetKernelArg(calclmodel2D->kernelBinaryXorReductioni, 1, sizeof(CALint), &i);
			clSetKernelArg(calclmodel2D->kernelBinaryXOrCopyi, 2, sizeof(CALint), &i);
			err = clEnqueueNDRangeKernel(calclmodel2D->queue, calclmodel2D->kernelBinaryXOrCopyi, 1, NULL, &tmp, NULL, 0, NULL, NULL);
			calclHandleError(err);
			calclComputeReduction2Di(calclmodel2D, i, REDUCTION_BINARY_XOR, rounded);
		}
	}

	for (i = 0; i < calclmodel2D->host_CA->sizeof_pQr_array; i++) {

		if (calclmodel2D->reductionFlagsMinr[i]) {
			clSetKernelArg(calclmodel2D->kernelMinReductionr, 1, sizeof(CALint), &i);
			clSetKernelArg(calclmodel2D->kernelMinCopyr, 2, sizeof(CALint), &i);
			err = clEnqueueNDRangeKernel(calclmodel2D->queue, calclmodel2D->kernelMinCopyr, 1, NULL, &tmp, NULL, 0, NULL, NULL);
			calclHandleError(err);
			calclComputeReduction2Dr(calclmodel2D, i, REDUCTION_MIN, rounded);
		}
		if (calclmodel2D->reductionFlagsMaxr[i]) {
			clSetKernelArg(calclmodel2D->kernelMaxReductionr, 1, sizeof(CALint), &i);
			clSetKernelArg(calclmodel2D->kernelMaxCopyr, 2, sizeof(CALint), &i);
			err = clEnqueueNDRangeKernel(calclmodel2D->queue, calclmodel2D->kernelMaxCopyr, 1, NULL, &tmp, NULL, 0, NULL, NULL);
			calclHandleError(err);
			calclComputeReduction2Dr(calclmodel2D, i, REDUCTION_MAX, rounded);
		}
		if (calclmodel2D->reductionFlagsSumr[i]) {
			clSetKernelArg(calclmodel2D->kernelSumReductionr, 1, sizeof(CALint), &i);
			clSetKernelArg(calclmodel2D->kernelSumCopyr, 2, sizeof(CALint), &i);
			err = clEnqueueNDRangeKernel(calclmodel2D->queue, calclmodel2D->kernelSumCopyr, 1, NULL, &tmp, NULL, 0, NULL, NULL);
			calclHandleError(err);
			calclComputeReduction2Dr(calclmodel2D, i, REDUCTION_SUM, rounded);
		}
		if (calclmodel2D->reductionFlagsLogicalAndr[i]) {
			clSetKernelArg(calclmodel2D->kernelLogicalAndReductionr, 1, sizeof(CALint), &i);
			clSetKernelArg(calclmodel2D->kernelLogicalAndCopyr, 2, sizeof(CALint), &i);
			err = clEnqueueNDRangeKernel(calclmodel2D->queue, calclmodel2D->kernelLogicalAndCopyr, 1, NULL, &tmp, NULL, 0, NULL, NULL);
			calclHandleError(err);
			calclComputeReduction2Dr(calclmodel2D, i, REDUCTION_LOGICAL_AND, rounded);
		}
		if (calclmodel2D->reductionFlagsLogicalOrr[i]) {
			clSetKernelArg(calclmodel2D->kernelLogicalOrReductionr, 1, sizeof(CALint), &i);
			clSetKernelArg(calclmodel2D->kernelLogicalOrCopyr, 2, sizeof(CALint), &i);
			err = clEnqueueNDRangeKernel(calclmodel2D->queue, calclmodel2D->kernelLogicalOrCopyr, 1, NULL, &tmp, NULL, 0, NULL, NULL);
			calclHandleError(err);
			calclComputeReduction2Dr(calclmodel2D, i, REDUCTION_LOGICAL_OR, rounded);
		}
		if (calclmodel2D->reductionFlagsLogicalXOrr[i]) {
			clSetKernelArg(calclmodel2D->kernelLogicalXOrReductionr, 1, sizeof(CALint), &i);
			clSetKernelArg(calclmodel2D->kernelLogicalXOrCopyr, 2, sizeof(CALint), &i);
			err = clEnqueueNDRangeKernel(calclmodel2D->queue, calclmodel2D->kernelLogicalXOrCopyr, 1, NULL, &tmp, NULL, 0, NULL, NULL);
			calclHandleError(err);
			calclComputeReduction2Dr(calclmodel2D, i, REDUCTION_LOGICAL_XOR, rounded);
		}
		if (calclmodel2D->reductionFlagsBinaryAndr[i]) {
			clSetKernelArg(calclmodel2D->kernelBinaryAndReductionr, 1, sizeof(CALint), &i);
			clSetKernelArg(calclmodel2D->kernelBinaryAndCopyr, 2, sizeof(CALint), &i);
			err = clEnqueueNDRangeKernel(calclmodel2D->queue, calclmodel2D->kernelBinaryAndCopyr, 1, NULL, &tmp, NULL, 0, NULL, NULL);
			calclHandleError(err);
			calclComputeReduction2Dr(calclmodel2D, i, REDUCTION_BINARY_AND, rounded);
		}
		if (calclmodel2D->reductionFlagsBinaryOrr[i]) {
			clSetKernelArg(calclmodel2D->kernelBinaryOrReductionr, 1, sizeof(CALint), &i);
			clSetKernelArg(calclmodel2D->kernelBinaryOrCopyr, 2, sizeof(CALint), &i);
			err = clEnqueueNDRangeKernel(calclmodel2D->queue, calclmodel2D->kernelBinaryOrCopyr, 1, NULL, &tmp, NULL, 0, NULL, NULL);
			calclHandleError(err);
			calclComputeReduction2Dr(calclmodel2D, i, REDUCTION_BINARY_OR, rounded);
		}
		if (calclmodel2D->reductionFlagsBinaryXOrr[i]) {
			clSetKernelArg(calclmodel2D->kernelBinaryXorReductionr, 1, sizeof(CALint), &i);
			clSetKernelArg(calclmodel2D->kernelBinaryXOrCopyr, 2, sizeof(CALint), &i);
			err = clEnqueueNDRangeKernel(calclmodel2D->queue, calclmodel2D->kernelBinaryXOrCopyr, 1, NULL, &tmp, NULL, 0, NULL, NULL);
			calclHandleError(err);
			calclComputeReduction2Dr(calclmodel2D, i, REDUCTION_BINARY_XOR, rounded);
		}
	}

}

CALbyte calclSingleStep2D(struct CALCLModel2D* calclmodel2D, size_t * threadsNum, int dimNum) {

	CALbyte activeCells = calclmodel2D->opt == CAL_OPT_ACTIVE_CELLS;
	int j;
	if (activeCells == CAL_TRUE) {
		for (j = 0; j < calclmodel2D->elementaryProcessesNum; j++) {

			calclKernelCall2D(calclmodel2D, calclmodel2D->elementaryProcesses[j], dimNum, threadsNum,
			NULL);
			calclComputeStreamCompaction2D(calclmodel2D);
			calclResizeThreadsNum2D(calclmodel2D, threadsNum);
			calclKernelCall2D(calclmodel2D, calclmodel2D->kernelUpdateSubstate, dimNum, threadsNum, NULL);

		}

		calclExecuteReduction2D(calclmodel2D, calclmodel2D->roundedDimensions);

		if (calclmodel2D->kernelSteering != NULL) {
			calclKernelCall2D(calclmodel2D, calclmodel2D->kernelSteering, dimNum, threadsNum, NULL);
			calclKernelCall2D(calclmodel2D, calclmodel2D->kernelUpdateSubstate, dimNum, threadsNum, NULL);
		}

	} else {
		for (j = 0; j < calclmodel2D->elementaryProcessesNum; j++) {
			calclKernelCall2D(calclmodel2D, calclmodel2D->elementaryProcesses[j], dimNum, threadsNum,
			NULL);
			copySubstatesBuffers2D(calclmodel2D);

		}

		calclExecuteReduction2D(calclmodel2D, calclmodel2D->roundedDimensions);

		if (calclmodel2D->kernelSteering != NULL) {
			calclKernelCall2D(calclmodel2D, calclmodel2D->kernelSteering, dimNum, threadsNum, NULL);
			copySubstatesBuffers2D(calclmodel2D);
		}

	}
	if (calclmodel2D->cl_update_substates != NULL && calclmodel2D->steps % calclmodel2D->callbackSteps == 0) {
		calclGetSubstatesDeviceToHost2D(calclmodel2D);
		calclmodel2D->cl_update_substates(calclmodel2D->host_CA);
	}

	calclmodel2D->steps++;

	if (calclmodel2D->kernelStopCondition != NULL) {
		return checkStopCondition2D(calclmodel2D, dimNum, threadsNum);
	}

	return CAL_FALSE;

}

FILE * file;
void calclKernelCall2D(struct CALCLModel2D* calclmodel2D, CALCLkernel ker, int numDim, size_t * dimSize, size_t * localDimSize) {

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

void calclComputeStreamCompaction2D(struct CALCLModel2D * calclmodel2D) {
	CALCLqueue queue = calclmodel2D->queue;
	calclKernelCall2D(calclmodel2D, calclmodel2D->kernelComputeCounts, 1, &calclmodel2D->streamCompactionThreadsNum, NULL);
	cl_int err;
	int iterations = calclmodel2D->streamCompactionThreadsNum;
	size_t tmpThreads = iterations;
	int i;

	for (i = iterations / 2; i > 0; i /= 2) {
		tmpThreads = i;
		err = clEnqueueNDRangeKernel(queue, calclmodel2D->kernelUpSweep, 1,
		NULL, &tmpThreads, NULL, 0, NULL, NULL);
		calclHandleError(err);
	}

	iterations = calclmodel2D->streamCompactionThreadsNum;

	for (i = 1; i < iterations; i *= 2) {
		tmpThreads = i;
		err = clEnqueueNDRangeKernel(queue, calclmodel2D->kernelDownSweep, 1,
		NULL, &tmpThreads, NULL, 0, NULL, NULL);
		calclHandleError(err);
	}

	calclKernelCall2D(calclmodel2D, calclmodel2D->kernelCompact, 1, &calclmodel2D->streamCompactionThreadsNum, NULL);
}

void calclSetKernelArgs2D(CALCLkernel * kernel, CALCLmem * args, cl_uint numArgs) {
	unsigned int i;
	for (i = 0; i < numArgs; i++)
		clSetKernelArg(*kernel, MODEL_ARGS_NUM + i, sizeof(CALCLmem), &args[i]);
}

void calclAddStopConditionFunc2D(struct CALCLModel2D * calclmodel2D, CALCLkernel * kernel) {
	calclmodel2D->kernelStopCondition = *kernel;
	calclSetModelParameters2D(calclmodel2D, kernel);
}

void calclAddInitFunc2D(struct CALCLModel2D* calclmodel2D, CALCLkernel * kernel) {
	calclmodel2D->kernelInitSubstates = *kernel;
	calclSetModelParameters2D(calclmodel2D, kernel);
}

void calclAddSteeringFunc2D(struct CALCLModel2D* calclmodel2D, CALCLkernel * kernel) {
	calclmodel2D->kernelSteering = *kernel;
	calclSetModelParameters2D(calclmodel2D, kernel);
}

void calclBackToHostFunc2D(struct CALCLModel2D* calclmodel2D, void (*cl_update_substates)(struct CALModel2D*), int callbackSteps) {
	calclmodel2D->cl_update_substates = cl_update_substates;
	calclmodel2D->callbackSteps = callbackSteps;
}

void calclAddElementaryProcess2D(struct CALCLModel2D* calclmodel2D, CALCLkernel * kernel) {

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

	calclSetModelParameters2D(calclmodel2D, kernel);
}

void calclFinalize2D(struct CALCLModel2D * calclmodel2D) {

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

	if (calclmodel2D->elementaryProcessesNum > 0)
		free(calclmodel2D->elementaryProcesses);
	free(calclmodel2D);

}

CALCLprogram calclLoadProgram2D(CALCLcontext context, CALCLdevice device, char* path_user_kernel, char* path_user_include) {
	//printf("OK1\n");
	char* u = " -cl-denorms-are-zero -cl-finite-math-only ";
	char* pathOpenCALCL = getenv("OPENCALCL_PATH");
	//printf("pathOpenCALCL %s \n", pathOpenCALCL);
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
	char* tmp2 = (char*) malloc(sizeof(char) * (strlen(pathOpenCALCL) + strlen(KERNEL_SOURCE_DIR)) + 1);
	strcpy(tmp2, pathOpenCALCL);
	strcat(tmp2, KERNEL_SOURCE_DIR);
	//	printf("source %s \n", tmp2);

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

int calclSetKernelArg2D(CALCLkernel* kernel, cl_uint arg_index, size_t arg_size, const void *arg_value) {
	return clSetKernelArg(*kernel, MODEL_ARGS_NUM + arg_index, arg_size, arg_value);
}

void calclAddMinReduction2Di(struct CALCLModel2D * calclmodel2D, int numSubstate) {
	calclmodel2D->reductionFlagsMini[numSubstate] = CAL_TRUE;
}
void calclAddMinReduction2Db(struct CALCLModel2D * calclmodel2D, int numSubstate) {
	calclmodel2D->reductionFlagsMinb[numSubstate] = CAL_TRUE;
}
void calclAddMinReduction2Dr(struct CALCLModel2D * calclmodel2D, int numSubstate) {
	calclmodel2D->reductionFlagsMinr[numSubstate] = CAL_TRUE;
}

void calclAddMaxReduction2Di(struct CALCLModel2D * calclmodel2D, int numSubstate) {
	calclmodel2D->reductionFlagsMaxi[numSubstate] = CAL_TRUE;
}
void calclAddMaxReduction2Db(struct CALCLModel2D * calclmodel2D, int numSubstate) {
	calclmodel2D->reductionFlagsMaxb[numSubstate] = CAL_TRUE;
}
void calclAddMaxReduction2Dr(struct CALCLModel2D * calclmodel2D, int numSubstate) {
	calclmodel2D->reductionFlagsMaxr[numSubstate] = CAL_TRUE;
}

void calclAddSumReduction2Di(struct CALCLModel2D * calclmodel2D, int numSubstate) {
	calclmodel2D->reductionFlagsSumi[numSubstate] = CAL_TRUE;
}
void calclAddSumReduction2Db(struct CALCLModel2D * calclmodel2D, int numSubstate) {
	calclmodel2D->reductionFlagsSumb[numSubstate] = CAL_TRUE;
}
void calclAddSumReduction2Dr(struct CALCLModel2D * calclmodel2D, int numSubstate) {
	calclmodel2D->reductionFlagsSumr[numSubstate] = CAL_TRUE;
}

void calclAddLogicalAndReduction2Di(struct CALCLModel2D * calclmodel2D, int numSubstate) {
	calclmodel2D->reductionFlagsLogicalAndi[numSubstate] = CAL_TRUE;
}
void calclAddLogicalAndReduction2Db(struct CALCLModel2D * calclmodel2D, int numSubstate) {
	calclmodel2D->reductionFlagsLogicalAndb[numSubstate] = CAL_TRUE;
}
void calclAddLogicalAndReduction2Dr(struct CALCLModel2D * calclmodel2D, int numSubstate) {
	calclmodel2D->reductionFlagsLogicalAndr[numSubstate] = CAL_TRUE;
}

void calclAddLogicalOrReduction2Di(struct CALCLModel2D * calclmodel2D, int numSubstate) {
	calclmodel2D->reductionFlagsLogicalOri[numSubstate] = CAL_TRUE;
}
void calclAddLogicalOrReduction2Db(struct CALCLModel2D * calclmodel2D, int numSubstate) {
	calclmodel2D->reductionFlagsLogicalOrb[numSubstate] = CAL_TRUE;
}
void calclAddLogicalOrReduction2Dr(struct CALCLModel2D * calclmodel2D, int numSubstate) {
	calclmodel2D->reductionFlagsLogicalOrr[numSubstate] = CAL_TRUE;
}

void calclAddLogicalXOrReduction2Di(struct CALCLModel2D * calclmodel2D, int numSubstate) {
	calclmodel2D->reductionFlagsLogicalXOri[numSubstate] = CAL_TRUE;
}
void calclAddLogicalXOrReduction2Db(struct CALCLModel2D * calclmodel2D, int numSubstate) {
	calclmodel2D->reductionFlagsLogicalXOrb[numSubstate] = CAL_TRUE;
}
void calclAddLogicalXOrReduction2Dr(struct CALCLModel2D * calclmodel2D, int numSubstate) {
	calclmodel2D->reductionFlagsLogicalXOrr[numSubstate] = CAL_TRUE;
}

void calclAddBinaryAndReduction2Di(struct CALCLModel2D * calclmodel2D, int numSubstate) {
	calclmodel2D->reductionFlagsBinaryAndi[numSubstate] = CAL_TRUE;
}
void calclAddBinaryAndReduction2Db(struct CALCLModel2D * calclmodel2D, int numSubstate) {
	calclmodel2D->reductionFlagsBinaryAndb[numSubstate] = CAL_TRUE;
}
void calclAddBinaryAndReduction2Dr(struct CALCLModel2D * calclmodel2D, int numSubstate) {
	calclmodel2D->reductionFlagsBinaryAndr[numSubstate] = CAL_TRUE;
}

void calclAddBinaryOrReduction2Di(struct CALCLModel2D * calclmodel2D, int numSubstate) {
	calclmodel2D->reductionFlagsBinaryOri[numSubstate] = CAL_TRUE;
}
void calclAddBinaryOrReduction2Db(struct CALCLModel2D * calclmodel2D, int numSubstate) {
	calclmodel2D->reductionFlagsBinaryOrb[numSubstate] = CAL_TRUE;
}
void calclAddBinaryOrReduction2Dr(struct CALCLModel2D * calclmodel2D, int numSubstate) {
	calclmodel2D->reductionFlagsBinaryOrr[numSubstate] = CAL_TRUE;
}

void calclAddBinaryXOrReduction2Di(struct CALCLModel2D * calclmodel2D, int numSubstate) {
	calclmodel2D->reductionFlagsBinaryXOri[numSubstate] = CAL_TRUE;
}
void calclAddBinaryXOrReduction2Db(struct CALCLModel2D * calclmodel2D, int numSubstate) {
	calclmodel2D->reductionFlagsBinaryXOrb[numSubstate] = CAL_TRUE;
}
void calclAddBinaryXOrReduction2Dr(struct CALCLModel2D * calclmodel2D, int numSubstate) {
	calclmodel2D->reductionFlagsBinaryXOrr[numSubstate] = CAL_TRUE;
}
