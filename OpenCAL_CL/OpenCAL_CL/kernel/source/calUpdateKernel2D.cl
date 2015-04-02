#include <cal2DActive.h>
#ifndef __OPENCL_VERSION__
#define __kernel
#define __global
#define __local
#define get_global_id (int)
#define get_global_size (int)
#define CLK_LOCAL_MEM_FENCE
#define barrier(int)
#endif

__kernel void calclkernelUpdateSubstates2D(
		CALint columns,
		CALint rows,
		CALint byteSubstateNum,
		CALint intSubstateNum,
		CALint realSubstateNum,
		__global CALbyte * currentByteSubstate,
		__global CALint * currentIntSubstate,
		__global CALreal * currentRealSubstate,
		__global CALbyte * nextByteSubstate,
		__global CALint * nextIntSubstate,
		__global CALreal * nextRealSubstate,
		__global struct CALCell2D * activeCells,
		__global CALint * activeCellsNum
) {

	int k;

	int threadID = get_global_id (0);

	if(threadID>=*activeCellsNum)
		return;


	for (k = 0; k < byteSubstateNum; k++)
		calCopyBufferActiveCells2Db(nextByteSubstate + rows * columns * k, currentByteSubstate + rows * columns * k, columns, activeCells, threadID);
	for (k = 0; k < intSubstateNum; k++)
		calCopyBufferActiveCells2Di(nextIntSubstate + rows * columns * k, currentIntSubstate + rows * columns * k, columns, activeCells, threadID);
	for (k = 0; k < realSubstateNum; k++)
		calCopyBufferActiveCells2Dr(nextRealSubstate + rows * columns * k, currentRealSubstate + rows * columns * k, columns, activeCells, threadID);
}


