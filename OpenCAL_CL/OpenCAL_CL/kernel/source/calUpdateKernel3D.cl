#include <cal3DActive.h>
#ifndef __OPENCL_VERSION__
#define __kernel
#define __global
#define __local
#define get_global_id (int)
#define get_global_size (int)
#define CLK_LOCAL_MEM_FENCE
#define barrier(int)
#endif

__kernel void calclkernelUpdateSubstates3D(
		CALint columns,
		CALint rows,
		CALint layers,
		CALint byteSubstateNum,
		CALint intSubstateNum,
		CALint realSubstateNum,
		__global CALbyte * currentByteSubstate,
		__global CALint * currentIntSubstate,
		__global CALreal * currentRealSubstate,
		__global CALbyte * nextByteSubstate,
		__global CALint * nextIntSubstate,
		__global CALreal * nextRealSubstate,
		__global struct CALCell3D * activeCells,
		__global CALint * activeCellsNum
) {

	int k;

	int threadID = get_global_id (0);

	if (threadID >= *activeCellsNum)
		return;

	for (k = 0; k < byteSubstateNum; k++)
		calCopyBufferActiveCells3Db(nextByteSubstate + rows * columns * layers * k, currentByteSubstate + rows * columns * layers * k, columns,rows, activeCells, threadID);
	for (k = 0; k < intSubstateNum; k++)
		calCopyBufferActiveCells3Di(nextIntSubstate + rows * columns * layers * k, currentIntSubstate + rows * columns * layers * k, columns,rows, activeCells, threadID);
	for (k = 0; k < realSubstateNum; k++)
		calCopyBufferActiveCells3Dr(nextRealSubstate + rows * columns * layers * k, currentRealSubstate + rows * columns * layers * k, columns,rows, activeCells, threadID);
}


