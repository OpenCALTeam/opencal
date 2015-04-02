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

__kernel void calclkernelComputeCounts3D(CALint dim, __global CALbyte * flags, __global CALint * counts, __global CALint * offsets, __global CALbyte * diff) {

	int id = get_global_id (0);

	if(diff[id] == CAL_FALSE){
		offsets[id] = counts[id];
		return;
	}

	diff[id] = CAL_FALSE;

	int threadsNum = get_global_size (0);
	int chunk = ceil((CALreal)dim/threadsNum);
	int startOffset = chunk * id;
	int count = 0;
	int i;

	if (startOffset > dim) {
		offsets[id] = 0;
		counts[id] = 0;
		return;
	}

	if (id == dim/chunk)
		chunk = dim - startOffset;

	for (i = 0; i < chunk; i++) {
		count += flags[startOffset + i];
	}
	offsets[id] = count;
	counts[id] = count;

}

__kernel void calclkernelUpSweep3D( __global CALint * offsets, CALint numElements) {

	int offset = numElements / get_global_size (0);
	int id = get_global_id (0);
	int ai = offset * (2 * id + 1) - 1;
	int bi = offset * (2 * id + 2) - 1;
	offsets[bi] += offsets[ai];

}
__kernel void calclkernelDownSweep3D( __global CALint * offsets, CALint numElements) {

	int numThreads = get_global_size (0);
	int offset = numElements / numThreads;
	int id = get_global_id (0);
	if (numThreads == 1)
		offsets[numElements * 2 - 1] = 0;
	int ai = offset * (2 * id + 1) - 1;
	int bi = offset * (2 * id + 2) - 1;
	int tmp = offsets[ai];
	offsets[ai] = offsets[bi];
	offsets[bi] += tmp;

}
__kernel void calclkernelCompact3D(
		CALint dim,
		CALint rows,
		CALint cols,
		__global CALbyte * flags,
		__global CALint * numActiveCells,
		__global struct CALCell3D * activeCells,
		__global CALint * counts,
		__global CALint * offsets
		) {

	int id = get_global_id (0);
	int threadsNum = get_global_size (0);
	int cellIndex = 0;
	int chunk = ceil((CALreal)dim/threadsNum);
	int startOffset = chunk * id;
	int startPoint = 0;

	if (id == threadsNum - 1){
		chunk = dim - startOffset;
		(*numActiveCells) = offsets[id]+counts[id];
	}

	if (counts[id] == 0)
		return;

	if (id > 0)
		startPoint = offsets[id];

	int i;
	int layerDim = rows*cols;
	for (i = 0; i < chunk; i++) {
		cellIndex = startOffset + i;
		if (flags[cellIndex] == CAL_TRUE) {
			activeCells[startPoint].i = (cellIndex % layerDim) / cols;
			activeCells[startPoint].j = cellIndex % cols;
			activeCells[startPoint].k = cellIndex / layerDim;
			startPoint++;
		}
	}
}
