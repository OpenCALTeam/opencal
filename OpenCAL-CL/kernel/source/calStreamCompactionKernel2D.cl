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

#include <OpenCAL-CL/calcl2D.h>
#ifndef __OPENCL_VERSION__
#define __kernel
#define __global
#define __local
#define get_global_id (int)
#define get_global_size (int)
#define CLK_LOCAL_MEM_FENCE
#define barrier(int)
#endif

__kernel void calclkernelComputeCounts2D(CALint dim, __global CALbyte * flags, __global CALint * counts, __global CALint * offsets, __global CALbyte * diff) {

	int id = get_global_id (0);

	if (diff[id] == CAL_FALSE) {
		offsets[id] = counts[id];
		return;
	}

	diff[id] = CAL_FALSE;

	int threadsNum = get_global_size (0);
	int chunk = ceil((double) dim / threadsNum);
	int startOffset = chunk * id;
	int count = 0;
	int i;

	if (startOffset > dim) {
		offsets[id] = 0;
		counts[id] = 0;
		return;
	}

	if (id == dim / chunk)
		chunk = dim - startOffset;

	for (i = 0; i < chunk; i++) {
		count += flags[startOffset + i];
	}
	offsets[id] = count;
	counts[id] = count;

}

__kernel void calclkernelUpSweep2D( __global CALint * offsets, CALint numElements) {

	int offset = numElements / get_global_size (0);
	int id = get_global_id (0);
	int ai = offset * (2 * id + 1) - 1;
	int bi = offset * (2 * id + 2) - 1;
	offsets[bi] += offsets[ai];

}
__kernel void calclkernelDownSweep2D( __global CALint * offsets, CALint numElements) {

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
__kernel void calclkernelCompact2D(
		CALint dim,
		CALint cols,
		__global CALbyte * flags,
		__global CALint * numActiveCells,
		__global struct CALCell2D * activeCells,
		__global CALint * counts,
		__global CALint * offsets
		) {

	int id = get_global_id (0);
	int threadsNum = get_global_size (0);
	int cellIndex = 0;
//	int chunk = dim / threadsNum;
	int chunk = ceil((double) dim / threadsNum);
	int startOffset = chunk * id;
	int startPoint = 0;

	if (id == threadsNum - 1) {
		chunk = dim - startOffset;
		(*numActiveCells) = offsets[id] + counts[id];
	}

	if (counts[id] == 0)
		return;

	if (id > 0)
		startPoint = offsets[id];

	int i;
	for (i = 0; i < chunk; i++) {
		cellIndex = startOffset + i;
		if (flags[cellIndex] == CAL_TRUE) {
			activeCells[startPoint].i = cellIndex / cols;
			activeCells[startPoint].j = cellIndex % cols;
			startPoint++;
		}
	}
}
