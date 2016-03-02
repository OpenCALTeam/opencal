#ifndef __OPENCL_VERSION__
#define __kernel
#define __global
#define __local
#define get_global_id (int)
#endif

#include <kernel.h>

//first elementary process
__kernel void sciddicaT_flows_computation(__CALCL_MODEL_2D, __global CALParameterr * Pepsilon, __global CALParameterr * Pr

) {

	initActiveThreads2D();

	int threadID = getRow();
	int i = getActiveCellRow(threadID);
	int j = getActiveCellCol(threadID);

	CALbyte eliminated_cells[5] = { CAL_FALSE, CAL_FALSE, CAL_FALSE, CAL_FALSE, CAL_FALSE };
	CALbyte again;
	CALint cells_count;
	CALreal average;
	CALreal m;
	CALreal u[5];
	CALint n;
	CALreal z, h;
	CALint sizeOfX_ = get_neighborhoods_size();
	CALParameterr eps = *Pepsilon;

	if (calGet2Dr(MODEL_2D,H, i, j) <= eps)
		return;

	m = calGet2Dr(MODEL_2D,H, i, j) - eps;
	u[0] = calGet2Dr(MODEL_2D, Z, i, j) + eps;
	for (n = 1; n < sizeOfX_; n++) {
		z = calGetX2Dr(MODEL_2D, Z, i, j, n);
		h = calGetX2Dr(MODEL_2D, H, i, j, n);
		u[n] = z + h;
	}

	do {
		again = CAL_FALSE;
		average = m;
		cells_count = 0;

		for (n = 0; n < sizeOfX_; n++)
			if (!eliminated_cells[n]) {
				average += u[n];
				cells_count++;
			}

		if (cells_count != 0)
			average /= cells_count;

		for (n = 0; n < sizeOfX_; n++)
			if ((average <= u[n]) && (!eliminated_cells[n])) {
				eliminated_cells[n] = CAL_TRUE;
				again = CAL_TRUE;
			}

	} while (again);

	//__global CALreal * fsubstate;

	for (n = 1; n < sizeOfX_; n++) {
		if (eliminated_cells[n])
			calSet2Dr(MODEL_2D, n-1, i, j, 0.0);
		else {
			calSet2Dr(MODEL_2D, n-1, i, j, (average - u[n]) * (*Pr));
			calAddActiveCellX2D(MODEL_2D, i, j, n);
		}
	}
}


__kernel void sciddicaT_width_update(__CALCL_MODEL_2D) {

	initActiveThreads2D();

	CALint neighborhoodSize = get_neighborhoods_size();

	int threadID = getRow();
	int i = getActiveCellRow(threadID);
	int j = getActiveCellCol(threadID);

	CALreal h_next;
	CALint n;

	h_next = calGet2Dr(MODEL_2D,H, i, j);


	for (n = 1; n < neighborhoodSize; n++)
		h_next += ( calGetX2Dr(MODEL_2D, NUMBER_OF_OUTFLOWS-n, i, j, n) - calGet2Dr(MODEL_2D, n-1, i, j) );

	calSet2Dr(MODEL_2D, H, i, j, h_next);

}

__kernel void sciddicaT_remove_inactive_cells(__CALCL_MODEL_2D, __global CALParameterr * Pepsilon) {

	initActiveThreads2D();

	int threadID = getRow();
	int i = getActiveCellRow(threadID);
	int j = getActiveCellCol(threadID);

	if (calGet2Dr(MODEL_2D, H, i, j) <= *Pepsilon)
		calRemoveActiveCell2D(MODEL_2D,i,j);
}

__kernel void sciddicaTSteering(__CALCL_MODEL_2D) {

	initActiveThreads2D();

	int threadID = getRow();

	int dim = get_columns() * get_rows();
	int i;
	for (i = 0; i < NUMBER_OF_OUTFLOWS; ++i)
		calInitSubstateActiveCell2Dr(MODEL_2D, i, threadID, 0);

}

