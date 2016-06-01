// The SciddicaT debris flows XCA transition function kernels

#include <kernel.h>

//first elementary process
__kernel void flowsComputation(__CALCL_MODEL_2D, __global CALParameterr * Pepsilon, __global CALParameterr * Pr)
{
	calclActiveThreadCheck2D();

	int threadID = calclGlobalRow();
	int i = calclActiveCellRow(threadID);
	int j = calclActiveCellColumn(threadID);

	CALbyte eliminated_cells[5] = { CAL_FALSE, CAL_FALSE, CAL_FALSE, CAL_FALSE, CAL_FALSE };
	CALbyte again;
	CALint cells_count;
	CALreal average;
	CALreal m;
	CALreal u[5];
	CALint n;
	CALreal z, h;
	CALint sizeOfX_ = calclGetNeighborhoodSize();
	CALParameterr eps = *Pepsilon;

	if (calclGet2Dr(MODEL_2D,H, i, j) <= eps)
		return;

	m = calclGet2Dr(MODEL_2D,H, i, j) - eps;
	u[0] = calclGet2Dr(MODEL_2D, Z, i, j) + eps;
	for (n = 1; n < sizeOfX_; n++) {
		z = calclGetX2Dr(MODEL_2D, Z, i, j, n);
		h = calclGetX2Dr(MODEL_2D, H, i, j, n);
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
			calclSet2Dr(MODEL_2D, n-1, i, j, 0.0);
		else {
			calclSet2Dr(MODEL_2D, n-1, i, j, (average - u[n]) * (*Pr));
			calclAddActiveCellX2D(MODEL_2D, i, j, n);
		}
	}
}


__kernel void widthUpdate(__CALCL_MODEL_2D)
{
	calclActiveThreadCheck2D();

	CALint neighborhoodSize = calclGetNeighborhoodSize();

	int threadID = calclGlobalRow();
	int i = calclActiveCellRow(threadID);
	int j = calclActiveCellColumn(threadID);

	CALreal h_next;
	CALint n;

	h_next = calclGet2Dr(MODEL_2D,H, i, j);


	for (n = 1; n < neighborhoodSize; n++)
		h_next += ( calclGetX2Dr(MODEL_2D, NUMBER_OF_OUTFLOWS-n, i, j, n) - calclGet2Dr(MODEL_2D, n-1, i, j) );

	calclSet2Dr(MODEL_2D, H, i, j, h_next);
}

__kernel void removeInactiveCells(__CALCL_MODEL_2D, __global CALParameterr * Pepsilon)
{
	calclActiveThreadCheck2D();

	int threadID = calclGlobalRow();
	int i = calclActiveCellRow(threadID);
	int j = calclActiveCellColumn(threadID);

	if (calclGet2Dr(MODEL_2D, H, i, j) <= *Pepsilon)
		calclRemoveActiveCell2D(MODEL_2D,i,j);
}

__kernel void steering(__CALCL_MODEL_2D)
{
	calclActiveThreadCheck2D();

	int threadID = calclGlobalRow();

	int dim = calclGetColumns() * calclGetRows();
	int i;
	for (i = 0; i < NUMBER_OF_OUTFLOWS; ++i)
		calclInitSubstateActiveCell2Dr(MODEL_2D, i, threadID, 0);
}
