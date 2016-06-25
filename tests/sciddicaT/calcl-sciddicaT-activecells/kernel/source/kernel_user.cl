/*
 * Copyright (c) 2016 OpenCALTeam (https://github.com/OpenCALTeam),
 * Telesio Research Group,
 * Department of Mathematics and Computer Science,
 * University of Calabria, Italy.
 *
 * This file is part of OpenCAL (Open Computing Abstraction Layer).
 *
 * OpenCAL is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3 of
 * the License, or (at your option) any later version.
 *
 * OpenCAL is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with OpenCAL. If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef __OPENCL_VERSION__
#define __kernel
#define __global
#define __local
#define get_global_id (int)
#endif

#include <kernel.h>

//first elementary process
__kernel void sciddicaT_flows_computation(MODEL_DEFINITION2D, __global CALParameterr * Pepsilon, __global CALParameterr * Pr

) {

	initThreads2D();

	__global CALbyte * activeCellsFlags = get_active_cells_flags();
	CALint cols_ = get_columns();

	int i = getX();
	int j = getY();

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

	if (calGet2Dr(MODEL2D, H, i, j) <= eps)
		return;

	m = calGet2Dr(MODEL2D, H, i, j) - eps;
	u[0] = calGet2Dr(MODEL2D, Z , i, j) + eps;
	for (n = 1; n < sizeOfX_; n++) {
		z = calGetX2Dr(MODEL2D,Z, i, j, n);
		h = calGetX2Dr(MODEL2D,H, i, j, n);
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

	for (n = 1; n < sizeOfX_; n++) {
		if (eliminated_cells[n])
			calSet2Dr(MODEL2D, n-1, i, j, 0.0);
		else
			calSet2Dr(MODEL2D, n-1, i, j,(average - u[n]) * (*Pr));
	}
}

__kernel void sciddicaT_width_update(MODEL_DEFINITION2D) {

	initThreads2D();

	__global CALbyte * activeCellsFlags = get_active_cells_flags();
	CALint cols_ = get_columns();

	int i = getX();
	int j = getY();

	CALreal h_next;
	CALint n;

	h_next = calGet2Dr(MODEL2D, H, i, j);

	for (n = 1; n < get_neighborhoods_size(); n++) {
		h_next += ( calGetX2Dr(MODEL2D, NUMBER_OF_OUTFLOWS-n, i, j, n) - calGet2Dr(MODEL2D, n-1, i, j) );
	}
	calSet2Dr(MODEL2D, H, i, j, h_next);

}

__kernel void sciddicaTSteering(MODEL_DEFINITION2D) {

	initThreads2D();

	__global CALbyte * activeCellsFlags = get_active_cells_flags();
	CALint cols_ = get_columns();
	CALint rows_ = get_rows();

	int i = getX();
	int j = getY();

	int dim = cols_ * rows_;
	int s;
	for (s = 0; s < NUMBER_OF_OUTFLOWS; ++s)
		calInitSubstate2Dr(MODEL2D, s, i, j, 0);

}
