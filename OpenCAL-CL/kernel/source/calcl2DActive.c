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

#include "OpenCAL-CL/calcl2DActive.h"

void calclAddActiveCell2D(__CALCL_MODEL_2D, int i, int j) {

	if (CAL_FALSE == calclGetBufferElement2D(calclGetActiveCellsFlags(), calclGetColumns(), i, j)) {
		calclSetBufferElement2D(calclGetActiveCellsFlags(), calclGetColumns(), i, j, CAL_TRUE);
		CALint chunkNum = (i * calclGetColumns() + j) / CALCLchunk;
		CALCLdiff[chunkNum] = CAL_TRUE;
	}

}
void calclAddActiveCellX2D(__CALCL_MODEL_2D, int i, int j, int n) {

	if ((calclGetNeighborhoodId() == CAL_HEXAGONAL_NEIGHBORHOOD_2D && j % 2 == 1) || (calclGetNeighborhoodId() == CAL_HEXAGONAL_NEIGHBORHOOD_ALT_2D && i % 2 == 1))
		n += CAL_HEXAGONAL_SHIFT;

	CALint ix;
	CALint jx;

	if (calclGetBoundaryCondition() == CAL_SPACE_FLAT) {
		ix = i + calclGetNeighborhood()[n].i;
		jx = j + calclGetNeighborhood()[n].j;
	} else {
		ix = calclGetToroidalX(i + calclGetNeighborhood()[n].i, calclGetRows());
		jx = calclGetToroidalX(j + calclGetNeighborhood()[n].j, calclGetColumns());
	}

	if (CAL_FALSE == calclGetBufferElement2D(calclGetActiveCellsFlags(), calclGetColumns(), ix, jx)) {
		calclSetBufferElement2D(calclGetActiveCellsFlags(), calclGetColumns(), ix, jx, CAL_TRUE);
		CALint chunkNum = (ix * calclGetColumns() + jx) / CALCLchunk;
		CALCLdiff[chunkNum] = CAL_TRUE;
	}
}

void calclRemoveActiveCell2D(__CALCL_MODEL_2D, int i, int j) {

	if (CAL_TRUE == calclGetBufferElement2D(calclGetActiveCellsFlags(), calclGetColumns(), i, j)) {
		calclSetBufferElement2D(calclGetActiveCellsFlags(), calclGetColumns(), i, j, CAL_FALSE);
		CALint chunkNum = (i * calclGetColumns() + j) / CALCLchunk;
		CALCLdiff[chunkNum] = CAL_TRUE;
	}

}

void calclInitSubstateActiveCell2Db(__CALCL_MODEL_2D, int substateNum, int n, CALbyte value) {
	__global CALbyte * current = calclGetCurrentSubstate2Db(MODEL_2D, substateNum);
	__global CALbyte * next = calclGetNextSubstate2Db(MODEL_2D, substateNum);
	calclSetBufferActiveCells2Db(current, calclGetColumns(), value, calclGetActiveCells(), n);
	calclSetBufferActiveCells2Db(next, calclGetColumns(), value, calclGetActiveCells(), n);
}
void calclInitSubstateActiveCell2Di(__CALCL_MODEL_2D, int substateNum, int n, CALint value) {
	__global CALint * current = calclGetCurrentSubstate2Di(MODEL_2D, substateNum);
	__global CALint * next = calclGetNextSubstate2Di(MODEL_2D, substateNum);
	calclSetBufferActiveCells2Di(current, calclGetColumns(), value, calclGetActiveCells(), n);
	calclSetBufferActiveCells2Di(next, calclGetColumns(), value, calclGetActiveCells(), n);
}
void calclInitSubstateActiveCell2Dr(__CALCL_MODEL_2D, int substateNum, int n, CALreal value) {
	__global CALreal * current = calclGetCurrentSubstate2Dr(MODEL_2D, substateNum);
	__global CALreal * next = calclGetNextSubstate2Dr(MODEL_2D, substateNum);
	calclSetBufferActiveCells2Dr(current, calclGetColumns(), value, calclGetActiveCells(), n);
	calclSetBufferActiveCells2Dr(next, calclGetColumns(), value, calclGetActiveCells(), n);
}