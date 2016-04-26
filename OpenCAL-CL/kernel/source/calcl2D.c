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

#include <OpenCAL-CL/calcl2D.h>

void calclInitSubstate2Db(__CALCL_MODEL_2D, CALint substateNum, int i, int j, CALbyte value) {
	__global CALbyte * current = calclGetCurrentSubstate2Db(MODEL_2D,substateNum);
	__global CALbyte * next = calclGetNextSubstate2Db(MODEL_2D,substateNum);
	calclSetBufferElement2D(current, calclGetColumns(), i, j, value);
	calclSetBufferElement2D(next, calclGetColumns(), i, j, value);
}
void calclInitSubstate2Di(__CALCL_MODEL_2D, CALint substateNum, int i, int j, CALint value) {
	__global CALint * current = calclGetCurrentSubstate2Di(MODEL_2D,substateNum);
	__global CALint * next = calclGetNextSubstate2Di(MODEL_2D,substateNum);
	calclSetBufferElement2D(current, calclGetColumns(), i, j, value);
	calclSetBufferElement2D(next, calclGetColumns(), i, j, value);
}
void calclInitSubstate2Dr(__CALCL_MODEL_2D, CALint substateNum, int i, int j, CALreal value) {
	__global CALreal * current = calclGetCurrentSubstate2Dr(MODEL_2D,substateNum);
	__global CALreal * next = calclGetNextSubstate2Dr(MODEL_2D,substateNum);
	calclSetBufferElement2D(current, calclGetColumns(), i, j, value);
	calclSetBufferElement2D(next, calclGetColumns(), i, j, value);
}
__global CALbyte * calclGetCurrentSubstate2Db(__CALCL_MODEL_2D, CALint substateNum) {
	return (CALCLcurrentByteSubstates + calclGetRows() * calclGetColumns() * substateNum);
}
__global CALreal * calclGetCurrentSubstate2Dr(__CALCL_MODEL_2D, CALint substateNum) {
	return (CALCLcurrentRealSubstates + calclGetRows() * calclGetColumns() * substateNum);
}
__global CALint * calclGetCurrentSubstate2Di(__CALCL_MODEL_2D, CALint substateNum) {
	return (CALCLcurrentIntSubstates + calclGetRows() * calclGetColumns() * substateNum);
}
__global CALbyte * calclGetNextSubstate2Db(__CALCL_MODEL_2D, CALint substateNum) {
	return (CALCLnextByteSubstates + calclGetRows() * calclGetColumns() * substateNum);
}
__global CALreal * calclGetNextSubstate2Dr(__CALCL_MODEL_2D, CALint substateNum) {
	return (CALCLnextRealSubstates + calclGetRows() * calclGetColumns() * substateNum);
}
__global CALint * calclGetNextSubstate2Di(__CALCL_MODEL_2D, CALint substateNum) {
	return (CALCLnextIntSubstates + calclGetRows() * calclGetColumns() * substateNum);
}

CALbyte calclGet2Db(__CALCL_MODEL_2D,CALint substateNum, int i, int j) {
	__global CALbyte * current = calclGetCurrentSubstate2Db(MODEL_2D,substateNum);
	return calclGetBufferElement2D(current, calclGetColumns(), i, j);
}
CALint calclGet2Di(__CALCL_MODEL_2D,CALint substateNum, int i, int j) {
	__global CALint * current = calclGetCurrentSubstate2Di(MODEL_2D,substateNum);
	return calclGetBufferElement2D(current, calclGetColumns(), i, j);
}
CALreal calclGet2Dr(__CALCL_MODEL_2D,CALint substateNum, int i, int j) {
	__global CALreal * current = calclGetCurrentSubstate2Dr(MODEL_2D,substateNum);
	return calclGetBufferElement2D(current, calclGetColumns(), i, j);
}

void calclSet2Dr(__CALCL_MODEL_2D,CALint substateNum, int i, int j, CALreal value) {
	__global CALreal * next = calclGetNextSubstate2Dr(MODEL_2D,substateNum);
	calclSetBufferElement2D(next, calclGetColumns(), i, j, value);
}
void calclSet2Di(__CALCL_MODEL_2D,CALint substateNum, int i, int j, CALint value) {
	__global CALint * next = calclGetNextSubstate2Di(MODEL_2D,substateNum);
	calclSetBufferElement2D(next, calclGetColumns(), i, j, value);
}
void calclSet2Db(__CALCL_MODEL_2D,CALint substateNum, int i, int j, CALbyte value) {
	__global CALbyte * next = calclGetNextSubstate2Db(MODEL_2D,substateNum);
	calclSetBufferElement2D(next, calclGetColumns(), i, j, value);
}

CALbyte calclGetX2Db(__CALCL_MODEL_2D,CALint substateNum, int i, int j, int n) {
	__global CALbyte * current = calclGetCurrentSubstate2Db(MODEL_2D,substateNum);
	if ((calclGetNeighborhoodId() == CAL_HEXAGONAL_NEIGHBORHOOD_2D && j % 2 == 1) || (calclGetNeighborhoodId() == CAL_HEXAGONAL_NEIGHBORHOOD_ALT_2D && i % 2 == 1))
		n += CAL_HEXAGONAL_SHIFT;

	if (calclGetBoundaryCondition() == CAL_SPACE_FLAT)
		return calclGetBufferElement2D(current, calclGetColumns(), i + CALCLneighborhood[n].i, j + CALCLneighborhood[n].j);
	else
		return calclGetBufferElement2D(current, calclGetColumns(), calclGetToroidalX(i + CALCLneighborhood[n].i, calclGetRows()), calclGetToroidalX(j + CALCLneighborhood[n].j, calclGetColumns()));
}
CALint calclGetX2Di(__CALCL_MODEL_2D,CALint substateNum, int i, int j, int n) {
	__global CALint * current = calclGetCurrentSubstate2Di(MODEL_2D,substateNum);
	if ((calclGetNeighborhoodId() == CAL_HEXAGONAL_NEIGHBORHOOD_2D && j % 2 == 1) || (calclGetNeighborhoodId() == CAL_HEXAGONAL_NEIGHBORHOOD_ALT_2D && i % 2 == 1))
		n += CAL_HEXAGONAL_SHIFT;

	if (calclGetBoundaryCondition() == CAL_SPACE_FLAT)
		return calclGetBufferElement2D(current, calclGetColumns(), i + CALCLneighborhood[n].i, j + CALCLneighborhood[n].j);
	else
		return calclGetBufferElement2D(current, calclGetColumns(), calclGetToroidalX(i + CALCLneighborhood[n].i, calclGetRows()), calclGetToroidalX(j + CALCLneighborhood[n].j, calclGetColumns()));
}
CALreal calclGetX2Dr(__CALCL_MODEL_2D,CALint substateNum, int i, int j, int n) {
	__global CALreal * current = calclGetCurrentSubstate2Dr(MODEL_2D,substateNum);
	if ((calclGetNeighborhoodId() == CAL_HEXAGONAL_NEIGHBORHOOD_2D && j % 2 == 1) || (calclGetNeighborhoodId() == CAL_HEXAGONAL_NEIGHBORHOOD_ALT_2D && i % 2 == 1))
		n += CAL_HEXAGONAL_SHIFT;

	if (calclGetBoundaryCondition() == CAL_SPACE_FLAT)
		return calclGetBufferElement2D(current, calclGetColumns(), i + CALCLneighborhood[n].i, j + CALCLneighborhood[n].j);
	else
		return calclGetBufferElement2D(current, calclGetColumns(), calclGetToroidalX(i + CALCLneighborhood[n].i, calclGetRows()), calclGetToroidalX(j + CALCLneighborhood[n].j, calclGetColumns()));
}


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
