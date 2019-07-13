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

#include <OpenCAL-CL/calcl3D.h>

void calclInitSubstate3Db(__CALCL_MODEL_3D, CALint substateNum, int i, int j, int k, CALbyte value) {
	__global CALbyte * current = calclGetCurrentSubstate3Db(MODEL_3D, substateNum);
	__global CALbyte * next = calclGetNextSubstate3Db(MODEL_3D, substateNum);

	calclSetBufferElement3D(current, calclGetRows(), calclGetColumns(), i, j, k, value);
	calclSetBufferElement3D(next, calclGetRows(), calclGetColumns(), i, j, k, value);
}
void calclInitSubstate3Di(__CALCL_MODEL_3D, CALint substateNum, int i, int j, int k, CALint value) {
	__global CALint * current = calclGetCurrentSubstate3Di(MODEL_3D, substateNum);
	__global CALint * next = calclGetNextSubstate3Di(MODEL_3D, substateNum);

	calclSetBufferElement3D(current, calclGetRows(), calclGetColumns(), i, j, k, value);
	calclSetBufferElement3D(next, calclGetRows(), calclGetColumns(), i, j, k, value);
}
void calclInitSubstate3Dr(__CALCL_MODEL_3D, CALint substateNum, int i, int j, int k, CALreal value) {
	__global CALreal * current = calclGetCurrentSubstate3Dr(MODEL_3D, substateNum);
	__global CALreal * next = calclGetNextSubstate3Dr(MODEL_3D, substateNum);

	calclSetBufferElement3D(current, calclGetRows(), calclGetColumns(), i, j, k, value);
	calclSetBufferElement3D(next, calclGetRows(), calclGetColumns(), i, j, k, value);
}

__global CALbyte * calclGetCurrentSubstate3Db(__CALCL_MODEL_3D, CALint substateNum) {
	return (calclGetCurrentByteSubstates() + calclGetRows() * calclGetColumns() * calclGetSlices() * substateNum);
}
__global CALreal * calclGetCurrentSubstate3Dr(__CALCL_MODEL_3D, CALint substateNum) {
	return (calclGetCurrentRealSubstates() + calclGetRows() * calclGetColumns() * calclGetSlices() * substateNum);
}
__global CALint * calclGetCurrentSubstate3Di(__CALCL_MODEL_3D, CALint substateNum) {
	return (calclGetCurrentIntSubstates() + calclGetRows() * calclGetColumns() * calclGetSlices() * substateNum);
}

__global CALbyte * calclGetNextSubstate3Db(__CALCL_MODEL_3D, CALint substateNum) {
	return (calclGetNextByteSubstates() + calclGetRows() * calclGetColumns() * calclGetSlices() * substateNum);
}
__global CALreal * calclGetNextSubstate3Dr(__CALCL_MODEL_3D, CALint substateNum) {
	return (calclGetNextRealSubstates() + calclGetRows() * calclGetColumns() * calclGetSlices() * substateNum);
}
__global CALint * calclGetNextSubstate3Di(__CALCL_MODEL_3D, CALint substateNum) {
	return (calclGetNextIntSubstates() + calclGetRows() * calclGetColumns() * calclGetSlices() * substateNum);
}

CALbyte calclGet3Db(__CALCL_MODEL_3D, CALint substateNum, int i, int j, int k) {
	__global CALbyte * current = calclGetCurrentSubstate3Db(MODEL_3D, substateNum);
	return calclGetBufferElement3D(current, calclGetRows(), calclGetColumns(), i, j, k);
}
CALint calclGet3Di(__CALCL_MODEL_3D, CALint substateNum, int i, int j, int k ) {
	__global CALint * current = calclGetCurrentSubstate3Di(MODEL_3D, substateNum);
	return calclGetBufferElement3D(current, calclGetRows(), calclGetColumns(), i, j, k);
}
CALreal calclGet3Dr(__CALCL_MODEL_3D, CALint substateNum, int i, int j, int k) {
	__global CALreal * current = calclGetCurrentSubstate3Dr(MODEL_3D, substateNum);
	return calclGetBufferElement3D(current, calclGetRows(), calclGetColumns(), i, j, k);
}

void calclSet3Db(__CALCL_MODEL_3D, CALint substateNum, int i, int j, int k, CALbyte value) {
	__global CALbyte * next = calclGetNextSubstate3Db(MODEL_3D, substateNum);
	calclSetBufferElement3D(next, calclGetRows(), calclGetColumns(), i, j, k, value);
}
void calclSet3Di(__CALCL_MODEL_3D, CALint substateNum, int i, int j, int k, CALint value) {
	__global CALint * next = calclGetNextSubstate3Di(MODEL_3D, substateNum);
	calclSetBufferElement3D(next, calclGetRows(), calclGetColumns(), i, j, k, value);
}
void calclSet3Dr(__CALCL_MODEL_3D, CALint substateNum, int i, int j, int k, CALreal value) {
	__global CALreal * next = calclGetNextSubstate3Dr(MODEL_3D, substateNum);
	calclSetBufferElement3D(next, calclGetRows(), calclGetColumns(), i, j, k, value);
}

CALbyte calclGetX3Db(__CALCL_MODEL_3D, CALint substateNum, int i, int j, int k, int n) {
	__global CALbyte * current = calclGetCurrentSubstate3Db(MODEL_3D, substateNum);
	if (calclGetBoundaryCondition() == CAL_SPACE_FLAT)
		return calclGetBufferElement3D(current, calclGetRows(), calclGetColumns(), i + calclGetNeighborhood()[n].i, j + calclGetNeighborhood()[n].j, k + calclGetNeighborhood()[n].k);
	else
		return calclGetBufferElement3D(current, calclGetRows(), calclGetColumns(), calclGetToroidalX(i + calclGetNeighborhood()[n].i, calclGetRows()), calclGetToroidalX(j + calclGetNeighborhood()[n].j, calclGetColumns()), calclGetToroidalX(k + calclGetNeighborhood()[n].k, calclGetSlices()));
}
CALint calclGetX3Di(__CALCL_MODEL_3D, CALint substateNum, int i, int j, int k, int n) {
	__global CALint * current = calclGetCurrentSubstate3Di(MODEL_3D, substateNum);
	if (calclGetBoundaryCondition() == CAL_SPACE_FLAT)
		return calclGetBufferElement3D(current, calclGetRows(), calclGetColumns(), i + calclGetNeighborhood()[n].i, j + calclGetNeighborhood()[n].j, k + calclGetNeighborhood()[n].k);
	else
		return calclGetBufferElement3D(current, calclGetRows(), calclGetColumns(), calclGetToroidalX(i + calclGetNeighborhood()[n].i, calclGetRows()), calclGetToroidalX(j + calclGetNeighborhood()[n].j, calclGetColumns()), calclGetToroidalX(k + calclGetNeighborhood()[n].k, calclGetSlices()));
}
CALreal calclGetX3Dr(__CALCL_MODEL_3D, CALint substateNum, int i, int j, int k, int n) {
	__global CALreal * current = calclGetCurrentSubstate3Dr(MODEL_3D, substateNum);
	if (calclGetBoundaryCondition() == CAL_SPACE_FLAT)
		return calclGetBufferElement3D(current, calclGetRows(), calclGetColumns(), i + calclGetNeighborhood()[n].i, j + calclGetNeighborhood()[n].j, k + calclGetNeighborhood()[n].k);
	else
		return calclGetBufferElement3D(current, calclGetRows(), calclGetColumns(), calclGetToroidalX(i + calclGetNeighborhood()[n].i, calclGetRows()), calclGetToroidalX(j + calclGetNeighborhood()[n].j, calclGetColumns()), calclGetToroidalX(k + calclGetNeighborhood()[n].k, calclGetSlices()));
}


void calclAddActiveCell3D(__CALCL_MODEL_3D, int i, int j, int k) {

	if (CAL_FALSE == calclGetBufferElement3D(calclGetActiveCellsFlags(), calclGetRows(), calclGetColumns(), i, j, k)){
		calclSetBufferElement3D(calclGetActiveCellsFlags(), calclGetRows(), calclGetColumns(), i, j, k, CAL_TRUE);
		CALint chunkNum = (k*calclGetRows()*calclGetColumns()+ i*calclGetColumns() + j)/CALCLchunk;
		CALCLdiff[chunkNum] = CAL_TRUE;
	}

}
void calclAddActiveCellX3D(__CALCL_MODEL_3D, int i, int j, int k, int n) {

	CALint ix;
	CALint jx;
	CALint kx;

	k=  k - borderSize;

	if (calclGetBoundaryCondition() == CAL_SPACE_FLAT) {
		ix = i + calclGetNeighborhood()[n].i;
		jx = j + calclGetNeighborhood()[n].j;
		kx = k + calclGetNeighborhood()[n].k;
	} else {
		ix = calclGetToroidalX(i + calclGetNeighborhood()[n].i, calclGetRows());
		jx = calclGetToroidalX(j + calclGetNeighborhood()[n].j, calclGetColumns());
		kx = calclGetToroidalX(k + calclGetNeighborhood()[n].k, calclGetSlices());
	}

	if (CAL_FALSE == calclGetBufferElement3D(calclGetActiveCellsFlags(), calclGetRows(), calclGetColumns(), ix, jx, kx)){
		calclSetBufferElement3D(calclGetActiveCellsFlags(), calclGetRows(), calclGetColumns(), ix, jx, kx, CAL_TRUE);
		CALint chunkNum = (kx*calclGetRows()*calclGetColumns()+ ix*calclGetColumns() + jx)/CALCLchunk;
		CALCLdiff[chunkNum] = CAL_TRUE;
	}
}

void calclRemoveActiveCell3D(__CALCL_MODEL_3D, int i, int j, int k) {

	if (CAL_TRUE == calclGetBufferElement3D(calclGetActiveCellsFlags(), calclGetRows(), calclGetColumns(), i, j, k)){
		calclSetBufferElement3D(calclGetActiveCellsFlags(), calclGetRows(), calclGetColumns(), i, j, k, CAL_FALSE);
		CALint chunkNum = (k*calclGetRows()*calclGetColumns()+ i*calclGetColumns() + j)/CALCLchunk;
		CALCLdiff[chunkNum] = CAL_TRUE;
	}

}

void calclInitSubstateActiveCell3Db(__CALCL_MODEL_3D, int substateNum, int n, CALbyte value) {
	__global CALbyte * current = calclGetCurrentSubstate3Db(MODEL_3D, substateNum);
	__global CALbyte * next = calclGetNextSubstate3Db(MODEL_3D, substateNum);
	calclSetBufferActiveCells3Db(current, calclGetRows(), calclGetColumns(), value, calclGetActiveCells(), n);
	calclSetBufferActiveCells3Db(next, calclGetRows(), calclGetColumns(), value, calclGetActiveCells(), n);
}
void calclInitSubstateActiveCell3Di(__CALCL_MODEL_3D, int substateNum, int n, CALint value) {
	__global CALint * current = calclGetCurrentSubstate3Di(MODEL_3D, substateNum);
	__global CALint * next = calclGetNextSubstate3Di(MODEL_3D, substateNum);
	calclSetBufferActiveCells3Di(current, calclGetRows(), calclGetColumns(), value, calclGetActiveCells(), n);
	calclSetBufferActiveCells3Di(next, calclGetRows(), calclGetColumns(), value, calclGetActiveCells(), n);
}
void calclInitSubstateActiveCell3Dr(__CALCL_MODEL_3D, int substateNum, int n, CALreal value) {
	__global CALreal * current = calclGetCurrentSubstate3Dr(MODEL_3D, substateNum);
	__global CALreal * next = calclGetNextSubstate3Dr(MODEL_3D, substateNum);
	calclSetBufferActiveCells3Dr(current, calclGetRows(), calclGetColumns(), value, calclGetActiveCells(), n);
	calclSetBufferActiveCells3Dr(next, calclGetRows(), calclGetColumns(), value, calclGetActiveCells(), n);
}
