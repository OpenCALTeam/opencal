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

#include "OpenCAL-CL/calcl3DActive.h"

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
