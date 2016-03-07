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
