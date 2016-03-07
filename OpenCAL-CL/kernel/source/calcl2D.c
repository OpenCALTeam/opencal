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
