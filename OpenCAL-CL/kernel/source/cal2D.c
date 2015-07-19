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

#include <OpenCAL/cal2D.h>

void calInitSubstate2Db(MODEL_DEFINITION2D, CALbyte value, int i, int j, CALint substateNum) {
	__global CALbyte * current = calGetCurrentSubstate2Db(MODEL2D,substateNum);
	__global CALbyte * next = calGetNextSubstate2Db(MODEL2D,substateNum);
	calSetBufferElement2D(current, get_columns(), i, j, value);
	calSetBufferElement2D(next, get_columns(), i, j, value);
}
void calInitSubstate2Di(MODEL_DEFINITION2D, CALint value, int i, int j, CALint substateNum) {
	__global CALint * current = calGetCurrentSubstate2Di(MODEL2D,substateNum);
	__global CALint * next = calGetNextSubstate2Di(MODEL2D,substateNum);
	calSetBufferElement2D(current, get_columns(), i, j, value);
	calSetBufferElement2D(next, get_columns(), i, j, value);
}
void calInitSubstate2Dr(MODEL_DEFINITION2D, CALreal value, int i, int j, CALint substateNum) {
	__global CALreal * current = calGetCurrentSubstate2Dr(MODEL2D,substateNum);
	__global CALreal * next = calGetNextSubstate2Dr(MODEL2D,substateNum);
	calSetBufferElement2D(current, get_columns(), i, j, value);
	calSetBufferElement2D(next, get_columns(), i, j, value);
}
__global CALbyte * calGetCurrentSubstate2Db(MODEL_DEFINITION2D, CALint substateNum) {
	return (CALCLcurrentByteSubstates + get_rows() * get_columns() * substateNum);
}
__global CALreal * calGetCurrentSubstate2Dr(MODEL_DEFINITION2D, CALint substateNum) {
	return (CALCLcurrentRealSubstates + get_rows() * get_columns() * substateNum);
}
__global CALint * calGetCurrentSubstate2Di(MODEL_DEFINITION2D, CALint substateNum) {
	return (CALCLcurrentIntSubstates + get_rows() * get_columns() * substateNum);
}
__global CALbyte * calGetNextSubstate2Db(MODEL_DEFINITION2D, CALint substateNum) {
	return (CALCLnextByteSubstates + get_rows() * get_columns() * substateNum);
}
__global CALreal * calGetNextSubstate2Dr(MODEL_DEFINITION2D, CALint substateNum) {
	return (CALCLnextRealSubstates + get_rows() * get_columns() * substateNum);
}
__global CALint * calGetNextSubstate2Di(MODEL_DEFINITION2D, CALint substateNum) {
	return (CALCLnextIntSubstates + get_rows() * get_columns() * substateNum);
}

CALbyte calGet2Db(MODEL_DEFINITION2D, int i, int j,CALint substateNum) {
	__global CALbyte * current = calGetCurrentSubstate2Db(MODEL2D,substateNum);
	return calGetBufferElement2D(current, get_columns(), i, j);
}
CALint calGet2Di(MODEL_DEFINITION2D, int i, int j,CALint substateNum) {
	__global CALint * current = calGetCurrentSubstate2Di(MODEL2D,substateNum);
	return calGetBufferElement2D(current, get_columns(), i, j);
}
CALreal calGet2Dr(MODEL_DEFINITION2D, int i, int j,CALint substateNum) {
	__global CALreal * current = calGetCurrentSubstate2Dr(MODEL2D,substateNum);
	return calGetBufferElement2D(current, get_columns(), i, j);
}

void calSet2Dr(MODEL_DEFINITION2D, CALreal value, int i, int j,CALint substateNum) {
	__global CALreal * next = calGetNextSubstate2Dr(MODEL2D,substateNum);
	calSetBufferElement2D(next, get_columns(), i, j, value);
}
void calSet2Di(MODEL_DEFINITION2D, CALint value, int i, int j,CALint substateNum) {
	__global CALint * next = calGetNextSubstate2Di(MODEL2D,substateNum);
	calSetBufferElement2D(next, get_columns(), i, j, value);
}
void calSet2Db(MODEL_DEFINITION2D, CALbyte value, int i, int j,CALint substateNum) {
	__global CALbyte * next = calGetNextSubstate2Db(MODEL2D,substateNum);
	calSetBufferElement2D(next, get_columns(), i, j, value);
}

CALbyte calGetX2Db(MODEL_DEFINITION2D, int i, int j, int n,CALint substateNum) {
	__global CALbyte * current = calGetCurrentSubstate2Db(MODEL2D,substateNum);
	if ((get_neighborhood_id() == CAL_HEXAGONAL_NEIGHBORHOOD_2D && j % 2 == 1) || (get_neighborhood_id() == CAL_HEXAGONAL_NEIGHBORHOOD_ALT_2D && i % 2 == 1))
		n += CAL_HEXAGONAL_SHIFT;

	if (get_boundary_condition() == CAL_SPACE_FLAT)
		return calGetBufferElement2D(current, get_columns(), i + CALCLneighborhood[n].i, j + CALCLneighborhood[n].j);
	else
		return calGetBufferElement2D(current, get_columns(), calGetToroidalX(i + CALCLneighborhood[n].i, get_rows()), calGetToroidalX(j + CALCLneighborhood[n].j, get_columns()));
}
CALint calGetX2Di(MODEL_DEFINITION2D, int i, int j, int n,CALint substateNum) {
	__global CALint * current = calGetCurrentSubstate2Di(MODEL2D,substateNum);
	if ((get_neighborhood_id() == CAL_HEXAGONAL_NEIGHBORHOOD_2D && j % 2 == 1) || (get_neighborhood_id() == CAL_HEXAGONAL_NEIGHBORHOOD_ALT_2D && i % 2 == 1))
		n += CAL_HEXAGONAL_SHIFT;

	if (get_boundary_condition() == CAL_SPACE_FLAT)
		return calGetBufferElement2D(current, get_columns(), i + CALCLneighborhood[n].i, j + CALCLneighborhood[n].j);
	else
		return calGetBufferElement2D(current, get_columns(), calGetToroidalX(i + CALCLneighborhood[n].i, get_rows()), calGetToroidalX(j + CALCLneighborhood[n].j, get_columns()));
}
CALreal calGetX2Dr(MODEL_DEFINITION2D, int i, int j, int n,CALint substateNum) {
	__global CALreal * current = calGetCurrentSubstate2Dr(MODEL2D,substateNum);
	if ((get_neighborhood_id() == CAL_HEXAGONAL_NEIGHBORHOOD_2D && j % 2 == 1) || (get_neighborhood_id() == CAL_HEXAGONAL_NEIGHBORHOOD_ALT_2D && i % 2 == 1))
		n += CAL_HEXAGONAL_SHIFT;

	if (get_boundary_condition() == CAL_SPACE_FLAT)
		return calGetBufferElement2D(current, get_columns(), i + CALCLneighborhood[n].i, j + CALCLneighborhood[n].j);
	else
		return calGetBufferElement2D(current, get_columns(), calGetToroidalX(i + CALCLneighborhood[n].i, get_rows()), calGetToroidalX(j + CALCLneighborhood[n].j, get_columns()));
}



