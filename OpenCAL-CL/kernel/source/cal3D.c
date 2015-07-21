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

#include <cal3D.h>

void calInitSubstate3Db(MODEL_DEFINITION3D, CALint substateNum, int i, int j, int k, CALbyte value) {
	__global CALbyte * current = calGetCurrentSubstate3Db(MODEL3D, substateNum);
	__global CALbyte * next = calGetNextSubstate3Db(MODEL3D, substateNum);

	calSetBufferElement3D(current, get_rows(), get_columns(), i, j, k, value);
	calSetBufferElement3D(next, get_rows(), get_columns(), i, j, k, value);
}
void calInitSubstate3Di(MODEL_DEFINITION3D, CALint substateNum, int i, int j, int k, CALint value) {
	__global CALint * current = calGetCurrentSubstate3Di(MODEL3D, substateNum);
	__global CALint * next = calGetNextSubstate3Di(MODEL3D, substateNum);

	calSetBufferElement3D(current, get_rows(), get_columns(), i, j, k, value);
	calSetBufferElement3D(next, get_rows(), get_columns(), i, j, k, value);
}
void calInitSubstate3Dr(MODEL_DEFINITION3D, CALint substateNum, int i, int j, int k, CALreal value) {
	__global CALreal * current = calGetCurrentSubstate3Dr(MODEL3D, substateNum);
	__global CALreal * next = calGetNextSubstate3Dr(MODEL3D, substateNum);

	calSetBufferElement3D(current, get_rows(), get_columns(), i, j, k, value);
	calSetBufferElement3D(next, get_rows(), get_columns(), i, j, k, value);
}

__global CALbyte * calGetCurrentSubstate3Db(MODEL_DEFINITION3D, CALint substateNum) {
	return (get_current_byte_substates() + get_rows() * get_columns() * get_slices() * substateNum);
}
__global CALreal * calGetCurrentSubstate3Dr(MODEL_DEFINITION3D, CALint substateNum) {
	return (get_current_real_substates() + get_rows() * get_columns() * get_slices() * substateNum);
}
__global CALint * calGetCurrentSubstate3Di(MODEL_DEFINITION3D, CALint substateNum) {
	return (get_current_int_substates() + get_rows() * get_columns() * get_slices() * substateNum);
}

__global CALbyte * calGetNextSubstate3Db(MODEL_DEFINITION3D, CALint substateNum) {
	return (get_next_byte_substates() + get_rows() * get_columns() * get_slices() * substateNum);
}
__global CALreal * calGetNextSubstate3Dr(MODEL_DEFINITION3D, CALint substateNum) {
	return (get_next_real_substates() + get_rows() * get_columns() * get_slices() * substateNum);
}
__global CALint * calGetNextSubstate3Di(MODEL_DEFINITION3D, CALint substateNum) {
	return (get_next_int_substates() + get_rows() * get_columns() * get_slices() * substateNum);
}

CALbyte calGet3Db(MODEL_DEFINITION3D, CALint substateNum, int i, int j, int k) {
	__global CALbyte * current = calGetCurrentSubstate3Db(MODEL3D, substateNum);
	return calGetBufferElement3D(current, get_rows(), get_columns(), i, j, k);
}
CALint calGet3Di(MODEL_DEFINITION3D, CALint substateNum, int i, int j, int k ) {
	__global CALint * current = calGetCurrentSubstate3Di(MODEL3D, substateNum);
	return calGetBufferElement3D(current, get_rows(), get_columns(), i, j, k);
}
CALreal calGet3Dr(MODEL_DEFINITION3D, CALint substateNum, int i, int j, int k) {
	__global CALreal * current = calGetCurrentSubstate3Dr(MODEL3D, substateNum);
	return calGetBufferElement3D(current, get_rows(), get_columns(), i, j, k);
}

void calSet3Db(MODEL_DEFINITION3D, CALint substateNum, int i, int j, int k, CALbyte value) {
	__global CALbyte * next = calGetNextSubstate3Db(MODEL3D, substateNum);
	calSetBufferElement3D(next, get_rows(), get_columns(), i, j, k, value);
}
void calSet3Di(MODEL_DEFINITION3D, CALint substateNum, int i, int j, int k, CALint value) {
	__global CALint * next = calGetNextSubstate3Di(MODEL3D, substateNum);
	calSetBufferElement3D(next, get_rows(), get_columns(), i, j, k, value);
}
void calSet3Dr(MODEL_DEFINITION3D, CALint substateNum, int i, int j, int k, CALreal value) {
	__global CALreal * next = calGetNextSubstate3Dr(MODEL3D, substateNum);
	calSetBufferElement3D(next, get_rows(), get_columns(), i, j, k, value);
}

CALbyte calGetX3Db(MODEL_DEFINITION3D, CALint substateNum, int i, int j, int k, int n) {
	__global CALbyte * current = calGetCurrentSubstate3Db(MODEL3D, substateNum);
	if (get_boundary_condition() == CAL_SPACE_FLAT)
		return calGetBufferElement3D(current, get_rows(), get_columns(), i + get_neighborhood()[n].i, j + get_neighborhood()[n].j, k + get_neighborhood()[n].k);
	else
		return calGetBufferElement3D(current, get_rows(), get_columns(), calGetToroidalX(i + get_neighborhood()[n].i, get_rows()), calGetToroidalX(j + get_neighborhood()[n].j, get_columns()), calGetToroidalX(k + get_neighborhood()[n].k, get_slices()));
}
CALint calGetX3Di(MODEL_DEFINITION3D, CALint substateNum, int i, int j, int k, int n) {
	__global CALint * current = calGetCurrentSubstate3Di(MODEL3D, substateNum);
	if (get_boundary_condition() == CAL_SPACE_FLAT)
		return calGetBufferElement3D(current, get_rows(), get_columns(), i + get_neighborhood()[n].i, j + get_neighborhood()[n].j, k + get_neighborhood()[n].k);
	else
		return calGetBufferElement3D(current, get_rows(), get_columns(), calGetToroidalX(i + get_neighborhood()[n].i, get_rows()), calGetToroidalX(j + get_neighborhood()[n].j, get_columns()), calGetToroidalX(k + get_neighborhood()[n].k, get_slices()));
}
CALreal calGetX3Dr(MODEL_DEFINITION3D, CALint substateNum, int i, int j, int k, int n) {
	__global CALreal * current = calGetCurrentSubstate3Dr(MODEL3D, substateNum);
	if (get_boundary_condition() == CAL_SPACE_FLAT)
		return calGetBufferElement3D(current, get_rows(), get_columns(), i + get_neighborhood()[n].i, j + get_neighborhood()[n].j, k + get_neighborhood()[n].k);
	else
		return calGetBufferElement3D(current, get_rows(), get_columns(), calGetToroidalX(i + get_neighborhood()[n].i, get_rows()), calGetToroidalX(j + get_neighborhood()[n].j, get_columns()), calGetToroidalX(k + get_neighborhood()[n].k, get_slices()));
}









