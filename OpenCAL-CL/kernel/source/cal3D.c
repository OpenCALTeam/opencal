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

void calInitSubstate3Db(__CALCL_MODEL_3D, CALint substateNum, int i, int j, int k, CALbyte value) {
	__global CALbyte * current = calGetCurrentSubstate3Db(MODEL_3D, substateNum);
	__global CALbyte * next = calGetNextSubstate3Db(MODEL_3D, substateNum);

	calSetBufferElement3D(current, get_rows(), get_columns(), i, j, k, value);
	calSetBufferElement3D(next, get_rows(), get_columns(), i, j, k, value);
}
void calInitSubstate3Di(__CALCL_MODEL_3D, CALint substateNum, int i, int j, int k, CALint value) {
	__global CALint * current = calGetCurrentSubstate3Di(MODEL_3D, substateNum);
	__global CALint * next = calGetNextSubstate3Di(MODEL_3D, substateNum);

	calSetBufferElement3D(current, get_rows(), get_columns(), i, j, k, value);
	calSetBufferElement3D(next, get_rows(), get_columns(), i, j, k, value);
}
void calInitSubstate3Dr(__CALCL_MODEL_3D, CALint substateNum, int i, int j, int k, CALreal value) {
	__global CALreal * current = calGetCurrentSubstate3Dr(MODEL_3D, substateNum);
	__global CALreal * next = calGetNextSubstate3Dr(MODEL_3D, substateNum);

	calSetBufferElement3D(current, get_rows(), get_columns(), i, j, k, value);
	calSetBufferElement3D(next, get_rows(), get_columns(), i, j, k, value);
}

__global CALbyte * calGetCurrentSubstate3Db(__CALCL_MODEL_3D, CALint substateNum) {
	return (get_current_byte_substates() + get_rows() * get_columns() * get_slices() * substateNum);
}
__global CALreal * calGetCurrentSubstate3Dr(__CALCL_MODEL_3D, CALint substateNum) {
	return (get_current_real_substates() + get_rows() * get_columns() * get_slices() * substateNum);
}
__global CALint * calGetCurrentSubstate3Di(__CALCL_MODEL_3D, CALint substateNum) {
	return (get_current_int_substates() + get_rows() * get_columns() * get_slices() * substateNum);
}

__global CALbyte * calGetNextSubstate3Db(__CALCL_MODEL_3D, CALint substateNum) {
	return (get_next_byte_substates() + get_rows() * get_columns() * get_slices() * substateNum);
}
__global CALreal * calGetNextSubstate3Dr(__CALCL_MODEL_3D, CALint substateNum) {
	return (get_next_real_substates() + get_rows() * get_columns() * get_slices() * substateNum);
}
__global CALint * calGetNextSubstate3Di(__CALCL_MODEL_3D, CALint substateNum) {
	return (get_next_int_substates() + get_rows() * get_columns() * get_slices() * substateNum);
}

CALbyte calGet3Db(__CALCL_MODEL_3D, CALint substateNum, int i, int j, int k) {
	__global CALbyte * current = calGetCurrentSubstate3Db(MODEL_3D, substateNum);
	return calGetBufferElement3D(current, get_rows(), get_columns(), i, j, k);
}
CALint calGet3Di(__CALCL_MODEL_3D, CALint substateNum, int i, int j, int k ) {
	__global CALint * current = calGetCurrentSubstate3Di(MODEL_3D, substateNum);
	return calGetBufferElement3D(current, get_rows(), get_columns(), i, j, k);
}
CALreal calGet3Dr(__CALCL_MODEL_3D, CALint substateNum, int i, int j, int k) {
	__global CALreal * current = calGetCurrentSubstate3Dr(MODEL_3D, substateNum);
	return calGetBufferElement3D(current, get_rows(), get_columns(), i, j, k);
}

void calSet3Db(__CALCL_MODEL_3D, CALint substateNum, int i, int j, int k, CALbyte value) {
	__global CALbyte * next = calGetNextSubstate3Db(MODEL_3D, substateNum);
	calSetBufferElement3D(next, get_rows(), get_columns(), i, j, k, value);
}
void calSet3Di(__CALCL_MODEL_3D, CALint substateNum, int i, int j, int k, CALint value) {
	__global CALint * next = calGetNextSubstate3Di(MODEL_3D, substateNum);
	calSetBufferElement3D(next, get_rows(), get_columns(), i, j, k, value);
}
void calSet3Dr(__CALCL_MODEL_3D, CALint substateNum, int i, int j, int k, CALreal value) {
	__global CALreal * next = calGetNextSubstate3Dr(MODEL_3D, substateNum);
	calSetBufferElement3D(next, get_rows(), get_columns(), i, j, k, value);
}

CALbyte calGetX3Db(__CALCL_MODEL_3D, CALint substateNum, int i, int j, int k, int n) {
	__global CALbyte * current = calGetCurrentSubstate3Db(MODEL_3D, substateNum);
	if (get_boundary_condition() == CAL_SPACE_FLAT)
		return calGetBufferElement3D(current, get_rows(), get_columns(), i + get_neighborhood()[n].i, j + get_neighborhood()[n].j, k + get_neighborhood()[n].k);
	else
		return calGetBufferElement3D(current, get_rows(), get_columns(), calGetToroidalX(i + get_neighborhood()[n].i, get_rows()), calGetToroidalX(j + get_neighborhood()[n].j, get_columns()), calGetToroidalX(k + get_neighborhood()[n].k, get_slices()));
}
CALint calGetX3Di(__CALCL_MODEL_3D, CALint substateNum, int i, int j, int k, int n) {
	__global CALint * current = calGetCurrentSubstate3Di(MODEL_3D, substateNum);
	if (get_boundary_condition() == CAL_SPACE_FLAT)
		return calGetBufferElement3D(current, get_rows(), get_columns(), i + get_neighborhood()[n].i, j + get_neighborhood()[n].j, k + get_neighborhood()[n].k);
	else
		return calGetBufferElement3D(current, get_rows(), get_columns(), calGetToroidalX(i + get_neighborhood()[n].i, get_rows()), calGetToroidalX(j + get_neighborhood()[n].j, get_columns()), calGetToroidalX(k + get_neighborhood()[n].k, get_slices()));
}
CALreal calGetX3Dr(__CALCL_MODEL_3D, CALint substateNum, int i, int j, int k, int n) {
	__global CALreal * current = calGetCurrentSubstate3Dr(MODEL_3D, substateNum);
	if (get_boundary_condition() == CAL_SPACE_FLAT)
		return calGetBufferElement3D(current, get_rows(), get_columns(), i + get_neighborhood()[n].i, j + get_neighborhood()[n].j, k + get_neighborhood()[n].k);
	else
		return calGetBufferElement3D(current, get_rows(), get_columns(), calGetToroidalX(i + get_neighborhood()[n].i, get_rows()), calGetToroidalX(j + get_neighborhood()[n].j, get_columns()), calGetToroidalX(k + get_neighborhood()[n].k, get_slices()));
}









