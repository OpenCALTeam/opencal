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

#include <OpenCAL-CL/calcl3DBuffer.h>

void calclCopyBufferActiveCells3Db(__global CALbyte* M_src, __global CALbyte* M_dest, int columns, int rows, __global struct CALCell3D* active_cells, int n) {
	int c = active_cells[n].k * columns * rows + active_cells[n].i * columns + active_cells[n].j;
	if (M_dest[c] != M_src[c])
		M_dest[c] = M_src[c];
}

void calclCopyBufferActiveCells3Di(__global CALint* M_src, __global CALint* M_dest, int columns, int rows, __global struct CALCell3D* active_cells, int n) {
	int c = active_cells[n].k * columns * rows + active_cells[n].i * columns + active_cells[n].j;
	if (M_dest[c] != M_src[c])
		M_dest[c] = M_src[c];
}

void calclCopyBufferActiveCells3Dr(__global CALreal* M_src, __global CALreal* M_dest, int columns, int rows, __global struct CALCell3D* active_cells, int n) {
	int c = active_cells[n].k * columns * rows + active_cells[n].i * columns + active_cells[n].j;
	if (M_dest[c] != M_src[c])
		M_dest[c] = M_src[c];
}

void calclCopyBuffer3Db(__global CALbyte* M_src, __global CALbyte* M_dest, int columns, int rows, int i, int j, int k) {
	CALbyte value = calclGetBufferElement3D(M_src, rows, columns, i, j, k);
	calclSetBufferElement3D(M_dest, rows, columns, i, j, k, value);
}
void calclCopyBuffer3Di(__global CALint* M_src, __global CALint* M_dest, int columns, int rows, int i, int j, int k) {
	CALint value = calclGetBufferElement3D(M_src, rows, columns, i, j, k);
	calclSetBufferElement3D(M_dest, rows, columns, i, j, k, value);
}
void calclCopyBuffer3Dr(__global CALreal* M_src, __global CALreal* M_dest, int columns, int rows, int i, int j, int k) {
	CALreal value = calclGetBufferElement3D(M_src, rows, columns, i, j, k);
	calclSetBufferElement3D(M_dest, rows, columns, i, j, k, value);
}

void calclAddMatrices3Db(__global CALbyte* M_op1, __global CALbyte* M_op3, __global CALbyte* M_dest, int i, int j, int k, int columns, int rows) {
	CALbyte sum = calclGetBufferElement3D(M_op1, rows, columns, i, j, k);
	sum += calclGetBufferElement3D(M_op3, rows, columns, i, j, k);
	calclSetBufferElement3D(M_dest, rows, columns, i, j, k, sum);
}
void calclAddMatrices3Di(__global CALint* M_op1, __global CALint* M_op3, __global CALint* M_dest, int i, int j, int k, int columns, int rows) {
	CALint sum = calclGetBufferElement3D(M_op1, rows, columns, i, j, k);
	sum += calclGetBufferElement3D(M_op3, rows, columns, i, j, k);
	calclSetBufferElement3D(M_dest, rows, columns, i, j, k, sum);
}
void calclAddMatrices3Dr(__global CALreal* M_op1, __global CALreal* M_op3, __global CALreal* M_dest, int i, int j, int k, int columns, int rows) {
	CALreal sum = calclGetBufferElement3D(M_op1, rows, columns, i, j, k);
	sum += calclGetBufferElement3D(M_op3, rows, columns, i, j, k);
	calclSetBufferElement3D(M_dest, rows, columns, i, j, k, sum);
}

void calclSubtractMatrices3Db(__global CALbyte* M_op1, __global CALbyte* M_op3, __global CALbyte* M_dest, int i, int j, int k, int columns, int rows) {
	CALbyte sub = calclGetBufferElement3D(M_op1, rows, columns, i, j, k);
	sub -= calclGetBufferElement3D(M_op3, rows, columns, i, j, k);
	calclSetBufferElement3D(M_dest, rows, columns, i, j, k, sub);
}
void calclSubtractMatrices3Di(__global CALint* M_op1, __global CALint* M_op3, __global CALint* M_dest, int i, int j, int k, int columns, int rows) {
	CALint sub = calclGetBufferElement3D(M_op1, rows, columns, i, j, k);
	sub -= calclGetBufferElement3D(M_op3, rows, columns, i, j, k);
	calclSetBufferElement3D(M_dest, rows, columns, i, j, k, sub);
}
void calclSubtractMatrices3Dr(__global CALreal* M_op1, __global CALreal* M_op3, __global CALreal* M_dest, int i, int j, int k, int columns, int rows) {
	CALreal sub = calclGetBufferElement3D(M_op1, rows, columns, i, j, k);
	sub -= calclGetBufferElement3D(M_op3, rows, columns, i, j, k);
	calclSetBufferElement3D(M_dest, rows, columns, i, j, k, sub);
}

void calclSetBufferActiveCells3Db(__global CALbyte* M, int columns, int rows, CALbyte value, __global struct CALCell3D* active_cells, int n) {
	calclSetBufferElement3D(M, columns, rows, active_cells[n].i, active_cells[n].j, active_cells[n].k, value);
}
void calclSetBufferActiveCells3Di(__global CALint* M, int columns, int rows, CALint value, __global struct CALCell3D* active_cells, int n) {
	calclSetBufferElement3D(M, columns, rows, active_cells[n].i, active_cells[n].j, active_cells[n].k, value);
}
void calclSetBufferActiveCells3Dr(__global CALreal* M, int columns, int rows, CALreal value, __global struct CALCell3D* active_cells, int n) {
	calclSetBufferElement3D(M, columns, rows, active_cells[n].i, active_cells[n].j, active_cells[n].k, value);
}
