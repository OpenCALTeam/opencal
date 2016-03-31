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

#include <OpenCAL-CL/calcl2DBuffer.h>

void calclCopyBufferActiveCells2Db(__global CALbyte* M_src, __global CALbyte* M_dest, int columns, __global struct CALCell2D* active_cells, int n) {
	int c = active_cells[n].i * columns + active_cells[n].j;
	if (M_dest[c] != M_src[c])
		M_dest[c] = M_src[c];
}

void calclCopyBufferActiveCells2Di(__global CALint* M_src, __global CALint* M_dest, int columns, __global struct CALCell2D* active_cells, int n) {
	int c = active_cells[n].i * columns + active_cells[n].j;
	if (M_dest[c] != M_src[c])
		M_dest[c] = M_src[c];
}

void calclCopyBufferActiveCells2Dr(__global CALreal* M_src, __global CALreal* M_dest, int columns, __global struct CALCell2D* active_cells, int n) {
	int c = active_cells[n].i * columns + active_cells[n].j;
	if (M_dest[c] != M_src[c])
		M_dest[c] = M_src[c];
}

void calclCopyBuffer2Db(__global CALbyte* M_src, __global CALbyte* M_dest, int columns, int i, int j) {
	CALbyte value = calclGetBufferElement2D(M_src, columns, i, j);
	calclSetBufferElement2D(M_dest, columns, i, j, value);
}
void calclCopyBuffer2Di(__global CALint* M_src, __global CALint* M_dest, int columns, int i, int j) {
	CALint value = calclGetBufferElement2D(M_src, columns, i, j);
	calclSetBufferElement2D(M_dest, columns, i, j, value);
}
void calclCopyBuffer2Dr(__global CALreal* M_src, __global CALreal* M_dest, int columns, int i, int j) {
	CALreal value = calclGetBufferElement2D(M_src, columns, i, j);
	calclSetBufferElement2D(M_dest, columns, i, j, value);
}

void calclAddMatrices2Db(__global CALbyte* M_op1, __global CALbyte* M_op2, __global CALbyte* M_dest, int i, int j, int columns) {
	CALbyte sum = calclGetBufferElement2D(M_op1, columns, i, j);
	sum += calclGetBufferElement2D(M_op2, columns, i, j);
	calclSetBufferElement2D(M_dest, columns, i, j, sum);
}
void calclAddMatrices2Di(__global CALint* M_op1, __global CALint* M_op2, __global CALint* M_dest, int i, int j, int columns) {
	CALint sum = calclGetBufferElement2D(M_op1, columns, i, j);
	sum += calclGetBufferElement2D(M_op2, columns, i, j);
	calclSetBufferElement2D(M_dest, columns, i, j, sum);
}
void calclAddMatrices2Dr(__global CALreal* M_op1, __global CALreal* M_op2, __global CALreal* M_dest, int i, int j, int columns) {
	CALreal sum = calclGetBufferElement2D(M_op1, columns, i, j);
	sum += calclGetBufferElement2D(M_op2, columns, i, j);
	calclSetBufferElement2D(M_dest, columns, i, j, sum);
}

void calclSubtractMatrices2Db(__global CALbyte* M_op1, __global CALbyte* M_op2, __global CALbyte* M_dest, int i, int j, int columns) {
	CALbyte sub = calclGetBufferElement2D(M_op1, columns, i, j);
	sub -= calclGetBufferElement2D(M_op2, columns, i, j);
	calclSetBufferElement2D(M_dest, columns, i, j, sub);
}

void calclSubtractMatrices2Di(__global CALint* M_op1, __global CALint* M_op2, __global CALint* M_dest, int i, int j, int columns) {
	CALint sub = calclGetBufferElement2D(M_op1, columns, i, j);
	sub -= calclGetBufferElement2D(M_op2, columns, i, j);
	calclSetBufferElement2D(M_dest, columns, i, j, sub);
}

void calclSubtractMatrices2Dr(__global CALreal* M_op1, __global CALreal* M_op2, __global CALreal* M_dest, int i, int j, int columns) {
	CALreal sub = calclGetBufferElement2D(M_op1, columns, i, j);
	sub -= calclGetBufferElement2D(M_op2, columns, i, j);
	calclSetBufferElement2D(M_dest, columns, i, j, sub);
}

void calclSetBufferActiveCells2Db(__global CALbyte* M, int columns, CALbyte value, __global struct CALCell2D* active_cells, int n) 
{
	calclSetBufferElement2D(M, columns, active_cells[n].i, active_cells[n].j, value);
}

void calclSetBufferActiveCells2Di(__global CALint* M, int columns, CALint value, __global struct CALCell2D* active_cells, int n) 
{
	calclSetBufferElement2D(M, columns, active_cells[n].i, active_cells[n].j, value);
}

void calclSetBufferActiveCells2Dr(__global CALreal* M, int columns, CALreal value, __global struct CALCell2D* active_cells, int n) 
{
	calclSetBufferElement2D(M, columns, active_cells[n].i, active_cells[n].j, value);
}

