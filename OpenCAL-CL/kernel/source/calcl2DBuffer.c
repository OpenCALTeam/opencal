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

#include <OpenCAL-CL/calcl2DBuffer.h>

void calclCopyBufferActiveCells2Db(__global CALbyte* M_src, __global CALbyte* M_dest, int columns, __global struct CALCell2D* active_cells, int n, CALint borderSize) {
	int c = (active_cells[n].i + borderSize) * columns + active_cells[n].j;
	if (M_dest[c] != M_src[c])
		M_dest[c] = M_src[c];
}

void calclCopyBufferActiveCells2Di(__global CALint* M_src, __global CALint* M_dest, int columns, __global struct CALCell2D* active_cells, int n, CALint borderSize) {
	int c = (active_cells[n].i + borderSize) * columns + active_cells[n].j;
	if (M_dest[c] != M_src[c])
		M_dest[c] = M_src[c];
}

void calclCopyBufferActiveCells2Dr(__global CALreal* M_src, __global CALreal* M_dest, int columns, __global struct CALCell2D* active_cells, int n, CALint borderSize) {
	int c = (active_cells[n].i + borderSize) * columns + active_cells[n].j;
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
