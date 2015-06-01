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

#include <cal2DBuffer.h>
#include <stdlib.h>
#include <string.h>



CALbyte* calAllocBuffer2Db(int rows, int columns) {
	return (CALbyte*)malloc(sizeof(CALbyte)*rows*columns);
}
CALint* calAllocBuffer2Di(int rows, int columns) {
	return (CALint*)malloc(sizeof(CALint)*rows*columns);
}
CALreal* calAllocBuffer2Dr(int rows, int columns) {
	return (CALreal*)malloc(sizeof(CALreal)*rows*columns);
}



void calDeleteBuffer2Db(CALbyte* M) {
	free(M);
}
void calDeleteBuffer2Di(CALint* M) {
	free(M);
}
void calDeleteBuffer2Dr(CALreal* M) {
	free(M);
}



void calCopyBuffer2Db(CALbyte* M_src, CALbyte* M_dest, int rows, int columns)
{
	memcpy(M_dest, M_src, sizeof(CALbyte)*rows*columns);
}
void calCopyBuffer2Di(CALint* M_src, CALint* M_dest, int rows, int columns)
{	
	memcpy(M_dest, M_src, sizeof(CALint)*rows*columns);
}
void calCopyBuffer2Dr(CALreal* M_src, CALreal* M_dest, int rows, int columns)
{
	memcpy(M_dest, M_src, sizeof(CALreal)*rows*columns);
}


void calCopyActiveCellsBuffer2Db(CALbyte* M_src, CALbyte* M_dest, int rows, int columns, struct CALCell2D* active_cells, int sizeof_active_cells) {
	int c, n;

	for(n=0; n<sizeof_active_cells; n++)
	{
		c = active_cells[n].i * columns + active_cells[n].j;
		if (M_dest[c] != M_src[c])
			M_dest[c] = M_src[c];
	}
}

void calCopyActiveCellsBuffer2Di(CALint* M_src, CALint* M_dest, int rows, int columns, struct CALCell2D* active_cells, int sizeof_active_cells) {
	int c, n;

	for(n=0; n<sizeof_active_cells; n++)
	{
		c = active_cells[n].i * columns + active_cells[n].j;
		if (M_dest[c] != M_src[c])
			M_dest[c] = M_src[c];
	}
}

void calCopyActiveCellsBuffer2Dr(CALreal* M_src, CALreal* M_dest, int rows, int columns, struct CALCell2D* active_cells, int sizeof_active_cells) {
	int c, n;

	for(n=0; n<sizeof_active_cells; n++)
	{
		c = active_cells[n].i * columns + active_cells[n].j;
		if (M_dest[c] != M_src[c])
			M_dest[c] = M_src[c];
	}
}


void calAddBuffer2Db(CALbyte* M_op1, CALbyte* M_op2,  CALbyte* M_dest, int rows, int columns) {
	int size = rows * columns;
	int i;

	for (i=0; i<size; i++)
		M_dest[i] = M_op1[i] + M_op2[i];
}
void calAddBuffer2Di(CALint* M_op1, CALint* M_op2,  CALint* M_dest, int rows, int columns) {
	int size = rows * columns;
	int i;

	for (i=0; i<size; i++)
		M_dest[i] = M_op1[i] + M_op2[i];
}
void calAddBuffer2Dr(CALreal* M_op1, CALreal* M_op2,  CALreal* M_dest, int rows, int columns) {
	int size = rows * columns;
	int i;
	
	for (i=0; i<size; i++)
		M_dest[i] = M_op1[i] + M_op2[i];
}



void calSubtractBuffer2Db(CALbyte* M_op1, CALbyte* M_op2,  CALbyte* M_dest, int rows, int columns) {
	int size = rows * columns;
	int i;

	for (i=0; i<size; i++)
		M_dest[i] = M_op1[i] - M_op2[i];
}
void calSubtractBuffer2Di(CALint* M_op1, CALint* M_op2,  CALint* M_dest, int rows, int columns) {
	int size = rows * columns;
	int i;

	for (i=0; i<size; i++)
		M_dest[i] = M_op1[i] - M_op2[i];
}
void calSubtractBuffer2Dr(CALreal* M_op1, CALreal* M_op2,  CALreal* M_dest, int rows, int columns) {
	int size = rows * columns;
	int i;
	
	for (i=0; i<size; i++)
		M_dest[i] = M_op1[i] - M_op2[i];
}



void calSetBuffer2Db(CALbyte* M, int rows, int columns, CALbyte value)
{
	memset(M, value, sizeof(CALbyte)*rows*columns);
}
void calSetBuffer2Di(CALint* M, int rows, int columns, CALint value)
{
	memset(M, value, sizeof(CALint)*rows*columns);
}
void calSetBuffer2Dr(CALreal* M, int rows, int columns, CALreal value)
{
	int size = rows * columns;
	int i;
	
	for (i=0; i<size; i++)
		M[i] = value;
}



void calSetActiveCellsBuffer2Db(CALbyte* M, int rows, int columns, CALbyte value, struct CALCell2D* active_cells, int sizeof_active_cells) {
	int n;

	for(n=0; n<sizeof_active_cells; n++)
		M[active_cells[n].i * columns + active_cells[n].j] = value;
}
void calSetActiveCellsBuffer2Di(CALint* M, int rows, int columns, CALint value, struct CALCell2D* active_cells, int sizeof_active_cells) {
	int n;

	for(n=0; n<sizeof_active_cells; n++)
		M[active_cells[n].i * columns + active_cells[n].j] = value;
}
void calSetActiveCellsBuffer2Dr(CALreal* M, int rows, int columns, CALreal value, struct CALCell2D* active_cells, int sizeof_active_cells) {
	int n;

	for(n=0; n<sizeof_active_cells; n++)
		M[active_cells[n].i * columns + active_cells[n].j] = value;
}
