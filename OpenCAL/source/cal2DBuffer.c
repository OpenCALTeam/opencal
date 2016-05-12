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

#include <OpenCAL/cal2DBuffer.h>
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


void calCopyBufferActiveCells2Db(CALbyte* M_src, CALbyte* M_dest,  struct CALModel2D* ca2D) {

    if(ca2D->OPTIMIZATION == CAL_OPT_ACTIVE_CELLS_NAIVE)
        calCopyBufferActiveCellsNaive2Db(M_src, M_dest, ca2D);
    else if(ca2D->OPTIMIZATION == CAL_OPT_ACTIVE_CELLS)
        calCopyBufferActiveCellsCLL2Db(M_src, M_dest, ca2D);
}

void calCopyBufferActiveCells2Di(CALint* M_src, CALint* M_dest, CALModel2D* ca2D) {

    if(ca2D->OPTIMIZATION == CAL_OPT_ACTIVE_CELLS_NAIVE)
        calCopyBufferActiveCellsNaive2Di(M_src, M_dest, ca2D);
    else if(ca2D->OPTIMIZATION == CAL_OPT_ACTIVE_CELLS)
        calCopyBufferActiveCellsCLL2Di(M_src, M_dest, ca2D);
}

void calCopyBufferActiveCells2Dr(CALreal* M_src, CALreal* M_dest, CALModel2D* ca2D) {

    if(ca2D->OPTIMIZATION == CAL_OPT_ACTIVE_CELLS_NAIVE)
        calCopyBufferActiveCellsNaive2Dr(M_src, M_dest, ca2D);
    else if(ca2D->OPTIMIZATION == CAL_OPT_ACTIVE_CELLS)
        calCopyBufferActiveCellsCLL2Dr(M_src, M_dest, ca2D);
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
	int size = rows * columns;
	int i;

	for (i=0; i<size; i++)
			M[i] = value;
}
void calSetBuffer2Dr(CALreal* M, int rows, int columns, CALreal value)
{
    int size = rows * columns;
    int i;

    for (i=0; i<size; i++)
        M[i] = value;
}



void calSetActiveCellsBuffer2Db(CALbyte* M, CALbyte value, CALModel2D* ca2D) {

    if(ca2D->OPTIMIZATION == CAL_OPT_ACTIVE_CELLS_NAIVE)
        calSetActiveCellsNaiveBuffer2Db(M, value, ca2D);
    else if(ca2D->OPTIMIZATION == CAL_OPT_ACTIVE_CELLS )
        calSetActiveCellsCLLBuffer2Db(M, value, ca2D);
}
void calSetActiveCellsBuffer2Di(CALint* M, CALint value, CALModel2D* ca2D)
{
    if(ca2D->OPTIMIZATION == CAL_OPT_ACTIVE_CELLS_NAIVE)
        calSetActiveCellsNaiveBuffer2Di(M, value, ca2D);
    else if(ca2D->OPTIMIZATION == CAL_OPT_ACTIVE_CELLS )
        calSetActiveCellsCLLBuffer2Di(M, value, ca2D);
}
void calSetActiveCellsBuffer2Dr(CALreal* M, CALreal value, CALModel2D* ca2D)
{
    if(ca2D->OPTIMIZATION == CAL_OPT_ACTIVE_CELLS_NAIVE)
        calSetActiveCellsNaiveBuffer2Dr(M, value, ca2D);
    else if(ca2D->OPTIMIZATION == CAL_OPT_ACTIVE_CELLS )
        calSetActiveCellsCLLBuffer2Dr(M, value, ca2D);
}
