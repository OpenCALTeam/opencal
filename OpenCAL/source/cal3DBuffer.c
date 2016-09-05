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

#include <OpenCAL/cal3DBuffer.h>
#include <stdlib.h>
#include <string.h>



CALbyte* calAllocBuffer3Db(int rows, int columns, int slices) {
    return (CALbyte*)malloc(sizeof(CALbyte)*rows*columns*slices);
}
CALint* calAllocBuffer3Di(int rows, int columns, int slices) {
    return (CALint*)malloc(sizeof(CALint)*rows*columns*slices);
}
CALreal* calAllocBuffer3Dr(int rows, int columns, int slices) {
    return (CALreal*)malloc(sizeof(CALreal)*rows*columns*slices);
}



void calDeleteBuffer3Db(CALbyte* M) {
    free(M);
}
void calDeleteBuffer3Di(CALint* M) {
    free(M);
}
void calDeleteBuffer3Dr(CALreal* M) {
    free(M);
}



void calCopyBuffer3Db(CALbyte* M_src, CALbyte* M_dest, int rows, int columns, int slices)
{
    memcpy(M_dest, M_src, sizeof(CALbyte)*rows*columns*slices);
}
void calCopyBuffer3Di(CALint* M_src, CALint* M_dest, int rows, int columns, int slices)
{
    memcpy(M_dest, M_src, sizeof(CALint)*rows*columns*slices);
}
void calCopyBuffer3Dr(CALreal* M_src, CALreal* M_dest, int rows, int columns, int slices)
{
    memcpy(M_dest, M_src, sizeof(CALreal)*rows*columns*slices);
}


void calCopyBufferActiveCells3Db(CALbyte* M_src, CALbyte* M_dest, CALModel3D* ca3D) {

    if(ca3D->OPTIMIZATION == CAL_OPT_ACTIVE_CELLS_NAIVE)
        calCopyBufferActiveCellsNaive3Db(M_src, M_dest, ca3D);
    else if(ca3D->OPTIMIZATION == CAL_OPT_ACTIVE_CELLS)
        calCopyBufferActiveCellsCLL3Db(M_src, M_dest, ca3D);
}

void calCopyBufferActiveCells3Di(CALint* M_src, CALint* M_dest, CALModel3D* ca3D) {

    if(ca3D->OPTIMIZATION == CAL_OPT_ACTIVE_CELLS_NAIVE)
        calCopyBufferActiveCellsNaive3Di(M_src, M_dest, ca3D);
    else if(ca3D->OPTIMIZATION == CAL_OPT_ACTIVE_CELLS)
        calCopyBufferActiveCellsCLL3Di(M_src, M_dest, ca3D);
}

void calCopyBufferActiveCells3Dr(CALreal* M_src, CALreal* M_dest, CALModel3D* ca3D) {

    if(ca3D->OPTIMIZATION == CAL_OPT_ACTIVE_CELLS_NAIVE)
        calCopyBufferActiveCellsNaive3Dr(M_src, M_dest, ca3D);
    else if(ca3D->OPTIMIZATION == CAL_OPT_ACTIVE_CELLS)
        calCopyBufferActiveCellsCLL3Dr(M_src, M_dest, ca3D);
}


void calAddBuffer3Db(CALbyte* M_op1, CALbyte* M_op2,  CALbyte* M_dest, int rows, int columns, int slices) {
    int size = rows * columns * slices;
    int i;

    for (i=0; i<size; i++)
        M_dest[i] = M_op1[i] + M_op2[i];
}
void calAddBuffer3Di(CALint* M_op1, CALint* M_op2,  CALint* M_dest, int rows, int columns, int slices) {
    int size = rows * columns * slices;
    int i;

    for (i=0; i<size; i++)
        M_dest[i] = M_op1[i] + M_op2[i];
}
void calAddBuffer3Dr(CALreal* M_op1, CALreal* M_op2,  CALreal* M_dest, int rows, int columns, int slices) {
    int size = rows * columns * slices;
    int i;

    for (i=0; i<size; i++)
        M_dest[i] = M_op1[i] + M_op2[i];
}



void calSubtractBuffer3Db(CALbyte* M_op1, CALbyte* M_op2,  CALbyte* M_dest, int rows, int columns, int slices) {
    int size = rows * columns * slices;
    int i;

    for (i=0; i<size; i++)
        M_dest[i] = M_op1[i] - M_op2[i];
}
void calSubtractBuffer3Di(CALint* M_op1, CALint* M_op2,  CALint* M_dest, int rows, int columns, int slices) {
    int size = rows * columns * slices;
    int i;

    for (i=0; i<size; i++)
        M_dest[i] = M_op1[i] - M_op2[i];
}
void calSubtractBuffer3Dr(CALreal* M_op1, CALreal* M_op2,  CALreal* M_dest, int rows, int columns, int slices) {
    int size = rows * columns * slices;
    int i;

    for (i=0; i<size; i++)
        M_dest[i] = M_op1[i] - M_op2[i];
}



void calSetBuffer3Db(CALbyte* M, int rows, int columns, int slices, CALbyte value)
{
    memset(M, value, sizeof(CALbyte)*rows*columns*slices);
}
void calSetBuffer3Di(CALint* M, int rows, int columns, int slices, CALint value)
{
	int size = rows * columns * slices;
	int i;

	for (i=0; i<size; i++)
		M[i] = value;
}
void calSetBuffer3Dr(CALreal* M, int rows, int columns, int slices, CALreal value)
{
    int size = rows * columns * slices;
    int i;

    for (i=0; i<size; i++)
        M[i] = value;
}



void calSetActiveCellsBuffer3Db(CALbyte* M, CALbyte value, CALModel3D* ca3D) {

    if(ca3D->OPTIMIZATION == CAL_OPT_ACTIVE_CELLS_NAIVE)
        calSetActiveCellsNaiveBuffer3Db(M, value, ca3D);
    else if(ca3D->OPTIMIZATION == CAL_OPT_ACTIVE_CELLS )
        calSetActiveCellsCLLBuffer3Db(M, value, ca3D);
}
void calSetActiveCellsBuffer3Di(CALint* M, CALint value, CALModel3D* ca3D) {

    if(ca3D->OPTIMIZATION == CAL_OPT_ACTIVE_CELLS_NAIVE)
        calSetActiveCellsNaiveBuffer3Di(M, value, ca3D);
    else if(ca3D->OPTIMIZATION == CAL_OPT_ACTIVE_CELLS )
        calSetActiveCellsCLLBuffer3Di(M, value, ca3D);
}
void calSetActiveCellsBuffer3Dr(CALreal* M, CALreal value, CALModel3D* ca3D) {

    if(ca3D->OPTIMIZATION == CAL_OPT_ACTIVE_CELLS_NAIVE)
        calSetActiveCellsNaiveBuffer3Dr(M, value, ca3D);
    else if(ca3D->OPTIMIZATION == CAL_OPT_ACTIVE_CELLS )
        calSetActiveCellsCLLBuffer3Dr(M, value, ca3D);
}
