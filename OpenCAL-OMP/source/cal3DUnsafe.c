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

#include <OpenCAL-OMP/cal3DBuffer.h>
#include <OpenCAL-OMP/cal3DUnsafe.h>
#include <stdlib.h>



void calAddActiveCellX3D(struct CALModel3D* ca3D, int i, int j, int k, int n)
{
    if (ca3D->T == CAL_SPACE_FLAT)
    {
        if(ca3D->OPTIMIZATION == CAL_OPT_ACTIVE_CELLS_NAIVE)
            calAddActiveCellNaive3D(ca3D, (i + ca3D->X[n].i), (j + ca3D->X[n].j), (k + ca3D->X[n].k));
        else if( ca3D->OPTIMIZATION ==CAL_OPT_ACTIVE_CELLS )
            calAddActiveCellCLL3D(ca3D, (i + ca3D->X[n].i), (j + ca3D->X[n].j), (k + ca3D->X[n].k));
    }
    else
    {
        if(ca3D->OPTIMIZATION == CAL_OPT_ACTIVE_CELLS_NAIVE)
            calAddActiveCellNaive3D(ca3D,  calGetToroidalX(i + ca3D->X[n].i, ca3D->rows),
                               calGetToroidalX(j + ca3D->X[n].j, ca3D->columns),
                               calGetToroidalX(k + ca3D->X[n].k, ca3D->slices));
        else if( ca3D->OPTIMIZATION ==CAL_OPT_ACTIVE_CELLS )
            calAddActiveCellCLL3D(ca3D,  calGetToroidalX(i + ca3D->X[n].i, ca3D->rows),
                                    calGetToroidalX(j + ca3D->X[n].j, ca3D->columns),
                                    calGetToroidalX(k + ca3D->X[n].k, ca3D->slices));

    }
}



void calInitX3Db(struct CALModel3D* ca3D, struct CALSubstate3Db* Q, int i, int j, int k, int n, CALbyte value)
{
    if (ca3D->T == CAL_SPACE_FLAT)
    {
        calSetBuffer3DElement(Q->current, ca3D->rows, ca3D->columns, i+ca3D->X[n].i, j+ca3D->X[n].j, k+ca3D->X[n].k, value);
        calSetBuffer3DElement(Q->next, ca3D->rows, ca3D->columns, i+ca3D->X[n].i, j+ca3D->X[n].j, k+ca3D->X[n].k, value);
    }
    else
    {
        calSetBuffer3DElement(Q->current, ca3D->rows, ca3D->columns, calGetToroidalX(i+ca3D->X[n].i, ca3D->rows), calGetToroidalX(j+ca3D->X[n].j, ca3D->columns), calGetToroidalX(k+ca3D->X[n].k, ca3D->slices), value);
        calSetBuffer3DElement(Q->next, ca3D->rows, ca3D->columns, calGetToroidalX(i+ca3D->X[n].i, ca3D->rows), calGetToroidalX(j+ca3D->X[n].j, ca3D->columns), calGetToroidalX(k+ca3D->X[n].k, ca3D->slices), value);
    }
}

void calInitX3Di(struct CALModel3D* ca3D, struct CALSubstate3Di* Q, int i, int j, int k, int n, CALint value)
{
    if (ca3D->T == CAL_SPACE_FLAT)
    {
        calSetBuffer3DElement(Q->current, ca3D->rows, ca3D->columns, i+ca3D->X[n].i, j+ca3D->X[n].j, k+ca3D->X[n].k, value);
        calSetBuffer3DElement(Q->next, ca3D->rows, ca3D->columns, i+ca3D->X[n].i, j+ca3D->X[n].j, k+ca3D->X[n].k, value);
    }
    else
    {
        calSetBuffer3DElement(Q->current, ca3D->rows, ca3D->columns, calGetToroidalX(i+ca3D->X[n].i, ca3D->rows), calGetToroidalX(j+ca3D->X[n].j, ca3D->columns), calGetToroidalX(k+ca3D->X[n].k, ca3D->slices), value);
        calSetBuffer3DElement(Q->next, ca3D->rows, ca3D->columns, calGetToroidalX(i+ca3D->X[n].i, ca3D->rows), calGetToroidalX(j+ca3D->X[n].j, ca3D->columns), calGetToroidalX(k+ca3D->X[n].k, ca3D->slices), value);
    }
}

void calInitX3Dr(struct CALModel3D* ca3D, struct CALSubstate3Dr* Q, int i, int j, int k, int n, CALreal value)
{
    if (ca3D->T == CAL_SPACE_FLAT)
    {
        calSetBuffer3DElement(Q->current, ca3D->rows, ca3D->columns, i+ca3D->X[n].i, j+ca3D->X[n].j, k+ca3D->X[n].k, value);
        calSetBuffer3DElement(Q->next, ca3D->rows, ca3D->columns, i+ca3D->X[n].i, j+ca3D->X[n].j, k+ca3D->X[n].k, value);
    }
    else
    {
        calSetBuffer3DElement(Q->current, ca3D->rows, ca3D->columns, calGetToroidalX(i+ca3D->X[n].i, ca3D->rows), calGetToroidalX(j+ca3D->X[n].j, ca3D->columns), calGetToroidalX(k+ca3D->X[n].k, ca3D->slices), value);
        calSetBuffer3DElement(Q->next, ca3D->rows, ca3D->columns, calGetToroidalX(i+ca3D->X[n].i, ca3D->rows), calGetToroidalX(j+ca3D->X[n].j, ca3D->columns), calGetToroidalX(k+ca3D->X[n].k, ca3D->slices), value);
    }
}



CALbyte calGetNext3Db(struct CALModel3D* ca3D, struct CALSubstate3Db* Q, int i, int j, int k) {
    CALbyte ret;

    CAL_SET_CELL_LOCK_3D(i, j, k, ca3D);

    ret = calGetBuffer3DElement(Q->next, ca3D->rows, ca3D->columns, i, j, k);

    CAL_UNSET_CELL_LOCK_3D(i, j, k, ca3D);

    return ret;
}

CALint calGetNext3Di(struct CALModel3D* ca3D, struct CALSubstate3Di* Q, int i, int j, int k) {
    CALint ret;

    CAL_SET_CELL_LOCK_3D(i, j, k, ca3D);

    ret = calGetBuffer3DElement(Q->next, ca3D->rows, ca3D->columns, i, j, k);

    CAL_UNSET_CELL_LOCK_3D(i, j, k, ca3D);

    return ret;
}

CALreal calGetNext3Dr(struct CALModel3D* ca3D, struct CALSubstate3Dr* Q, int i, int j, int k) {
    CALreal ret;

    CAL_SET_CELL_LOCK_3D(i, j, k, ca3D);

    ret = calGetBuffer3DElement(Q->next, ca3D->rows, ca3D->columns, i, j, k);

    CAL_UNSET_CELL_LOCK_3D(i, j, k, ca3D);

    return ret;
}



CALbyte calGetNextX3Db(struct CALModel3D* ca3D, struct CALSubstate3Db* Q, int i, int j, int k, int n)
{
    if (ca3D->T == CAL_SPACE_FLAT)
        return calGetNext3Db(ca3D, Q, i + ca3D->X[n].i, j + ca3D->X[n].j, k + ca3D->X[n].k);
    else
        return calGetNext3Db(ca3D, Q, calGetToroidalX(i + ca3D->X[n].i, ca3D->rows),
                             calGetToroidalX(j + ca3D->X[n].j, ca3D->columns),
                             calGetToroidalX(k + ca3D->X[n].k, ca3D->columns));
}

CALint calGetNextX3Di(struct CALModel3D* ca3D, struct CALSubstate3Di* Q, int i, int j, int k, int n)
{
    if (ca3D->T == CAL_SPACE_FLAT)
        return calGetNext3Di(ca3D, Q, i + ca3D->X[n].i, j + ca3D->X[n].j, k + ca3D->X[n].k);
    else
        return calGetNext3Di(ca3D, Q, calGetToroidalX(i + ca3D->X[n].i, ca3D->rows),
                             calGetToroidalX(j + ca3D->X[n].j, ca3D->columns),
                             calGetToroidalX(k + ca3D->X[n].k, ca3D->columns));
}

CALreal calGetNextX3Dr(struct CALModel3D* ca3D, struct CALSubstate3Dr* Q, int i, int j, int k, int n)
{
    if (ca3D->T == CAL_SPACE_FLAT)
        return calGetNext3Dr(ca3D, Q, i + ca3D->X[n].i, j + ca3D->X[n].j, k + ca3D->X[n].k);
    else
        return calGetNext3Dr(ca3D, Q, calGetToroidalX(i + ca3D->X[n].i, ca3D->rows),
                             calGetToroidalX(j + ca3D->X[n].j, ca3D->columns),
                             calGetToroidalX(k + ca3D->X[n].k, ca3D->columns));
}



void calSetX3Db(struct CALModel3D* ca3D, struct CALSubstate3Db* Q, int i, int j, int k, int n, CALbyte value)
{
    if (ca3D->T == CAL_SPACE_FLAT)
        calSet3Db(ca3D, Q, i + ca3D->X[n].i, j + ca3D->X[n].j, k + ca3D->X[n].k, value);
    else
        calSet3Db(ca3D, Q, calGetToroidalX(i + ca3D->X[n].i, ca3D->rows),
                  calGetToroidalX(j + ca3D->X[n].j, ca3D->columns),
                  calGetToroidalX(k + ca3D->X[n].k, ca3D->slices), value);
}

void calSetX3Di(struct CALModel3D* ca3D, struct CALSubstate3Di* Q, int i, int j, int k, int n, CALint value)
{
    if (ca3D->T == CAL_SPACE_FLAT)
        calSet3Di(ca3D, Q, i + ca3D->X[n].i, j + ca3D->X[n].j, k + ca3D->X[n].k, value);
    else
        calSet3Di(ca3D, Q, calGetToroidalX(i + ca3D->X[n].i, ca3D->rows),
                  calGetToroidalX(j + ca3D->X[n].j, ca3D->columns),
                  calGetToroidalX(k + ca3D->X[n].k, ca3D->slices), value);
}

void calSetX3Dr(struct CALModel3D* ca3D, struct CALSubstate3Dr* Q, int i, int j, int k, int n, CALreal value)
{
    if (ca3D->T == CAL_SPACE_FLAT)
        calSet3Dr(ca3D, Q, i + ca3D->X[n].i, j + ca3D->X[n].j, k + ca3D->X[n].k, value);
    else
        calSet3Dr(ca3D, Q, calGetToroidalX(i + ca3D->X[n].i, ca3D->rows),
                  calGetToroidalX(j + ca3D->X[n].j, ca3D->columns),
                  calGetToroidalX(k + ca3D->X[n].k, ca3D->slices), value);
}



void calSetCurrentX3Db(struct CALModel3D* ca3D, struct CALSubstate3Db* Q, int i, int j, int k, int n, CALbyte value)
{
    if (ca3D->T == CAL_SPACE_FLAT)
        calSetBuffer3DElement(Q->current, ca3D->rows, ca3D->columns, i + ca3D->X[n].i, j + ca3D->X[n].j, k + ca3D->X[n].k, value);
    else
        calSetBuffer3DElement(Q->current, ca3D->rows, ca3D->columns, calGetToroidalX(i + ca3D->X[n].i, ca3D->rows), calGetToroidalX(j + ca3D->X[n].j, ca3D->columns), calGetToroidalX(k + ca3D->X[n].k, ca3D->slices), value);
}

void calSetCurrentX3Di(struct CALModel3D* ca3D, struct CALSubstate3Di* Q, int i, int j, int k, int n, CALint value)
{
    if (ca3D->T == CAL_SPACE_FLAT)
        calSetBuffer3DElement(Q->current, ca3D->rows, ca3D->columns, i + ca3D->X[n].i, j + ca3D->X[n].j, k + ca3D->X[n].k, value);
    else
        calSetBuffer3DElement(Q->current, ca3D->rows, ca3D->columns, calGetToroidalX(i + ca3D->X[n].i, ca3D->rows), calGetToroidalX(j + ca3D->X[n].j, ca3D->columns), calGetToroidalX(k + ca3D->X[n].k, ca3D->slices), value);
}

void calSetCurrentX3Dr(struct CALModel3D* ca3D, struct CALSubstate3Dr* Q, int i, int j,	int k, int n, CALreal value)
{
    if (ca3D->T == CAL_SPACE_FLAT)
        calSetBuffer3DElement(Q->current, ca3D->rows, ca3D->columns, i + ca3D->X[n].i, j + ca3D->X[n].j, k + ca3D->X[n].k, value);
    else
        calSetBuffer3DElement(Q->current, ca3D->rows, ca3D->columns, calGetToroidalX(i + ca3D->X[n].i, ca3D->rows), calGetToroidalX(j + ca3D->X[n].j, ca3D->columns), calGetToroidalX(k + ca3D->X[n].k, ca3D->slices), value);
}

void calAddNext3Dr(struct CALModel3D *ca3D, struct CALSubstate3Dr *Q, int i, int j, int k, CALreal value) {
    CALreal curr;
    CAL_SET_CELL_LOCK_3D(i, j, k, ca3D);

    curr = calGetBuffer3DElement(Q->next, ca3D->rows, ca3D->columns, i, j, k);
    calSetBuffer3DElement(Q->next, ca3D->rows, ca3D->columns, i, j, k, curr + value);

    CAL_UNSET_CELL_LOCK_3D(i, j, k, ca3D);
}

void calAddNextX3Dr(struct CALModel3D *ca3D, struct CALSubstate3Dr *Q, int i, int j, int k, int n, CALreal value) {
    if (ca3D->T == CAL_SPACE_FLAT)
        calAddNext3Dr(ca3D, Q, i + ca3D->X[n].i,  j + ca3D->X[n].j, k + ca3D->X[n].k, value);
    else
        calAddNext3Dr(ca3D, Q, calGetToroidalX(i + ca3D->X[n].i, ca3D->rows),
                      calGetToroidalX(j + ca3D->X[n].j, ca3D->columns),
                      calGetToroidalX(k + ca3D->X[n].k, ca3D->slices), value);
}

void calAddNext3Db(struct CALModel3D *ca3D, struct CALSubstate3Db *Q, int i, int j, int k, CALbyte value) {
    CALbyte curr;
    CAL_SET_CELL_LOCK_3D(i, j, k, ca3D);

    curr = calGetBuffer3DElement(Q->next, ca3D->rows, ca3D->columns, i, j, k);
    calSetBuffer3DElement(Q->next, ca3D->rows, ca3D->columns, i, j, k, curr + value);

    CAL_UNSET_CELL_LOCK_3D(i, j, k, ca3D);
}

void calAddNextX3Db(struct CALModel3D *ca3D, struct CALSubstate3Db *Q, int i, int j, int k, int n, CALbyte value) {
    if (ca3D->T == CAL_SPACE_FLAT)
        calAddNext3Db(ca3D, Q, i + ca3D->X[n].i,  j + ca3D->X[n].j, k + ca3D->X[n].k, value);
    else
        calAddNext3Db(ca3D, Q, calGetToroidalX(i + ca3D->X[n].i, ca3D->rows),
                      calGetToroidalX(j + ca3D->X[n].j, ca3D->columns),
                      calGetToroidalX(k + ca3D->X[n].k, ca3D->slices), value);

}

void calAddNext3Di(struct CALModel3D *ca3D, struct CALSubstate3Di *Q, int i, int j, int k, CALint value) {
    CALint curr;
    CAL_SET_CELL_LOCK_3D(i, j, k, ca3D);

    curr = calGetBuffer3DElement(Q->next, ca3D->rows, ca3D->columns, i, j, k);
    calSetBuffer3DElement(Q->next, ca3D->rows, ca3D->columns, i, j, k, curr + value);

    CAL_UNSET_CELL_LOCK_3D(i, j, k, ca3D);
}

void calAddNextX3Di(struct CALModel3D *ca3D, struct CALSubstate3Di *Q, int i, int j, int k, int n, CALint value) {
    if (ca3D->T == CAL_SPACE_FLAT)
        calAddNext3Di(ca3D, Q, i + ca3D->X[n].i,  j + ca3D->X[n].j, k + ca3D->X[n].k, value);
    else
        calAddNext3Di(ca3D, Q, calGetToroidalX(i + ca3D->X[n].i, ca3D->rows),
                      calGetToroidalX(j + ca3D->X[n].j, ca3D->columns),
                      calGetToroidalX(k + ca3D->X[n].k, ca3D->slices), value);
}
