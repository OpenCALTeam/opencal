/*
 * Copyright (c) 2016 OpenCALTeam (https://github.com/OpenCALTeam),
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
#include <OpenCAL/cal3DUnsafe.h>
#include <stdlib.h>



void calAddActiveCellX3D(struct CALModel3D* ca3D, int i, int j, int k, int n)
{
	if (ca3D->T == CAL_SPACE_FLAT)
	{
		if (!calGetBuffer3DElement(ca3D->A.flags, ca3D->rows, ca3D->columns, (i + ca3D->X[n].i), (j + ca3D->X[n].j), (k + ca3D->X[n].k)))
		{
			calSetBuffer3DElement(ca3D->A.flags, ca3D->rows, ca3D->columns, (i + ca3D->X[n].i), (j + ca3D->X[n].j), (k + ca3D->X[n].k), CAL_TRUE);
			ca3D->A.size_next++;
		}
	}
	else
	{
		if (!calGetBuffer3DElement(ca3D->A.flags, ca3D->rows, ca3D->columns, calGetToroidalX(i + ca3D->X[n].i, ca3D->rows), calGetToroidalX(j + ca3D->X[n].j, ca3D->columns), calGetToroidalX(k + ca3D->X[n].k, ca3D->slices)))
		{
			calSetBuffer3DElement(ca3D->A.flags, ca3D->rows, ca3D->columns, calGetToroidalX(i + ca3D->X[n].i, ca3D->rows), calGetToroidalX(j + ca3D->X[n].j, ca3D->columns), calGetToroidalX(k + ca3D->X[n].k, ca3D->slices), CAL_TRUE);
			ca3D->A.size_next++;
		}
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
	return calGetBuffer3DElement(Q->next, ca3D->rows, ca3D->columns, i, j, k);
}

CALint calGetNext3Di(struct CALModel3D* ca3D, struct CALSubstate3Di* Q, int i, int j, int k) {
	return calGetBuffer3DElement(Q->next, ca3D->rows, ca3D->columns, i, j, k);
}

CALreal calGetNext3Dr(struct CALModel3D* ca3D, struct CALSubstate3Dr* Q, int i, int j, int k) {
	return calGetBuffer3DElement(Q->next, ca3D->rows, ca3D->columns, i, j, k);
}



CALbyte calGetNextX3Db(struct CALModel3D* ca3D, struct CALSubstate3Db* Q, int i, int j, int k, int n)
{
	if (ca3D->T == CAL_SPACE_FLAT)
		return calGetBuffer3DElement(Q->next, ca3D->rows, ca3D->columns, i + ca3D->X[n].i, j + ca3D->X[n].j, k + ca3D->X[n].k);
	else
		return calGetBuffer3DElement(Q->next, ca3D->rows, ca3D->columns, calGetToroidalX(i + ca3D->X[n].i, ca3D->rows), calGetToroidalX(j + ca3D->X[n].j, ca3D->columns), calGetToroidalX(k + ca3D->X[n].k, ca3D->columns));
}

CALint calGetNextX3Di(struct CALModel3D* ca3D, struct CALSubstate3Di* Q, int i, int j, int k, int n)
{
	if (ca3D->T == CAL_SPACE_FLAT)
		return calGetBuffer3DElement(Q->next, ca3D->rows, ca3D->columns, i + ca3D->X[n].i, j + ca3D->X[n].j, k + ca3D->X[n].k);
	else
		return calGetBuffer3DElement(Q->next, ca3D->rows, ca3D->columns, calGetToroidalX(i + ca3D->X[n].i, ca3D->rows), calGetToroidalX(j + ca3D->X[n].j, ca3D->columns), calGetToroidalX(k + ca3D->X[n].k, ca3D->columns));
}

CALreal calGetNextX3Dr(struct CALModel3D* ca3D, struct CALSubstate3Dr* Q, int i, int j, int k, int n)
{
	if (ca3D->T == CAL_SPACE_FLAT)
		return calGetBuffer3DElement(Q->next, ca3D->rows, ca3D->columns, i + ca3D->X[n].i, j + ca3D->X[n].j, k + ca3D->X[n].k);
	else
		return calGetBuffer3DElement(Q->next, ca3D->rows, ca3D->columns, calGetToroidalX(i + ca3D->X[n].i, ca3D->rows), calGetToroidalX(j + ca3D->X[n].j, ca3D->columns), calGetToroidalX(k + ca3D->X[n].k, ca3D->columns));
}



void calSetX3Db(struct CALModel3D* ca3D, struct CALSubstate3Db* Q, int i, int j, int k, int n, CALbyte value)
{
	if (ca3D->T == CAL_SPACE_FLAT)
		calSetBuffer3DElement(Q->next, ca3D->rows, ca3D->columns, i + ca3D->X[n].i, j + ca3D->X[n].j, k + ca3D->X[n].k, value);
	else
		calSetBuffer3DElement(Q->next, ca3D->rows, ca3D->columns, calGetToroidalX(i + ca3D->X[n].i, ca3D->rows), calGetToroidalX(j + ca3D->X[n].j, ca3D->columns), calGetToroidalX(k + ca3D->X[n].k, ca3D->slices), value);
}

void calSetX3Di(struct CALModel3D* ca3D, struct CALSubstate3Di* Q, int i, int j, int k, int n, CALint value)
{
	if (ca3D->T == CAL_SPACE_FLAT)
		calSetBuffer3DElement(Q->next, ca3D->rows, ca3D->columns, i + ca3D->X[n].i, j + ca3D->X[n].j, k + ca3D->X[n].k, value);
	else
		calSetBuffer3DElement(Q->next, ca3D->rows, ca3D->columns, calGetToroidalX(i + ca3D->X[n].i, ca3D->rows), calGetToroidalX(j + ca3D->X[n].j, ca3D->columns), calGetToroidalX(k + ca3D->X[n].k, ca3D->slices), value);
}

void calSetX3Dr(struct CALModel3D* ca3D, struct CALSubstate3Dr* Q, int i, int j, int k, int n, CALreal value)
{
	if (ca3D->T == CAL_SPACE_FLAT)
		calSetBuffer3DElement(Q->next, ca3D->rows, ca3D->columns, i + ca3D->X[n].i, j + ca3D->X[n].j, k + ca3D->X[n].k, value);
	else
		calSetBuffer3DElement(Q->next, ca3D->rows, ca3D->columns, calGetToroidalX(i + ca3D->X[n].i, ca3D->rows), calGetToroidalX(j + ca3D->X[n].j, ca3D->columns), calGetToroidalX(k + ca3D->X[n].k, ca3D->slices), value);
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
