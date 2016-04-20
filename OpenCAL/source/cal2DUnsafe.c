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

#include <OpenCAL/cal2DBuffer.h>
#include <OpenCAL/cal2DUnsafe.h>
#include <stdlib.h>



void calAddActiveCellX2D(struct CALModel2D* ca2D, int i, int j, int n)
{
	if ((ca2D->X_id == CAL_HEXAGONAL_NEIGHBORHOOD_2D && j%2 ==1) || (ca2D->X_id == CAL_HEXAGONAL_NEIGHBORHOOD_ALT_2D && i%2 ==1))
		n += CAL_HEXAGONAL_SHIFT;

	if (ca2D->T == CAL_SPACE_FLAT)
	{
		if (!calGetMatrixElement(ca2D->A.flags, ca2D->columns, (i + ca2D->X[n].i), (j + ca2D->X[n].j)))
		{
			calSetMatrixElement(ca2D->A.flags, ca2D->columns, (i + ca2D->X[n].i), (j + ca2D->X[n].j), CAL_TRUE);
			ca2D->A.size_next++;
		}
	}
	else
	{
		if (!calGetMatrixElement(ca2D->A.flags, ca2D->columns, calGetToroidalX(i + ca2D->X[n].i, ca2D->rows), calGetToroidalX(j + ca2D->X[n].j, ca2D->columns)))
		{
			calSetMatrixElement(ca2D->A.flags, ca2D->columns, calGetToroidalX(i + ca2D->X[n].i, ca2D->rows), calGetToroidalX(j + ca2D->X[n].j, ca2D->columns), CAL_TRUE);
			ca2D->A.size_next++;
		}
	}
}



void calInitX2Db(struct CALModel2D* ca2D, struct CALSubstate2Db* Q, int i, int j, int n, CALbyte value)
{
	if ((ca2D->X_id == CAL_HEXAGONAL_NEIGHBORHOOD_2D && j%2 ==1) || (ca2D->X_id == CAL_HEXAGONAL_NEIGHBORHOOD_ALT_2D && i%2 ==1))
		n += CAL_HEXAGONAL_SHIFT;

	if (ca2D->T == CAL_SPACE_FLAT)
	{
		calSetMatrixElement(Q->current, ca2D->columns, i + ca2D->X[n].i, j + ca2D->X[n].j, value);
		calSetMatrixElement(Q->next, ca2D->columns, i + ca2D->X[n].i, j + ca2D->X[n].j, value);
	}
	else
	{
		calSetMatrixElement(Q->current, ca2D->columns, calGetToroidalX(i + ca2D->X[n].i, ca2D->rows), calGetToroidalX(j + ca2D->X[n].j, ca2D->columns), value);
		calSetMatrixElement(Q->next, ca2D->columns, calGetToroidalX(i + ca2D->X[n].i, ca2D->rows), calGetToroidalX(j + ca2D->X[n].j, ca2D->columns), value);
	}
}

void calInitX2Di(struct CALModel2D* ca2D, struct CALSubstate2Di* Q, int i, int j, int n, CALint value)
{
	if ((ca2D->X_id == CAL_HEXAGONAL_NEIGHBORHOOD_2D && j%2 ==1) || (ca2D->X_id == CAL_HEXAGONAL_NEIGHBORHOOD_ALT_2D && i%2 ==1))
		n += CAL_HEXAGONAL_SHIFT;

	if (ca2D->T == CAL_SPACE_FLAT)
	{
		calSetMatrixElement(Q->current, ca2D->columns, i + ca2D->X[n].i, j + ca2D->X[n].j, value);
		calSetMatrixElement(Q->next, ca2D->columns, i + ca2D->X[n].i, j + ca2D->X[n].j, value);
	}
	else
	{
		calSetMatrixElement(Q->current, ca2D->columns, calGetToroidalX(i + ca2D->X[n].i, ca2D->rows), calGetToroidalX(j + ca2D->X[n].j, ca2D->columns), value);
		calSetMatrixElement(Q->next, ca2D->columns, calGetToroidalX(i + ca2D->X[n].i, ca2D->rows), calGetToroidalX(j + ca2D->X[n].j, ca2D->columns), value);
	}
}

void calInitX2Dr(struct CALModel2D* ca2D, struct CALSubstate2Dr* Q, int i, int j, int n, CALreal value)
{
	if ((ca2D->X_id == CAL_HEXAGONAL_NEIGHBORHOOD_2D && j%2 ==1) || (ca2D->X_id == CAL_HEXAGONAL_NEIGHBORHOOD_ALT_2D && i%2 ==1))
		n += CAL_HEXAGONAL_SHIFT;

	if (ca2D->T == CAL_SPACE_FLAT)
	{
		calSetMatrixElement(Q->current, ca2D->columns, i + ca2D->X[n].i, j + ca2D->X[n].j, value);
		calSetMatrixElement(Q->next, ca2D->columns, i + ca2D->X[n].i, j + ca2D->X[n].j, value);
	}
	else
	{
		calSetMatrixElement(Q->current, ca2D->columns, calGetToroidalX(i + ca2D->X[n].i, ca2D->rows), calGetToroidalX(j + ca2D->X[n].j, ca2D->columns), value);
		calSetMatrixElement(Q->next, ca2D->columns, calGetToroidalX(i + ca2D->X[n].i, ca2D->rows), calGetToroidalX(j + ca2D->X[n].j, ca2D->columns), value);
	}
}



CALbyte calGetNext2Db(struct CALModel2D* ca2D, struct CALSubstate2Db* Q, int i, int j) {
	return calGetMatrixElement(Q->next, ca2D->columns, i, j);
}

CALint calGetNext2Di(struct CALModel2D* ca2D, struct CALSubstate2Di* Q, int i, int j) {
	return calGetMatrixElement(Q->next, ca2D->columns, i, j);
}

CALreal calGetNext2Dr(struct CALModel2D* ca2D, struct CALSubstate2Dr* Q, int i, int j) {
	return calGetMatrixElement(Q->next, ca2D->columns, i, j);
}



CALbyte calGetNextX2Db(struct CALModel2D* ca2D, struct CALSubstate2Db* Q, int i, int j, int n)
{
	if ((ca2D->X_id == CAL_HEXAGONAL_NEIGHBORHOOD_2D && j%2 ==1) || (ca2D->X_id == CAL_HEXAGONAL_NEIGHBORHOOD_ALT_2D && i%2 ==1))
		n += CAL_HEXAGONAL_SHIFT;

	if (ca2D->T == CAL_SPACE_FLAT)
		return calGetMatrixElement(Q->next, ca2D->columns, i + ca2D->X[n].i, j + ca2D->X[n].j);
	else
		return calGetMatrixElement(Q->next, ca2D->columns, calGetToroidalX(i + ca2D->X[n].i, ca2D->rows), calGetToroidalX(j + ca2D->X[n].j, ca2D->columns));
}

CALint calGetNextX2Di(struct CALModel2D* ca2D, struct CALSubstate2Di* Q, int i, int j, int n)
{
	if ((ca2D->X_id == CAL_HEXAGONAL_NEIGHBORHOOD_2D && j%2 ==1) || (ca2D->X_id == CAL_HEXAGONAL_NEIGHBORHOOD_ALT_2D && i%2 ==1))
		n += CAL_HEXAGONAL_SHIFT;

	if (ca2D->T == CAL_SPACE_FLAT)
		return calGetMatrixElement(Q->next, ca2D->columns, i + ca2D->X[n].i, j + ca2D->X[n].j);
	else
		return calGetMatrixElement(Q->next, ca2D->columns, calGetToroidalX(i + ca2D->X[n].i, ca2D->rows), calGetToroidalX(j + ca2D->X[n].j, ca2D->columns));
}

CALreal calGetNextX2Dr(struct CALModel2D* ca2D, struct CALSubstate2Dr* Q, int i, int j, int n)
{
	if ((ca2D->X_id == CAL_HEXAGONAL_NEIGHBORHOOD_2D && j%2 ==1) || (ca2D->X_id == CAL_HEXAGONAL_NEIGHBORHOOD_ALT_2D && i%2 ==1))
		n += CAL_HEXAGONAL_SHIFT;

	if (ca2D->T == CAL_SPACE_FLAT)
		return calGetMatrixElement(Q->next, ca2D->columns, i + ca2D->X[n].i, j + ca2D->X[n].j);
	else
		return calGetMatrixElement(Q->next, ca2D->columns, calGetToroidalX(i + ca2D->X[n].i, ca2D->rows), calGetToroidalX(j + ca2D->X[n].j, ca2D->columns));
}



void calSetX2Db(struct CALModel2D* ca2D, struct CALSubstate2Db* Q, int i, int j, int n, CALbyte value)
{
	if ((ca2D->X_id == CAL_HEXAGONAL_NEIGHBORHOOD_2D && j%2 ==1) || (ca2D->X_id == CAL_HEXAGONAL_NEIGHBORHOOD_ALT_2D && i%2 ==1))
		n += CAL_HEXAGONAL_SHIFT;

	if (ca2D->T == CAL_SPACE_FLAT)
		calSetMatrixElement(Q->next, ca2D->columns, i + ca2D->X[n].i, j + ca2D->X[n].j, value);
	else
		calSetMatrixElement(Q->next, ca2D->columns, calGetToroidalX(i + ca2D->X[n].i, ca2D->rows), calGetToroidalX(j + ca2D->X[n].j, ca2D->columns), value);
}

void calSetX2Di(struct CALModel2D* ca2D, struct CALSubstate2Di* Q, int i, int j, int n, CALint value)
{
	if ((ca2D->X_id == CAL_HEXAGONAL_NEIGHBORHOOD_2D && j%2 ==1) || (ca2D->X_id == CAL_HEXAGONAL_NEIGHBORHOOD_ALT_2D && i%2 ==1))
		n += CAL_HEXAGONAL_SHIFT;

	if (ca2D->T == CAL_SPACE_FLAT)
		calSetMatrixElement(Q->next, ca2D->columns, i + ca2D->X[n].i, j + ca2D->X[n].j, value);
	else
		calSetMatrixElement(Q->next, ca2D->columns, calGetToroidalX(i + ca2D->X[n].i, ca2D->rows), calGetToroidalX(j + ca2D->X[n].j, ca2D->columns), value);
}

void calSetX2Dr(struct CALModel2D* ca2D, struct CALSubstate2Dr* Q, int i, int j, int n, CALreal value)
{
	if ((ca2D->X_id == CAL_HEXAGONAL_NEIGHBORHOOD_2D && j%2 ==1) || (ca2D->X_id == CAL_HEXAGONAL_NEIGHBORHOOD_ALT_2D && i%2 ==1))
		n += CAL_HEXAGONAL_SHIFT;

	if (ca2D->T == CAL_SPACE_FLAT)
		calSetMatrixElement(Q->next, ca2D->columns, i + ca2D->X[n].i, j + ca2D->X[n].j, value);
	else
		calSetMatrixElement(Q->next, ca2D->columns, calGetToroidalX(i + ca2D->X[n].i, ca2D->rows), calGetToroidalX(j + ca2D->X[n].j, ca2D->columns), value);
}



void calSetCurrentX2Db(struct CALModel2D* ca2D, struct CALSubstate2Db* Q, int i, int j, int n, CALbyte value)
{
	if ((ca2D->X_id == CAL_HEXAGONAL_NEIGHBORHOOD_2D && j%2 ==1) || (ca2D->X_id == CAL_HEXAGONAL_NEIGHBORHOOD_ALT_2D && i%2 ==1))
		n += CAL_HEXAGONAL_SHIFT;

	if (ca2D->T == CAL_SPACE_FLAT)
		calSetMatrixElement(Q->current, ca2D->columns, i + ca2D->X[n].i, j + ca2D->X[n].j, value);
	else
		calSetMatrixElement(Q->current, ca2D->columns, calGetToroidalX(i + ca2D->X[n].i, ca2D->rows), calGetToroidalX(j + ca2D->X[n].j, ca2D->columns), value);
}

void calSetCurrentX2Di(struct CALModel2D* ca2D, struct CALSubstate2Di* Q, int i, int j, int n, CALint value)
{
	if ((ca2D->X_id == CAL_HEXAGONAL_NEIGHBORHOOD_2D && j%2 ==1) || (ca2D->X_id == CAL_HEXAGONAL_NEIGHBORHOOD_ALT_2D && i%2 ==1))
		n += CAL_HEXAGONAL_SHIFT;

	if (ca2D->T == CAL_SPACE_FLAT)
		calSetMatrixElement(Q->current, ca2D->columns, i + ca2D->X[n].i, j + ca2D->X[n].j, value);
	else
		calSetMatrixElement(Q->current, ca2D->columns, calGetToroidalX(i + ca2D->X[n].i, ca2D->rows), calGetToroidalX(j + ca2D->X[n].j, ca2D->columns), value);
}

void calSetCurrentX2Dr(struct CALModel2D* ca2D, struct CALSubstate2Dr* Q, int i, int j,	int n, CALreal value)
{
	if ((ca2D->X_id == CAL_HEXAGONAL_NEIGHBORHOOD_2D && j%2 ==1) || (ca2D->X_id == CAL_HEXAGONAL_NEIGHBORHOOD_ALT_2D && i%2 ==1))
		n += CAL_HEXAGONAL_SHIFT;

	if (ca2D->T == CAL_SPACE_FLAT)
		calSetMatrixElement(Q->current, ca2D->columns, i + ca2D->X[n].i, j + ca2D->X[n].j, value);
	else
		calSetMatrixElement(Q->current, ca2D->columns, calGetToroidalX(i + ca2D->X[n].i, ca2D->rows), calGetToroidalX(j + ca2D->X[n].j, ca2D->columns), value);
}
