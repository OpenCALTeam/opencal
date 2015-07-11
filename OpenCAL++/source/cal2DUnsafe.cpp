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
#include <cal2DUnsafe.h>
#include <stdlib.h>



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
