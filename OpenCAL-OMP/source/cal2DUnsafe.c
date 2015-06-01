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
#include <calOmpDef.h>

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
	CALint ret;

	CAL_SET_CELL_LOCK(i, j, ca2D);

	ret = calGetMatrixElement(Q->next, ca2D->columns, i, j);

	CAL_UNSET_CELL_LOCK(i, j, ca2D);

	return ret;

}

CALint calGetNext2Di(struct CALModel2D* ca2D, struct CALSubstate2Di* Q, int i, int j) {
	CALint ret;

	CAL_SET_CELL_LOCK(i, j, ca2D);

	ret = calGetMatrixElement(Q->next, ca2D->columns, i, j);

	CAL_UNSET_CELL_LOCK(i, j, ca2D);

	return ret;
}

CALreal calGetNext2Dr(struct CALModel2D* ca2D, struct CALSubstate2Dr* Q, int i, int j) {
	CALreal ret;

	CAL_SET_CELL_LOCK(i, j, ca2D);

	ret = calGetMatrixElement(Q->next, ca2D->columns, i, j);

	CAL_UNSET_CELL_LOCK(i, j, ca2D);

	return ret;
}

CALbyte calGetNextX2Db(struct CALModel2D* ca2D, struct CALSubstate2Db* Q, int i, int j, int n)
{
	if ((ca2D->X_id == CAL_HEXAGONAL_NEIGHBORHOOD_2D && j%2 ==1) || (ca2D->X_id == CAL_HEXAGONAL_NEIGHBORHOOD_ALT_2D && i%2 ==1))
		n += CAL_HEXAGONAL_SHIFT;

	if (ca2D->T == CAL_SPACE_FLAT)
		return calGetNext2Db(ca2D, Q, i + ca2D->X[n].i, j + ca2D->X[n].j);
	else
		return calGetNext2Db(ca2D, Q, calGetToroidalX(i + ca2D->X[n].i, ca2D->rows), calGetToroidalX(j + ca2D->X[n].j, ca2D->columns));

}

CALint calGetNextX2Di(struct CALModel2D* ca2D, struct CALSubstate2Di* Q, int i, int j, int n)
{
	if ((ca2D->X_id == CAL_HEXAGONAL_NEIGHBORHOOD_2D && j%2 ==1) || (ca2D->X_id == CAL_HEXAGONAL_NEIGHBORHOOD_ALT_2D && i%2 ==1))
		n += CAL_HEXAGONAL_SHIFT;

	if (ca2D->T == CAL_SPACE_FLAT)
		return calGetNext2Di(ca2D, Q, i + ca2D->X[n].i, j + ca2D->X[n].j);
	else
		return calGetNext2Di(ca2D, Q, calGetToroidalX(i + ca2D->X[n].i, ca2D->rows), calGetToroidalX(j + ca2D->X[n].j, ca2D->columns));
}

CALreal calGetNextX2Dr(struct CALModel2D* ca2D, struct CALSubstate2Dr* Q, int i, int j, int n)
{
	if ((ca2D->X_id == CAL_HEXAGONAL_NEIGHBORHOOD_2D && j%2 ==1) || (ca2D->X_id == CAL_HEXAGONAL_NEIGHBORHOOD_ALT_2D && i%2 ==1))
		n += CAL_HEXAGONAL_SHIFT;

	if (ca2D->T == CAL_SPACE_FLAT)
		return calGetNext2Dr(ca2D, Q, i + ca2D->X[n].i, j + ca2D->X[n].j);
	else
		return calGetNext2Dr(ca2D, Q, calGetToroidalX(i + ca2D->X[n].i, ca2D->rows), calGetToroidalX(j + ca2D->X[n].j, ca2D->columns));
}


void calSetX2Db(struct CALModel2D* ca2D, struct CALSubstate2Db* Q, int i, int j, int n, CALbyte value)
{
	if ((ca2D->X_id == CAL_HEXAGONAL_NEIGHBORHOOD_2D && j % 2 == 1) || (ca2D->X_id == CAL_HEXAGONAL_NEIGHBORHOOD_ALT_2D && i % 2 == 1))
		n += CAL_HEXAGONAL_SHIFT;

	if (ca2D->T == CAL_SPACE_FLAT)
		calSet2Db(ca2D, Q, i + ca2D->X[n].i, j + ca2D->X[n].j, value);
	else
		calSet2Db(ca2D, Q, calGetToroidalX(i + ca2D->X[n].i, ca2D->rows), calGetToroidalX(j + ca2D->X[n].j, ca2D->columns), value);
}

void calSetX2Di(struct CALModel2D* ca2D, struct CALSubstate2Di* Q, int i, int j, int n, CALint value)
{
	if ((ca2D->X_id == CAL_HEXAGONAL_NEIGHBORHOOD_2D && j%2 ==1) || (ca2D->X_id == CAL_HEXAGONAL_NEIGHBORHOOD_ALT_2D && i%2 ==1))
		n += CAL_HEXAGONAL_SHIFT;

	if (ca2D->T == CAL_SPACE_FLAT)
		calSet2Di(ca2D, Q, i + ca2D->X[n].i, j + ca2D->X[n].j, value);
	else
		calSet2Di(ca2D, Q, calGetToroidalX(i + ca2D->X[n].i, ca2D->rows), calGetToroidalX(j + ca2D->X[n].j, ca2D->columns), value);

}

void calSetX2Dr(struct CALModel2D* ca2D, struct CALSubstate2Dr* Q, int i, int j, int n, CALreal value)
{
	if ((ca2D->X_id == CAL_HEXAGONAL_NEIGHBORHOOD_2D && j%2 ==1) || (ca2D->X_id == CAL_HEXAGONAL_NEIGHBORHOOD_ALT_2D && i%2 ==1))
		n += CAL_HEXAGONAL_SHIFT;

	if (ca2D->T == CAL_SPACE_FLAT)
		calSet2Dr(ca2D, Q, i + ca2D->X[n].i, j + ca2D->X[n].j, value);
	else
		calSet2Dr(ca2D, Q, calGetToroidalX(i + ca2D->X[n].i, ca2D->rows), calGetToroidalX(j + ca2D->X[n].j, ca2D->columns), value);
}

void calSetCurrentX2Db(struct CALModel2D* ca2D, struct CALSubstate2Db* Q, int i, int j, int n, CALbyte value)
{
	if ((ca2D->X_id == CAL_HEXAGONAL_NEIGHBORHOOD_2D && j%2 ==1) || (ca2D->X_id == CAL_HEXAGONAL_NEIGHBORHOOD_ALT_2D && i%2 ==1))
		n += CAL_HEXAGONAL_SHIFT;

	if (ca2D->T == CAL_SPACE_FLAT)
		calSetCurrent2Db(ca2D, Q, i + ca2D->X[n].i, j + ca2D->X[n].j, value);
	else
		calSetCurrent2Db(ca2D, Q, calGetToroidalX(i + ca2D->X[n].i, ca2D->rows),
				 calGetToroidalX(j + ca2D->X[n].j, ca2D->columns), value);
}

void calSetCurrentX2Di(struct CALModel2D* ca2D, struct CALSubstate2Di* Q, int i, int j, int n, CALint value)
{
	if ((ca2D->X_id == CAL_HEXAGONAL_NEIGHBORHOOD_2D && j%2 ==1) || (ca2D->X_id == CAL_HEXAGONAL_NEIGHBORHOOD_ALT_2D && i%2 ==1))
		n += CAL_HEXAGONAL_SHIFT;

	if (ca2D->T == CAL_SPACE_FLAT)
		calSetCurrent2Di(ca2D, Q, i + ca2D->X[n].i, j + ca2D->X[n].j, value);
	else
		calSetCurrent2Di(ca2D, Q, calGetToroidalX(i + ca2D->X[n].i, ca2D->rows),
				 calGetToroidalX(j + ca2D->X[n].j, ca2D->columns), value);
}

void calSetCurrentX2Dr(struct CALModel2D* ca2D, struct CALSubstate2Dr* Q, int i, int j,	int n, CALreal value)
{
	if ((ca2D->X_id == CAL_HEXAGONAL_NEIGHBORHOOD_2D && j%2 ==1) || (ca2D->X_id == CAL_HEXAGONAL_NEIGHBORHOOD_ALT_2D && i%2 ==1))
		n += CAL_HEXAGONAL_SHIFT;

	if (ca2D->T == CAL_SPACE_FLAT)
		calSetCurrent2Dr(ca2D, Q, i + ca2D->X[n].i, j + ca2D->X[n].j, value);
	else
		calSetCurrent2Dr(ca2D, Q, calGetToroidalX(i + ca2D->X[n].i, ca2D->rows),
				 calGetToroidalX(j + ca2D->X[n].j, ca2D->columns), value);

}

void calAddNext2Dr(struct CALModel2D *ca2D, struct CALSubstate2Dr *Q, int i, int j, CALreal value) {
	CALreal curr;
	CAL_SET_CELL_LOCK(i, j, ca2D);

	curr = calGetMatrixElement(Q->next, ca2D->columns, i, j);
	calSetMatrixElement(Q->next, ca2D->columns, i, j, curr + value);

	CAL_UNSET_CELL_LOCK(i, j, ca2D);
}

void calAddNextX2Dr(struct CALModel2D *ca2D, struct CALSubstate2Dr *Q, int i, int j, int n, CALreal value) {
	if ((ca2D->X_id == CAL_HEXAGONAL_NEIGHBORHOOD_2D && j%2 ==1) || (ca2D->X_id == CAL_HEXAGONAL_NEIGHBORHOOD_ALT_2D && i%2 ==1))
		n += CAL_HEXAGONAL_SHIFT;

	if (ca2D->T == CAL_SPACE_FLAT)
		calAddNext2Dr(ca2D, Q, i + ca2D->X[n].i,  j + ca2D->X[n].j, value);
	else
		calAddNext2Dr(ca2D, Q, calGetToroidalX(i + ca2D->X[n].i, ca2D->rows),
			      calGetToroidalX(j + ca2D->X[n].j, ca2D->columns), value);

}
void calAddNext2Db(struct CALModel2D *ca2D, struct CALSubstate2Db *Q, int i, int j, CALbyte value) {
	CALbyte curr;
	CAL_SET_CELL_LOCK(i, j, ca2D);

	curr = calGetMatrixElement(Q->next, ca2D->columns, i, j);
	calSetMatrixElement(Q->next, ca2D->columns, i, j, curr + value);

	CAL_UNSET_CELL_LOCK(i, j, ca2D);
}

void calAddNextX2Db(struct CALModel2D *ca2D, struct CALSubstate2Db *Q, int i, int j, int n, CALbyte value) {
	if ((ca2D->X_id == CAL_HEXAGONAL_NEIGHBORHOOD_2D && j%2 ==1) || (ca2D->X_id == CAL_HEXAGONAL_NEIGHBORHOOD_ALT_2D && i%2 ==1))
		n += CAL_HEXAGONAL_SHIFT;

	if (ca2D->T == CAL_SPACE_FLAT)
		calAddNext2Db(ca2D, Q, i + ca2D->X[n].i,  j + ca2D->X[n].j, value);
	else
		calAddNext2Db(ca2D, Q, calGetToroidalX(i + ca2D->X[n].i, ca2D->rows),
			      calGetToroidalX(j + ca2D->X[n].j, ca2D->columns), value);

}
void calAddNext2Di(struct CALModel2D *ca2D, struct CALSubstate2Di *Q, int i, int j, CALint value) {
	CALint curr;
	CAL_SET_CELL_LOCK(i, j, ca2D);

	curr = calGetMatrixElement(Q->next, ca2D->columns, i, j);
	calSetMatrixElement(Q->next, ca2D->columns, i, j, curr + value);

	CAL_UNSET_CELL_LOCK(i, j, ca2D);
}

void calAddNextX2Di(struct CALModel2D *ca2D, struct CALSubstate2Di *Q, int i, int j, int n, CALint value) {
	if ((ca2D->X_id == CAL_HEXAGONAL_NEIGHBORHOOD_2D && j%2 ==1) || (ca2D->X_id == CAL_HEXAGONAL_NEIGHBORHOOD_ALT_2D && i%2 ==1))
		n += CAL_HEXAGONAL_SHIFT;

	if (ca2D->T == CAL_SPACE_FLAT)
		calAddNext2Di(ca2D, Q, i + ca2D->X[n].i,  j + ca2D->X[n].j, value);
	else
		calAddNext2Di(ca2D, Q, calGetToroidalX(i + ca2D->X[n].i, ca2D->rows),
			      calGetToroidalX(j + ca2D->X[n].j, ca2D->columns), value);

}
