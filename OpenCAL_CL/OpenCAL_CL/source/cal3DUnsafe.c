#include <cal3DBuffer.h>
#include <cal3DUnsafe.h>
#include <stdlib.h>



void calInitX3Db(struct CALModel3D* ca3D, struct CALSubstate3Db* Q, int i, int j, int k, int n, CALbyte value)
{
	if (ca3D->T == CAL_SPACE_FLAT)
	{
		calSetBuffer3DElement(Q->current, ca3D->rows, ca3D->columns, i+ca3D->X[n].i, j+ca3D->X[n].j, k+ca3D->X[n].k, value);
		calSetBuffer3DElement(Q->next, ca3D->rows, ca3D->columns, i+ca3D->X[n].i, j+ca3D->X[n].j, k+ca3D->X[n].k, value);
	}
	else 
	{
		calSetBuffer3DElement(Q->current, ca3D->rows, ca3D->columns, calGetToroidalX(i+ca3D->X[n].i, ca3D->rows), calGetToroidalX(j+ca3D->X[n].j, ca3D->columns), calGetToroidalX(k+ca3D->X[n].k, ca3D->layers), value);
		calSetBuffer3DElement(Q->next, ca3D->rows, ca3D->columns, calGetToroidalX(i+ca3D->X[n].i, ca3D->rows), calGetToroidalX(j+ca3D->X[n].j, ca3D->columns), calGetToroidalX(k+ca3D->X[n].k, ca3D->layers), value);
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
		calSetBuffer3DElement(Q->current, ca3D->rows, ca3D->columns, calGetToroidalX(i+ca3D->X[n].i, ca3D->rows), calGetToroidalX(j+ca3D->X[n].j, ca3D->columns), calGetToroidalX(k+ca3D->X[n].k, ca3D->layers), value);
		calSetBuffer3DElement(Q->next, ca3D->rows, ca3D->columns, calGetToroidalX(i+ca3D->X[n].i, ca3D->rows), calGetToroidalX(j+ca3D->X[n].j, ca3D->columns), calGetToroidalX(k+ca3D->X[n].k, ca3D->layers), value);
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
		calSetBuffer3DElement(Q->current, ca3D->rows, ca3D->columns, calGetToroidalX(i+ca3D->X[n].i, ca3D->rows), calGetToroidalX(j+ca3D->X[n].j, ca3D->columns), calGetToroidalX(k+ca3D->X[n].k, ca3D->layers), value);
		calSetBuffer3DElement(Q->next, ca3D->rows, ca3D->columns, calGetToroidalX(i+ca3D->X[n].i, ca3D->rows), calGetToroidalX(j+ca3D->X[n].j, ca3D->columns), calGetToroidalX(k+ca3D->X[n].k, ca3D->layers), value);
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
		calSetBuffer3DElement(Q->next, ca3D->rows, ca3D->columns, calGetToroidalX(i + ca3D->X[n].i, ca3D->rows), calGetToroidalX(j + ca3D->X[n].j, ca3D->columns), calGetToroidalX(k + ca3D->X[n].k, ca3D->layers), value);
}

void calSetX3Di(struct CALModel3D* ca3D, struct CALSubstate3Di* Q, int i, int j, int k, int n, CALint value)
{
	if (ca3D->T == CAL_SPACE_FLAT)
		calSetBuffer3DElement(Q->next, ca3D->rows, ca3D->columns, i + ca3D->X[n].i, j + ca3D->X[n].j, k + ca3D->X[n].k, value);
	else
		calSetBuffer3DElement(Q->next, ca3D->rows, ca3D->columns, calGetToroidalX(i + ca3D->X[n].i, ca3D->rows), calGetToroidalX(j + ca3D->X[n].j, ca3D->columns), calGetToroidalX(k + ca3D->X[n].k, ca3D->layers), value);
}

void calSetX3Dr(struct CALModel3D* ca3D, struct CALSubstate3Dr* Q, int i, int j, int k, int n, CALreal value)
{
	if (ca3D->T == CAL_SPACE_FLAT)
		calSetBuffer3DElement(Q->next, ca3D->rows, ca3D->columns, i + ca3D->X[n].i, j + ca3D->X[n].j, k + ca3D->X[n].k, value);
	else
		calSetBuffer3DElement(Q->next, ca3D->rows, ca3D->columns, calGetToroidalX(i + ca3D->X[n].i, ca3D->rows), calGetToroidalX(j + ca3D->X[n].j, ca3D->columns), calGetToroidalX(k + ca3D->X[n].k, ca3D->layers), value);
}



void calSetCurrentX3Db(struct CALModel3D* ca3D, struct CALSubstate3Db* Q, int i, int j, int k, int n, CALbyte value)
{
	if (ca3D->T == CAL_SPACE_FLAT)
		calSetBuffer3DElement(Q->current, ca3D->rows, ca3D->columns, i + ca3D->X[n].i, j + ca3D->X[n].j, k + ca3D->X[n].k, value);
	else
		calSetBuffer3DElement(Q->current, ca3D->rows, ca3D->columns, calGetToroidalX(i + ca3D->X[n].i, ca3D->rows), calGetToroidalX(j + ca3D->X[n].j, ca3D->columns), calGetToroidalX(k + ca3D->X[n].k, ca3D->layers), value);
}

void calSetCurrentX3Di(struct CALModel3D* ca3D, struct CALSubstate3Di* Q, int i, int j, int k, int n, CALint value)
{
if (ca3D->T == CAL_SPACE_FLAT)
		calSetBuffer3DElement(Q->current, ca3D->rows, ca3D->columns, i + ca3D->X[n].i, j + ca3D->X[n].j, k + ca3D->X[n].k, value);
	else
		calSetBuffer3DElement(Q->current, ca3D->rows, ca3D->columns, calGetToroidalX(i + ca3D->X[n].i, ca3D->rows), calGetToroidalX(j + ca3D->X[n].j, ca3D->columns), calGetToroidalX(k + ca3D->X[n].k, ca3D->layers), value);
}

void calSetCurrentX3Dr(struct CALModel3D* ca3D, struct CALSubstate3Dr* Q, int i, int j,	int k, int n, CALreal value)
{
if (ca3D->T == CAL_SPACE_FLAT)
		calSetBuffer3DElement(Q->current, ca3D->rows, ca3D->columns, i + ca3D->X[n].i, j + ca3D->X[n].j, k + ca3D->X[n].k, value);
	else
		calSetBuffer3DElement(Q->current, ca3D->rows, ca3D->columns, calGetToroidalX(i + ca3D->X[n].i, ca3D->rows), calGetToroidalX(j + ca3D->X[n].j, ca3D->columns), calGetToroidalX(k + ca3D->X[n].k, ca3D->layers), value);
}
