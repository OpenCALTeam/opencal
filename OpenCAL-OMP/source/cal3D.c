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

#include <OpenCAL-OMP/calOmpDef.h>
#include <OpenCAL-OMP/cal3D.h>
#include <OpenCAL-OMP/cal3DBuffer.h>
#include <stdlib.h>
#include <string.h>



/******************************************************************************
							PRIVATE FUNCIONS

*******************************************************************************/



/*! \brief Builds the 3D pre-defined von Neumann neighborhood.
*/
void calDefineVonNeumannNeighborhood3D(struct CALModel3D* ca3D	//!< Pointer to the cellular automaton structure.
									   )
{
	/*
	     slice -1       slice 0       slice 1

	   |   |         | 1 |         |   |
	---|---|---   ---|---|---   ---|---|---
	   | 5 |       2 | 0 | 3       | 6 |
	---|---|---   ---|---|---   ---|---|---
	   |   |         | 4 |         |   |
   */

	//slice  0
	calAddNeighbor3D(ca3D,   0,   0,   0);
	calAddNeighbor3D(ca3D, - 1,   0,   0);
	calAddNeighbor3D(ca3D,   0, - 1,   0);
	calAddNeighbor3D(ca3D,   0, + 1,   0);
	calAddNeighbor3D(ca3D, + 1,   0,   0);
	//slice -1
	calAddNeighbor3D(ca3D,   0,   0, - 1);
	//slice +1
	calAddNeighbor3D(ca3D,   0,   0, + 1);
}



/*! \brief Builds the 3D pre-defined Moore neighborhood.
*/
void calDefineMooreNeighborhood3D(struct CALModel3D* ca3D		//!< Pointer to the cellular automaton structure.
								  )
{
	/*
	     slice -1       slice 0       slice 1

	14 |10 | 17    5 | 1 | 8    23 |19 | 26
	---|---|---   ---|---|---   ---|---|---
	11 | 9 | 12    2 | 0 | 3    20 |18 | 21
	---|---|---   ---|---|---   ---|---|---
	15 |13 | 16    6 | 4 | 7    24 |22 | 25
   */

	//slice  0
	calAddNeighbor3D(ca3D,   0,   0,   0);
    calAddNeighbor3D(ca3D, - 1,   0,   0);
    calAddNeighbor3D(ca3D,   0, - 1,   0);
	calAddNeighbor3D(ca3D,   0, + 1,   0);
    calAddNeighbor3D(ca3D, + 1,   0,   0);
    calAddNeighbor3D(ca3D, - 1, - 1,   0);
    calAddNeighbor3D(ca3D, + 1, - 1,   0);
	calAddNeighbor3D(ca3D, + 1, + 1,   0);
	calAddNeighbor3D(ca3D, - 1, + 1,   0);
	//slice -1
	calAddNeighbor3D(ca3D,   0,   0, - 1);
    calAddNeighbor3D(ca3D, - 1,   0, - 1);
    calAddNeighbor3D(ca3D,   0, - 1, - 1);
	calAddNeighbor3D(ca3D,   0, + 1, - 1);
    calAddNeighbor3D(ca3D, + 1,   0, - 1);
    calAddNeighbor3D(ca3D, - 1, - 1, - 1);
    calAddNeighbor3D(ca3D, + 1, - 1, - 1);
	calAddNeighbor3D(ca3D, + 1, + 1, - 1);
	calAddNeighbor3D(ca3D, - 1, + 1, - 1);
	//slice +1
	calAddNeighbor3D(ca3D,   0,   0, + 1);
    calAddNeighbor3D(ca3D, - 1,   0, + 1);
    calAddNeighbor3D(ca3D,   0, - 1, + 1);
	calAddNeighbor3D(ca3D,   0, + 1, + 1);
    calAddNeighbor3D(ca3D, + 1,   0, + 1);
    calAddNeighbor3D(ca3D, - 1, - 1, + 1);
    calAddNeighbor3D(ca3D, + 1, - 1, + 1);
	calAddNeighbor3D(ca3D, + 1, + 1, + 1);
	calAddNeighbor3D(ca3D, - 1, + 1, + 1);
}



/*! \brief 8 bit (256 values) integer substates allocation function.
*/
CALbyte calAllocSubstate3Db(struct CALModel3D* ca3D,	//!< Pointer to the cellular automaton structure.
							struct CALSubstate3Db* Q	//!< Pointer to a 3D byte substate.
							)
{
	Q->current = calAllocBuffer3Db(ca3D->rows, ca3D->columns, ca3D->slices);
	Q->next = calAllocBuffer3Db(ca3D->rows, ca3D->columns, ca3D->slices);

	if (!Q->current || !Q->next)
		return CAL_FALSE;

	return CAL_TRUE;
}

/*! \brief Integer substates allocation function.
*/
CALbyte calAllocSubstate3Di(struct CALModel3D* ca3D,	//!< Pointer to the cellular automaton structure.
							struct CALSubstate3Di* Q	//!< Pointer to a 3D int substate.
							)
{
	Q->current = calAllocBuffer3Di(ca3D->rows, ca3D->columns, ca3D->slices);
	Q->next = calAllocBuffer3Di(ca3D->rows, ca3D->columns, ca3D->slices);

	if (!Q->current || !Q->next)
		return CAL_FALSE;

	return CAL_TRUE;
}

/*! \brief Real (floating point) substates allocation function.
*/
CALbyte calAllocSubstate3Dr(struct CALModel3D* ca3D,	//!< Pointer to the cellular automaton structure.
							struct CALSubstate3Dr* Q	//!< Pointer to a 3D real (floating point) substate.
							)
{
	Q->current = calAllocBuffer3Dr(ca3D->rows, ca3D->columns, ca3D->slices);
	Q->next = calAllocBuffer3Dr(ca3D->rows, ca3D->columns, ca3D->slices);

	if (!Q->current || !Q->next)
		return CAL_FALSE;

	return CAL_TRUE;
}



/*! \brief Deletes the memory associated to a byte substate.
*/
void calDeleteSubstate3Db(struct CALModel3D* ca3D,	//!< Pointer to the cellular automaton structure.
						  struct CALSubstate3Db* Q	//!< Pointer to a 3D byte substate.
						  )
{
	calDeleteBuffer3Db(Q->current);
	calDeleteBuffer3Db(Q->next);
}

/*! \brief Deletes the memory associated to an int substate.
*/
void calDeleteSubstate3Di(struct CALModel3D* ca3D,	//!< Pointer to the cellular automaton structure.
						  struct CALSubstate3Di* Q)	//!< Pointer to a 3D int substate.
{
	calDeleteBuffer3Di(Q->current);
	calDeleteBuffer3Di(Q->next);
}

/*! \brief Deletes the memory associated to a real (floating point) substate.
*/
void calDeleteSubstate3Dr(struct CALModel3D* ca3D,	//!< Pointer to the cellular automaton structure.
						  struct CALSubstate3Dr* Q	//!< Pointer to a 3D real (floating point) substate.
						  )
{
	calDeleteBuffer3Dr(Q->current);
	calDeleteBuffer3Dr(Q->next);
}


/******************************************************************************
							PUBLIC FUNCIONS

*******************************************************************************/


struct CALModel3D* calCADef3D(int rows,
							  int columns,
							  int slices,
							  enum CALNeighborhood3D CAL_NEIGHBORHOOD_3D,
							  enum CALSpaceBoundaryCondition CAL_TOROIDALITY,
							  enum CALOptimization CAL_OPTIMIZATION
							  )
{
	int i;
	struct CALModel3D *ca3D = (struct CALModel3D *)malloc(sizeof(struct CALModel3D));
	if (!ca3D)
		return NULL;

	ca3D->rows = rows;
	ca3D->columns = columns;
	ca3D->slices = slices;

	ca3D->T = CAL_TOROIDALITY;

	ca3D->OPTIMIZATION = CAL_OPTIMIZATION;
	if (ca3D->OPTIMIZATION == CAL_OPT_ACTIVE_CELLS) {
		ca3D->A.flags = calAllocBuffer3Db(ca3D->rows, ca3D->columns, ca3D->slices);
		calSetBuffer3Db(ca3D->A.flags, ca3D->rows, ca3D->columns, ca3D->slices, CAL_FALSE);
	}
	else
		ca3D->A.flags = NULL;

#pragma omp parallel
	{
#pragma omp single
		ca3D->A.num_threads = CAL_GET_NUM_THREADS();
	}


	ca3D->A.size_next = (int*)malloc(sizeof(int) * ca3D->A.num_threads);
	for(i=0;i<ca3D->A.num_threads;i++)
		ca3D->A.size_next[i] = 0;


	ca3D->A.cells = NULL;
	ca3D->A.size_current = 0;

	ca3D->X = NULL;
	ca3D->sizeof_X = 0;

	ca3D->X_id = CAL_NEIGHBORHOOD_3D;
	switch (CAL_NEIGHBORHOOD_3D) {
		case CAL_VON_NEUMANN_NEIGHBORHOOD_3D:
			calDefineVonNeumannNeighborhood3D(ca3D);
			break;
		case CAL_MOORE_NEIGHBORHOOD_3D:
			calDefineMooreNeighborhood3D(ca3D);
			break;
	}

	ca3D->pQb_array = NULL;
	ca3D->pQi_array = NULL;
	ca3D->pQr_array = NULL;
	ca3D->sizeof_pQb_array = 0;
	ca3D->sizeof_pQi_array = 0;
	ca3D->sizeof_pQr_array = 0;

	ca3D->elementary_processes = NULL;
	ca3D->num_of_elementary_processes = 0;

	ca3D->is_safe = CAL_UNSAFE_INACTIVE;

	CAL_ALLOC_LOCKS_3D(ca3D);
	CAL_INIT_LOCKS_3D(ca3D, i);


	return ca3D;
}

void calSetUnsafe3D(struct CALModel3D* ca3D) {
	ca3D->is_safe = CAL_UNSAFE_ACTIVE;
}


void calAddActiveCell3D(struct CALModel3D* ca3D, int i, int j, int k)
{
	CAL_SET_CELL_LOCK_3D(i, j, k, ca3D);

	if (!calGetBuffer3DElement(ca3D->A.flags, ca3D->rows, ca3D->columns, i, j, k))
	{
		calSetBuffer3DElement(ca3D->A.flags, ca3D->rows, ca3D->columns, i, j, k, CAL_TRUE);
		CAL_UNSET_CELL_LOCK_3D(i, j, k, ca3D);
		ca3D->A.size_next[CAL_GET_THREAD_NUM()]++;
	}

	CAL_UNSET_CELL_LOCK_3D(i, j, k, ca3D);
}

void calRemoveActiveCell3D(struct CALModel3D* ca3D, int i, int j, int k)
{
	CAL_SET_CELL_LOCK_3D(i, j, k, ca3D);

	if (calGetBuffer3DElement(ca3D->A.flags, ca3D->rows, ca3D->columns, i, j, k))
	{
		calSetBuffer3DElement(ca3D->A.flags, ca3D->rows, ca3D->columns, i, j, k, CAL_FALSE);
		CAL_UNSET_CELL_LOCK_3D(i, j, k, ca3D);
		ca3D->A.size_next[CAL_GET_THREAD_NUM()]--;
		return;
	}

	CAL_UNSET_CELL_LOCK_3D(i, j, k, ca3D);
}

void calUpdateActiveCells3D(struct CALModel3D* ca3D)
{
	int i, j, k, n;

	int tn;
	int diff;
	struct CALCell3D **tcells;
	int *tsize;

	free(ca3D->A.cells);
	ca3D->A.cells = NULL;

	diff = 0;
	for (i = 0; i < ca3D->A.num_threads; i++) {
		diff += ca3D->A.size_next[i];
		ca3D->A.size_next[i] = 0;
	}

	ca3D->A.size_current += diff;
	if (ca3D->A.size_current == 0)
		return;

	ca3D->A.cells = (struct CALCell3D*)malloc(sizeof(struct CALCell3D)*ca3D->A.size_current);

	tcells = (struct CALCell3D**)malloc(sizeof(struct CALCell3D*) * ca3D->A.num_threads);
	for (i = 0; i < ca3D->A.num_threads; i++) {
		tcells[i] = (struct CALCell3D*)malloc(sizeof(struct CALCell3D) * ca3D->A.size_current);
	}

	tsize = (int *)malloc(sizeof(int) * ca3D->A.num_threads);

#pragma omp parallel shared(tcells, tsize) private (i, j, k, tn)
	{
		tn = CAL_GET_THREAD_NUM();
		tsize[tn] = 0;

#pragma omp for
		for (i = 0; i < ca3D->rows; i++)
			for (j = 0; j < ca3D->columns; j++)
				for (k = 0; k < ca3D->slices; k++)
					if (calGetBuffer3DElement(ca3D->A.flags, ca3D->rows, ca3D->columns, i, j, k))
					{
						tcells[tn][tsize[tn]].i = i;
						tcells[tn][tsize[tn]].j = j;
						tcells[tn][tsize[tn]].k = k;
						tsize[tn]++;
					}

#pragma omp single
		{
			n = 0;
			for (i = 0; i < ca3D->A.num_threads; i++) {
				memcpy(&ca3D->A.cells[n],
					tcells[i], sizeof(struct CALCell3D) * tsize[i]);
				n += tsize[i];
			}
		}
	}

}



struct CALCell3D* calAddNeighbor3D(struct CALModel3D* ca3D, int i, int j, int k) {
	struct CALCell3D* X_tmp = ca3D->X;
	struct CALCell3D* X_new;
	int n;

	X_new = (struct CALCell3D*)malloc(sizeof(struct CALCell3D)*(ca3D->sizeof_X + 1));
	if (!X_new)
		return NULL;

	for (n = 0; n < ca3D->sizeof_X; n++) {
		X_new[n].i = ca3D->X[n].i;
		X_new[n].j = ca3D->X[n].j;
		X_new[n].k = ca3D->X[n].k;
	}
	X_new[ca3D->sizeof_X].i = i;
	X_new[ca3D->sizeof_X].j = j;
	X_new[ca3D->sizeof_X].k = k;

	ca3D->X = X_new;
	free(X_tmp);

	ca3D->sizeof_X++;

	return ca3D->X;
}



struct CALSubstate3Db* calAddSubstate3Db(struct CALModel3D* ca3D){
	struct CALSubstate3Db* Q;
	struct CALSubstate3Db** pQb_array_tmp = ca3D->pQb_array;
	struct CALSubstate3Db** pQb_array_new;
	int i;

	pQb_array_new = (struct CALSubstate3Db**)malloc(sizeof(struct CALSubstate3Db*)*(ca3D->sizeof_pQb_array + 1));
	if (!pQb_array_new)
		return NULL;

	for (i = 0; i < ca3D->sizeof_pQb_array; i++)
		pQb_array_new[i] = ca3D->pQb_array[i];

	Q = (struct CALSubstate3Db*)malloc(sizeof(struct CALSubstate3Db));
	if (!Q)
		return NULL;
	if (!calAllocSubstate3Db(ca3D, Q))
		return NULL;

	pQb_array_new[ca3D->sizeof_pQb_array] = Q;
	ca3D->sizeof_pQb_array++;

	ca3D->pQb_array = pQb_array_new;
	free(pQb_array_tmp);

	return Q;
}

struct CALSubstate3Di* calAddSubstate3Di(struct CALModel3D* ca3D){
	struct CALSubstate3Di* Q;
	struct CALSubstate3Di** pQi_array_tmp = ca3D->pQi_array;
	struct CALSubstate3Di** pQi_array_new;
	int i;

	pQi_array_new = (struct CALSubstate3Di**)malloc(sizeof(struct CALSubstate3Di*)*(ca3D->sizeof_pQi_array + 1));
	if(!pQi_array_new)
		return NULL;

	for (i = 0; i < ca3D->sizeof_pQi_array; i++)
		pQi_array_new[i] = ca3D->pQi_array[i];

	Q = (struct CALSubstate3Di*)malloc(sizeof(struct CALSubstate3Di));
	if (!Q)
		return NULL;
	if (!calAllocSubstate3Di(ca3D, Q))
		return NULL;

	pQi_array_new[ca3D->sizeof_pQi_array] = Q;
	ca3D->sizeof_pQi_array++;

	ca3D->pQi_array = pQi_array_new;
	free(pQi_array_tmp);

	return Q;
}

struct CALSubstate3Dr* calAddSubstate3Dr(struct CALModel3D* ca3D){
	struct CALSubstate3Dr* Q;
	struct CALSubstate3Dr** pQr_array_tmp = ca3D->pQr_array;
	struct CALSubstate3Dr** pQr_array_new;
	int i;

	pQr_array_new = (struct CALSubstate3Dr**)malloc(sizeof(struct CALSubstate3Dr*)*(ca3D->sizeof_pQr_array + 1));
	if (!pQr_array_new)
		return NULL;

	for (i = 0; i < ca3D->sizeof_pQr_array; i++)
		pQr_array_new[i] = ca3D->pQr_array[i];

	Q = (struct CALSubstate3Dr*)malloc(sizeof(struct CALSubstate3Dr));
	if (!Q)
		return NULL;
	if (!calAllocSubstate3Dr(ca3D, Q))
		return NULL;

	pQr_array_new[ca3D->sizeof_pQr_array] = Q;
	ca3D->sizeof_pQr_array++;

	ca3D->pQr_array = pQr_array_new;
	free(pQr_array_tmp);

	return Q;
}



struct CALSubstate3Db* calAddSingleLayerSubstate3Db(struct CALModel3D* ca3D){

	struct CALSubstate3Db* Q;
	Q = (struct CALSubstate3Db*)malloc(sizeof(struct CALSubstate3Db));
	if (!Q)
		return NULL;
	Q->current = calAllocBuffer3Db(ca3D->rows, ca3D->columns, ca3D->slices);
	if (!Q->current)
		return NULL;
	Q->next = NULL;

	return Q;
}

struct CALSubstate3Di* calAddSingleLayerSubstate3Di(struct CALModel3D* ca3D){

	struct CALSubstate3Di* Q;
	Q = (struct CALSubstate3Di*)malloc(sizeof(struct CALSubstate3Di));
	if (!Q)
		return NULL;
	Q->current = calAllocBuffer3Di(ca3D->rows, ca3D->columns, ca3D->slices);
	if (!Q->current)
		return NULL;
	Q->next = NULL;

	return Q;
}

struct CALSubstate3Dr* calAddSingleLayerSubstate3Dr(struct CALModel3D* ca3D){

	struct CALSubstate3Dr* Q;
	Q = (struct CALSubstate3Dr*)malloc(sizeof(struct CALSubstate3Dr));
	if (!Q)
		return NULL;
	Q->current = calAllocBuffer3Dr(ca3D->rows, ca3D->columns, ca3D->slices);
	if (!Q->current)
		return NULL;
	Q->next = NULL;

	return Q;
}



CALCallbackFunc3D* calAddElementaryProcess3D(struct CALModel3D* ca3D,	//!< Pointer to the cellular automaton structure.
											 CALCallbackFunc3D elementary_process
											 )
{
	CALCallbackFunc3D* callbacks_temp = ca3D->elementary_processes;
	CALCallbackFunc3D* callbacks_new = (CALCallbackFunc3D*)malloc(sizeof(CALCallbackFunc3D)*(ca3D->num_of_elementary_processes + 1));
	int n;

	if (!callbacks_new)
		return NULL;

	for (n = 0; n < ca3D->num_of_elementary_processes; n++)
		callbacks_new[n] = ca3D->elementary_processes[n];
	callbacks_new[ca3D->num_of_elementary_processes] = elementary_process;

	ca3D->elementary_processes = callbacks_new;
	free(callbacks_temp);

	ca3D->num_of_elementary_processes++;

	return ca3D->elementary_processes;
}



void calInitSubstate3Db(struct CALModel3D* ca3D, struct CALSubstate3Db* Q, CALbyte value) {
	if (ca3D->A.cells)
	{
		calSetActiveCellsBuffer3Db(Q->current, ca3D->rows, ca3D->columns, ca3D->slices, value, ca3D->A.cells, ca3D->A.size_current);
		if(Q->next)
			calSetActiveCellsBuffer3Db(Q->next, ca3D->rows, ca3D->columns, ca3D->slices, value, ca3D->A.cells, ca3D->A.size_current);
	}
	else
	{
		calSetBuffer3Db(Q->current, ca3D->rows, ca3D->columns, ca3D->slices, value);
		if(Q->next)
			calSetBuffer3Db(Q->next, ca3D->rows, ca3D->columns, ca3D->slices, value);
	}
}

void calInitSubstate3Di(struct CALModel3D* ca3D, struct CALSubstate3Di* Q, CALint value) {
	if (ca3D->A.cells)
	{
		calSetActiveCellsBuffer3Di(Q->current, ca3D->rows, ca3D->columns, ca3D->slices, value, ca3D->A.cells, ca3D->A.size_current);
		if(Q->next)
			calSetActiveCellsBuffer3Di(Q->next, ca3D->rows, ca3D->columns, ca3D->slices, value, ca3D->A.cells, ca3D->A.size_current);
	}
	else
	{
		calSetBuffer3Di(Q->current, ca3D->rows, ca3D->columns, ca3D->slices, value);
		if(Q->next)
			calSetBuffer3Di(Q->next, ca3D->rows, ca3D->columns, ca3D->slices, value);
	}
}

void calInitSubstate3Dr(struct CALModel3D* ca3D, struct CALSubstate3Dr* Q, CALreal value) {
	if (ca3D->A.cells)
	{
		calSetActiveCellsBuffer3Dr(Q->current, ca3D->rows, ca3D->columns, ca3D->slices, value, ca3D->A.cells, ca3D->A.size_current);
		if(Q->next)
			calSetActiveCellsBuffer3Dr(Q->next, ca3D->rows, ca3D->columns, ca3D->slices, value, ca3D->A.cells, ca3D->A.size_current);
	}
	else
	{
		calSetBuffer3Dr(Q->current, ca3D->rows, ca3D->columns, ca3D->slices, value);
		if(Q->next)
			calSetBuffer3Dr(Q->next, ca3D->rows, ca3D->columns, ca3D->slices, value);
	}
}



void calInitSubstateNext3Db(struct CALModel3D* ca3D, struct CALSubstate3Db* Q, CALbyte value) {
	if (ca3D->A.cells)
		calSetActiveCellsBuffer3Db(Q->next, ca3D->rows, ca3D->columns, ca3D->slices, value, ca3D->A.cells, ca3D->A.size_current);
	else
		calSetBuffer3Db(Q->next, ca3D->rows, ca3D->columns, ca3D->slices, value);
}

void calInitSubstateNext3Di(struct CALModel3D* ca3D, struct CALSubstate3Di* Q, CALint value) {
	if (ca3D->A.cells)
		calSetActiveCellsBuffer3Di(Q->next, ca3D->rows, ca3D->columns, ca3D->slices, value, ca3D->A.cells, ca3D->A.size_current);
	else
		calSetBuffer3Di(Q->next, ca3D->rows, ca3D->columns, ca3D->slices, value);
}

void calInitSubstateNext3Dr(struct CALModel3D* ca3D, struct CALSubstate3Dr* Q, CALreal value) {
	if (ca3D->A.cells)
		calSetActiveCellsBuffer3Dr(Q->next, ca3D->rows, ca3D->columns, ca3D->slices, value, ca3D->A.cells, ca3D->A.size_current);
	else
		calSetBuffer3Dr(Q->next, ca3D->rows, ca3D->columns, ca3D->slices, value);
}



void calUpdateSubstate3Db(struct CALModel3D* ca3D, struct CALSubstate3Db* Q) {
	if (ca3D->A.cells)
		calCopyActiveCellsBuffer3Db(Q->next, Q->current, ca3D->rows, ca3D->columns, ca3D->slices, ca3D->A.cells, ca3D->A.size_current);
	else
		calCopyBuffer3Db(Q->next, Q->current, ca3D->rows, ca3D->columns, ca3D->slices);
}

void calUpdateSubstate3Di(struct CALModel3D* ca3D, struct CALSubstate3Di* Q) {
	if (ca3D->A.cells)
		calCopyActiveCellsBuffer3Di(Q->next, Q->current, ca3D->rows, ca3D->columns, ca3D->slices, ca3D->A.cells, ca3D->A.size_current);
	else
		calCopyBuffer3Di(Q->next, Q->current, ca3D->rows, ca3D->columns, ca3D->slices);
}

void calUpdateSubstate3Dr(struct CALModel3D* ca3D, struct CALSubstate3Dr* Q) {
	if (ca3D->A.cells)
		calCopyActiveCellsBuffer3Dr(Q->next, Q->current, ca3D->rows, ca3D->columns, ca3D->slices, ca3D->A.cells, ca3D->A.size_current);
	else
		calCopyBuffer3Dr(Q->next, Q->current, ca3D->rows, ca3D->columns, ca3D->slices);
}



void calApplyElementaryProcess3D(struct CALModel3D* ca3D,	//!< Pointer to the cellular automaton structure.
								CALCallbackFunc3D elementary_process //!< Pointer to a transition function's elementary process.
								 )
{
	int i, j, k, n;

	if (ca3D->A.cells) //Computationally active cells optimization.
#pragma omp parallel for private (n)
		for (n = 0; n < ca3D->A.size_current; n++)
			elementary_process(ca3D, ca3D->A.cells[n].i, ca3D->A.cells[n].j, ca3D->A.cells[n].k);
	else //Standart cicle of the transition function
#pragma omp parallel for private (k, i, j)
		for (i = 0; i < ca3D->rows; i++)
			for (j = 0; j < ca3D->columns; j++)
				for (k = 0; k < ca3D->slices; k++)
					elementary_process(ca3D, i, j, k);
}


void calGlobalTransitionFunction3D(struct CALModel3D* ca3D)
{
	//The global transition function.
	//It applies transition function elementary processes sequentially.
	//Note that a substates' update is performed after each elementary process.

	int b;

	for (b=0; b<ca3D->num_of_elementary_processes; b++)
	{
		//applying the b-th elementary process
		calApplyElementaryProcess3D(ca3D, ca3D->elementary_processes[b]);

		//updating substates
		calUpdate3D(ca3D);
	}
}



void calUpdate3D(struct CALModel3D* ca3D)
{
	int i;

	//updating active cells
	if (ca3D->OPTIMIZATION == CAL_OPT_ACTIVE_CELLS)
		calUpdateActiveCells3D(ca3D);

	//updating substates
	for (i=0; i < ca3D->sizeof_pQb_array; i++)
		calUpdateSubstate3Db(ca3D, ca3D->pQb_array[i]);

	for (i=0; i < ca3D->sizeof_pQi_array; i++)
		calUpdateSubstate3Di(ca3D, ca3D->pQi_array[i]);

	for (i=0; i < ca3D->sizeof_pQr_array; i++)
		calUpdateSubstate3Dr(ca3D, ca3D->pQr_array[i]);
}



void calInit3Db(struct CALModel3D* ca3D, struct CALSubstate3Db* Q, int i, int j, int k, CALbyte value) {
	calSetBuffer3DElement(Q->current, ca3D->rows, ca3D->columns, i, j, k, value);
	calSetBuffer3DElement(Q->next, ca3D->rows, ca3D->columns, i, j, k, value);
}

void calInit3Di(struct CALModel3D* ca3D, struct CALSubstate3Di* Q, int i, int j, int k, CALint value) {
	calSetBuffer3DElement(Q->current, ca3D->rows, ca3D->columns, i, j, k, value);
	calSetBuffer3DElement(Q->next, ca3D->rows, ca3D->columns, i, j, k, value);
}

void calInit3Dr(struct CALModel3D* ca3D, struct CALSubstate3Dr* Q, int i, int j, int k, CALreal value) {
	calSetBuffer3DElement(Q->current, ca3D->rows, ca3D->columns, i, j, k, value);
	calSetBuffer3DElement(Q->next, ca3D->rows, ca3D->columns, i, j, k, value);
}



CALbyte calGet3Db(struct CALModel3D* ca3D, struct CALSubstate3Db* Q, int i, int j, int k) {
	CALbyte ret;
	if (ca3D->is_safe == CAL_UNSAFE_ACTIVE)
		CAL_SET_CELL_LOCK_3D(i, j, k, ca3D);

	ret = calGetBuffer3DElement(Q->current, ca3D->rows, ca3D->columns, i, j, k);

	if (ca3D->is_safe == CAL_UNSAFE_ACTIVE)
		CAL_UNSET_CELL_LOCK_3D(i, j, k, ca3D);

	return ret;
}

CALint calGet3Di(struct CALModel3D* ca3D, struct CALSubstate3Di* Q, int i, int j, int k) {
	CALint ret;
	if (ca3D->is_safe == CAL_UNSAFE_ACTIVE)
		CAL_SET_CELL_LOCK_3D(i, j, k, ca3D);

	ret = calGetBuffer3DElement(Q->current, ca3D->rows, ca3D->columns, i, j, k);

	if (ca3D->is_safe == CAL_UNSAFE_ACTIVE)
		CAL_UNSET_CELL_LOCK_3D(i, j, k, ca3D);

	return ret;
}

CALreal calGet3Dr(struct CALModel3D* ca3D, struct CALSubstate3Dr* Q, int i, int j, int k) {
	CALreal ret;
	if (ca3D->is_safe == CAL_UNSAFE_ACTIVE)
		CAL_SET_CELL_LOCK_3D(i, j, k, ca3D);

	ret = calGetBuffer3DElement(Q->current, ca3D->rows, ca3D->columns, i, j, k);

	if (ca3D->is_safe == CAL_UNSAFE_ACTIVE)
		CAL_UNSET_CELL_LOCK_3D(i, j, k, ca3D);

	return ret;
}



CALbyte calGetX3Db(struct CALModel3D* ca3D, struct CALSubstate3Db* Q, int i, int j, int k, int n)
{
	if (ca3D->T == CAL_SPACE_FLAT)
		return calGetBuffer3DElement(Q->current, ca3D->rows, ca3D->columns, i + ca3D->X[n].i, j + ca3D->X[n].j, k + ca3D->X[n].k);
	else
		return calGetBuffer3DElement(Q->current, ca3D->rows, ca3D->columns, calGetToroidalX(i + ca3D->X[n].i, ca3D->rows), calGetToroidalX(j + ca3D->X[n].j, ca3D->columns), calGetToroidalX(k + ca3D->X[n].k, ca3D->slices));
}

CALint calGetX3Di(struct CALModel3D* ca3D, struct CALSubstate3Di* Q, int i, int j, int k, int n)
{
	if (ca3D->T == CAL_SPACE_FLAT)
		return calGetBuffer3DElement(Q->current, ca3D->rows, ca3D->columns, i + ca3D->X[n].i, j + ca3D->X[n].j, k + ca3D->X[n].k);
	else
		return calGetBuffer3DElement(Q->current, ca3D->rows, ca3D->columns, calGetToroidalX(i + ca3D->X[n].i, ca3D->rows), calGetToroidalX(j + ca3D->X[n].j, ca3D->columns), calGetToroidalX(k + ca3D->X[n].k, ca3D->slices));
}

CALreal calGetX3Dr(struct CALModel3D* ca3D, struct CALSubstate3Dr* Q, int i, int j, int k, int n)
{
	if (ca3D->T == CAL_SPACE_FLAT)
		return calGetBuffer3DElement(Q->current, ca3D->rows, ca3D->columns, i + ca3D->X[n].i, j + ca3D->X[n].j, k + ca3D->X[n].k);
	else
		return calGetBuffer3DElement(Q->current, ca3D->rows, ca3D->columns, calGetToroidalX(i + ca3D->X[n].i, ca3D->rows), calGetToroidalX(j + ca3D->X[n].j, ca3D->columns), calGetToroidalX(k + ca3D->X[n].k, ca3D->slices));
}


void calSet3Db(struct CALModel3D* ca3D, struct CALSubstate3Db* Q, int i, int j, int k, CALbyte value) {
	if (ca3D->is_safe == CAL_UNSAFE_ACTIVE)
		CAL_SET_CELL_LOCK_3D(i, j, k, ca3D);

	calSetBuffer3DElement(Q->next, ca3D->rows, ca3D->columns, i, j, k, value);

	if (ca3D->is_safe == CAL_UNSAFE_ACTIVE)
		CAL_UNSET_CELL_LOCK_3D(i, j, k, ca3D);
}

void calSet3Di(struct CALModel3D* ca3D, struct CALSubstate3Di* Q, int i, int j, int k, CALint value) {
	if (ca3D->is_safe == CAL_UNSAFE_ACTIVE)
		CAL_SET_CELL_LOCK_3D(i, j, k, ca3D);

	calSetBuffer3DElement(Q->next, ca3D->rows, ca3D->columns, i, j, k, value);

	if (ca3D->is_safe == CAL_UNSAFE_ACTIVE)
		CAL_UNSET_CELL_LOCK_3D(i, j, k, ca3D);
}

void calSet3Dr(struct CALModel3D* ca3D, struct CALSubstate3Dr* Q, int i, int j, int k, CALreal value) {
	if (ca3D->is_safe == CAL_UNSAFE_ACTIVE)
		CAL_SET_CELL_LOCK_3D(i, j, k, ca3D);

	calSetBuffer3DElement(Q->next, ca3D->rows, ca3D->columns, i, j, k, value);

	if (ca3D->is_safe == CAL_UNSAFE_ACTIVE)
		CAL_UNSET_CELL_LOCK_3D(i, j, k, ca3D);

}



void calSetCurrent3Db(struct CALModel3D* ca3D, struct CALSubstate3Db* Q, int i, int j, int k, CALbyte value){
	CAL_SET_CELL_LOCK_3D(i, j, k, ca3D);

	calSetBuffer3DElement(Q->current, ca3D->rows, ca3D->columns, i, j, k, value);

	CAL_UNSET_CELL_LOCK_3D(i, j, k, ca3D);
}

void calSetCurrent3Di(struct CALModel3D* ca3D, struct CALSubstate3Di* Q, int i, int j, int k, CALint value){
	CAL_SET_CELL_LOCK_3D(i, j, k, ca3D);

	calSetBuffer3DElement(Q->current, ca3D->rows, ca3D->columns, i, j, k, value);

	CAL_UNSET_CELL_LOCK_3D(i, j, k, ca3D);
}

void calSetCurrent3Dr(struct CALModel3D* ca3D, struct CALSubstate3Dr* Q, int i, int j, int k, CALreal value){
	CAL_SET_CELL_LOCK_3D(i, j, k, ca3D);

	calSetBuffer3DElement(Q->current, ca3D->rows, ca3D->columns, i, j, k, value);

	CAL_UNSET_CELL_LOCK_3D(i, j, k, ca3D);
}



void calFinalize3D(struct CALModel3D* ca3D)
{
	int i;

	free(ca3D->A.flags);
	free(ca3D->A.cells);

	free(ca3D->X);

	for (i=0; i < ca3D->sizeof_pQb_array; i++)
		calDeleteSubstate3Db(ca3D, ca3D->pQb_array[i]);

	for (i=0; i < ca3D->sizeof_pQi_array; i++)
		calDeleteSubstate3Di(ca3D, ca3D->pQi_array[i]);

	for (i=0; i < ca3D->sizeof_pQr_array; i++)
		calDeleteSubstate3Dr(ca3D, ca3D->pQr_array[i]);

	free(ca3D->elementary_processes);

	CAL_DESTROY_LOCKS(ca3D, i);
	CAL_FREE_LOCKS(ca3D);

	free(ca3D);

	ca3D = NULL;
}
