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

#include <cal3D.h>
#include <cal3DBuffer.h>
#include <stdlib.h>


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
		 (sopra)						(sotto)

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
	ca3D->A.size_next = 0;
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

	return ca3D;
}



void calAddActiveCell3D(struct CALModel3D* ca3D, int i, int j, int k)
{
	if (!calGetBuffer3DElement(ca3D->A.flags, ca3D->rows, ca3D->columns, i, j, k))
	{
		calSetBuffer3DElement(ca3D->A.flags, ca3D->rows, ca3D->columns, i, j, k, CAL_TRUE);
		ca3D->A.size_next++;
	}
}

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

void calRemoveActiveCell3D(struct CALModel3D* ca3D, int i, int j, int k)
{
	if (calGetBuffer3DElement(ca3D->A.flags, ca3D->rows, ca3D->columns, i, j, k))
	{
		calSetBuffer3DElement(ca3D->A.flags, ca3D->rows, ca3D->columns, i, j, k, CAL_FALSE);
		ca3D->A.size_next--;
	}
}

void calUpdateActiveCells3D(struct CALModel3D* ca3D)
{
	int i, j, k, n;

	free(ca3D->A.cells);
	ca3D->A.cells = NULL;

	ca3D->A.size_current = ca3D->A.size_next;
	if (ca3D->A.size_current == 0)
		return;
	
	ca3D->A.cells = (struct CALCell3D*)malloc(sizeof(struct CALCell3D)*ca3D->A.size_current);

	n = 0;
	for (i=0; i<ca3D->rows; i++)
		for (j=0; j<ca3D->columns; j++)
			for (k = 0; k<ca3D->slices; k++)
				if (calGetBuffer3DElement(ca3D->A.flags, ca3D->rows, ca3D->columns, i, j, k))
				{
					ca3D->A.cells[n].i = i;
					ca3D->A.cells[n].j = j;
					ca3D->A.cells[n].k = k;
					n++;
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
			for (n=0; n<ca3D->A.size_current; n++)
				(*elementary_process)(ca3D, ca3D->A.cells[n].i, ca3D->A.cells[n].j, ca3D->A.cells[n].k);
		else //Standart cicle of the transition function
			for (i=0; i<ca3D->rows; i++)
				for (j=0; j<ca3D->columns; j++)	
					for (k = 0; k<ca3D->slices; k++)
						(*elementary_process)(ca3D, i, j, k);
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
	return calGetBuffer3DElement(Q->current, ca3D->rows, ca3D->columns, i, j, k);
}

CALint calGet3Di(struct CALModel3D* ca3D, struct CALSubstate3Di* Q, int i, int j, int k) {
	return calGetBuffer3DElement(Q->current, ca3D->rows, ca3D->columns, i, j, k);
}

CALreal calGet3Dr(struct CALModel3D* ca3D, struct CALSubstate3Dr* Q, int i, int j, int k) {
	return calGetBuffer3DElement(Q->current, ca3D->rows, ca3D->columns, i, j, k);
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
	calSetBuffer3DElement(Q->next, ca3D->rows, ca3D->columns, i, j, k, value);
}

void calSet3Di(struct CALModel3D* ca3D, struct CALSubstate3Di* Q, int i, int j, int k, CALint value) {
	calSetBuffer3DElement(Q->next, ca3D->rows, ca3D->columns, i, j, k, value);
}

void calSet3Dr(struct CALModel3D* ca3D, struct CALSubstate3Dr* Q, int i, int j, int k, CALreal value) {
	calSetBuffer3DElement(Q->next, ca3D->rows, ca3D->columns, i, j, k, value);
}



void calSetCurrent3Db(struct CALModel3D* ca3D, struct CALSubstate3Db* Q, int i, int j, int k, CALbyte value){
	calSetBuffer3DElement(Q->current, ca3D->rows, ca3D->columns, i, j, k, value);
}

void calSetCurrent3Di(struct CALModel3D* ca3D, struct CALSubstate3Di* Q, int i, int j, int k, CALint value){
	calSetBuffer3DElement(Q->current, ca3D->rows, ca3D->columns, i, j, k, value);
}

void calSetCurrent3Dr(struct CALModel3D* ca3D, struct CALSubstate3Dr* Q, int i, int j, int k, CALreal value){
	calSetBuffer3DElement(Q->current, ca3D->rows, ca3D->columns, i, j, k, value);
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

	free(ca3D);
	ca3D = NULL;
}
