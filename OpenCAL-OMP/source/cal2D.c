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

#include <OpenCAL-OMP/cal2D.h>
#include <OpenCAL-OMP/cal2DBuffer.h>
#include <OpenCAL-OMP/calOmpDef.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/******************************************************************************
                            PRIVATE FUNCIONS

*******************************************************************************/



/*! \brief Builds the pre-defined von Neumann neighborhood.
*/
void calDefineVonNeumannNeighborhood2D(struct CALModel2D* ca2D	//!< Pointer to the cellular automaton structure.
                                       )
{
    /*
           | 1 |
        ---|---|---
         2 | 0 | 3
        ---|---|---
           | 4 |
   */

    calAddNeighbor2D(ca2D,   0,   0);
    calAddNeighbor2D(ca2D, - 1,   0);
    calAddNeighbor2D(ca2D,   0, - 1);
    calAddNeighbor2D(ca2D,   0, + 1);
    calAddNeighbor2D(ca2D, + 1,   0);
}



/*! \brief Builds the pre-defined Moore neighborhood.
*/
void calDefineMooreNeighborhood2D(struct CALModel2D* ca2D		//!< Pointer to the cellular automaton structure.
                                  )
{
    /*
         5 | 1 | 8
        ---|---|---
         2 | 0 | 3
        ---|---|---
         6 | 4 | 7
   */

    calAddNeighbor2D(ca2D,   0,   0);
    calAddNeighbor2D(ca2D, - 1,   0);
    calAddNeighbor2D(ca2D,   0, - 1);
    calAddNeighbor2D(ca2D,   0, + 1);
    calAddNeighbor2D(ca2D, + 1,   0);
    calAddNeighbor2D(ca2D, - 1, - 1);
    calAddNeighbor2D(ca2D, + 1, - 1);
    calAddNeighbor2D(ca2D, + 1, + 1);
    calAddNeighbor2D(ca2D, - 1, + 1);
}


/*! \brief Builds the pre-defined Moore hexagonal neighborhood.
*/
void calDefineHexagonalNeighborhood2D(struct CALModel2D* ca2D		//!< Pointer to the cellular automaton structure.
                                      )
{
    /*
        cell orientation
             __
            /  \
            \__/
    */
    /*
         3 | 2 | 1
        ---|---|---
         4 | 0 | 6		if (j%2 == 0), i.e. even columns
        ---|---|---
           | 5 |
    */

    calAddNeighbor2D(ca2D,   0,   0);
    calAddNeighbor2D(ca2D, - 1, + 1);
    calAddNeighbor2D(ca2D, - 1,   0);
    calAddNeighbor2D(ca2D, - 1, - 1);
    calAddNeighbor2D(ca2D,   0, - 1);
    calAddNeighbor2D(ca2D, + 1,   0);
    calAddNeighbor2D(ca2D,   0, + 1);

    /*
           | 2 |
        ---|---|---
         3 | 0 | 1		if (j%2 == 1), i.e. odd columns
        ---|---|---
         4 | 5 | 6
    */

    calAddNeighbor2D(ca2D,   0,   0);
    calAddNeighbor2D(ca2D,   0, + 1);
    calAddNeighbor2D(ca2D, - 1,   0);
    calAddNeighbor2D(ca2D,   0, - 1);
    calAddNeighbor2D(ca2D, + 1, - 1);
    calAddNeighbor2D(ca2D, + 1,   0);
    calAddNeighbor2D(ca2D, + 1, + 1);

    ca2D->sizeof_X = 7;
}

/*! \brief Builds the pre-defined Moore hexagonal neighborhood.
*/
void calDefineAlternativeHexagonalNeighborhood2D(struct CALModel2D* ca2D	//!< Pointer to the cellular automaton structure.
                                                 )
{
    /*
        cell orientation

            /\
           /  \
           |  |
           \  /
            \/
    */
    /*
         2 | 1 |
        ---|---|---
         3 | 0 | 6		if (i%2 == 0), i.e. even rows
        ---|---|---
         4 | 5 |
    */

    calAddNeighbor2D(ca2D,   0,   0);
    calAddNeighbor2D(ca2D, - 1,   0);
    calAddNeighbor2D(ca2D, - 1, - 1);
    calAddNeighbor2D(ca2D,   0, - 1);
    calAddNeighbor2D(ca2D, + 1, - 1);
    calAddNeighbor2D(ca2D, + 1,   0);
    calAddNeighbor2D(ca2D,   0, + 1);

    /*
           | 2 | 1
        ---|---|---
         3 | 0 | 6		if (i%2 == 1), i.e. odd rows
        ---|---|---
           | 4 | 5
    */

    calAddNeighbor2D(ca2D,   0,   0);
    calAddNeighbor2D(ca2D, - 1, + 1);
    calAddNeighbor2D(ca2D, - 1,   0);
    calAddNeighbor2D(ca2D,   0, - 1);
    calAddNeighbor2D(ca2D, + 1,   0);
    calAddNeighbor2D(ca2D, + 1, + 1);
    calAddNeighbor2D(ca2D,   0, + 1);

    ca2D->sizeof_X = 7;
}


/*! \brief 8 bit (256 values) integer substates allocation function.
*/
CALbyte calAllocSubstate2Db(struct CALModel2D* ca2D,	//!< Pointer to the cellular automaton structure.
                            struct CALSubstate2Db* Q	//!< Pointer to a 2D byte substate.
                            )
{
    Q->current = calAllocBuffer2Db(ca2D->rows, ca2D->columns);
    Q->next = calAllocBuffer2Db(ca2D->rows, ca2D->columns);

    if (!Q->current || !Q->next)
        return CAL_FALSE;

    return CAL_TRUE;
}

/*! \brief Integer substates allocation function.
*/
CALbyte calAllocSubstate2Di(struct CALModel2D* ca2D,	//!< Pointer to the cellular automaton structure.
                            struct CALSubstate2Di* Q	//!< Pointer to a 2D int substate.
                            )
{
    Q->current = calAllocBuffer2Di(ca2D->rows, ca2D->columns);
    Q->next = calAllocBuffer2Di(ca2D->rows, ca2D->columns);

    if (!Q->current || !Q->next)
        return CAL_FALSE;

    return CAL_TRUE;
}

/*! \brief Real (floating point) substates allocation function.
*/
CALbyte calAllocSubstate2Dr(struct CALModel2D* ca2D,	//!< Pointer to the cellular automaton structure.
                            struct CALSubstate2Dr* Q	//!< Pointer to a 2D real (floating point) substate.
                            )
{
    Q->current = calAllocBuffer2Dr(ca2D->rows, ca2D->columns);
    Q->next = calAllocBuffer2Dr(ca2D->rows, ca2D->columns);

    if (!Q->current || !Q->next)
        return CAL_FALSE;

    return CAL_TRUE;
}



/*! \brief Deletes the memory associated to a byte substate.
*/
void calDeleteSubstate2Db(struct CALModel2D* ca2D,	//!< Pointer to the cellular automaton structure.
                          struct CALSubstate2Db* Q	//!< Pointer to a 2D byte substate.
                          )
{
    calDeleteBuffer2Db(Q->current);
    calDeleteBuffer2Db(Q->next);
}

/*! \brief Deletes the memory associated to an int substate.
*/
void calDeleteSubstate2Di(struct CALModel2D* ca2D,	//!< Pointer to the cellular automaton structure.
                          struct CALSubstate2Di* Q)	//!< Pointer to a 2D int substate.
{
    calDeleteBuffer2Di(Q->current);
    calDeleteBuffer2Di(Q->next);
}

/*! \brief Deletes the memory associated to a real (floating point) substate.
*/
void calDeleteSubstate2Dr(struct CALModel2D* ca2D,	//!< Pointer to the cellular automaton structure.
                          struct CALSubstate2Dr* Q	//!< Pointer to a 2D real (floating point) substate.
                          )
{
    calDeleteBuffer2Dr(Q->current);
    calDeleteBuffer2Dr(Q->next);
}


/******************************************************************************
                            PUBLIC FUNCIONS

*******************************************************************************/
struct CALModel2D* calCADef2DMN(int rows,
                              int columns,
                              enum CALNeighborhood2D CAL_NEIGHBORHOOD_2D,
                              enum CALSpaceBoundaryCondition CAL_TOROIDALITY,
                              enum CALOptimization CAL_OPTIMIZATION,
                              int borderSizeInRows
                              )
{
    int i;
    struct CALModel2D *ca2D = (struct CALModel2D *)malloc(sizeof(struct CALModel2D));
    if (!ca2D)
        return NULL;

    
    ca2D->borderSizeInRows = borderSizeInRows;
    ca2D->rows = rows+borderSizeInRows*2;
    ca2D->columns = columns;

    ca2D->T = CAL_TOROIDALITY;

    ca2D->A = NULL;
    ca2D->contiguousLinkedList = NULL;
    ca2D->OPTIMIZATION = CAL_OPTIMIZATION;
    if (ca2D->OPTIMIZATION == CAL_OPT_ACTIVE_CELLS_NAIVE) {
        ca2D->A = malloc( sizeof(struct CALActiveCells2D));
        ca2D->A->flags = calAllocBuffer2Db(ca2D->rows, ca2D->columns);
        ca2D->A->cells = NULL;
        ca2D->A->size_current = 0;
#pragma omp parallel
        {
#pragma omp single
            ca2D->A->num_threads = CAL_GET_NUM_THREADS();
        }
        ca2D->A->size_next = (int*)malloc(sizeof(int) * ca2D->A->num_threads);

        for(i=0;i<ca2D->A->num_threads;i++)
            ca2D->A->size_next[i] = 0;

        calSetBuffer2Db(ca2D->A->flags, ca2D->rows, ca2D->columns, CAL_FALSE);
    }
    else if(ca2D->OPTIMIZATION == CAL_OPT_ACTIVE_CELLS)
        ca2D->contiguousLinkedList = calMakeContiguousLinkedList2D(ca2D);

    ca2D->X = NULL;
    ca2D->sizeof_X = 0;

    ca2D->X_id = CAL_NEIGHBORHOOD_2D;
    switch (CAL_NEIGHBORHOOD_2D) {
    case CAL_VON_NEUMANN_NEIGHBORHOOD_2D:
        calDefineVonNeumannNeighborhood2D(ca2D);
        break;
    case CAL_MOORE_NEIGHBORHOOD_2D:
        calDefineMooreNeighborhood2D(ca2D);
        break;
    case CAL_HEXAGONAL_NEIGHBORHOOD_2D:
        calDefineHexagonalNeighborhood2D(ca2D);
        break;
    case CAL_HEXAGONAL_NEIGHBORHOOD_ALT_2D:
        calDefineAlternativeHexagonalNeighborhood2D(ca2D);
        break;
    }

    ca2D->pQb_array = NULL;
    ca2D->pQi_array = NULL;
    ca2D->pQr_array = NULL;
    ca2D->sizeof_pQb_array = 0;
    ca2D->sizeof_pQi_array = 0;
    ca2D->sizeof_pQr_array = 0;

    ca2D->pQb_single_layer_array = NULL;
    ca2D->pQi_single_layer_array = NULL;
    ca2D->pQr_single_layer_array = NULL;
    ca2D->sizeof_pQb_single_layer_array = 0;
    ca2D->sizeof_pQi_single_layer_array = 0;
    ca2D->sizeof_pQr_single_layer_array = 0;

    ca2D->elementary_processes = NULL;
    ca2D->num_of_elementary_processes = 0;

    ca2D->is_safe = CAL_UNSAFE_INACTIVE;

    CAL_ALLOC_LOCKS(ca2D);
    CAL_INIT_LOCKS(ca2D, i);

    return ca2D;
}


struct CALModel2D* calCADef2D(int rows,
                              int columns,
                              enum CALNeighborhood2D CAL_NEIGHBORHOOD_2D,
                              enum CALSpaceBoundaryCondition CAL_TOROIDALITY,
                              enum CALOptimization CAL_OPTIMIZATION
                              )
{
    int i;
    struct CALModel2D *ca2D = (struct CALModel2D *)malloc(sizeof(struct CALModel2D));
    if (!ca2D)
        return NULL;

    ca2D->rows = rows;
    ca2D->columns = columns;

    ca2D->T = CAL_TOROIDALITY;

    ca2D->A = NULL;
    ca2D->contiguousLinkedList = NULL;
    ca2D->OPTIMIZATION = CAL_OPTIMIZATION;
    if (ca2D->OPTIMIZATION == CAL_OPT_ACTIVE_CELLS_NAIVE) {
        ca2D->A = malloc( sizeof(struct CALActiveCells2D));
        ca2D->A->flags = calAllocBuffer2Db(ca2D->rows, ca2D->columns);
        ca2D->A->cells = NULL;
        ca2D->A->size_current = 0;
#pragma omp parallel
        {
#pragma omp single
            ca2D->A->num_threads = CAL_GET_NUM_THREADS();
        }
        ca2D->A->size_next = (int*)malloc(sizeof(int) * ca2D->A->num_threads);

        for(i=0;i<ca2D->A->num_threads;i++)
            ca2D->A->size_next[i] = 0;

        calSetBuffer2Db(ca2D->A->flags, ca2D->rows, ca2D->columns, CAL_FALSE);
    }
    else if(ca2D->OPTIMIZATION == CAL_OPT_ACTIVE_CELLS)
        ca2D->contiguousLinkedList = calMakeContiguousLinkedList2D(ca2D);

    ca2D->X = NULL;
    ca2D->sizeof_X = 0;

    ca2D->X_id = CAL_NEIGHBORHOOD_2D;
    switch (CAL_NEIGHBORHOOD_2D) {
    case CAL_VON_NEUMANN_NEIGHBORHOOD_2D:
        calDefineVonNeumannNeighborhood2D(ca2D);
        break;
    case CAL_MOORE_NEIGHBORHOOD_2D:
        calDefineMooreNeighborhood2D(ca2D);
        break;
    case CAL_HEXAGONAL_NEIGHBORHOOD_2D:
        calDefineHexagonalNeighborhood2D(ca2D);
        break;
    case CAL_HEXAGONAL_NEIGHBORHOOD_ALT_2D:
        calDefineAlternativeHexagonalNeighborhood2D(ca2D);
        break;
    }

    ca2D->pQb_array = NULL;
    ca2D->pQi_array = NULL;
    ca2D->pQr_array = NULL;
    ca2D->sizeof_pQb_array = 0;
    ca2D->sizeof_pQi_array = 0;
    ca2D->sizeof_pQr_array = 0;

    ca2D->pQb_single_layer_array = NULL;
    ca2D->pQi_single_layer_array = NULL;
    ca2D->pQr_single_layer_array = NULL;
    ca2D->sizeof_pQb_single_layer_array = 0;
    ca2D->sizeof_pQi_single_layer_array = 0;
    ca2D->sizeof_pQr_single_layer_array = 0;

    ca2D->elementary_processes = NULL;
    ca2D->num_of_elementary_processes = 0;

    ca2D->is_safe = CAL_UNSAFE_INACTIVE;

    CAL_ALLOC_LOCKS(ca2D);
    CAL_INIT_LOCKS(ca2D, i);

    return ca2D;
}

void calSetUnsafe2D(struct CALModel2D* ca2D) {
    ca2D->is_safe = CAL_UNSAFE_ACTIVE;
}

void calAddActiveCell2D(struct CALModel2D* ca2D, int i, int j)
{
    if(ca2D->OPTIMIZATION == CAL_OPT_ACTIVE_CELLS_NAIVE )
        calAddActiveCellNaive2D( ca2D, i, j );
    else if(ca2D->OPTIMIZATION == CAL_OPT_ACTIVE_CELLS )
        calAddActiveCellCLL2D(ca2D, i, j);
}

void calRemoveActiveCell2D(struct CALModel2D* ca2D, int i, int j)
{
    if(ca2D->OPTIMIZATION == CAL_OPT_ACTIVE_CELLS_NAIVE )
        calRemoveActiveCellNaive2D( ca2D, i, j );
    else if(ca2D->OPTIMIZATION == CAL_OPT_ACTIVE_CELLS )
        calRemoveActiveCellCLL2D(ca2D, i, j);
}

void calUpdateActiveCells2D(struct CALModel2D* ca2D)
{
    if(ca2D->OPTIMIZATION == CAL_OPT_ACTIVE_CELLS)
        calUpdateContiguousLinkedList2D(ca2D->contiguousLinkedList);
    else if(ca2D->OPTIMIZATION == CAL_OPT_ACTIVE_CELLS_NAIVE)
        calUpdateActiveCellsNaive2D(ca2D);
}



struct CALCell2D* calAddNeighbor2D(struct CALModel2D* ca2D, int i, int j) {
    struct CALCell2D* X_tmp = ca2D->X;
    struct CALCell2D* X_new;
    int n;

    X_new = (struct CALCell2D*)malloc(sizeof(struct CALCell2D)*(ca2D->sizeof_X + 1));
    if (!X_new)
        return NULL;

    for (n = 0; n < ca2D->sizeof_X; n++) {
        X_new[n].i = ca2D->X[n].i;
        X_new[n].j = ca2D->X[n].j;
    }
    X_new[ca2D->sizeof_X].i = i;
    X_new[ca2D->sizeof_X].j = j;

    ca2D->X = X_new;
    free(X_tmp);

    ca2D->sizeof_X++;

    return ca2D->X;
}



struct CALSubstate2Db* calAddSubstate2Db(struct CALModel2D* ca2D){
    struct CALSubstate2Db* Q;
    struct CALSubstate2Db** pQb_array_tmp = ca2D->pQb_array;
    struct CALSubstate2Db** pQb_array_new;
    int i;

    pQb_array_new = (struct CALSubstate2Db**)malloc(sizeof(struct CALSubstate2Db*)*(ca2D->sizeof_pQb_array + 1));
    if (!pQb_array_new)
        return NULL;

    for (i = 0; i < ca2D->sizeof_pQb_array; i++)
        pQb_array_new[i] = ca2D->pQb_array[i];

    Q = (struct CALSubstate2Db*)malloc(sizeof(struct CALSubstate2Db));
    if (!Q)
        return NULL;
    if (!calAllocSubstate2Db(ca2D, Q))
        return NULL;

    pQb_array_new[ca2D->sizeof_pQb_array] = Q;
    ca2D->sizeof_pQb_array++;

    ca2D->pQb_array = pQb_array_new;
    free(pQb_array_tmp);

    return Q;
}

struct CALSubstate2Di* calAddSubstate2Di(struct CALModel2D* ca2D){
    struct CALSubstate2Di* Q;
    struct CALSubstate2Di** pQi_array_tmp = ca2D->pQi_array;
    struct CALSubstate2Di** pQi_array_new;
    int i;

    pQi_array_new = (struct CALSubstate2Di**)malloc(sizeof(struct CALSubstate2Di*)*(ca2D->sizeof_pQi_array + 1));
    if(!pQi_array_new)
        return NULL;

    for (i = 0; i < ca2D->sizeof_pQi_array; i++)
        pQi_array_new[i] = ca2D->pQi_array[i];

    Q = (struct CALSubstate2Di*)malloc(sizeof(struct CALSubstate2Di));
    if (!Q)
        return NULL;
    if (!calAllocSubstate2Di(ca2D, Q))
        return NULL;

    pQi_array_new[ca2D->sizeof_pQi_array] = Q;
    ca2D->sizeof_pQi_array++;

    ca2D->pQi_array = pQi_array_new;
    free(pQi_array_tmp);

    return Q;
}

struct CALSubstate2Dr* calAddSubstate2Dr(struct CALModel2D* ca2D){
    struct CALSubstate2Dr* Q;
    struct CALSubstate2Dr** pQr_array_tmp = ca2D->pQr_array;
    struct CALSubstate2Dr** pQr_array_new;
    int i;

    pQr_array_new = (struct CALSubstate2Dr**)malloc(sizeof(struct CALSubstate2Dr*)*(ca2D->sizeof_pQr_array + 1));
    if (!pQr_array_new)
        return NULL;

    for (i = 0; i < ca2D->sizeof_pQr_array; i++)
        pQr_array_new[i] = ca2D->pQr_array[i];

    Q = (struct CALSubstate2Dr*)malloc(sizeof(struct CALSubstate2Dr));
    if (!Q)
        return NULL;
    if (!calAllocSubstate2Dr(ca2D, Q))
        return NULL;

    pQr_array_new[ca2D->sizeof_pQr_array] = Q;
    ca2D->sizeof_pQr_array++;

    ca2D->pQr_array = pQr_array_new;
    free(pQr_array_tmp);

    return Q;
}



struct CALSubstate2Db* calAddSingleLayerSubstate2Db(struct CALModel2D* ca2D){

    struct CALSubstate2Db* Q;
    struct CALSubstate2Db** pQb_single_layer_array_tmp = ca2D->pQb_single_layer_array;
    struct CALSubstate2Db** pQb_single_layer_array_new;
    int i;

    pQb_single_layer_array_new = (struct CALSubstate2Db**)malloc(sizeof(struct CALSubstate2Db*)*(ca2D->sizeof_pQb_single_layer_array + 1));
    if (!pQb_single_layer_array_new)
        return NULL;

    for (i = 0; i < ca2D->sizeof_pQb_single_layer_array; i++)
        pQb_single_layer_array_new[i] = ca2D->pQb_single_layer_array[i];

    Q = (struct CALSubstate2Db*)malloc(sizeof(struct CALSubstate2Db));
    if (!Q)
        return NULL;
    Q->current = calAllocBuffer2Db(ca2D->rows, ca2D->columns);
    if (!Q->current)
        return NULL;
    Q->next = NULL;

    pQb_single_layer_array_new[ca2D->sizeof_pQb_single_layer_array] = Q;
    ca2D->sizeof_pQb_single_layer_array++;

    ca2D->pQb_single_layer_array = pQb_single_layer_array_new;
    free(pQb_single_layer_array_tmp);
    return Q;
}

struct CALSubstate2Di* calAddSingleLayerSubstate2Di(struct CALModel2D* ca2D){

    struct CALSubstate2Di* Q;
    struct CALSubstate2Di** pQi_single_layer_array_tmp = ca2D->pQi_single_layer_array;
    struct CALSubstate2Di** pQi_single_layer_array_new;
    int i;

    pQi_single_layer_array_new = (struct CALSubstate2Di**)malloc(sizeof(struct CALSubstate2Di*)*(ca2D->sizeof_pQi_single_layer_array + 1));
    if (!pQi_single_layer_array_new)
        return NULL;

    for (i = 0; i < ca2D->sizeof_pQi_single_layer_array; i++)
        pQi_single_layer_array_new[i] = ca2D->pQi_single_layer_array[i];

    Q = (struct CALSubstate2Di*)malloc(sizeof(struct CALSubstate2Di));
    if (!Q)
        return NULL;
    Q->current = calAllocBuffer2Di(ca2D->rows, ca2D->columns);
    if (!Q->current)
        return NULL;
    Q->next = NULL;

    pQi_single_layer_array_new[ca2D->sizeof_pQi_single_layer_array] = Q;
    ca2D->sizeof_pQi_single_layer_array++;

    ca2D->pQi_single_layer_array = pQi_single_layer_array_new;
    free(pQi_single_layer_array_tmp);
    return Q;
}

struct CALSubstate2Dr* calAddSingleLayerSubstate2Dr(struct CALModel2D* ca2D){

    struct CALSubstate2Dr* Q;
    struct CALSubstate2Dr** pQr_single_layer_array_tmp = ca2D->pQr_single_layer_array;
    struct CALSubstate2Dr** pQr_single_layer_array_new;
    int i;

    pQr_single_layer_array_new = (struct CALSubstate2Dr**)malloc(sizeof(struct CALSubstate2Dr*)*(ca2D->sizeof_pQr_single_layer_array + 1));
    if (!pQr_single_layer_array_new)
        return NULL;

    for (i = 0; i < ca2D->sizeof_pQr_single_layer_array; i++)
        pQr_single_layer_array_new[i] = ca2D->pQr_single_layer_array[i];

    Q = (struct CALSubstate2Dr*)malloc(sizeof(struct CALSubstate2Dr));
    if (!Q)
        return NULL;
    Q->current = calAllocBuffer2Dr(ca2D->rows, ca2D->columns);
    if (!Q->current)
        return NULL;
    Q->next = NULL;

    pQr_single_layer_array_new[ca2D->sizeof_pQr_single_layer_array] = Q;
    ca2D->sizeof_pQr_single_layer_array++;

    ca2D->pQr_single_layer_array = pQr_single_layer_array_new;
    free(pQr_single_layer_array_tmp);
    return Q;
}



CALCallbackFunc2D* calAddElementaryProcess2D(struct CALModel2D* ca2D,	//!< Pointer to the cellular automaton structure.
                                             CALCallbackFunc2D elementary_process
                                             )
{
    CALCallbackFunc2D* callbacks_temp= ca2D->elementary_processes;
    CALCallbackFunc2D* callbacks_new = (CALCallbackFunc2D*)malloc(sizeof(CALCallbackFunc2D)*(ca2D->num_of_elementary_processes + 1));
    int n;

    if (!callbacks_new)
        return NULL;

    for (n = 0; n < ca2D->num_of_elementary_processes; n++)
        callbacks_new[n] = ca2D->elementary_processes[n];
    callbacks_new[ca2D->num_of_elementary_processes] = elementary_process;

    ca2D->elementary_processes = callbacks_new;
    free(callbacks_temp);

    ca2D->num_of_elementary_processes++;

    return ca2D->elementary_processes;
}



void calInitSubstate2Db(struct CALModel2D* ca2D, struct CALSubstate2Db* Q, CALbyte value) {
    if ( (ca2D->OPTIMIZATION == CAL_OPT_ACTIVE_CELLS && ca2D->contiguousLinkedList->size_current > 0 ) ||
         ( ca2D->OPTIMIZATION == CAL_OPT_ACTIVE_CELLS_NAIVE && ca2D->A->size_current > 0) )
    {
        calSetActiveCellsBuffer2Db(Q->current, value, ca2D);
        if(Q->next)
            calSetActiveCellsBuffer2Db(Q->next, value, ca2D);
    }
    else
    {
        calSetBuffer2Db(Q->current, ca2D->rows, ca2D->columns, value);
        if(Q->next)
            calSetBuffer2Db(Q->next, ca2D->rows, ca2D->columns, value);
    }
}

void calInitSubstate2Di(struct CALModel2D* ca2D, struct CALSubstate2Di* Q, CALint value) {
    if ( (ca2D->OPTIMIZATION == CAL_OPT_ACTIVE_CELLS && ca2D->contiguousLinkedList->size_current > 0 ) ||
         ( ca2D->OPTIMIZATION == CAL_OPT_ACTIVE_CELLS_NAIVE && ca2D->A->size_current > 0) )
    {
        calSetActiveCellsBuffer2Di(Q->current, value, ca2D);
        if(Q->next)
            calSetActiveCellsBuffer2Di(Q->next, value, ca2D);
    }
    else
    {
        calSetBuffer2Di(Q->current, ca2D->rows, ca2D->columns, value);
        if(Q->next)
            calSetBuffer2Di(Q->next, ca2D->rows, ca2D->columns, value);
    }
}

void calInitSubstate2Dr(struct CALModel2D* ca2D, struct CALSubstate2Dr* Q, CALreal value) {
    if ( (ca2D->OPTIMIZATION == CAL_OPT_ACTIVE_CELLS && ca2D->contiguousLinkedList->size_current > 0 ) ||
         ( ca2D->OPTIMIZATION == CAL_OPT_ACTIVE_CELLS_NAIVE && ca2D->A->size_current > 0) )
    {
        calSetActiveCellsBuffer2Dr(Q->current, value, ca2D);
        if(Q->next)
            calSetActiveCellsBuffer2Dr(Q->next, value, ca2D);
    }
    else
    {
        calSetBuffer2Dr(Q->current, ca2D->rows, ca2D->columns, value);
        if(Q->next)
            calSetBuffer2Dr(Q->next, ca2D->rows, ca2D->columns, value);
    }
}



void calInitSubstateNext2Db(struct CALModel2D* ca2D, struct CALSubstate2Db* Q, CALbyte value) {
    if ( (ca2D->OPTIMIZATION == CAL_OPT_ACTIVE_CELLS && ca2D->contiguousLinkedList->size_current > 0 ) ||
         ( ca2D->OPTIMIZATION == CAL_OPT_ACTIVE_CELLS_NAIVE && ca2D->A->size_current > 0) )
        calSetActiveCellsBuffer2Db(Q->next, value, ca2D);
    else
        calSetBuffer2Db(Q->next, ca2D->rows, ca2D->columns, value);
}

void calInitSubstateNext2Di(struct CALModel2D* ca2D, struct CALSubstate2Di* Q, CALint value) {
    if ( (ca2D->OPTIMIZATION == CAL_OPT_ACTIVE_CELLS && ca2D->contiguousLinkedList->size_current > 0 ) ||
         ( ca2D->OPTIMIZATION == CAL_OPT_ACTIVE_CELLS_NAIVE && ca2D->A->size_current > 0) )
        calSetActiveCellsBuffer2Di(Q->next, value, ca2D);
    else
        calSetBuffer2Di(Q->next, ca2D->rows, ca2D->columns, value);
}

void calInitSubstateNext2Dr(struct CALModel2D* ca2D, struct CALSubstate2Dr* Q, CALreal value) {
    if ( (ca2D->OPTIMIZATION == CAL_OPT_ACTIVE_CELLS && ca2D->contiguousLinkedList->size_current > 0 ) ||
         ( ca2D->OPTIMIZATION == CAL_OPT_ACTIVE_CELLS_NAIVE && ca2D->A->size_current > 0) )
        calSetActiveCellsBuffer2Dr(Q->next, value, ca2D);
    else
        calSetBuffer2Dr(Q->next, ca2D->rows, ca2D->columns, value);
}


void calUpdateSubstate2Db(struct CALModel2D* ca2D, struct CALSubstate2Db* Q) {
    if ( (ca2D->OPTIMIZATION == CAL_OPT_ACTIVE_CELLS && ca2D->contiguousLinkedList->size_current > 0 ) ||
         ( ca2D->OPTIMIZATION == CAL_OPT_ACTIVE_CELLS_NAIVE && ca2D->A->size_current > 0) )
        calCopyBufferActiveCells2Db(Q->next, Q->current, ca2D);
    else
        calCopyBuffer2Db(Q->next, Q->current, ca2D->rows, ca2D->columns);
}

void calUpdateSubstate2Di(struct CALModel2D* ca2D, struct CALSubstate2Di* Q) {
    if ( (ca2D->OPTIMIZATION == CAL_OPT_ACTIVE_CELLS && ca2D->contiguousLinkedList->size_current > 0 ) ||
         ( ca2D->OPTIMIZATION == CAL_OPT_ACTIVE_CELLS_NAIVE && ca2D->A->size_current > 0) )
        calCopyBufferActiveCells2Di(Q->next, Q->current, ca2D);
    else
        calCopyBuffer2Di(Q->next, Q->current, ca2D->rows, ca2D->columns);
}

void calUpdateSubstate2Dr(struct CALModel2D* ca2D, struct CALSubstate2Dr* Q) {
    if ( (ca2D->OPTIMIZATION == CAL_OPT_ACTIVE_CELLS && ca2D->contiguousLinkedList->size_current > 0 ) ||
         ( ca2D->OPTIMIZATION == CAL_OPT_ACTIVE_CELLS_NAIVE && ca2D->A->size_current > 0) )
        calCopyBufferActiveCells2Dr(Q->next, Q->current, ca2D);
    else
        calCopyBuffer2Dr(Q->next, Q->current, ca2D->rows, ca2D->columns);

}



void calApplyElementaryProcess2D(struct CALModel2D* ca2D,	//!< Pointer to the cellular automaton structure.
                                 CALCallbackFunc2D elementary_process  //!< Pointer to a transition function's elementary process.
                                 )
{
    int i, j;

    if (ca2D->OPTIMIZATION == CAL_OPT_ACTIVE_CELLS_NAIVE && ca2D->A->size_current > 0) //Computationally active cells optimization(naive).
        calApplyElementaryProcessActiveCellsNaive2D( ca2D, elementary_process);
    else if(ca2D->OPTIMIZATION == CAL_OPT_ACTIVE_CELLS && ca2D->contiguousLinkedList->size_current > 0) //Computationally active cells optimization(optimal).
        calApplyElementaryProcessActiveCellsCLL2D(ca2D, elementary_process);
    else //Standart cicle of the transition function { //Standart cicle of the transition function
#pragma omp parallel private (i,j)
{
    int tid = omp_get_thread_num();
    printf("Hello World from thread = %d\n", tid);
    #pragma omp parallel for
        for (i = 0; i < ca2D->rows; i++){
            for (j = 0; j < ca2D->columns; j++){
                elementary_process(ca2D, i, j);
            }
        }
    }
                
}




void calGlobalTransitionFunction2D(struct CALModel2D* ca2D)
{
    //The global transition function.
    //It applies transition function elementary processes sequentially.
    //Note that a substates' update is performed after each elementary process.

    int b;

    for (b=0; b<ca2D->num_of_elementary_processes; b++)
    {
        //applying the b-th elementary process
        calApplyElementaryProcess2D(ca2D, ca2D->elementary_processes[b]);

        //updating substates
        calUpdate2D(ca2D);
    }
}



void calUpdate2D(struct CALModel2D* ca2D)
{
    int i;

    //updating active cells
    if (ca2D->OPTIMIZATION == CAL_OPT_ACTIVE_CELLS_NAIVE)
        calUpdateActiveCellsNaive2D(ca2D);
    else if(ca2D->OPTIMIZATION == CAL_OPT_ACTIVE_CELLS)
        calUpdateContiguousLinkedList2D(ca2D->contiguousLinkedList);

    //updating substates
    for (i=0; i < ca2D->sizeof_pQb_array; i++)
        calUpdateSubstate2Db(ca2D, ca2D->pQb_array[i]);

    for (i=0; i < ca2D->sizeof_pQi_array; i++)
        calUpdateSubstate2Di(ca2D, ca2D->pQi_array[i]);

    for (i=0; i < ca2D->sizeof_pQr_array; i++)
        calUpdateSubstate2Dr(ca2D, ca2D->pQr_array[i]);
}



void calInit2Db(struct CALModel2D* ca2D, struct CALSubstate2Db* Q, int i, int j, CALbyte value) {
    calSetMatrixElement(Q->current, ca2D->columns, i, j, value);
    calSetMatrixElement(Q->next, ca2D->columns, i, j, value);
}

void calInit2Di(struct CALModel2D* ca2D, struct CALSubstate2Di* Q, int i, int j, CALint value) {
    calSetMatrixElement(Q->current, ca2D->columns, i, j, value);
    calSetMatrixElement(Q->next, ca2D->columns, i, j, value);
}

void calInit2Dr(struct CALModel2D* ca2D, struct CALSubstate2Dr* Q, int i, int j, CALreal value) {
    calSetMatrixElement(Q->current, ca2D->columns, i, j, value);
    calSetMatrixElement(Q->next, ca2D->columns, i, j, value);
}


CALbyte calGet2Db(struct CALModel2D* ca2D, struct CALSubstate2Db* Q, int i, int j) {
    CALbyte ret;
    if (ca2D->is_safe == CAL_UNSAFE_ACTIVE)
        CAL_SET_CELL_LOCK(i, j, ca2D);

    ret = calGetMatrixElement(Q->current, ca2D->columns, i, j);

    if (ca2D->is_safe == CAL_UNSAFE_ACTIVE)
        CAL_UNSET_CELL_LOCK(i, j, ca2D);

    return ret;
}

CALint calGet2Di(struct CALModel2D* ca2D, struct CALSubstate2Di* Q, int i, int j) {
    CALint ret;
    if (ca2D->is_safe == CAL_UNSAFE_ACTIVE)
        CAL_SET_CELL_LOCK(i, j, ca2D);

    ret = calGetMatrixElement(Q->current, ca2D->columns, i, j);

    if (ca2D->is_safe == CAL_UNSAFE_ACTIVE)
        CAL_UNSET_CELL_LOCK(i, j, ca2D);

    return ret;
}

CALreal calGet2Dr(struct CALModel2D* ca2D, struct CALSubstate2Dr* Q, int i, int j) {
    CALreal ret;
    if (ca2D->is_safe == CAL_UNSAFE_ACTIVE)
        CAL_SET_CELL_LOCK(i, j, ca2D);

    ret = calGetMatrixElement(Q->current, ca2D->columns, i, j);

    if (ca2D->is_safe == CAL_UNSAFE_ACTIVE)
        CAL_UNSET_CELL_LOCK(i, j, ca2D);

    return ret;
}

CALbyte calGetX2Db(struct CALModel2D* ca2D, struct CALSubstate2Db* Q, int i, int j, int n)
{
    if ((ca2D->X_id == CAL_HEXAGONAL_NEIGHBORHOOD_2D && j%2 ==1) || (ca2D->X_id == CAL_HEXAGONAL_NEIGHBORHOOD_ALT_2D && i%2 ==1))
        n += CAL_HEXAGONAL_SHIFT;

    if (ca2D->T == CAL_SPACE_FLAT)
        return calGet2Db(ca2D, Q, i + ca2D->X[n].i, j + ca2D->X[n].j);
    else
        return calGet2Db(ca2D, Q, calGetToroidalX(i + ca2D->X[n].i, ca2D->rows),
                         calGetToroidalX(j + ca2D->X[n].j, ca2D->columns));
}

CALint calGetX2Di(struct CALModel2D* ca2D, struct CALSubstate2Di* Q, int i, int j, int n)
{
    if ((ca2D->X_id == CAL_HEXAGONAL_NEIGHBORHOOD_2D && j%2 ==1) || (ca2D->X_id == CAL_HEXAGONAL_NEIGHBORHOOD_ALT_2D && i%2 ==1))
        n += CAL_HEXAGONAL_SHIFT;

    if (ca2D->T == CAL_SPACE_FLAT)
        return calGet2Di(ca2D, Q, i + ca2D->X[n].i, j + ca2D->X[n].j);
    else
        return calGet2Di(ca2D, Q, calGetToroidalX(i + ca2D->X[n].i, ca2D->rows),
                         calGetToroidalX(j + ca2D->X[n].j, ca2D->columns));
}

CALreal calGetX2Dr(struct CALModel2D* ca2D, struct CALSubstate2Dr* Q, int i, int j, int n)
{
    if ((ca2D->X_id == CAL_HEXAGONAL_NEIGHBORHOOD_2D && j%2 ==1) || (ca2D->X_id == CAL_HEXAGONAL_NEIGHBORHOOD_ALT_2D && i%2 ==1))
        n += CAL_HEXAGONAL_SHIFT;

    if (ca2D->T == CAL_SPACE_FLAT)
        return calGet2Dr(ca2D, Q, i + ca2D->X[n].i, j + ca2D->X[n].j);
    else
        return calGet2Dr(ca2D, Q, calGetToroidalX(i + ca2D->X[n].i, ca2D->rows),
                         calGetToroidalX(j + ca2D->X[n].j, ca2D->columns));

}

void calSet2Db(struct CALModel2D* ca2D, struct CALSubstate2Db* Q, int i, int j, CALbyte value) {

    if (ca2D->is_safe == CAL_UNSAFE_ACTIVE)
        CAL_SET_CELL_LOCK(i, j, ca2D);

    calSetMatrixElement(Q->next, ca2D->columns, i, j, value);

    if (ca2D->is_safe == CAL_UNSAFE_ACTIVE)
        CAL_UNSET_CELL_LOCK(i, j, ca2D);
}

void calSet2Di(struct CALModel2D* ca2D, struct CALSubstate2Di* Q, int i, int j, CALint value) {
    if (ca2D->is_safe == CAL_UNSAFE_ACTIVE)
        CAL_SET_CELL_LOCK(i, j, ca2D);

    calSetMatrixElement(Q->next, ca2D->columns, i, j, value);

    if (ca2D->is_safe == CAL_UNSAFE_ACTIVE)
        CAL_UNSET_CELL_LOCK(i, j, ca2D);
}

void calSet2Dr(struct CALModel2D* ca2D, struct CALSubstate2Dr* Q, int i, int j, CALreal value) {
    if (ca2D->is_safe == CAL_UNSAFE_ACTIVE)
        CAL_SET_CELL_LOCK(i, j, ca2D);

    calSetMatrixElement(Q->next, ca2D->columns, i, j, value);

    if (ca2D->is_safe == CAL_UNSAFE_ACTIVE)
        CAL_UNSET_CELL_LOCK(i, j, ca2D);
}



void calSetCurrent2Db(struct CALModel2D* ca2D, struct CALSubstate2Db* Q, int i, int j, CALbyte value){

    CAL_SET_CELL_LOCK(i, j, ca2D);

    calSetMatrixElement(Q->current, ca2D->columns, i, j, value);

    CAL_UNSET_CELL_LOCK(i, j, ca2D);
}

void calSetCurrent2Di(struct CALModel2D* ca2D, struct CALSubstate2Di* Q, int i, int j, CALint value){

    CAL_SET_CELL_LOCK(i, j, ca2D);

    calSetMatrixElement(Q->current, ca2D->columns, i, j, value);

    CAL_UNSET_CELL_LOCK(i, j, ca2D);
}

void calSetCurrent2Dr(struct CALModel2D* ca2D, struct CALSubstate2Dr* Q, int i, int j, CALreal value){

    CAL_SET_CELL_LOCK(i, j, ca2D);

    calSetMatrixElement(Q->current, ca2D->columns, i, j, value);

    CAL_UNSET_CELL_LOCK(i, j, ca2D);
}


void calFinalize2D(struct CALModel2D* ca2D)
{
    int i;

    if(ca2D->OPTIMIZATION == CAL_OPT_ACTIVE_CELLS_NAIVE)
        calFreeActiveCellsNaive2D(ca2D->A);
    else if(ca2D->OPTIMIZATION == CAL_OPT_ACTIVE_CELLS)
        calFreeContiguousLinkedList2D(ca2D->contiguousLinkedList);
    free(ca2D->X);

    for (i=0; i < ca2D->sizeof_pQb_array; i++)
        calDeleteSubstate2Db(ca2D, ca2D->pQb_array[i]);

    for (i=0; i < ca2D->sizeof_pQi_array; i++)
        calDeleteSubstate2Di(ca2D, ca2D->pQi_array[i]);

    for (i=0; i < ca2D->sizeof_pQr_array; i++)
        calDeleteSubstate2Dr(ca2D, ca2D->pQr_array[i]);

    for (i=0; i < ca2D->sizeof_pQb_single_layer_array; i++)
        calDeleteSubstate2Db(ca2D, ca2D->pQb_single_layer_array[i]);

    for (i=0; i < ca2D->sizeof_pQi_single_layer_array; i++)
        calDeleteSubstate2Di(ca2D, ca2D->pQi_single_layer_array[i]);

    for (i=0; i < ca2D->sizeof_pQr_single_layer_array; i++)
        calDeleteSubstate2Dr(ca2D, ca2D->pQr_single_layer_array[i]);

    free(ca2D->elementary_processes);

    CAL_DESTROY_LOCKS(ca2D, i);
    CAL_FREE_LOCKS(ca2D);

    free(ca2D);
    ca2D = NULL;
}
