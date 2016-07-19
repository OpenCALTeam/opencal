#include <OpenCAL-CPU/calModel.h>
#include <OpenCAL-CPU/calRun.h>


#include <stdarg.h>


static CALbyte calAllocSubstate_b(struct CALModel* calModel, struct CALSubstate_b* Q)
{
    Q->current = calAllocBuffer_b(calModel->coordinatesDimensions, calModel->numberOfCoordinates);
    Q->next = calAllocBuffer_b(calModel->coordinatesDimensions, calModel->numberOfCoordinates);

    if (!Q->current || !Q->next)
        return CAL_FALSE;

    return CAL_TRUE;
}

static CALint calAllocSubstate_i(struct CALModel* calModel, struct CALSubstate_i* Q)
{
    Q->current = calAllocBuffer_i(calModel->coordinatesDimensions, calModel->numberOfCoordinates);
    Q->next = calAllocBuffer_i(calModel->coordinatesDimensions, calModel->numberOfCoordinates);

    if (!Q->current || !Q->next)
        return CAL_FALSE;

    return CAL_TRUE;
}

static CALreal calAllocSubstate_r(struct CALModel* calModel, struct CALSubstate_r* Q)
{
    Q->current = calAllocBuffer_r(calModel->coordinatesDimensions, calModel->numberOfCoordinates);
    Q->next = calAllocBuffer_r(calModel->coordinatesDimensions, calModel->numberOfCoordinates);

    if (!Q->current || !Q->next)
        return CAL_FALSE;

    return CAL_TRUE;
}

struct CALModel*calCADef(struct CALDimensions* dimensions, enum CALNeighborhood CAL_NEIGHBORHOOD, enum CALSpaceBoundaryCondition CAL_TOROIDALITY, enum CALOptimization CAL_OPTIMIZATION, int initial_step, int final_step)
{

    struct CALModel *calModel = (struct CALModel *)malloc(sizeof(struct CALModel));
    if (!calModel)
        return NULL;

    calModel->numberOfCoordinates = dimensions->number_of_dimensions;

    calModel->coordinatesDimensions = dimensions->coordinates_dimensions;

    int n;
    calModel->cellularSpaceDimension = 1;
    for(n = 0; n < calModel->numberOfCoordinates; n++)
        calModel->cellularSpaceDimension *= calModel->coordinatesDimensions[n];
    calModel->calIndexesPool =  calDefIndexesPool(calModel->coordinatesDimensions, calModel->numberOfCoordinates);

    //CALL calRun constructor and set optimization
    int ** cellPattern;
    switch (CAL_NEIGHBORHOOD) {
    case CAL_VON_NEUMANN_NEIGHBORHOOD:
        cellPattern = defineVonNeumannNeighborhood(1, calModel->numberOfCoordinates);
        break;
    case CAL_MOORE_NEIGHBORHOOD:
        cellPattern = defineMooreNeighborhood(1, calModel->numberOfCoordinates);
        break;
        //    case CAL_HEXAGONAL_NEIGHBORHOOD:
        //        break;
        //    case CAL_HEXAGONAL_NEIGHBORHOOD_ALT:
        //        break;
    }

    calModel->calNeighborPool = calDefNeighborPool(calModel->calIndexesPool,CAL_TOROIDALITY, cellPattern,1);



//   printf("STAMPA VICINATO %d\n",calModel->calNeighborPool->size_of_X);
//    for (int i = 0; i < calModel->calIndexesPool->cellular_space_dimension; i++)
//    {
//        for (int j = 0; j <  calModel->calNeighborPool->size_of_X; j++)
//        {
//            printf("%d ",calModel->calNeighborPool->neighborPool[i][j]);
//        }
//        printf("\n");
//    }
    calModel->A->OPTIMIZATION = CAL_OPTIMIZATION;

    calModel->calRun = makeCALRun(initial_step, final_step);
    //Manage Optimization

    free(dimensions);

    return calModel;
}

void calAddNeighbor(struct CALModel* calModel, CALIndices neighbourIndex)
{

}

struct CALSubstate_b*calAddSubstate_b(struct CALModel* calModel, enum CALInitMethod initMethod, CALbyte value)
{
    struct CALSubstate_b* Q;
    struct CALSubstate_b** pQb_array_tmp = calModel->pQb_array;
    struct CALSubstate_b** pQb_array_new;
    int i;

    pQb_array_new = (struct CALSubstate_b**)malloc(sizeof(struct CALSubstate_b*)*(calModel->sizeof_pQb_array + 1));
    if (!pQb_array_new)
        return NULL;

    for (i = 0; i < calModel->sizeof_pQb_array; i++)
        pQb_array_new[i] = calModel->pQb_array[i];

    Q = (struct CALSubstate_b*)malloc(sizeof(struct CALSubstate_b));
    if (!Q)
        return NULL;
    if (!calAllocSubstate_b(calModel, Q))
        return NULL;

    pQb_array_new[calModel->sizeof_pQb_array] = Q;
    calModel->sizeof_pQb_array++;

    calModel->pQb_array = pQb_array_new;
    free(pQb_array_tmp);

    if(initMethod == CAL_INIT_CURRENT)
        calSetBuffer_b(Q->current, calModel->cellularSpaceDimension, value);
    else if(initMethod == CAL_INIT_NEXT)
        calSetBuffer_b(Q->next, calModel->cellularSpaceDimension, value);
    else if(initMethod == CAL_INIT_BOTH)
    {
        calSetBuffer_b(Q->current, calModel->cellularSpaceDimension, value);
        calSetBuffer_b(Q->next, calModel->cellularSpaceDimension, value);
    }

    return Q;
}

struct CALSubstate_i*calAddSubstate_i(struct CALModel* calModel, enum CALInitMethod initMethod, CALint value)
{
    struct CALSubstate_i* Q;
    struct CALSubstate_i** pQi_array_tmp = calModel->pQi_array;
    struct CALSubstate_i** pQi_array_new;
    int i;

    pQi_array_new = (struct CALSubstate_i**)malloc(sizeof(struct CALSubstate_i*)*(calModel->sizeof_pQi_array + 1));
    if (!pQi_array_new)
        return NULL;

    for (i = 0; i < calModel->sizeof_pQi_array; i++)
        pQi_array_new[i] = calModel->pQi_array[i];

    Q = (struct CALSubstate_i*)malloc(sizeof(struct CALSubstate_i));
    if (!Q)
        return NULL;
    if (!calAllocSubstate_i(calModel, Q))
        return NULL;

    pQi_array_new[calModel->sizeof_pQi_array] = Q;
    calModel->sizeof_pQi_array++;

    calModel->pQi_array = pQi_array_new;
    free(pQi_array_tmp);

    if(initMethod == CAL_INIT_CURRENT)
        calSetBuffer_i(Q->current, calModel->cellularSpaceDimension, value);
    else if(initMethod == CAL_INIT_NEXT)
        calSetBuffer_i(Q->next, calModel->cellularSpaceDimension, value);
    else if(initMethod == CAL_INIT_BOTH)
    {
        calSetBuffer_i(Q->current, calModel->cellularSpaceDimension, value);
        calSetBuffer_i(Q->next, calModel->cellularSpaceDimension, value);
    }

    return Q;
}

struct CALSubstate_r*calAddSubstate_r(struct CALModel* calModel, enum CALInitMethod initMethod, CALreal value)
{
    struct CALSubstate_r* Q;
    struct CALSubstate_r** pQr_array_tmp = calModel->pQr_array;
    struct CALSubstate_r** pQr_array_new;
    int i;

    pQr_array_new = (struct CALSubstate_r**)malloc(sizeof(struct CALSubstate_r*)*(calModel->sizeof_pQr_array + 1));
    if (!pQr_array_new)
        return NULL;

    for (i = 0; i < calModel->sizeof_pQr_array; i++)
        pQr_array_new[i] = calModel->pQr_array[i];

    Q = (struct CALSubstate_r*)malloc(sizeof(struct CALSubstate_r));
    if (!Q)
        return NULL;
    if (!calAllocSubstate_r(calModel, Q))
        return NULL;

    pQr_array_new[calModel->sizeof_pQr_array] = Q;
    calModel->sizeof_pQr_array++;

    calModel->pQr_array = pQr_array_new;
    free(pQr_array_tmp);

    if(initMethod == CAL_INIT_CURRENT)
        calSetBuffer_r(Q->current, calModel->cellularSpaceDimension, value);
    else if(initMethod == CAL_INIT_NEXT)
        calSetBuffer_r(Q->next, calModel->cellularSpaceDimension, value);
    else if(initMethod == CAL_INIT_BOTH)
    {
        calSetBuffer_r(Q->current, calModel->cellularSpaceDimension, value);
        calSetBuffer_r(Q->next, calModel->cellularSpaceDimension, value);
    }

    return Q;
}

struct CALSubstate_b*calAddSingleLayerSubstate_b(struct CALModel* calModel, CALbyte init_value)
{
    struct CALSubstate_b* Q;
    Q = (struct CALSubstate_b*)malloc(sizeof(struct CALSubstate_b));
    if (!Q)
        return NULL;
    Q->current = calAllocBuffer_b(calModel->coordinatesDimensions, calModel->numberOfCoordinates);
    if (!Q->current)
        return NULL;
    Q->next = NULL;

    calSetBuffer_b(Q->current, calModel->cellularSpaceDimension, init_value);

    return Q;
}

struct CALSubstate_i*calAddSingleLayerSubstate_i(struct CALModel* calModel, CALint init_value)
{
    struct CALSubstate_i* Q;
    Q = (struct CALSubstate_i*)malloc(sizeof(struct CALSubstate_i));
    if (!Q)
        return NULL;
    Q->current = calAllocBuffer_i(calModel->coordinatesDimensions, calModel->numberOfCoordinates);
    if (!Q->current)
        return NULL;
    Q->next = NULL;

    calSetBuffer_i(Q->current, calModel->cellularSpaceDimension, init_value);

    return Q;
}

struct CALSubstate_r*calAddSingleLayerSubstate_r(struct CALModel* calModel, CALreal init_value)
{
    struct CALSubstate_r* Q;
    Q = (struct CALSubstate_r*)malloc(sizeof(struct CALSubstate_r));
    if (!Q)
        return NULL;
    Q->current = calAllocBuffer_r(calModel->coordinatesDimensions, calModel->numberOfCoordinates);
    if (!Q->current)
        return NULL;
    Q->next = NULL;

    calSetBuffer_r(Q->current, calModel->cellularSpaceDimension, init_value);

    return Q;
}

void calInitSubstate_b(struct CALModel* calModel, struct CALSubstate_b* Q, CALbyte value)
{
    if(Q->current)
        calSetBuffer_b(Q->current, calModel->cellularSpaceDimension, value);
    if(Q->next)
        calSetBuffer_b(Q->next, calModel->cellularSpaceDimension, value);
}

void calInitSubstate_i(struct CALModel* calModel, struct CALSubstate_i* Q, CALint value)
{
    if(Q->current)
        calSetBuffer_i(Q->current, calModel->cellularSpaceDimension, value);
    if(Q->next)
        calSetBuffer_i(Q->next, calModel->cellularSpaceDimension, value);
}

void calInitSubstate_r(struct CALModel* calModel, struct CALSubstate_r* Q, CALreal value)
{
    if(Q->current)
        calSetBuffer_r(Q->current, calModel->cellularSpaceDimension, value);
    if(Q->next)
        calSetBuffer_r(Q->next, calModel->cellularSpaceDimension, value);
}

void calInit_b(struct CALModel* calModel, struct CALSubstate_b* Q, CALIndices central_cell, CALbyte value)
{
    int linear_index = getLinearIndex(central_cell, calModel->coordinatesDimensions, calModel->numberOfCoordinates);
    Q->current[linear_index] = value;
    Q->next[linear_index] = value;
}

void calInit_i(struct CALModel* calModel, struct CALSubstate_i* Q, CALIndices central_cell, CALint value)
{
    int linear_index = getLinearIndex(central_cell, calModel->coordinatesDimensions, calModel->numberOfCoordinates);
    Q->current[linear_index] = value;
    Q->next[linear_index] = value;
}

void calInit_r(struct CALModel* calModel, struct CALSubstate_r* Q, CALIndices central_cell, CALreal value)
{
    int linear_index = getLinearIndex(central_cell, calModel->coordinatesDimensions, calModel->numberOfCoordinates);
    Q->current[linear_index] = value;
    Q->next[linear_index] = value;
}

CALbyte calGet_b(struct CALModel* calModel, struct CALSubstate_b* Q, CALIndices indexes)
{
    CALbyte ret;
    int linear_index = getLinearIndex(indexes, calModel->coordinatesDimensions, calModel->numberOfCoordinates);
#if CAL_PARALLEL == 1
    CAL_SET_CELL_LOCK(linear_index, calModel->calRun->locks);
#endif

    ret = calGetMatrixElement(Q->current, linear_index);

#if CAL_PARALLEL == 1
    CAL_UNSET_CELL_LOCK(linear_index, calModel->calRun->locks );
#endif

    return ret;
}

CALreal calGet_r(struct CALModel* calModel, struct CALSubstate_r* Q, CALIndices indexes)
{
    CALreal ret;
    int linear_index = getLinearIndex(indexes, calModel->coordinatesDimensions, calModel->numberOfCoordinates);
#if CAL_PARALLEL == 1
    CAL_SET_CELL_LOCK(linear_index, calModel->calRun->locks );
#endif

    ret = calGetMatrixElement(Q->current, linear_index);

#if CAL_PARALLEL == 1
    CAL_UNSET_CELL_LOCK(linear_index, calModel->calRun->locks );
#endif

    return ret;
}

CALint calGet_i(struct CALModel* calModel, struct CALSubstate_i* Q, CALIndices indexes)
{
    CALint ret;
    int linear_index = getLinearIndex(indexes, calModel->coordinatesDimensions, calModel->numberOfCoordinates);
#if CAL_PARALLEL == 1
    CAL_SET_CELL_LOCK(linear_index, calModel->calRun->locks );
#endif

    ret = calGetMatrixElement(Q->current, linear_index);

#if CAL_PARALLEL == 1
    CAL_UNSET_CELL_LOCK(linear_index, calModel->calRun->locks );
#endif

    return ret;
}

CALbyte calGetX_b(struct CALModel* calModel, struct CALSubstate_b* Q, CALIndices central_cell, int n)
{
    CALbyte ret;
    int linear_index_central_cell = getLinearIndex(central_cell, calModel->coordinatesDimensions, calModel->numberOfCoordinates);
    int neighbour_index = calModel->calNeighborPool->neighborPool[linear_index_central_cell][n];
#if CAL_PARALLEL == 1
    CAL_SET_CELL_LOCK(neighbour_index, calModel->calRun->locks );
#endif
    ret = Q->current[neighbour_index];

#if CAL_PARALLEL == 1
    CAL_UNSET_CELL_LOCK(neighbour_index, calModel->calRun->locks );
#endif

    return ret;

}

CALint calGetX_i(struct CALModel* calModel, struct CALSubstate_i* Q, CALIndices central_cell, int n)
{
    CALint ret;
    int linear_index_central_cell = getLinearIndex(central_cell, calModel->coordinatesDimensions, calModel->numberOfCoordinates);
    int neighbour_index = calModel->calNeighborPool->neighborPool[linear_index_central_cell][n];
#if CAL_PARALLEL == 1
    CAL_SET_CELL_LOCK(neighbour_index, calModel->calRun->locks );
#endif
    ret = Q->current[neighbour_index];

#if CAL_PARALLEL == 1
    CAL_UNSET_CELL_LOCK(neighbour_index, calModel->calRun->locks );
#endif

    return ret;
}

CALreal calGetX_r(struct CALModel* calModel, struct CALSubstate_r* Q, CALIndices central_cell, int n)
{
    CALreal ret;
    int linear_index_central_cell = getLinearIndex(central_cell, calModel->coordinatesDimensions, calModel->numberOfCoordinates);
    int neighbour_index = calModel->calNeighborPool->neighborPool[linear_index_central_cell][n];
#if CAL_PARALLEL == 1
    CAL_SET_CELL_LOCK(neighbour_index, calModel->calRun->locks );
#endif
    ret = Q->current[neighbour_index];

#if CAL_PARALLEL == 1
    CAL_UNSET_CELL_LOCK(neighbour_index, calModel->calRun->locks );
#endif

    return ret;
}

void calSet_b(struct CALModel* calModel, struct CALSubstate_b* Q, CALIndices central_cell, CALbyte value)
{
    int linear_index = getLinearIndex(central_cell, calModel->coordinatesDimensions, calModel->numberOfCoordinates);
#if CAL_PARALLEL == 1
    CAL_SET_CELL_LOCK(linear_index, calModel->calRun->locks );
#endif
    calSetMatrixElement(Q->next, linear_index, value);
#if CAL_PARALLEL == 1
    CAL_UNSET_CELL_LOCK(linear_index, calModel->calRun->locks );
#endif
}

void calSet_i(struct CALModel* calModel, struct CALSubstate_i* Q, CALIndices central_cell, CALint value)
{
    int linear_index = getLinearIndex(central_cell, calModel->coordinatesDimensions, calModel->numberOfCoordinates);
#if CAL_PARALLEL == 1
    CAL_SET_CELL_LOCK(linear_index, calModel->calRun->locks );
#endif
    calSetMatrixElement(Q->next, linear_index, value);
#if CAL_PARALLEL == 1
    CAL_UNSET_CELL_LOCK(linear_index, calModel->calRun->locks );
#endif
}

void calSet_r(struct CALModel* calModel, struct CALSubstate_r* Q, CALIndices central_cell, CALreal value)
{
    int linear_index = getLinearIndex(central_cell, calModel->coordinatesDimensions, calModel->numberOfCoordinates);
#if CAL_PARALLEL == 1
    CAL_SET_CELL_LOCK(linear_index, calModel->calRun->locks );
#endif
    calSetMatrixElement(Q->next, linear_index, value);
#if CAL_PARALLEL == 1
    CAL_UNSET_CELL_LOCK(linear_index, calModel->calRun->locks );
#endif
}

void calSetCurrent_b(struct CALModel* calModel, struct CALSubstate_b* Q, CALIndices central_cell, CALbyte value)
{
    int linear_index = getLinearIndex(central_cell, calModel->coordinatesDimensions, calModel->numberOfCoordinates);
#if CAL_PARALLEL == 1
    CAL_SET_CELL_LOCK(linear_index, calModel->calRun->locks );
#endif
    calSetMatrixElement(Q->current, linear_index, value);
#if CAL_PARALLEL == 1
    CAL_UNSET_CELL_LOCK(linear_index, calModel->calRun->locks );
#endif
}

void calSetCurrent_i(struct CALModel* calModel, struct CALSubstate_i* Q, CALIndices central_cell, CALint value)
{
    int linear_index = getLinearIndex(central_cell, calModel->coordinatesDimensions, calModel->numberOfCoordinates);
#if CAL_PARALLEL == 1
    CAL_SET_CELL_LOCK(linear_index, calModel->calRun->locks );
#endif
    calSetMatrixElement(Q->current, linear_index, value);
#if CAL_PARALLEL == 1
    CAL_UNSET_CELL_LOCK(linear_index, calModel->calRun->locks );
#endif
}

void calSetCurrent_r(struct CALModel* calModel, struct CALSubstate_r* Q, CALIndices central_cell, CALreal value)
{
    int linear_index = getLinearIndex(central_cell, calModel->coordinatesDimensions, calModel->numberOfCoordinates);
#if CAL_PARALLEL == 1
    CAL_SET_CELL_LOCK(linear_index, calModel->calRun->locks );
#endif
    calSetMatrixElement(Q->current, linear_index, value);
#if CAL_PARALLEL == 1
    CAL_UNSET_CELL_LOCK(linear_index, calModel->calRun->locks );
#endif
}

void calUpdateSubstate_b(struct CALModel* calModel, struct CALSubstate_b* Q)
{
    calCopyBuffer_b(Q->next, Q->current, calModel->cellularSpaceDimension);
}

void calUpdateSubstate_i(struct CALModel* calModel, struct CALSubstate_i* Q)
{
    calCopyBuffer_i(Q->next, Q->current, calModel->cellularSpaceDimension);
}

void calUpdateSubstate_r(struct CALModel* calModel, struct CALSubstate_r* Q)
{
    calCopyBuffer_r(Q->next, Q->current, calModel->cellularSpaceDimension);
}

void calUpdate(struct CALModel* calModel)
{

    int i;

    //updating substates
    for (i=0; i < calModel->sizeof_pQb_array; i++)
        calUpdateSubstate_b(calModel, calModel->pQb_array[i]);

    for (i=0; i < calModel->sizeof_pQi_array; i++)
        calUpdateSubstate_i(calModel, calModel->pQi_array[i]);

    for (i=0; i < calModel->sizeof_pQr_array; i++)
        calUpdateSubstate_r(calModel, calModel->pQr_array[i]);
}


int*calGetCell(struct CALModel* calModel, ...)
{
    va_list arguments;

    int n = calModel->numberOfCoordinates;
    int* cell = (int*) malloc(sizeof(int) * n);
    int x;
    va_start (arguments, n);
    for (x = 0; x < n; x++ )
    {
        cell[x] = va_arg( arguments, int );
    }
    va_end ( arguments );

    int* cell_to_return = calModel->calIndexesPool->pool[getLinearIndex(cell, calModel->coordinatesDimensions, n)];

    free(cell);

    return cell_to_return;
}

void calFinalize(struct CALModel* calModel)
{
    int i;

    for (i=0; i < calModel->sizeof_pQb_array; i++)
    {
        calDeleteBuffer_b(calModel->pQb_array[i]->current);
        calDeleteBuffer_b(calModel->pQb_array[i]->next);
    }

    for (i=0; i < calModel->sizeof_pQi_array; i++)
    {
        calDeleteBuffer_i(calModel->pQi_array[i]->current);
        calDeleteBuffer_i(calModel->pQi_array[i]->next);
    }

    for (i=0; i < calModel->sizeof_pQr_array; i++)
    {
        calDeleteBuffer_r(calModel->pQr_array[i]->current);
        calDeleteBuffer_r(calModel->pQr_array[i]->next);
    }

    free(calModel->pQb_array);
    free(calModel->pQi_array);
    free(calModel->pQr_array);

    for(i = 0; i < calModel->num_of_processes; i++)
    {
        if(calModel->model_processes[i].type == 'L')
            free(calModel->model_processes[i].localProcess);
        else
            free(calModel->model_processes[i].globalProcess);
    }
    free(calModel->model_processes);

    free(calModel);

    calModel = NULL;
}
