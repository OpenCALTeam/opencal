#include <OpenCAL-CPU/calActiveCellsNaive.h>

#include <string.h>
void calAddActiveCellNaive(struct CALModel* calModel, CALIndices cell)
{
    int linear_index = getLinearIndex(cell, calModel->coordinatesDimensions, calModel->numberOfCoordinates);
#if CAL_PARALLEL == 1
    CAL_SET_CELL_LOCK(linear_index, calModel->calRun->locks );
#endif

    if (!calGetMatrixElement(((struct CALActiveCellsNaive*)calModel->A)->flags, linear_index))
    {
        calSetMatrixElement(((struct CALActiveCellsNaive*)calModel->A)->flags, linear_index, CAL_TRUE);

        ((struct CALActiveCellsNaive*)calModel->A)->size_next[CAL_GET_THREAD_NUM()]++;
        return;
    }

#if CAL_PARALLEL == 1
    CAL_UNSET_CELL_LOCK(linear_index, calModel->calRun->locks );
#endif
}

void calRemoveActiveCellNaive(struct CALModel* calModel, CALIndices cell)
{
    int linear_index = getLinearIndex(cell, calModel->coordinatesDimensions, calModel->numberOfCoordinates);
#if CAL_PARALLEL == 1
    CAL_SET_CELL_LOCK(linear_index, calModel->calRun->locks );
#endif

    if (calGetMatrixElement(((struct CALActiveCellsNaive*)calModel->A)->flags, linear_index))
    {
        calSetMatrixElement(((struct CALActiveCellsNaive*)calModel->A)->flags, linear_index, CAL_FALSE);

        ((struct CALActiveCellsNaive*)calModel->A)->size_next[CAL_GET_THREAD_NUM()]--;
        return;
    }

#if CAL_PARALLEL == 1
    CAL_UNSET_CELL_LOCK(linear_index, calModel->calRun->locks );
#endif
}

void calApplyElementaryProcessActiveCellsNaive(struct CALModel *calModel, CALLocalProcess elementary_process)
{
    int n;
    int number_of_dimensions = calModel->numberOfCoordinates;
    struct CALActiveCellsNaive* A = ((struct CALActiveCellsNaive*)calModel->A);
#pragma omp parallel for private(n) firstprivate(calModel, A, number_of_dimensions)
    for (n = 0; n < A->size_current; n++)
        elementary_process(calModel, A->cells[n], number_of_dimensions);
}

void calFreeActiveCellsNaive(struct CALActiveCellsNaive* activeCells )
{
    free(activeCells->cells);
    free(activeCells->flags);
}

void calCopyBufferActiveCellsNaive_b(CALbyte* M_src, CALbyte* M_dest,  struct CALModel* calModel)
{
    int n;
    int linear_index;
    struct CALActiveCellsNaive* A = (struct CALActiveCellsNaive*)(calModel->A);
#pragma omp parallel for private (linear_index), firstprivate(A, calModel)
    for(n = 0; n < A->size_current; n++)
    {
        linear_index = getLinearIndex(A->cells[n], calModel->coordinatesDimensions, calModel->numberOfCoordinates);
        if (M_dest[linear_index] != M_src[linear_index])
            M_dest[linear_index] = M_src[linear_index];
    }
}

void calCopyBufferActiveCellsNaive_i(CALint* M_src, CALint* M_dest,  struct CALModel* calModel)
{
    int n;
    int linear_index;
    struct CALActiveCellsNaive* A = (struct CALActiveCellsNaive*)(calModel->A);
#pragma omp parallel for private (linear_index), firstprivate(A, calModel)
    for(n = 0; n < A->size_current; n++)
    {
        linear_index = getLinearIndex(A->cells[n], calModel->coordinatesDimensions, calModel->numberOfCoordinates);
        if (M_dest[linear_index] != M_src[linear_index])
            M_dest[linear_index] = M_src[linear_index];
    }
}

void calCopyBufferActiveCellsNaive_r(CALreal* M_src, CALreal* M_dest,  struct CALModel* calModel)
{
    int n;
    int linear_index;
    struct CALActiveCellsNaive* A = (struct CALActiveCellsNaive*)(calModel->A);
#pragma omp parallel for private (linear_index), firstprivate(A, calModel)
    for(n = 0; n < A->size_current; n++)
    {
        linear_index = getLinearIndex(A->cells[n], calModel->coordinatesDimensions, calModel->numberOfCoordinates);
        if (M_dest[linear_index] != M_src[linear_index])
            M_dest[linear_index] = M_src[linear_index];
    }
}


void calSetActiveCellsNaiveBuffer_b(CALbyte* M, CALbyte value, struct CALModel* calModel)
{
    int n;
    struct CALActiveCellsNaive* A = (struct CALActiveCellsNaive*)(calModel->A);

#pragma omp parallel for firstprivate(value)
    for( n = 0; n < A->size_current; n++)
        M[getLinearIndex(A->cells[n], calModel->coordinatesDimensions, calModel->numberOfCoordinates)] = value;
}

void calSetActiveCellsNaiveBuffer_i(CALint* M, CALint value, struct CALModel* calModel)
{
    int n;
    struct CALActiveCellsNaive* A = (struct CALActiveCellsNaive*)(calModel->A);

#pragma omp parallel for firstprivate(value)
    for( n = 0; n < A->size_current; n++)
        M[getLinearIndex(A->cells[n], calModel->coordinatesDimensions, calModel->numberOfCoordinates)] = value;
}

void calSetActiveCellsNaiveBuffer_r(CALreal* M, CALreal value, struct CALModel* calModel)
{
    int n;
    struct CALActiveCellsNaive* A = (struct CALActiveCellsNaive*)(calModel->A);

#pragma omp parallel for firstprivate(value)
    for( n = 0; n < A->size_current; n++)
        M[getLinearIndex(A->cells[n], calModel->coordinatesDimensions, calModel->numberOfCoordinates)] = value;
}


void calUpdateActiveCellsNaive(struct CALModel* calModel)
{
    struct CALActiveCellsNaive* A = (struct CALActiveCellsNaive*)(calModel->A);

    int i, n;
    int diff;


    int tn;
    CALIndices **tcells;
    int *tsize;

    free(A->cells);
    A->cells = NULL;

    diff = 0;
    for(i = 0;i < A->num_threads; i++) {
        diff += A->size_next[i];
        A->size_next[i] = 0;
    }

    A->size_current += diff;
    if (A->size_current == 0)
        return;


    A->cells = (CALIndices*) malloc(sizeof(CALIndices) * A->size_current);

    tcells = (CALIndices**) malloc (sizeof(CALIndices*) * A->num_threads);

    for(i = 0; i < A->num_threads; i++)
    {
        tcells[i] = (CALIndices*) malloc (sizeof(CALIndices) * A->size_current);
    }

    tsize = (int *)malloc(sizeof(int) * A->num_threads);

#pragma omp parallel shared(tcells, tsize) private (i, tn)
    {
        tn = CAL_GET_THREAD_NUM();
        tsize[tn] = 0;

        CALIndices* pool = calModel->calIndexesPool->pool;

#pragma omp for
        for (i = 0; i < calModel->cellularSpaceDimension; i++)
            if(calGetMatrixElement(A->flags, getLinearIndex(pool[i],
                                                            calModel->coordinatesDimensions, calModel->numberOfCoordinates)))
            {
                tcells[tn][ tsize[tn] ] = pool[i];
                tsize[tn]++;
            }

    }

    n = 0;
    for (i = 0; i < A->num_threads; i++) {
        memcpy(&A->cells[n],
               tcells[i], sizeof(CALIndices) * tsize[i]);
        n += tsize[i];
        free(tcells[i]);
    }

    free(tsize);
    free(tcells);
}
