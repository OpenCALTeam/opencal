#include <OpenCAL-CPU/calActiveCells.h>
#include <OpenCAL-CPU/calActiveCellsNaive.h>
#include <OpenCAL-CPU/calActiveCellsCLL.h>

struct CALActiveCells* calACDef(struct CALModel* calModel, enum CALOptimization CAL_OPTIMIZATION)
{
    if(CAL_OPTIMIZATION == CAL_OPT_ACTIVE_CELLS)
        return calMakeACCLL(calModel);
    else if(CAL_OPTIMIZATION == CAL_OPT_ACTIVE_CELLS_NAIVE)
        return calMakeACNaive(calModel);
}

void calAddActiveCell(struct CALModel* model, CALIndices cell)
{
    if(model->A->OPTIMIZATION == CAL_OPT_ACTIVE_CELLS)
        calAddActiveCellCLL(((struct CALActiveCellsCLL*) model->A), cell);
    else
        calAddActiveCellNaive(((struct CALActiveCellsNaive*) model->A), cell);
}

void calAddActiveCellX(struct CALModel* model, CALIndices cell, int n)
{
    if(model->A->OPTIMIZATION == CAL_OPT_ACTIVE_CELLS)
        calAddActiveCellCLL(((struct CALActiveCellsCLL*) model->A), model->calIndexesPool->pool[calGetNeighbourIndex(model, cell, n)]);
    else
        calAddActiveCellNaive(((struct CALActiveCellsNaive*) model->A), model->calIndexesPool->pool[calGetNeighbourIndex(model, cell, n)]);
}

void calRemoveActiveCell(struct CALModel* model, CALIndices cell)
{
    if(model->A->OPTIMIZATION == CAL_OPT_ACTIVE_CELLS)
        calRemoveActiveCellCLL(((struct CALActiveCellsCLL*) model->A), cell);
    else
        calRemoveActiveCellNaive(((struct CALActiveCellsNaive*) model->A), cell);

}

CALbyte calApplyLocalFunctionOpt(struct CALModel* model, CALLocalProcess local_process)
{
    if(model->A->OPTIMIZATION == CAL_OPT_ACTIVE_CELLS && ((struct CALActiveCellsCLL*) model->A)->size_current > 0)
    {
        calApplyElementaryProcessActiveCellsCLL(((struct CALActiveCellsCLL*) model->A), local_process);
        return CAL_TRUE;
    }
        else if(model->A->OPTIMIZATION == CAL_OPT_ACTIVE_CELLS_NAIVE && ((struct CALActiveCellsNaive*) model->A)->size_current > 0)
    {
        calApplyElementaryProcessActiveCellsNaive(((struct CALActiveCellsNaive*) model->A), local_process);
        return CAL_TRUE;
    }
    else
        return CAL_FALSE;
}

void calUpdateActiveCells(struct CALActiveCells* A)
{
    if(A->OPTIMIZATION == CAL_OPT_ACTIVE_CELLS)
        calUpdateActiveCellsCLL(((struct CALActiveCellsCLL*) A));
    else
        calUpdateActiveCellsNaive(((struct CALActiveCellsNaive*) A));

}

CALbyte calCopyBufferActiveCells_b(CALbyte* M_src, CALbyte* M_dest, struct CALActiveCells* A)
{
    if(A->OPTIMIZATION == CAL_OPT_ACTIVE_CELLS && ((struct CALActiveCellsCLL*) A)->size_current > 0)
    {
        calCopyBufferActiveCellsCLL_b(M_src, M_dest, ((struct CALActiveCellsCLL*) A));
        return CAL_TRUE;
    }
        else if(A->OPTIMIZATION == CAL_OPT_ACTIVE_CELLS_NAIVE && ((struct CALActiveCellsNaive*) A)->size_current > 0)
    {
        calCopyBufferActiveCellsNaive_b(M_src, M_dest, ((struct CALActiveCellsNaive*) A));
        return CAL_TRUE;
    }
    else
        return CAL_FALSE;
}

CALbyte calCopyBufferActiveCells_i(CALint* M_src, CALint* M_dest, struct CALActiveCells* A)
{
    if(A->OPTIMIZATION == CAL_OPT_ACTIVE_CELLS && ((struct CALActiveCellsCLL*) A)->size_current > 0)
    {
        calCopyBufferActiveCellsCLL_i(M_src, M_dest, ((struct CALActiveCellsCLL*) A));
        return CAL_TRUE;
    }
        else if(A->OPTIMIZATION == CAL_OPT_ACTIVE_CELLS_NAIVE && ((struct CALActiveCellsNaive*) A)->size_current > 0)
    {
        calCopyBufferActiveCellsNaive_i(M_src, M_dest, ((struct CALActiveCellsNaive*) A));
        return CAL_TRUE;
    }
    else
        return CAL_FALSE;

}

CALbyte calCopyBufferActiveCells_r(CALreal* M_src, CALreal* M_dest, struct CALActiveCells* A)
{
    if(A->OPTIMIZATION == CAL_OPT_ACTIVE_CELLS && ((struct CALActiveCellsCLL*) A)->size_current > 0)
    {
        calCopyBufferActiveCellsCLL_r(M_src, M_dest, ((struct CALActiveCellsCLL*) A));
        return CAL_TRUE;
    }
        else if(A->OPTIMIZATION == CAL_OPT_ACTIVE_CELLS_NAIVE && ((struct CALActiveCellsNaive*) A)->size_current > 0)
    {
        calCopyBufferActiveCellsNaive_r(M_src, M_dest, ((struct CALActiveCellsNaive*) A));
        return CAL_TRUE;
    }
    else
        return CAL_FALSE;
}

CALbyte calSetActiveCellsBuffer_b(CALbyte* M, CALbyte value, struct CALActiveCells* A)
{
    if(A->OPTIMIZATION == CAL_OPT_ACTIVE_CELLS && ((struct CALActiveCellsCLL*) A)->size_current > 0)
    {
        calSetActiveCellsCLLBuffer_b(M, value, ((struct CALActiveCellsCLL*) A));
        return CAL_TRUE;
    }
        else if(A->OPTIMIZATION == CAL_OPT_ACTIVE_CELLS_NAIVE && ((struct CALActiveCellsNaive*) A)->size_current > 0)
    {
        calSetActiveCellsNaiveBuffer_b(M, value, ((struct CALActiveCellsNaive*) A));
        return CAL_TRUE;
    }
    else
        return CAL_FALSE;
}

CALbyte calSetActiveCellsBuffer_i(CALint* M, CALint value, struct CALActiveCells* A)
{
    if(A->OPTIMIZATION == CAL_OPT_ACTIVE_CELLS && ((struct CALActiveCellsCLL*) A)->size_current > 0)
    {
        calSetActiveCellsCLLBuffer_i(M, value, ((struct CALActiveCellsCLL*) A));
        return CAL_TRUE;
    }
        else if(A->OPTIMIZATION == CAL_OPT_ACTIVE_CELLS_NAIVE && ((struct CALActiveCellsNaive*) A)->size_current > 0)
    {
        calSetActiveCellsNaiveBuffer_i(M, value, ((struct CALActiveCellsNaive*) A));
        return CAL_TRUE;
    }
    else
        return CAL_FALSE;
}

CALbyte calSetActiveCellsBuffer_r(CALreal* M, CALreal value, struct CALActiveCells* A)
{
    if(A->OPTIMIZATION == CAL_OPT_ACTIVE_CELLS && ((struct CALActiveCellsCLL*) A)->size_current > 0)
    {
        calSetActiveCellsCLLBuffer_r(M, value, ((struct CALActiveCellsCLL*) A));
        return CAL_TRUE;
    }
        else if(A->OPTIMIZATION == CAL_OPT_ACTIVE_CELLS_NAIVE && ((struct CALActiveCellsNaive*) A)->size_current > 0)
    {
        calSetActiveCellsNaiveBuffer_r(M, value, ((struct CALActiveCellsNaive*) A));
        return CAL_TRUE;
    }
    else
        return CAL_FALSE;
}

void calFreeActiveCells(struct CALActiveCells* A)
{
    if(A->OPTIMIZATION == CAL_OPT_ACTIVE_CELLS)
        calFreeContiguousLinkedList(((struct CALActiveCellsCLL*) A));
    else
        calFreeActiveCellsNaive(((struct CALActiveCellsNaive*) A));

}

void calCheckForActiveCells(struct CALModel* model, CALbyte (*active_cells_def)(struct CALModel*, CALIndices, int))
{
    CALIndices* pool = model->calIndexesPool->pool;
    int dim = model->cellularSpaceDimension;
    int numb_of_dim = model->numberOfCoordinates;
    int i;

    for(i = 0; i < dim; i++)
    {
        if(active_cells_def(model, pool[i], numb_of_dim))
            calAddActiveCell(model, pool[i]);
        else
            calRemoveActiveCell(model, pool[i]);
    }

}

void calRemoveInactiveCells(struct CALModel* model, CALbyte (*active_cells_def)(struct CALModel*, CALIndices, int))
{
        if(model->A->OPTIMIZATION == CAL_OPT_ACTIVE_CELLS)
        {
           struct CALActiveCellsCLL* A1 = ((struct CALActiveCellsCLL*) model->A);
           calRemoveInactiveCellsCLL(A1, active_cells_def);
        }
        else if(model->A->OPTIMIZATION == CAL_OPT_ACTIVE_CELLS_NAIVE)
        {
           struct CALActiveCellsNaive* A1 = ((struct CALActiveCellsNaive*) model->A);
           calRemoveInactiveCellsNaive(A1, active_cells_def);
        }
}


