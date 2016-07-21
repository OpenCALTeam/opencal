#include <OpenCAL-CPU/calActiveCells.h>
#include <OpenCAL-CPU/calActiveCellsNaive.h>
#include <OpenCAL-CPU/calActiveCellsCLL.h>

void calAddActiveCells(struct CALActiveCells* A, CALIndices cell)
{
    if(A->OPTIMIZATION == CAL_OPT_ACTIVE_CELLS)
        calAddActiveCellCLL(((struct CALActiveCellsCLL*) A), cell);
    else
        calAddActiveCellNaive(((struct CALActiveCellsNaive*) A), cell);
}

void calRemoveActiveCells(struct CALActiveCells* A, CALIndices cell)
{
    if(A->OPTIMIZATION == CAL_OPT_ACTIVE_CELLS)
        calRemoveActiveCellCLL(((struct CALActiveCellsCLL*) A), cell);
    else
        calRemoveActiveCellNaive(((struct CALActiveCellsNaive*) A), cell);

}

CALbyte calApplyLocalFunctionOpt(struct CALActiveCells* A, CALLocalProcess local_process)
{
    if(A->OPTIMIZATION == CAL_OPT_ACTIVE_CELLS && ((struct CALActiveCellsCLL*) A)->size_current > 0)
    {
        calApplyElementaryProcessActiveCellsCLL(((struct CALActiveCellsCLL*) A), local_process);
        return CAL_TRUE;
    }
        else if(A->OPTIMIZATION == CAL_OPT_ACTIVE_CELLS_NAIVE && ((struct CALActiveCellsNaive*) A)->size_current > 0)
    {
        calApplyElementaryProcessActiveCellsNaive(((struct CALActiveCellsNaive*) A), local_process);
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
