#include <OpenCAL/cal2DActiveCellsNaive.h>
#include <OpenCAL/cal2DBuffer.h>

void calAddActiveCellNaive2D(struct CALActiveCells2D* activeCells, int i, int j, int columns)
{
    if (!calGetMatrixElement(activeCells->flags, columns, i, j))
    {
        calSetMatrixElement(activeCells->flags, columns, i, j, CAL_TRUE);
        activeCells->size_next++;
    }
}

void calRemoveActiveCellNaive2D(struct CALActiveCells2D* activeCells, int i, int j, int columns)
{
    if (calGetMatrixElement( activeCells->flags, columns, i, j))
    {
        calSetMatrixElement(activeCells->flags, columns, i, j, CAL_FALSE);
        activeCells->size_next--;
    }
}

void calUpdateActiveCellsNaive2D(struct CALModel2D* ca2D)
{
    int i, j, n;

    free(ca2D->A->cells);
    ca2D->A->cells = NULL;

    ca2D->A->size_current = ca2D->A->size_next;
    if (ca2D->A->size_current == 0)
        return;

    ca2D->A->cells = (struct CALCell2D*)malloc(sizeof(struct CALCell2D)*ca2D->A->size_current);

    n = 0;
    for (i=0; i<ca2D->rows; i++)
        for (j=0; j<ca2D->columns; j++)
            if (calGetMatrixElement(ca2D->A->flags, ca2D->columns, i, j))
            {
                ca2D->A->cells[n].i = i;
                ca2D->A->cells[n].j = j;
                n++;
            }
}


void calFreeActiveCellsNaive2D(struct CALActiveCells2D* activeCells)
{
    free( activeCells->cells );
    free( activeCells->flags );
}

void calSetActiveCellsNaiveBuffer2Db(CALbyte* M, CALbyte value, CALModel2D* ca2D)
{
    int n;

    for(n=0; n<ca2D->A->size_current; n++)
        M[ca2D->A->cells[n].i * ca2D->columns + ca2D->A->cells[n].j] = value;
}

void calSetActiveCellsNaiveBuffer2Di(CALint* M, CALint value, CALModel2D* ca2D)
{
    int n;

    for(n=0; n<ca2D->A->size_current; n++)
        M[ca2D->A->cells[n].i * ca2D->columns + ca2D->A->cells[n].j] = value;
}

void calSetActiveCellsNaiveBuffer2Dr(CALreal* M, CALreal value, CALModel2D* ca2D)
{
    int n;

    for(n=0; n<ca2D->A->size_current; n++)
        M[ca2D->A->cells[n].i * ca2D->columns + ca2D->A->cells[n].j] = value;
}

void calCopyBufferActiveCellsNaive2Db(CALbyte* M_src, CALbyte* M_dest, CALModel2D* ca2D)
{
    int c, n;
    for(n=0; n<ca2D->A->size_current; n++)
    {
        c = ca2D->A->cells[n].i * ca2D->columns + ca2D->A->cells[n].j;
        if (M_dest[c] != M_src[c])
            M_dest[c] = M_src[c];
    }
}

void calCopyBufferActiveCellsNaive2Di(CALint* M_src, CALint* M_dest, CALModel2D* ca2D)
{
    int c, n;
    for(n=0; n<ca2D->A->size_current; n++)
    {
        c = ca2D->A->cells[n].i * ca2D->columns + ca2D->A->cells[n].j;
        if (M_dest[c] != M_src[c])
            M_dest[c] = M_src[c];
    }
}

void calCopyBufferActiveCellsNaive2Dr(CALreal* M_src, CALreal* M_dest, CALModel2D* ca2D)
{
    int c, n;
    for(n=0; n<ca2D->A->size_current; n++)
    {
        c = ca2D->A->cells[n].i * ca2D->columns + ca2D->A->cells[n].j;
        if (M_dest[c] != M_src[c])
            M_dest[c] = M_src[c];
    }
}

void calApplyElementaryProcessActiveCellsNaive2D(CALModel2D* ca2D, CALCallbackFunc2D elementary_process)
{
    int n;
    for (n=0; n<ca2D->A->size_current; n++)
        elementary_process(ca2D, ca2D->A->cells[n].i, ca2D->A->cells[n].j);
}
