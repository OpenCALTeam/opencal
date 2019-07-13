#include <OpenCAL/cal3DActiveCellsNaive.h>
#include <OpenCAL/cal3DBuffer.h>

void calAddActiveCellNaive3D(struct CALActiveCells3D* activeCells, int i, int j, int k, int columns, int rows)
{
    if (!calGetBuffer3DElement(activeCells->flags, rows, columns, i, j, k))
    {
        calSetBuffer3DElement(activeCells->flags, rows, columns, i, j, k, CAL_TRUE);
        activeCells->size_next++;
    }
}

void calRemoveActiveCellNaive3D(struct CALActiveCells3D* activeCells, int i, int j, int k, int columns, int rows)
{
    if (calGetBuffer3DElement(activeCells->flags, rows, columns, i, j, k))
    {
        calSetBuffer3DElement(activeCells->flags, rows, columns, i, j, k, CAL_FALSE);
        activeCells->size_next--;
    }
}

void calUpdateActiveCellsNaive3D(struct CALModel3D* ca3D)
{
    int i, j, k, n;

    free(ca3D->A->cells);
    ca3D->A->cells = NULL;

    ca3D->A->size_current = ca3D->A->size_next;
    if (ca3D->A->size_current == 0)
        return;

    ca3D->A->cells = (struct CALCell3D*)malloc(sizeof(struct CALCell3D)*ca3D->A->size_current);

    n = 0;
    for (i=0; i<ca3D->rows; i++)
        for (j=0; j<ca3D->columns; j++)
            for (k = 0; k<ca3D->slices; k++)
                if (calGetBuffer3DElement(ca3D->A->flags, ca3D->rows, ca3D->columns, i, j, k))
                {
                    ca3D->A->cells[n].i = i;
                    ca3D->A->cells[n].j = j;
                    ca3D->A->cells[n].k = k;
                    n++;
                }
}


void calFreeActiveCellsNaive3D(struct CALActiveCells3D* activeCells)
{
    free( activeCells->cells );
    free( activeCells->flags );
}

void calSetActiveCellsNaiveBuffer3Db(CALbyte* M, CALbyte value, CALModel3D* ca3D)
{
    int n;

    for(n=0; n<ca3D->A->size_current; n++)
        M[ca3D->A->cells[n].k * ca3D->rows * ca3D->columns + ca3D->A->cells[n].i * ca3D->columns + ca3D->A->cells[n].j] = value;
}

void calSetActiveCellsNaiveBuffer3Di(CALint* M, CALint value, CALModel3D* ca3D)
{
    int n;

    for(n=0; n<ca3D->A->size_current; n++)
        M[ca3D->A->cells[n].k * ca3D->rows * ca3D->columns + ca3D->A->cells[n].i * ca3D->columns + ca3D->A->cells[n].j] = value;
}

void calSetActiveCellsNaiveBuffer3Dr(CALreal* M, CALreal value, CALModel3D* ca3D)
{
    int n;

    for(n=0; n<ca3D->A->size_current; n++)
        M[ca3D->A->cells[n].k * ca3D->rows * ca3D->columns + ca3D->A->cells[n].i * ca3D->columns + ca3D->A->cells[n].j] = value;
}

void calCopyBufferActiveCellsNaive3Db(CALbyte* M_src, CALbyte* M_dest, CALModel3D* ca3D)
{
    int c, n;
    for(n=0; n<ca3D->A->size_current; n++)
    {
        c = ca3D->A->cells[n].k * ca3D->rows * ca3D->columns + ca3D->A->cells[n].i * ca3D->columns + ca3D->A->cells[n].j;
        if (M_dest[c] != M_src[c])
            M_dest[c] = M_src[c];
    }
}

void calCopyBufferActiveCellsNaive3Di(CALint* M_src, CALint* M_dest, CALModel3D* ca3D)
{
    int c, n;
    for(n=0; n<ca3D->A->size_current; n++)
    {
        c = ca3D->A->cells[n].k * ca3D->rows * ca3D->columns + ca3D->A->cells[n].i * ca3D->columns + ca3D->A->cells[n].j;
        if (M_dest[c] != M_src[c])
            M_dest[c] = M_src[c];
    }
}

void calCopyBufferActiveCellsNaive3Dr(CALreal* M_src, CALreal* M_dest, CALModel3D* ca3D)
{
    int c, n;
    for(n=0; n<ca3D->A->size_current; n++)
    {
        c = ca3D->A->cells[n].k * ca3D->rows * ca3D->columns + ca3D->A->cells[n].i * ca3D->columns + ca3D->A->cells[n].j;
        if (M_dest[c] != M_src[c])
            M_dest[c] = M_src[c];
    }
}

void calApplyElementaryProcessActiveCellsNaive3D(CALModel3D* ca3D, CALCallbackFunc3D elementary_process)
{
    int n;
    for (n=0; n<ca3D->A->size_current; n++)
        elementary_process(ca3D, ca3D->A->cells[n].i, ca3D->A->cells[n].j, ca3D->A->cells[n].k);
}
