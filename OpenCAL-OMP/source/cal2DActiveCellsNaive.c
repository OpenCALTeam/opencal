#include <OpenCAL-OMP/cal2DActiveCellsNaive.h>
#include <OpenCAL-OMP/cal2DBuffer.h>
#include <string.h>

void updateRectMinAndMax2D(int* minR, int* maxR, int celli)
{
    if(celli <= *minR)
        *minR = celli;
    else
        if(celli >= *maxR)
            *maxR = celli;
}

void calAddActiveCellNaive2D(struct CALModel2D* ca2D, int i, int j)
{
    CAL_SET_CELL_LOCK(i, j, ca2D);

    if (!calGetMatrixElement(ca2D->A->flags, ca2D->columns, i, j))
    {
        calSetMatrixElement(ca2D->A->flags, ca2D->columns, i, j, CAL_TRUE);

        CAL_UNSET_CELL_LOCK(i, j, ca2D);

        ca2D->A->size_next[CAL_GET_THREAD_NUM()]++;
        updateRectMinAndMax2D(&ca2D->mini, &ca2D->maxi, i);
        updateRectMinAndMax2D(&ca2D->minj, &ca2D->maxj, j);
        //printf("ca2D->A->size_next[CAL_GET_THREAD_NUM()]  = %d", ca2D->A->size_next[CAL_GET_THREAD_NUM()]);
        return;
    }

    CAL_UNSET_CELL_LOCK(i, j, ca2D);
}

void calRemoveActiveCellNaive2D(struct CALModel2D* ca2D, int i, int j)
{
    CAL_SET_CELL_LOCK(i, j, ca2D);

    if (calGetMatrixElement(ca2D->A->flags, ca2D->columns, i, j))
    {
        calSetMatrixElement(ca2D->A->flags, ca2D->columns, i, j, CAL_FALSE);
        CAL_UNSET_CELL_LOCK(i, j, ca2D);

        ca2D->A->size_next[CAL_GET_THREAD_NUM()]--;
        return;
    }

    CAL_UNSET_CELL_LOCK(i, j, ca2D);
}

void calUpdateActiveCellsNaive2D(struct CALModel2D* ca2D)
{
    
    int i, j, n;
    if(ca2D->maxi == -1 && ca2D->maxj==-1)
    {
            printf("Begin Update \n");
#pragma omp for
        for (i = 0; i < ca2D->rows; i++)
            for (j = 0; j < ca2D->columns; j++)
                    if (calGetMatrixElement(ca2D->A->flags,ca2D->columns, i, j))
                    {
                        if (i < ca2D->mini)
                        {
                            ca2D->mini = i;//(- ca2D->rectangleBorder;
                        }
                        if (i > ca2D->maxi)
                        {
                            ca2D->maxi = i;//+ ca2D->rectangleBorder;
                        }

                        if (j < ca2D->minj)
                        {
                            ca2D->minj = j;//- ca2D->rectangleBorder;
                        }
                        if (j > ca2D->maxj)
                        {
                            ca2D->maxj = j;//+ ca2D->rectangleBorder;
                        }
                    }

                    printf("Update \n");            
                    printf("(mini, maxi) = (%d,%d) \n", ca2D->mini, ca2D->maxi);
                    printf("(minj, maxj) = (%d,%d) \n", ca2D->minj, ca2D->maxj);
    // exit(0);
    }
    int diff;


    int tn;
    struct CALCell2D **tcells;
    int *tsize;

    free(ca2D->A->cells);
    ca2D->A->cells = NULL;

    diff=0;
    for(i=0;i<ca2D->A->num_threads;i++) {
        diff += ca2D->A->size_next[i];
        ca2D->A->size_next[i] = 0;
    }

    ca2D->A->size_current += diff;
    if (ca2D->A->size_current == 0)
        return;


    ca2D->A->cells = (struct CALCell2D*)malloc(sizeof(struct CALCell2D)*ca2D->A->size_current);

    tcells = (struct CALCell2D**)malloc(sizeof(struct CALCell2D*) * ca2D->A->num_threads);
    for(i = 0;i < ca2D->A->num_threads;i++) {
        tcells[i] = (struct CALCell2D*)malloc(sizeof(struct CALCell2D) * ca2D->A->size_current);
    }

    tsize = (int *)malloc(sizeof(int) * ca2D->A->num_threads);

#pragma omp parallel shared(tcells, tsize) private (i, j, tn)
    {
        tn = CAL_GET_THREAD_NUM();
        tsize[tn] = 0;

#pragma omp for
        for (i=0; i<ca2D->rows; i++)
            for (j=0; j<ca2D->columns; j++)
                if (calGetMatrixElement(ca2D->A->flags, ca2D->columns, i, j))
                {
                    tcells[tn][ tsize[tn] ].i = i;
                    tcells[tn][ tsize[tn] ].j = j;
                    tsize[tn]++;
                }

// #pragma omp for 
//         for (i=ca2D->mini; i<=ca2D->maxi; i++)
//             for (j=ca2D->minj; j<=ca2D->maxj; j++)
//                 if (calGetMatrixElement(ca2D->A->flags, ca2D->columns, i, j))
//                 {
//                     tcells[tn][ tsize[tn] ].i = i;
//                     tcells[tn][ tsize[tn] ].j = j;
//                     tsize[tn]++;
//                 }

      }

    n = 0;
    for (i = 0; i < ca2D->A->num_threads; i++) {
        memcpy(&ca2D->A->cells[n],
               tcells[i], sizeof(struct CALCell2D) * tsize[i]);
        n += tsize[i];
        free(tcells[i]);
    }
//printf("Begin Update (%d,%d)  (%d,%d) - n = %d \n",ca2D->mini, ca2D->maxi,ca2D->minj, ca2D->maxj, n);

    free(tsize);
    free(tcells);

}


void calFreeActiveCellsNaive2D(struct CALActiveCells2D* activeCells)
{
    free( activeCells->cells );
    free( activeCells->flags );
}

void calSetActiveCellsNaiveBuffer2Db(CALbyte* M, CALbyte value, struct CALModel2D* ca2D)
{
    int n;

#pragma omp for private(value)
    for(n=0; n<ca2D->A->size_current; n++)
        M[ca2D->A->cells[n].i * ca2D->columns + ca2D->A->cells[n].j] = value;
}

void calSetActiveCellsNaiveBuffer2Di(CALint* M, CALint value, struct CALModel2D* ca2D)
{
    int n;

#pragma omp for private(value)
    for(n=0; n<ca2D->A->size_current; n++)
        M[ca2D->A->cells[n].i * ca2D->columns + ca2D->A->cells[n].j] = value;
}

void calSetActiveCellsNaiveBuffer2Dr(CALreal* M, CALreal value, struct CALModel2D* ca2D)
{
    int n;

#pragma omp for private(value)
    for(n=0; n<ca2D->A->size_current; n++)
        M[ca2D->A->cells[n].i * ca2D->columns + ca2D->A->cells[n].j] = value;
}

void calCopyBufferActiveCellsNaive2Db(CALbyte* M_src, CALbyte* M_dest, struct CALModel2D* ca2D)
{
    int c, n;

#pragma omp parallel for private (c)
    for(n=0; n<ca2D->A->size_current; n++)
    {
        c = ca2D->A->cells[n].i * ca2D->columns + ca2D->A->cells[n].j;
        if (M_dest[c] != M_src[c])
            M_dest[c] = M_src[c];
    }
}

void calCopyBufferActiveCellsNaive2Di(CALint* M_src, CALint* M_dest, struct CALModel2D* ca2D)
{
    int c, n;

#pragma omp parallel for private (c)
    for(n=0; n<ca2D->A->size_current; n++)
    {
        c = ca2D->A->cells[n].i * ca2D->columns + ca2D->A->cells[n].j;
        if (M_dest[c] != M_src[c])
            M_dest[c] = M_src[c];
    }
}

void calCopyBufferActiveCellsNaive2Dr(CALreal* M_src, CALreal* M_dest, struct CALModel2D* ca2D)
{
    int c, n;

#pragma omp parallel for private (c)
    for(n=0; n<ca2D->A->size_current; n++)
    {
        // printf("ca2D->A->size_current %d, n %d, ca2D->A->cells[n].i %d, ca2D->A->cells[n].j %d \n", ca2D->A->size_current, n, ca2D->A->cells[n].i,ca2D->A->cells[n].j);
        c = ca2D->A->cells[n].i * ca2D->columns + ca2D->A->cells[n].j;
        if (M_dest[c] != M_src[c])
            M_dest[c] = M_src[c];
    }
}

void calApplyElementaryProcessActiveCellsNaive2D(struct CALModel2D* ca2D, CALCallbackFunc2D elementary_process)
{
    int n;
#pragma omp parallel for private(n)
    for (n=0; n<ca2D->A->size_current; n++)
        elementary_process(ca2D, ca2D->A->cells[n].i, ca2D->A->cells[n].j);
}
