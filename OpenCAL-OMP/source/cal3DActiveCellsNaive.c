#include <OpenCAL-OMP/cal3DActiveCellsNaive.h>
#include <OpenCAL-OMP/cal3DBuffer.h>
#include <string.h>

void updateRectMinAndMax(int* minR, int* maxR, int celli)
{
    if(celli < *minR)
        *minR = celli;
    else
        if(celli > *maxR)
            *maxR = celli;
}


void calAddActiveCellNaive3D(struct CALModel3D* ca3D, int i, int j, int k)
{
    CAL_SET_CELL_LOCK_3D(i, j, k, ca3D);

    if (!calGetBuffer3DElement(ca3D->A->flags, ca3D->rows, ca3D->columns, i, j, k))
    {
        calSetBuffer3DElement(ca3D->A->flags, ca3D->rows, ca3D->columns, i, j, k, CAL_TRUE);
        CAL_UNSET_CELL_LOCK_3D(i, j, k, ca3D);
        ca3D->A->size_next[CAL_GET_THREAD_NUM()]++;
        updateRectMinAndMax(&ca3D->mini, &ca3D->maxi, i);
        updateRectMinAndMax(&ca3D->minj, &ca3D->maxj, j);
        updateRectMinAndMax(&ca3D->mink, &ca3D->maxk, k);
    }

    CAL_UNSET_CELL_LOCK_3D(i, j, k, ca3D);
}

void calRemoveActiveCellNaive3D(CALModel3D* ca3D, int i, int j, int k)
{
    CAL_SET_CELL_LOCK_3D(i, j, k, ca3D);

    if (calGetBuffer3DElement(ca3D->A->flags, ca3D->rows, ca3D->columns, i, j, k))
    {
        calSetBuffer3DElement(ca3D->A->flags, ca3D->rows, ca3D->columns, i, j, k, CAL_FALSE);
        CAL_UNSET_CELL_LOCK_3D(i, j, k, ca3D);
        ca3D->A->size_next[CAL_GET_THREAD_NUM()]--;
        return;
    }

    CAL_UNSET_CELL_LOCK_3D(i, j, k, ca3D);
}

void calUpdateActiveCellsNaive3D(struct CALModel3D* ca3D)
{
    //printf("Begin Update \n");
    int i, j, k, n;
    if(ca3D->maxi == -1 && ca3D->maxj==-1 && ca3D->maxk == -1)
    {
#pragma omp for
        for (i = 0; i < ca3D->rows; i++)
            for (j = 0; j < ca3D->columns; j++)
                for (k = 0; k < ca3D->slices; k++)
                    if (calGetBuffer3DElement(ca3D->A->flags, ca3D->rows, ca3D->columns, i, j, k))
                    {
                        if (i < ca3D->mini)
                        {
                            ca3D->mini = i;//(- ca3D->rectangleBorder;
                        }
                        if (i > ca3D->maxi)
                        {
                            ca3D->maxi = i;//+ ca3D->rectangleBorder;
                        }

                        if (j < ca3D->minj)
                        {
                            ca3D->minj = j;//- ca3D->rectangleBorder;
                        }
                        if (j > ca3D->maxj)
                        {
                            ca3D->maxj = j;//+ ca3D->rectangleBorder;
                        }

                        if (k < ca3D->mink)
                        {
                            ca3D->mink = k;//- ca3D->rectangleBorder;
                        }
                        if (k > ca3D->maxk)
                        {
                            ca3D->maxk = k;//+ ca3D->rectangleBorder;
                        }
                    }

                    printf("Update \n");            
                    printf("(mini, maxi) = (%d,%d) \n", ca3D->mini, ca3D->maxi);
                    printf("(minj, maxj) = (%d,%d) \n", ca3D->minj, ca3D->maxj);
                    printf("(mink, maxk) = (%d,%d) \n", ca3D->mink, ca3D->maxk);
    // exit(0);
    }



    int tn;
    int diff;
    struct CALCell3D **tcells;
    int *tsize;

    free(ca3D->A->cells);

    ca3D->A->cells = NULL;

    diff = 0;
    for (i = 0; i < ca3D->A->num_threads; i++) {
        diff += ca3D->A->size_next[i];
        ca3D->A->size_next[i] = 0;
    }

    ca3D->A->size_current += diff;
    if (ca3D->A->size_current == 0)
        return;

    ca3D->A->cells = (struct CALCell3D*)malloc(sizeof(struct CALCell3D)*ca3D->A->size_current);

    tcells = (struct CALCell3D**)malloc(sizeof(struct CALCell3D*) * ca3D->A->num_threads);
    for (i = 0; i < ca3D->A->num_threads; i++) {
        tcells[i] = (struct CALCell3D*)malloc(sizeof(struct CALCell3D) * ca3D->A->size_current);
    }

    tsize = (int *)malloc(sizeof(int) * ca3D->A->num_threads);

#pragma omp parallel shared(tcells, tsize) private (i, j, k, tn)
    {
        tn = CAL_GET_THREAD_NUM();
        tsize[tn] = 0;

// #pragma omp for 
//         for (i = ca3D->mini; i <= ca3D->maxi; i++)
//             for (j = ca3D->minj; j <= ca3D->maxj; j++)
//                 for (k = ca3D->mink; k <= ca3D->maxk; k++){
//                     if (calGetBuffer3DElement(ca3D->A->flags, ca3D->rows, ca3D->columns, i, j, k))
//                     {
//                         tcells[tn][tsize[tn]].i = i;
//                         tcells[tn][tsize[tn]].j = j;
//                         tcells[tn][tsize[tn]].k = k;
//                         tsize[tn]++;
//                     }
//                 }


#pragma omp for 
        for (i = 0; i < ca3D->rows; i++)
            for (j = 0; j < ca3D->columns; j++)
                for (k = 0; k < ca3D->slices; k++)
                    if (calGetBuffer3DElement(ca3D->A->flags, ca3D->rows, ca3D->columns, i, j, k))
                    {
                        tcells[tn][tsize[tn]].i = i;
                        tcells[tn][tsize[tn]].j = j;
                        tcells[tn][tsize[tn]].k = k;
                        tsize[tn]++;
                    }

#pragma omp single
        {
            n = 0;
            for (i = 0; i < ca3D->A->num_threads; i++) {
                memcpy(&ca3D->A->cells[n],
                    tcells[i], sizeof(struct CALCell3D) * tsize[i]);
                n += tsize[i];
                //printf("tsize[i] = %d\n", tsize[i]);
            }
        }
    }
    for (i = 0; i < ca3D->A->num_threads; i++) 
        free(tcells[i]);
    free(tcells);
    free(tsize);

    // printf("End Update \n");            
    // printf("(mini, maxi) = (%d,%d) \n", ca3D->mini, ca3D->maxi);
    // printf("(minj, maxj) = (%d,%d) \n", ca3D->minj, ca3D->maxj);
    // printf("(mink, maxk) = (%d,%d) \n", ca3D->mink, ca3D->maxk);

}


void calFreeActiveCellsNaive3D(struct CALActiveCells3D* activeCells)
{
    free( activeCells->cells );
    free( activeCells->flags );
    free( activeCells->size_next );
}

void calSetActiveCellsNaiveBuffer3Db(CALbyte* M, CALbyte value, CALModel3D* ca3D)
{
    int n;
#pragma omp parallel for private( n ) firstprivate( M, value, ca3D )
    for(n=0; n<ca3D->A->size_current; n++)
        M[ca3D->A->cells[n].k * ca3D->rows * ca3D->columns + ca3D->A->cells[n].i * ca3D->columns + ca3D->A->cells[n].j] = value;
}

void calSetActiveCellsNaiveBuffer3Di(CALint* M, CALint value, CALModel3D* ca3D)
{
    int n;

#pragma omp parallel for private( n ) firstprivate( M, value, ca3D )
    for(n=0; n<ca3D->A->size_current; n++)
        M[ca3D->A->cells[n].k * ca3D->rows * ca3D->columns + ca3D->A->cells[n].i * ca3D->columns + ca3D->A->cells[n].j] = value;
}

void calSetActiveCellsNaiveBuffer3Dr(CALreal* M, CALreal value, CALModel3D* ca3D)
{
    int n;

#pragma omp parallel for private( n ) firstprivate( M, value, ca3D )
    for(n=0; n<ca3D->A->size_current; n++)
        M[ca3D->A->cells[n].k * ca3D->rows * ca3D->columns + ca3D->A->cells[n].i * ca3D->columns + ca3D->A->cells[n].j] = value;
}

void calCopyBufferActiveCellsNaive3Db(CALbyte* M_src, CALbyte* M_dest, CALModel3D* ca3D)
{
    int c, n;

#pragma omp parallel for private( c, n ) firstprivate( M_src, M_dest, ca3D )
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

#pragma omp parallel for private( c, n ) firstprivate( M_src, M_dest, ca3D )
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

#pragma omp parallel for private( c, n ) firstprivate( M_src, M_dest, ca3D )
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
#pragma omp parallel for private( n ) firstprivate( ca3D )
    for (n=0; n<ca3D->A->size_current; n++)
    {   
       elementary_process(ca3D, ca3D->A->cells[n].i, ca3D->A->cells[n].j, ca3D->A->cells[n].k);
    }
       
}
