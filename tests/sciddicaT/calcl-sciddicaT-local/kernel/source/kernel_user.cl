// The SciddicaT debris flows XCA transition function kernels

#include <kernel.h>

#define MAX_GROUP_WORK_SIZE 7
#define SHIFT 1
#define NUMBER_OF_LOOPS 200

__kernel void flowsComputation(__CALCL_MODEL_2D, __global CALParameterr * Pepsilon, __global CALParameterr * Pr)
{
    calclThreadCheck2D();

    int iGlobal = calclGlobalRow();
    int jGlobal = calclGlobalColumn();

    int iLocal = calclLocalRow();
    int jLocal = calclLocalColumn();

    int localRow = get_local_id(0);
    int localCol = get_local_id(1);

    //    __local double localMatrixH[(MAX_GROUP_WORK_SIZE+1)*(MAX_GROUP_WORK_SIZE+1)];//[MAX_GROUP_WORK_SIZE+1];
    //    __local double localMatrixZ[(MAX_GROUP_WORK_SIZE+1)*(MAX_GROUP_WORK_SIZE+1)];//[MAX_GROUP_WORK_SIZE+1];
    __local double localMatrixH[(MAX_GROUP_WORK_SIZE+1)][MAX_GROUP_WORK_SIZE+1];
    __local double localMatrixZ[(MAX_GROUP_WORK_SIZE+1)][MAX_GROUP_WORK_SIZE+1];

    localMatrixZ[iLocal+SHIFT][jLocal+SHIFT] = calclGet2Dr(MODEL_2D,Z, iGlobal, jGlobal);
    localMatrixH[iLocal+SHIFT][jLocal+SHIFT] = calclGet2Dr(MODEL_2D,H, iGlobal, jGlobal);

    localMatrixZ[iLocal+SHIFT-1][jLocal+SHIFT] = calclGetX2Dr(MODEL_2D,Z, iGlobal, jGlobal,1);
    localMatrixH[iLocal+SHIFT-1][jLocal+SHIFT] = calclGetX2Dr(MODEL_2D,H, iGlobal, jGlobal,1);

    //4
    //    localMatrixZ[(iLocal+SHIFT+1)*(MAX_GROUP_WORK_SIZE+1)+jLocal+SHIFT] = calclGetX2Dr(MODEL_2D,Z, iGlobal, jGlobal,4);
    //    localMatrixH[(iLocal+SHIFT+1)*(MAX_GROUP_WORK_SIZE+1)+jLocal+SHIFT] = calclGetX2Dr(MODEL_2D,H, iGlobal, jGlobal,4);
    localMatrixZ[iLocal+SHIFT+1][jLocal+SHIFT] = calclGetX2Dr(MODEL_2D,Z, iGlobal, jGlobal,4);
    localMatrixH[iLocal+SHIFT+1][jLocal+SHIFT] = calclGetX2Dr(MODEL_2D,H, iGlobal, jGlobal,4);

    //2
    //    localMatrixZ[iLocal+SHIFT*(MAX_GROUP_WORK_SIZE+1)+jLocal+SHIFT-1] = calclGetX2Dr(MODEL_2D,Z, iGlobal, jGlobal,2);
    //    localMatrixH[iLocal+SHIFT*(MAX_GROUP_WORK_SIZE+1)+jLocal+SHIFT-1] = calclGetX2Dr(MODEL_2D,H, iGlobal, jGlobal,2);
    localMatrixZ[iLocal+SHIFT][jLocal+SHIFT-1] = calclGetX2Dr(MODEL_2D,Z, iGlobal, jGlobal,2);
    localMatrixH[iLocal+SHIFT][jLocal+SHIFT-1] = calclGetX2Dr(MODEL_2D,H, iGlobal, jGlobal,2);

    //3
    //    localMatrixZ[iLocal+SHIFT*(MAX_GROUP_WORK_SIZE+1)+jLocal+SHIFT+1] = calclGetX2Dr(MODEL_2D,Z, iGlobal, jGlobal,3);
    //    localMatrixH[iLocal+SHIFT*(MAX_GROUP_WORK_SIZE+1)+jLocal+SHIFT+1] = calclGetX2Dr(MODEL_2D,H, iGlobal, jGlobal,3);
    localMatrixZ[iLocal+SHIFT][jLocal+SHIFT+1] = calclGetX2Dr(MODEL_2D,Z, iGlobal, jGlobal,3);
    localMatrixH[iLocal+SHIFT][jLocal+SHIFT+1] = calclGetX2Dr(MODEL_2D,H, iGlobal, jGlobal,3);

    //            if(iGlobal == 0 && jGlobal ==  0){
    //                printf("%f - \n",localMatrixH[iLocal][jLocal]);
    //            }

    // barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

    //inizializza localMatrix Ogni Thread leggi il proprio valore


    CALbyte eliminated_cells[5] = { CAL_FALSE, CAL_FALSE, CAL_FALSE, CAL_FALSE, CAL_FALSE };
    CALbyte again;
    CALint cells_count;
    CALreal average;
    CALreal m;
    CALreal u[5];
    CALint n;
    CALreal z, h;

    CALint sizeOfX_ = calclGetNeighborhoodSize();
    CALParameterr eps = *Pepsilon;

    for(int k = 0; k < NUMBER_OF_LOOPS; k++){
        if (localMatrixH[iLocal+SHIFT][jLocal+SHIFT] <= eps)
            return;


        //        m = localMatrixH[(iLocal+SHIFT)*(MAX_GROUP_WORK_SIZE+1)+(jLocal+SHIFT)] - eps;
        //        u[0] = localMatrixZ[(iLocal+SHIFT)*(MAX_GROUP_WORK_SIZE+1)+(jLocal+SHIFT)] + eps;
        m = localMatrixH[iLocal+SHIFT][jLocal+SHIFT] - eps;
        u[0] = localMatrixZ[iLocal+SHIFT][jLocal+SHIFT] + eps;

        //        z = localMatrixZ[(iLocal-1+SHIFT)*(MAX_GROUP_WORK_SIZE+1)+(jLocal+SHIFT)]; //calclGetX2Dr(MODEL_2D,Z, i, j, n);
        //        h = localMatrixH[(iLocal-1+SHIFT)*(MAX_GROUP_WORK_SIZE+1)+(jLocal+SHIFT)]; //calclGetX2Dr(MODEL_2D,H, i, j, n);
        z = localMatrixZ[iLocal-1+SHIFT][jLocal+SHIFT]; //calclGetX2Dr(MODEL_2D,Z, i, j, n);
        h = localMatrixH[iLocal-1+SHIFT][jLocal+SHIFT]; //calclGetX2Dr(MODEL_2D,H, i, j, n);
        u[1] = z + h;

        //        z = localMatrixZ[iLocal+SHIFT*(MAX_GROUP_WORK_SIZE+1)+(jLocal-1+SHIFT)];
        //        h = localMatrixH[iLocal+SHIFT*(MAX_GROUP_WORK_SIZE+1)+(jLocal-1+SHIFT)];
        z = localMatrixZ[iLocal+SHIFT][jLocal-1+SHIFT];
        h = localMatrixH[iLocal+SHIFT][jLocal-1+SHIFT];
        u[2] = z + h;

        //        z = localMatrixZ[iLocal+SHIFT*(MAX_GROUP_WORK_SIZE+1)+(jLocal+SHIFT+1)];
        //        h = localMatrixH[iLocal+SHIFT*(MAX_GROUP_WORK_SIZE+1)+(jLocal+1+SHIFT)];
        z = localMatrixZ[iLocal+SHIFT][jLocal+SHIFT+1];
        h = localMatrixH[iLocal+SHIFT][jLocal+1+SHIFT];
        u[3] = z + h;

        //        z = localMatrixZ[(iLocal+SHIFT+1)*(MAX_GROUP_WORK_SIZE+1)+(jLocal+SHIFT)];
        //        h = localMatrixH[(iLocal+SHIFT+1)*(MAX_GROUP_WORK_SIZE+1)+(jLocal+SHIFT)];
        z = localMatrixZ[iLocal+SHIFT+1][jLocal+SHIFT];
        h = localMatrixH[iLocal+SHIFT+1][jLocal+SHIFT];
        u[4] = z + h;


        do {
            again = CAL_FALSE;
            average = m;
            cells_count = 0;

            for (n = 0; n < sizeOfX_; n++)
                if (!eliminated_cells[n]) {
                    average += u[n];
                    cells_count++;
                }

            if (cells_count != 0)
                average /= cells_count;

            for (n = 0; n < sizeOfX_; n++)
                if ((average <= u[n]) && (!eliminated_cells[n])) {
                    eliminated_cells[n] = CAL_TRUE;
                    again = CAL_TRUE;
                }

        } while (again);

        for (n = 1; n < sizeOfX_; n++) {
            if (eliminated_cells[n])
                calclSet2Dr(MODEL_2D, n-1, iGlobal, jGlobal, 0.0);
            else
                calclSet2Dr(MODEL_2D, n-1, iGlobal, jGlobal,(average - u[n]) * (*Pr));
        }
    }
}

__kernel void widthUpdate(__CALCL_MODEL_2D)
{
    calclThreadCheck2D();

    int iGlobal = calclGlobalRow();
    int jGlobal = calclGlobalColumn();

    int iLocal = calclLocalRow();
    int jLocal = calclLocalColumn();

    CALreal h_next;
    CALint n;

    __local double localMatrixIN[MAX_GROUP_WORK_SIZE+1][MAX_GROUP_WORK_SIZE+1];
    __local double localMatrixOUT[MAX_GROUP_WORK_SIZE+1][MAX_GROUP_WORK_SIZE+1];

    localMatrixIN[iLocal+SHIFT][jLocal+SHIFT] = calclGet2Dr(MODEL_2D,H, iGlobal, jGlobal);

    //    if(iGlobal == 0 && jGlobal ==  0){
    //        printf("%f - \n",localMatrixH[iLocal][jLocal]);
    //    }
    localMatrixIN[iLocal+SHIFT-1][jLocal+SHIFT] = calclGetX2Dr(MODEL_2D, NUMBER_OF_OUTFLOWS-1, iGlobal, jGlobal, 1);

    localMatrixIN[iLocal+SHIFT+1][jLocal+SHIFT] = calclGetX2Dr(MODEL_2D, NUMBER_OF_OUTFLOWS-4, iGlobal, jGlobal, 4);

    localMatrixIN[iLocal+SHIFT][jLocal+SHIFT-1] = calclGetX2Dr(MODEL_2D, NUMBER_OF_OUTFLOWS-2, iGlobal, jGlobal, 2);

    localMatrixIN[iLocal+SHIFT][jLocal+SHIFT+1] = calclGetX2Dr(MODEL_2D, NUMBER_OF_OUTFLOWS-3, iGlobal, jGlobal, 3);


    localMatrixOUT[iLocal+SHIFT-1][jLocal+SHIFT] =  calclGet2Dr(MODEL_2D, 0, iGlobal, jGlobal);

    localMatrixOUT[iLocal+SHIFT+1][jLocal+SHIFT] =  calclGet2Dr(MODEL_2D, 3, iGlobal, jGlobal);

    localMatrixOUT[iLocal+SHIFT][jLocal+SHIFT-1] =  calclGet2Dr(MODEL_2D, 1, iGlobal, jGlobal);

    localMatrixOUT[iLocal+SHIFT][jLocal+SHIFT+1] =  calclGet2Dr(MODEL_2D, 2, iGlobal, jGlobal);

    for(int k =0; k < NUMBER_OF_LOOPS; k ++){
        h_next = localMatrixIN[iLocal+SHIFT][jLocal+SHIFT];

        h_next +=localMatrixIN[iLocal+SHIFT-1][jLocal+SHIFT] -localMatrixOUT[iLocal+SHIFT-1][jLocal+SHIFT];

        h_next +=localMatrixIN[iLocal+SHIFT+1][jLocal+SHIFT] -localMatrixOUT[iLocal+SHIFT+1][jLocal+SHIFT];

        h_next +=localMatrixIN[iLocal+SHIFT][jLocal+SHIFT-1] -localMatrixOUT[iLocal+SHIFT][jLocal+SHIFT-1];

        h_next +=localMatrixIN[iLocal+SHIFT][jLocal+SHIFT+1] -localMatrixOUT[iLocal+SHIFT][jLocal+SHIFT+1];

        calclSet2Dr(MODEL_2D, H, iGlobal, jGlobal, h_next);
    }

}

__kernel void steering(__CALCL_MODEL_2D)
{
    calclThreadCheck2D();

    CALint cols_ = calclGetColumns();
    CALint rows_ = calclGetRows();

    int i = calclGlobalRow();
    int j = calclGlobalColumn();
    int s;
    //	for (s = 0; s < NUMBER_OF_OUTFLOWS; ++s)
    calclInitSubstate2Dr(MODEL_2D, 0, i, j, 0);
    calclInitSubstate2Dr(MODEL_2D, 1, i, j, 0);
    calclInitSubstate2Dr(MODEL_2D, 2, i, j, 0);
    calclInitSubstate2Dr(MODEL_2D, 3, i, j, 0);
}
