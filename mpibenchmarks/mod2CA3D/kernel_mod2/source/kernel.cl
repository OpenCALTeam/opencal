#include <OpenCAL-CL/calcl3D.h>

#define Q 0

__kernel void mod2TransitionFunction(__CALCL_MODEL_3D) {

    //calclThreadCheck3D();
    //printf(" r = %d, c = %d, s = %d  \n", calclGetRows(), calclGetColumns(), calclGetSlices());
    if( (calclGlobalSlice() < 1) || (calclGlobalSlice() >= calclGetSlices()-1) || get_global_id(0)>=calclGetRows() || get_global_id(1)>=calclGetColumns()|| get_global_id(2)>=calclGetSlices()) return;

    int i = calclGlobalRow();
    int j = calclGlobalColumn();
    int k = calclGlobalSlice();
//    if(calclGetSlices() == 5)
//        printf("%d, %d, %d  \n", i, j, k);
    int sum = 0, n;
    CALint sizeOf_X = calclGetNeighborhoodSize();

//    if(i == 1 && j ==1 && k ==1){
//        printf("calclGetSlices() %d\n", calclGetSlices());
//        if(calclGetSlices() == 5){
//        for (int kk = 1; kk < calclGetSlices()-1; ++kk) {
//            for (int ii = 0; ii < 5; ++ii) {
//                for (int jj = 0; jj < 5; ++jj) {
//                    printf("%d ", calclGet3Di(MODEL_3D, Q, ii, jj,kk+1));
//                }
//                printf("\n");
//            }
//            printf("\n\n");

//        }
//        printf(" finish \n\n");
//        }

//    }

    for (n=0; n<sizeOf_X; n++)
        sum += calclGetX3Di(MODEL_3D, Q, i, j, k, n);

//    if(calclGet3Di(MODEL_3D, Q, i, j,k) > 0 && calclGetSlices() == 5){
//        printf("%d, %d, %d -- sum = %d , %d - %d\n", i, j, k, sum,calclGet3Di(MODEL_3D, Q, i, j,k),   calclGetX3Di(MODEL_3D, Q, i, j, k, 0));
//        int ti = calclGetToroidalX(i + calclGetNeighborhood()[n].i, calclGetRows());
//        int tj = calclGetToroidalX(j + calclGetNeighborhood()[n].j, calclGetColumns());
//        int tk = calclGetToroidalX(k + calclGetNeighborhood()[n].k, calclGetSlices());
//        printf("toro ==> %d, %d, %d \n", ti, tj, tk);
////        for (n=0; n<sizeOf_X; n++)
////            printf("%d ==> %d \n", n, calclGetX3Di(MODEL_3D, Q, i, j, k, n));
//    }
    calclSet3Di(MODEL_3D, Q, i, j, k, sum%2);


}
