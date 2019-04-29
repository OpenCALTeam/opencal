#include <OpenCAL-CL/calcl3D.h>

#define Q 0

__kernel void mod2TransitionFunction(__CALCL_MODEL_3D) {
    calclActiveThreadCheck3D();
    int threadID = calclGlobalRow();
    int i = calclGetActiveCells()[threadID].i;//calclActiveCellRow(threadID);
    int j = calclGetActiveCells()[threadID].j;// calclActiveCellColumn(threadID);
    int k = calclGetActiveCells()[threadID].k;//getActiveCellSlice(threadID);

    int sum = 0, n;
    CALint sizeOf_X = calclGetNeighborhoodSize();

    for (n=0; n<sizeOf_X; n++)
        sum += calclGetX3Di(MODEL_3D, Q, i, j, k, n);

    calclSet3Di(MODEL_3D, Q, i, j, k, sum%2);
    if(sum%2==1){
        calclAddActiveCell3D(MODEL_3D, i, j, k);
        printf("%d, %d, %d, %d  \n",borderSize, i, j, k);
    }
}
