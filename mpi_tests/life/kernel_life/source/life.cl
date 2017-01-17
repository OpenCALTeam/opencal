// Conway's game of Life transition function kernel

#include <OpenCAL-CL/calcl2D.h>
 
#define DEVICE_Q 0

__kernel void lifeTransitionFunction(__CALCL_MODEL_2D)
{

	calclThreadCheck2D();
    int i = calclGlobalRow()+borderSize;   
	int j = calclGlobalColumn();

/*
         printf(" i: %d -- j: %d \n", i,j);

	
        if(i == 2 && j == 0 ){
            printf("stampo matrice\n");

            for (int i = 0; i < calclGetRows(); ++i) {
                for (int j = 0; j < calclGetColumns(); ++j) {

                    printf(" %d ", calclGet2Di(MODEL_2D, DEVICE_Q, i, j));

                }
                printf("\n");
            }

            printf("\n");
            printf("\n");
            printf("\n");
            printf("\n");

        }
*/
 	CALint sizeOfX_ = calclGetNeighborhoodSize();

	int sum = 0, n;

        for (n=1; n<sizeOfX_; n++){
			sum += calclGetX2Di(MODEL_2D, DEVICE_Q, i, j, n);

        }

        if ((sum == 3) || (sum == 2 && calclGet2Di(MODEL_2D, DEVICE_Q, i, j) == 1)){
            //printf("rows = %d       i = %d   j = %d    sum = %d \n",calclGetRows(), i ,j , sum);
			calclSet2Di(MODEL_2D, DEVICE_Q, i, j, 1);
        }
	else
		calclSet2Di(MODEL_2D, DEVICE_Q, i, j, 0);

//        if(calclGet2Di(MODEL_2D, DEVICE_Q, i, j) == 1){
//            printf("i = %d   j = %d \n", i ,j);
//        }

}
