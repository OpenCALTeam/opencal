#ifndef __OPENCL_VERSION__
#define __kernel
#define __global
#define __local
#define get_global_id (int)
#endif

//#include <life.h>
#define Q 0
#include <OpenCAL-CL/calcl2D.h>


__kernel void life_transition_function(__CALCL_MODEL_2D) {

	calclThreadCheck2D();

	CALint cols_ = calclGetColumns();
	CALint rows_ = calclGetRows();

	int i = calclGlobalRow();
	int j = calclGlobalColumns();
 	CALint sizeOfX_ = calclGetNeighborhoodSize();
  	int sum = 0, n;

	for (n=1; n<sizeOfX_; n++)
		sum += calclGetX2Di(MODEL_2D, Q, i, j, n);

	if ((sum == 3) || (sum == 2 && calclGet2Di(MODEL_2D, Q, i, j) == 1))
		calclSet2Di(MODEL_2D, Q, i, j, 1);
	else
		calclSet2Di(MODEL_2D, Q, i, j, 0);

}
