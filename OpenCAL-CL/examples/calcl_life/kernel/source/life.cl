#ifndef __OPENCL_VERSION__
#define __kernel
#define __global
#define __local
#define get_global_id (int)
#endif

#include <life.h>

__kernel void life_transition_function(__CALCL_MODEL_2D) {

	initThreads2D();

	CALint cols_ = get_columns();
	CALint rows_ = get_rows();

	int i = getRow();
	int j = getCol();
 	CALint sizeOfX_ = get_neighborhoods_size();
  	int sum = 0, n;
	
	for (n=1; n<sizeOfX_; n++)
		sum += calGetX2Di(MODEL_2D, Q, i, j, n);

	if ((sum == 3) || (sum == 2 && calGet2Di(MODEL_2D, Q, i, j) == 1))
		calSet2Di(MODEL_2D, Q, i, j, 1);
	else
		calSet2Di(MODEL_2D, Q, i, j, 0);

}
