#ifndef __OPENCL_VERSION__
#define __kernel
#define __global
#define __local
#define get_global_id (int)
#endif

#define Q 0
#include <cal3D.h>

__kernel void mod2_transition_function(MODEL_DEFINITION3D) {

	initThreads3D();


	int i = getRow();
	int j = getCol();
	int k = getSlice();

	int sum = 0, n;
	CALint sizeOf_X = get_neighborhoods_size();

	for (n=0; n<sizeOf_X; n++)
		sum += calGetX3Db(MODEL3D, Q, i, j, k, n);

	calSet3Db(MODEL3D, Q, i, j, k, sum%2);

}
