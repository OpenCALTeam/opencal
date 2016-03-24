#include <OpenCAL-CL/calcl3D.h>

#define Q 0

__kernel void mod2TransitionFunction(__CALCL_MODEL_3D)
{
	calclThreadCheck3D();

	int i = calclGlobalRow();
	int j = calclGlobalColumns();
	int k = calclGlobalSlice();

	int sum = 0, n;
	CALint sizeOf_X = calclGetNeighborhoodSize();

	for (n=0; n<sizeOf_X; n++)
		sum += calclGetX3Db(MODEL_3D, Q, i, j, k, n);

	calclSet3Db(MODEL_3D, Q, i, j, k, sum%2);
}
