#include <OpenCAL-CL/calcl2D.h>

__kernel void calclMinReductionKernelb(__global double *  minima, CALint substates,  __global CALbyte * offsets,unsigned offset ,int size,int count){

	int id = get_global_id (0);
	unsigned i = offset * ((id << 1) + 1) - 1,
             j = i + offset;
            if(count > 0){
             	i = i - (count);
            	j = j - (count);
             }
	if((i >= size && j >= size) || (i < size && j >= size))
		return;
//	printf("offsets[%d] : %f \n",i,offsets[i] );
//	printf("offsets[%d] : %f \n",j,offsets[j] );

	if(offsets[i] > offsets[j])
		offsets[i] = offsets[j];

	if(id == 0)
	{
	  if((get_global_size (0) /2) <= 0)
	  {
		minima[substates] = offsets[i];
	  }
	}

}

__kernel void calclMaxReductionKernelb(__global double *  maxima, CALint substates,  __global CALbyte * offsets,unsigned offset ,int size,int count){

	int id = get_global_id (0);
	unsigned i = offset * ((id << 1) + 1) - 1,
             j = i + offset;
            if(count > 0){
             	i = i - (count);
            	j = j - (count);
             }
	if((i >= size && j >= size) || (i < size && j >= size))
		return;
	printf("offsets[%d] : %d \n",i,offsets[i] );
	printf("offsets[%d] : %d \n",j,offsets[j] );

	if(offsets[i] < offsets[j])
		offsets[i] = offsets[j];

	if(id == 0)
	{
	  if((get_global_size (0) /2) <= 0)
	  {
		maxima[substates] = offsets[i];
	  }
	}

}

__kernel void calclSumReductionKernelb(__global double *  sum, CALint substates,  __global CALbyte * offsets,unsigned offset ,int size,int count){

	int id = get_global_id (0);
	unsigned i = offset * ((id << 1) + 1) - 1,
             j = i + offset;
            if(count > 0){
             	i = i - (count);
            	j = j - (count);
             }
	if((i >= size && j >= size) || (i < size && j >= size))
		return;
		offsets[i] += offsets[j];

	if(id == 0)
	{
	  if((get_global_size (0) /2) <= 0)
	  {
		sum[substates] = offsets[i];
	  }
	}

}

__kernel void calclProdReductionKernelb(__global double *  result, CALint substates,  __global CALbyte * offsets,unsigned offset ,int size,int count){

	int id = get_global_id (0);
	unsigned i = offset * ((id << 1) + 1) - 1,
             j = i + offset;
            if(count > 0){
             	i = i - (count);
            	j = j - (count);
             }
	if((i >= size && j >= size) || (i < size && j >= size))
		return;

	offsets[i] *= offsets[j];

	if(id == 0)
	{
	  if((get_global_size (0) /2) <= 0)
	  {
		result[substates] = offsets[i];
	  }
	}

}

__kernel void calclLogicAndReductionKernelb(__global double *  result, CALint substates,  __global CALbyte * offsets,unsigned offset ,int size,int count){

	int id = get_global_id (0);
	unsigned i = offset * ((id << 1) + 1) - 1,
             j = i + offset;
            if(count > 0){
             	i = i - (count);
            	j = j - (count);
             }
	if((i >= size && j >= size) || (i < size && j >= size))
		return;

	offsets[i] = (offsets[i] && offsets[j]);

	if(id == 0)
	{
	  if((get_global_size (0) /2) <= 0)
	  {
		result[substates] = offsets[i];
	  }
	}

}

__kernel void calclLogicOrReductionKernelb(__global double *  result, CALint substates,  __global CALbyte * offsets,unsigned offset ,int size,int count){

	int id = get_global_id (0);
	unsigned i = offset * ((id << 1) + 1) - 1,
             j = i + offset;
            if(count > 0){
             	i = i - (count);
            	j = j - (count);
             }
	if((i >= size && j >= size) || (i < size && j >= size))
		return;

	offsets[i] = (offsets[i] || offsets[j]);

	if(id == 0)
	{
	  if((get_global_size (0) /2) <= 0)
	  {
		result[substates] = offsets[i];
	  }
	}

}

__kernel void calclLogicXOrReductionKernelb(__global double *  result, CALint substates,  __global CALbyte * offsets,unsigned offset ,int size,int count){

	int id = get_global_id (0);
	unsigned i = offset * ((id << 1) + 1) - 1,
             j = i + offset;
            if(count > 0){
             	i = i - (count);
            	j = j - (count);
             }
	if((i >= size && j >= size) || (i < size && j >= size))
		return;

	offsets[i] = (!!offsets[i] != !!offsets[j]);

	if(id == 0)
	{
	  if((get_global_size (0) /2) <= 0)
	  {
		result[substates] = offsets[i];
	  }
	}

}

__kernel void calclBinaryAndReductionKernelb(__global double *  result, CALint substates,  __global CALbyte * offsets,unsigned offset ,int size,int count){

	int id = get_global_id (0);
	unsigned i = offset * ((id << 1) + 1) - 1,
             j = i + offset;
            if(count > 0){
             	i = i - (count);
            	j = j - (count);
             }
	if((i >= size && j >= size) || (i < size && j >= size))
		return;

	offsets[i] = (offsets[i] & offsets[j]);

	if(id == 0)
	{
	  if((get_global_size (0) /2) <= 0)
	  {
		result[substates] = offsets[i];
	  }
	}

}

__kernel void calclBinaryOrReductionKernelb(__global double *  result, CALint substates,  __global CALbyte * offsets,unsigned offset ,int size,int count){

	int id = get_global_id (0);
	unsigned i = offset * ((id << 1) + 1) - 1,
             j = i + offset;
            if(count > 0){
             	i = i - (count);
            	j = j - (count);
             }
	if((i >= size && j >= size) || (i < size && j >= size))
		return;

	offsets[i] = (offsets[i] | offsets[j]);

	if(id == 0)
	{
	  if((get_global_size (0) /2) <= 0)
	  {
		result[substates] = offsets[i];
	  }
	}

}

__kernel void calclBinaryXOrReductionKernelb(__global double *  result, CALint substates,  __global CALbyte * offsets,unsigned offset ,int size,int count){

	int id = get_global_id (0);
	unsigned i = offset * ((id << 1) + 1) - 1,
             j = i + offset;
            if(count > 0){
             	i = i - (count);
            	j = j - (count);
             }
	if((i >= size && j >= size) || (i < size && j >= size))
		return;

	offsets[i] = (offsets[i] ^ offsets[j]);

	if(id == 0)
	{
	  if((get_global_size (0) /2) <= 0)
	  {
		result[substates] = offsets[i];
	  }
	}

}





__kernel void calclMinReductionKerneli(__global double *  minima, CALint substates,  __global CALint * offsets,unsigned offset ,int size,int count){

	int id = get_global_id (0);
	unsigned i = offset * ((id << 1) + 1) - 1,
             j = i + offset;
            if(count > 0){
             	i = i - (count);
            	j = j - (count);
             }
	if((i >= size && j >= size) || (i < size && j >= size))
		return;
//	printf("offsets[%d] : %f \n",i,offsets[i] );
//	printf("offsets[%d] : %f \n",j,offsets[j] );

	if(offsets[i] > offsets[j])
		offsets[i] = offsets[j];

	if(id == 0)
	{
	  if((get_global_size (0) /2) <= 0)
	  {
		minima[substates] = offsets[i];
	  }
	}

}

__kernel void calclMaxReductionKerneli(__global double *  maxima, CALint substates,  __global CALint * offsets,unsigned offset ,int size,int count){

	int id = get_global_id (0);
	unsigned i = offset * ((id << 1) + 1) - 1,
             j = i + offset;
            if(count > 0){
             	i = i - (count);
            	j = j - (count);
             }
	if((i >= size && j >= size) || (i < size && j >= size))
		return;
	printf("offsets[%d] : %d \n",i,offsets[i] );
	printf("offsets[%d] : %d \n",j,offsets[j] );

	if(offsets[i] < offsets[j])
		offsets[i] = offsets[j];


	if(id == 0)
	{
	  if((get_global_size (0) /2) <= 0)
	  {
			maxima[substates] = offsets[i];
			printf("FInal result %d : %f , %d \n",i,maxima[substates],substates );
	  }
	}

}

__kernel void calclSumReductionKerneli(__global double *  sum, CALint substates,  __global CALint * offsets,unsigned offset ,int size,int count){

	int id = get_global_id (0);
	unsigned i = offset * ((id << 1) + 1) - 1,
             j = i + offset;
            if(count > 0){
             	i = i - (count);
            	j = j - (count);
             }
	if((i >= size && j >= size) || (i < size && j >= size))
		return;

	//printf("offsets[%d] + offsets[%d]  	 \n", i, j );

	offsets[i] += offsets[j];


	if(id == 0)
	{
	  if((get_global_size (0) /2) <= 0)
	  {
		sum[substates] = offsets[i];
		//printf("Sum -> %f \n", sum[substates]);
	  }
	}

}

__kernel void calclProdReductionKerneli(__global double *  result, CALint substates,  __global CALint * offsets,unsigned offset ,int size,int count){

	int id = get_global_id (0);
	unsigned i = offset * ((id << 1) + 1) - 1,
             j = i + offset;
            if(count > 0){
             	i = i - (count);
            	j = j - (count);
             }
	if((i >= size && j >= size) || (i < size && j >= size))
		return;

	offsets[i] *= offsets[j];

	if(id == 0)
	{
	  if((get_global_size (0) /2) <= 0)
	  {
		result[substates] = offsets[i];
	  }
	}

}

__kernel void calclLogicAndReductionKerneli(__global double *  result, CALint substates,  __global CALint * offsets,unsigned offset ,int size,int count){

	int id = get_global_id (0);
	unsigned i = offset * ((id << 1) + 1) - 1,
             j = i + offset;
            if(count > 0){
             	i = i - (count);
            	j = j - (count);
             }
	if((i >= size && j >= size) || (i < size && j >= size))
		return;

	offsets[i] = (offsets[i] && offsets[j]);

	if(id == 0)
	{
	  if((get_global_size (0) /2) <= 0)
	  {
		result[substates] = offsets[i];
	  }
	}

}

__kernel void calclLogicOrReductionKerneli(__global double *  result, CALint substates,  __global CALint * offsets,unsigned offset ,int size,int count){

	int id = get_global_id (0);
	unsigned i = offset * ((id << 1) + 1) - 1,
             j = i + offset;
            if(count > 0){
             	i = i - (count);
            	j = j - (count);
             }
	if((i >= size && j >= size) || (i < size && j >= size))
		return;

	offsets[i] = (offsets[i] || offsets[j]);

	if(id == 0)
	{
	  if((get_global_size (0) /2) <= 0)
	  {
		result[substates] = offsets[i];
	  }
	}

}

__kernel void calclLogicXOrReductionKerneli(__global double *  result, CALint substates,  __global CALint * offsets,unsigned offset ,int size,int count){

	int id = get_global_id (0);
	unsigned i = offset * ((id << 1) + 1) - 1,
             j = i + offset;
            if(count > 0){
             	i = i - (count);
            	j = j - (count);
             }
	if((i >= size && j >= size) || (i < size && j >= size))
		return;

	offsets[i] = (!!offsets[i] != !!offsets[j]);

	if(id == 0)
	{
	  if((get_global_size (0) /2) <= 0)
	  {
		result[substates] = offsets[i];
	  }
	}

}

__kernel void calclBinaryAndReductionKerneli(__global double *  result, CALint substates,  __global CALint * offsets,unsigned offset ,int size,int count){

	int id = get_global_id (0);
	unsigned i = offset * ((id << 1) + 1) - 1,
             j = i + offset;
            if(count > 0){
             	i = i - (count);
            	j = j - (count);
             }
	if((i >= size && j >= size) || (i < size && j >= size))
		return;

	offsets[i] = (offsets[i] & offsets[j]);

	if(id == 0)
	{
	  if((get_global_size (0) /2) <= 0)
	  {
		result[substates] = offsets[i];
	  }
	}

}

__kernel void calclBinaryOrReductionKerneli(__global double *  result, CALint substates,  __global CALint * offsets,unsigned offset ,int size,int count){

	int id = get_global_id (0);
	unsigned i = offset * ((id << 1) + 1) - 1,
             j = i + offset;
            if(count > 0){
             	i = i - (count);
            	j = j - (count);
             }
	if((i >= size && j >= size) || (i < size && j >= size))
		return;

	offsets[i] = (offsets[i] | offsets[j]);

	if(id == 0)
	{
	  if((get_global_size (0) /2) <= 0)
	  {
		result[substates] = offsets[i];
	  }
	}

}

__kernel void calclBinaryXOrReductionKerneli(__global double *  result, CALint substates,  __global CALint * offsets,unsigned offset ,int size,int count){

	int id = get_global_id (0);
	unsigned i = offset * ((id << 1) + 1) - 1,
             j = i + offset;
            if(count > 0){
             	i = i - (count);
            	j = j - (count);
             }
	if((i >= size && j >= size) || (i < size && j >= size))
		return;

	offsets[i] = (offsets[i] ^ offsets[j]);

	if(id == 0)
	{
	  if((get_global_size (0) /2) <= 0)
	  {
		result[substates] = offsets[i];
	  }
	}

}


__kernel void calclMinReductionKernelr(__global CALreal *  minima, CALint substates,  __global CALreal * offsets,unsigned offset ,int size,int count){

	int id = get_global_id (0);
	unsigned i = offset * ((id << 1) + 1) - 1,
             j = i + offset;
            if(count > 0){
             	i = i - (count);
            	j = j - (count);
             }

	if((i >= size && j >= size) || (i < size && j >= size))
		return;

/*	if(i < 200 && j < 200)
		printf("offsets[%d] : %f \n",i,offsets[i] );
		printf("offsets[%d] : %f \n",j,offsets[j] );
*/
	if(offsets[i] > offsets[j])
		offsets[i] = offsets[j];

	if(id == 0)
	{
	  if((get_global_size (0) /2) <= 0)
	  {
		minima[substates] = offsets[i];
	  }
	}

}

__kernel void calclMaxReductionKernelr(__global CALreal *  maxima, CALint substates,  __global CALreal * offsets,unsigned offset ,int size,int count){

	int id = get_global_id (0);
	unsigned i = offset * ((id << 1) + 1) - 1,
             j = i + offset;
            if(count > 0){
             	i = i - (count);
            	j = j - (count);
             }
	if((i >= size && j >= size) || (i < size && j >= size)){
			return;
	}
//	printf("offsets[%d] - size %d - offsets[%d]  \n",i, size,j );
//	printf("offsets[%d] : %f \n",i,offsets[i] );
//	printf("offsets[%d] : %f \n",j,offsets[j] );

//	if(offsets[i] < offsets[j])
//		offsets[i] = offsets[j];

	if(id == 0)
	{
	  if((get_global_size (0) /2) <= 0)
	  {
		maxima[substates] = offsets[i];
	  }
	}

}

__kernel void calclSumReductionKernelr(__global CALreal *  sum, CALint substates,  __global CALreal * offsets,unsigned offset ,int size,int count){

	int id = get_global_id (0);
	unsigned i = offset * ((id << 1) + 1) - 1,
             j = i + offset;
            if(count > 0){
             	i = i - (count);
            	j = j - (count);
             }
	if((i >= size && j >= size) || (i < size && j >= size))
		return;
		offsets[i] += offsets[j];
//	printf("offsets[%d] : %f \n",i,offsets[i] );
//	printf("offsets[%d] : %f \n",j,offsets[j] );

	if(id == 0)
	{
	  if((get_global_size (0) /2) <= 0)
	  {

		sum[substates] = offsets[i];

	  }
	}

}

__kernel void calclProdReductionKernelr(__global CALreal *  result, CALint substates,  __global CALreal * offsets,unsigned offset ,int size,int count){

	int id = get_global_id (0);
	unsigned i = offset * ((id << 1) + 1) - 1,
             j = i + offset;
            if(count > 0){
             	i = i - (count);
            	j = j - (count);
             }
	if((i >= size && j >= size) || (i < size && j >= size))
		return;

	offsets[i] *= offsets[j];

	if(id == 0)
	{
	  if((get_global_size (0) /2) <= 0)
	  {
		result[substates] = offsets[i];
	  }
	}

}

__kernel void calclLogicAndReductionKernelr(__global CALreal *  result, CALint substates,  __global CALreal * offsets,unsigned offset ,int size,int count){

	int id = get_global_id (0);
	unsigned i = offset * ((id << 1) + 1) - 1,
             j = i + offset;
            if(count > 0){
             	i = i - (count);
            	j = j - (count);
             }
	if((i >= size && j >= size) || (i < size && j >= size))
		return;

	offsets[i] = (offsets[i] && offsets[j]);

	if(id == 0)
	{
	  if((get_global_size (0) /2) <= 0)
	  {
		result[substates] = offsets[i];
	  }
	}

}

__kernel void calclLogicOrReductionKernelr(__global CALreal *  result, CALint substates,  __global CALreal * offsets,unsigned offset ,int size,int count){

	int id = get_global_id (0);
	unsigned i = offset * ((id << 1) + 1) - 1,
             j = i + offset;
            if(count > 0){
             	i = i - (count);
            	j = j - (count);
             }
	if((i >= size && j >= size) || (i < size && j >= size))
		return;

	offsets[i] = (offsets[i] || offsets[j]);

	if(id == 0)
	{
	  if((get_global_size (0) /2) <= 0)
	  {
		result[substates] = offsets[i];
	  }
	}

}

__kernel void calclLogicXOrReductionKernelr(__global CALreal *  result, CALint substates,  __global CALreal * offsets,unsigned offset ,int size,int count){

	int id = get_global_id (0);
	unsigned i = offset * ((id << 1) + 1) - 1,
             j = i + offset;
            if(count > 0){
             	i = i - (count);
            	j = j - (count);
             }
	if((i >= size && j >= size) || (i < size && j >= size))
		return;

	offsets[i] = (!!offsets[i] != !!offsets[j]);

	if(id == 0)
	{
	  if((get_global_size (0) /2) <= 0)
	  {
		result[substates] = offsets[i];
	  }
	}

}

__kernel void calclBinaryAndReductionKernelr(__global double *  result, CALint substates,  __global CALint * offsets,unsigned offset ,int size,int count){

	int id = get_global_id (0);
	unsigned i = offset * ((id << 1) + 1) - 1,
             j = i + offset;
            if(count > 0){
             	i = i - (count);
            	j = j - (count);
             }
	if((i >= size && j >= size) || (i < size && j >= size))
		return;

	offsets[i] = (offsets[i] & offsets[j]);

	if(id == 0)
	{
	  if((get_global_size (0) /2) <= 0)
	  {
		result[substates] = offsets[i];
	  }
	}

}

__kernel void calclBinaryOrReductionKernelr(__global double *  result, CALint substates,  __global CALint * offsets,unsigned offset ,int size,int count){

	int id = get_global_id (0);
	unsigned i = offset * ((id << 1) + 1) - 1,
             j = i + offset;
            if(count > 0){
             	i = i - (count);
            	j = j - (count);
             }
	if((i >= size && j >= size) || (i < size && j >= size))
		return;

	offsets[i] = (offsets[i] | offsets[j]);

	if(id == 0)
	{
	  if((get_global_size (0) /2) <= 0)
	  {
		result[substates] = offsets[i];
	  }
	}

}

__kernel void calclBinaryXOrReductionKernelr(__global double *  result, CALint substates,  __global CALint * offsets,unsigned offset ,int size,int count){

	int id = get_global_id (0);
	unsigned i = offset * ((id << 1) + 1) - 1,
             j = i + offset;
            if(count > 0){
             	i = i - (count);
            	j = j - (count);
             }
	if((i >= size && j >= size) || (i < size && j >= size))
		return;

	offsets[i] = (offsets[i] ^ offsets[j]);

	if(id == 0)
	{
	  if((get_global_size (0) /2) <= 0)
	  {
		result[substates] = offsets[i];
	  }
	}

}


__kernel void copy2Di(__global CALint *  end,__global CALint *  start, CALint substateNum, CALint rows, CALint columns){
		int id = get_global_id (0);
		//printf("*(start + rows * columns * substateNum + id) %d \n", start[id]);
		end[id] = *(start + rows * columns * substateNum + id);//  calclGetBufferElement((start + rows * columns * substateNum), columns, i, j);
}

__kernel void copy2Db(__global CALbyte *  end,__global CALbyte *  start, CALint substateNum, CALint rows, CALint columns){
		int id = get_global_id (0);
		//printf("*(start + rows * columns * substateNum + id) %d \n", start[id]);
		end[id] = *(start + rows * columns * substateNum + id);//  calclGetBufferElement((start + rows * columns * substateNum), columns, i, j);
}

__kernel void copy2Dr(__global CALreal *  end,__global CALreal *  start, CALint substateNum, CALint rows, CALint columns){
		int id = get_global_id (0);
		//printf("*(start + rows * columns * substateNum + id) %d \n", start[id]);
		end[id] = *(start + rows * columns * substateNum + id);//  calclGetBufferElement((start + rows * columns * substateNum), columns, i, j);
}


__kernel void copy3Di(__global CALint *  end,__global CALint *  start, CALint substateNum, CALint rows, CALint columns, CALint layers){
		int id = get_global_id (0);
		end[id] = *(start + rows * columns * layers * substateNum + id);
}

__kernel void copy3Db(__global CALbyte *  end,__global CALbyte *  start, CALint substateNum, CALint rows, CALint columns, CALint layers){
		int id = get_global_id (0);
		end[id] = *(start + rows * columns * layers *substateNum + id);
}

__kernel void copy3Dr(__global CALreal *  end,__global CALreal *  start, CALint substateNum, CALint rows, CALint columns, CALint layers){
		int id = get_global_id (0);
		end[id] = *(start + rows * columns * layers *substateNum + id);
}
