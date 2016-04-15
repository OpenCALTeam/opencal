#include <OpenCAL-CL/calcl2D.h>

__kernel void calclMinReductionKernel2Db(__global double *  minima, CALint substates,  __global CALbyte * offsets,unsigned offset ,int size,int count){

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

__kernel void calclMaxReductionKernel2Db(__global double *  maxima, CALint substates,  __global CALbyte * offsets,unsigned offset ,int size,int count){

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

__kernel void calclSumReductionKernel2Db(__global double *  sum, CALint substates,  __global CALbyte * offsets,unsigned offset ,int size,int count){

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

__kernel void calclProdReductionKernel2Db(__global double *  result, CALint substates,  __global CALbyte * offsets,unsigned offset ,int size,int count){

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

__kernel void calclLogicAndReductionKernel2Db(__global double *  result, CALint substates,  __global CALbyte * offsets,unsigned offset ,int size,int count){

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

__kernel void calclLogicOrReductionKernel2Db(__global double *  result, CALint substates,  __global CALbyte * offsets,unsigned offset ,int size,int count){

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

__kernel void calclLogicXOrReductionKernel2Db(__global double *  result, CALint substates,  __global CALbyte * offsets,unsigned offset ,int size,int count){

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

__kernel void calclBinaryAndReductionKernel2Db(__global double *  result, CALint substates,  __global CALbyte * offsets,unsigned offset ,int size,int count){

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

__kernel void calclBinaryOrReductionKernel2Db(__global double *  result, CALint substates,  __global CALbyte * offsets,unsigned offset ,int size,int count){

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

__kernel void calclBinaryXOrReductionKernel2Db(__global double *  result, CALint substates,  __global CALbyte * offsets,unsigned offset ,int size,int count){

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





__kernel void calclMinReductionKernel2Di(__global double *  minima, CALint substates,  __global CALint * offsets,unsigned offset ,int size,int count){

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

__kernel void calclMaxReductionKernel2Di(__global double *  maxima, CALint substates,  __global CALint * offsets,unsigned offset ,int size,int count){

	int id = get_global_id (0);
	unsigned i = offset * ((id << 1) + 1) - 1,
             j = i + offset;
            if(count > 0){
             	i = i - (count);
            	j = j - (count);
             }
	if((i >= size && j >= size) || (i < size && j >= size))
		return;
//	printf("offsets[%d] : %d \n",i,offsets[i] );
//	printf("offsets[%d] : %d \n",j,offsets[j] );

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

__kernel void calclSumReductionKernel2Di(__global double *  sum, CALint substates,  __global CALint * offsets,unsigned offset ,int size,int count){

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

__kernel void calclProdReductionKernel2Di(__global double *  result, CALint substates,  __global CALint * offsets,unsigned offset ,int size,int count){

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

__kernel void calclLogicAndReductionKernel2Di(__global double *  result, CALint substates,  __global CALint * offsets,unsigned offset ,int size,int count){

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

__kernel void calclLogicOrReductionKernel2Di(__global double *  result, CALint substates,  __global CALint * offsets,unsigned offset ,int size,int count){

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

__kernel void calclLogicXOrReductionKernel2Di(__global double *  result, CALint substates,  __global CALint * offsets,unsigned offset ,int size,int count){

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

__kernel void calclBinaryAndReductionKernel2Di(__global double *  result, CALint substates,  __global CALint * offsets,unsigned offset ,int size,int count){

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

__kernel void calclBinaryOrReductionKernel2Di(__global double *  result, CALint substates,  __global CALint * offsets,unsigned offset ,int size,int count){

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

__kernel void calclBinaryXOrReductionKernel2Di(__global double *  result, CALint substates,  __global CALint * offsets,unsigned offset ,int size,int count){

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


__kernel void calclMinReductionKernel2Dr(__global CALreal *  minima, CALint substates,  __global CALreal * offsets,unsigned offset ,int size,int count){

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

__kernel void calclMaxReductionKernel2Dr(__global CALreal *  maxima, CALint substates,  __global CALreal * offsets,unsigned offset ,int size,int count){

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

__kernel void calclSumReductionKernel2Dr(__global CALreal *  sum, CALint substates,  __global CALreal * offsets,unsigned offset ,int size,int count){

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

__kernel void calclProdReductionKernel2Dr(__global CALreal *  result, CALint substates,  __global CALreal * offsets,unsigned offset ,int size,int count){

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

__kernel void calclLogicAndReductionKernel2Dr(__global CALreal *  result, CALint substates,  __global CALreal * offsets,unsigned offset ,int size,int count){

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

__kernel void calclLogicOrReductionKernel2Dr(__global CALreal *  result, CALint substates,  __global CALreal * offsets,unsigned offset ,int size,int count){

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

__kernel void calclLogicXOrReductionKernel2Dr(__global CALreal *  result, CALint substates,  __global CALreal * offsets,unsigned offset ,int size,int count){

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

__kernel void calclBinaryAndReductionKernel2Dr(__global double *  result, CALint substates,  __global CALint * offsets,unsigned offset ,int size,int count){

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

__kernel void calclBinaryOrReductionKernel2Dr(__global double *  result, CALint substates,  __global CALint * offsets,unsigned offset ,int size,int count){

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

__kernel void calclBinaryXOrReductionKernel2Dr(__global double *  result, CALint substates,  __global CALint * offsets,unsigned offset ,int size,int count){

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


__kernel void copyi(__global CALint *  end,__global CALint *  start, CALint substateNum, CALint rows, CALint columns){
		int id = get_global_id (0);
		//printf("*(start + rows * columns * substateNum + id) %d \n", start[id]);
		end[id] = *(start + rows * columns * substateNum + id);//  calclGetBufferElement2D((start + rows * columns * substateNum), columns, i, j);
}

__kernel void copyb(__global CALbyte *  end,__global CALbyte *  start, CALint substateNum, CALint rows, CALint columns){
		int id = get_global_id (0);
		//printf("*(start + rows * columns * substateNum + id) %d \n", start[id]);
		end[id] = *(start + rows * columns * substateNum + id);//  calclGetBufferElement2D((start + rows * columns * substateNum), columns, i, j);
}

__kernel void copyr(__global CALreal *  end,__global CALreal *  start, CALint substateNum, CALint rows, CALint columns){
		int id = get_global_id (0);
		//printf("*(start + rows * columns * substateNum + id) %d \n", start[id]);
		end[id] = *(start + rows * columns * substateNum + id);//  calclGetBufferElement2D((start + rows * columns * substateNum), columns, i, j);
}

__kernel void copySumi(__global CALint *  end,__global CALint *  start, CALint substateNum, CALint rows, CALint columns){
		int id = get_global_id (0);
//		if(id == 0){
//		}
		end[id] = *(start + rows * columns * substateNum + id);//  calclGetBufferElement2D((start + rows * columns * substateNum), columns, i, j);
		//printf("%d=%d \n", id, end[id]);
}
