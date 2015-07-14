// (C) Copyright University of Calabria and others.
// All rights reserved. This program and the accompanying materials
// are made available under the terms of the GNU Lesser General Public License
// (LGPL) version 2.1 which accompanies this distribution, and is available at
// http://www.gnu.org/licenses/lgpl-2.1.html
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
// Lesser General Public License for more details.

#include ".\..\include\cal2DToolkit.cuh"
#include <stdio.h>
#include ".\..\include\cal2DBufferIO.cuh"
#include ".\..\include\cal2DBuffer.cuh"
#include <iostream>
using namespace std;

struct CudaCALModel2D *copy_model;


struct CudaCALModel2D* calCudaAlloc(){
	struct CudaCALModel2D* device_object;
	cudaMalloc((void**)&device_object, sizeof(CudaCALModel2D));
	return device_object;
}

struct CudaCALModel2D* calCudaHostAlloc(struct CudaCALModel2D *model){
	struct CudaCALModel2D* device_object;
	cudaHostAlloc((void**)&device_object, sizeof(CudaCALModel2D), cudaHostAllocDefault);

	cudaMalloc((void**)&device_object->i,model->sizeof_X*sizeof(int));
	cudaMalloc((void**)&device_object->j,model->sizeof_X*sizeof(int));

	if(model->OPTIMIZATION == CAL_OPT_ACTIVE_CELLS){
		cudaMalloc((void**)&device_object->activecell_flags,model->rows*model->columns*sizeof(CALbyte));
		cudaMalloc((void**)&device_object->activecell_index,model->rows*model->columns*sizeof(CALint));
	}

	if(model->sizeof_pQb_array > 0){
		cudaMalloc((void**)&device_object->pQb_array_current,model->sizeof_pQb_array*model->rows*model->columns*sizeof(CALbyte));
		cudaMalloc((void**)&device_object->pQb_array_next,model->sizeof_pQb_array*model->rows*model->columns*sizeof(CALbyte));
	}
	if(model->sizeof_pQi_array > 0){
		cudaMalloc((void**)&device_object->pQi_array_current,model->sizeof_pQi_array*model->rows*model->columns*sizeof(CALint));
		cudaMalloc((void**)&device_object->pQi_array_next,model->sizeof_pQi_array*model->rows*model->columns*sizeof(CALint));
	}
	if(model->sizeof_pQr_array > 0){
		cudaMalloc((void**)&device_object->pQr_array_current,model->sizeof_pQr_array*model->rows*model->columns*sizeof(CALreal));
		cudaMalloc((void**)&device_object->pQr_array_next,model->sizeof_pQr_array*model->rows*model->columns*sizeof(CALreal));
	}
	return device_object;
}

struct CudaCALModel2D* calCudaAllocatorModel(struct CudaCALModel2D *model){

	cudaMallocHost((void**)&copy_model, sizeof(struct CudaCALModel2D), cudaHostAllocPortable);

	memcpy(copy_model,model,sizeof(struct CudaCALModel2D));

	cudaMalloc((void**)&copy_model->i,model->sizeof_X*sizeof(int));
	cudaMalloc((void**)&copy_model->j,model->sizeof_X*sizeof(int));

	if(model->OPTIMIZATION == CAL_OPT_ACTIVE_CELLS){
		cudaMalloc((void**)&copy_model->activecell_flags,model->rows*model->columns*sizeof(CALbyte));
		cudaMalloc((void**)&copy_model->activecell_index,model->rows*model->columns*sizeof(CALint));
		cudaMalloc((void**)&copy_model->array_of_index_result, model->rows*model->columns*sizeof(CALint));
	}

	if(model->sizeof_pQb_array > 0){
		cudaMalloc((void**)&copy_model->pQb_array_current,model->sizeof_pQb_array*model->rows*model->columns*sizeof(CALbyte));
		cudaMalloc((void**)&copy_model->pQb_array_next,model->sizeof_pQb_array*model->rows*model->columns*sizeof(CALbyte));
	}
	if(model->sizeof_pQi_array > 0){
		cudaMalloc((void**)&copy_model->pQi_array_current,model->sizeof_pQi_array*model->rows*model->columns*sizeof(CALint));
		cudaMalloc((void**)&copy_model->pQi_array_next,model->sizeof_pQi_array*model->rows*model->columns*sizeof(CALint));
	}
	if(model->sizeof_pQr_array > 0){
		cudaMalloc((void**)&copy_model->pQr_array_current,model->sizeof_pQr_array*model->rows*model->columns*sizeof(CALreal));
		cudaMalloc((void**)&copy_model->pQr_array_next,model->sizeof_pQr_array*model->rows*model->columns*sizeof(CALreal));
	}

	return copy_model;
}

void calCudaFinalizeModel(){

	cudaFree(copy_model->i);
	cudaFree(copy_model->j);

	if(copy_model->OPTIMIZATION == CAL_OPT_ACTIVE_CELLS){
		cudaFree(copy_model->activecell_flags);
		cudaFree(copy_model->activecell_index);
		cudaFree(copy_model->array_of_index_result);
	}
	cudaFree(copy_model->pQb_array_current);
	cudaFree(copy_model->pQb_array_next);
	cudaFree(copy_model->pQi_array_current);
	cudaFree(copy_model->pQi_array_next);
	cudaFree(copy_model->pQr_array_current);
	cudaFree(copy_model->pQr_array_next); 
	cudaFreeHost(copy_model);
}

void calCudaFreeModel2D(struct CudaCALModel2D *copy_model){

	cudaFree(copy_model->i);
	cudaFree(copy_model->j);

	if(copy_model->OPTIMIZATION == CAL_OPT_ACTIVE_CELLS){
		cudaFree(copy_model->activecell_flags);
		cudaFree(copy_model->activecell_index);
		cudaFree(copy_model->array_of_index_result);
	}
	cudaFree(copy_model->pQb_array_current);
	cudaFree(copy_model->pQb_array_next);
	cudaFree(copy_model->pQi_array_current);
	cudaFree(copy_model->pQi_array_next);
	cudaFree(copy_model->pQr_array_current);
	cudaFree(copy_model->pQr_array_next); 
	cudaFreeHost(copy_model);
}

__device__
	CALint calCudaGetIndex(CudaCALModel2D* ca2D){

		if(ca2D->OPTIMIZATION == CAL_OPT_ACTIVE_CELLS){
			return ca2D->array_of_index_result[calCudaGetSimpleOffset()];
		}else{
			return calCudaGetSimpleOffset();
		}
}
__device__
	CALint calCudaGetSimpleOffset(){

		CALint i = blockIdx.x*blockDim.x + threadIdx.x;
		CALint j = blockIdx.y*blockDim.y + threadIdx.y;

		return i + j*blockDim.x*gridDim.x;
}

__device__ CALint calCudaGetIndexRow(CudaCALModel2D* model, CALint offset){
	return offset / (model->columns);
}

__device__ CALint calCudaGetIndexColumn(CudaCALModel2D* model, CALint offset){
	return offset % (model->columns);
}


CALbyte calInitializeInGPU2D(struct CudaCALModel2D* model, struct CudaCALModel2D *d_model){

	CALbyte result = CAL_TRUE;

	calCudaAllocatorModel(model);

	cudaMemcpy(copy_model->i,model->i, sizeof(CALint)*model->sizeof_X, cudaMemcpyHostToDevice);
	cudaMemcpy(copy_model->j,model->j, sizeof(CALint)*model->sizeof_X, cudaMemcpyHostToDevice);

	if(model->OPTIMIZATION == CAL_OPT_ACTIVE_CELLS){
		cudaMemcpy(copy_model->activecell_flags,model->activecell_flags, sizeof(CALbyte)*model->rows*model->columns, cudaMemcpyHostToDevice);
		cudaMemcpy(copy_model->activecell_index,model->activecell_index, sizeof(CALint)*model->rows*model->columns, cudaMemcpyHostToDevice);
		cudaMemcpy(copy_model->array_of_index_result,model->array_of_index_result, sizeof(CALint)*model->rows*model->columns, cudaMemcpyHostToDevice);
	}

	if(model->sizeof_pQb_array > 0){
		cudaMemcpy(copy_model->pQb_array_current,model->pQb_array_current, model->sizeof_pQb_array*model->rows*model->columns*sizeof(CALbyte), cudaMemcpyHostToDevice);
		cudaMemcpy(copy_model->pQb_array_next,model->pQb_array_next, model->sizeof_pQb_array*model->rows*model->columns*sizeof(CALbyte), cudaMemcpyHostToDevice);
	}
	if(model->sizeof_pQi_array > 0){
		cudaMemcpy(copy_model->pQi_array_current,model->pQi_array_current, model->sizeof_pQi_array*model->rows*model->columns*sizeof(CALint), cudaMemcpyHostToDevice);
		cudaMemcpy(copy_model->pQi_array_next,model->pQi_array_next, model->sizeof_pQi_array*model->rows*model->columns*sizeof(CALint), cudaMemcpyHostToDevice);
	}
	if(model->sizeof_pQr_array > 0){
		cudaMemcpy(copy_model->pQr_array_current,model->pQr_array_current, model->sizeof_pQr_array*model->rows*model->columns*sizeof(CALreal), cudaMemcpyHostToDevice);
		cudaMemcpy(copy_model->pQr_array_next,model->pQr_array_next, model->sizeof_pQr_array*model->rows*model->columns*sizeof(CALreal), cudaMemcpyHostToDevice);
	}
	cudaMemcpy(d_model, copy_model, sizeof(struct CudaCALModel2D), cudaMemcpyHostToDevice);

	return result;
}

CALbyte calSendDataGPUtoCPU(struct CudaCALModel2D* model, struct CudaCALModel2D *d_model){

	CALbyte result = CAL_TRUE;

	cudaMemcpy(copy_model, d_model, sizeof(struct CudaCALModel2D), cudaMemcpyDeviceToHost);

	if(model->sizeof_pQb_array > 0){
		cudaMemcpy(model->pQb_array_current,copy_model->pQb_array_current,model->sizeof_pQb_array*model->rows*model->columns*sizeof(CALbyte), cudaMemcpyDeviceToHost);
		cudaMemcpy(model->pQb_array_next,copy_model->pQb_array_next,model->sizeof_pQb_array*model->rows*model->columns*sizeof(CALbyte), cudaMemcpyDeviceToHost);		
	}
	if(model->sizeof_pQi_array > 0){
		cudaMemcpy(model->pQi_array_current,copy_model->pQi_array_current,model->sizeof_pQi_array*model->rows*model->columns*sizeof(CALint), cudaMemcpyDeviceToHost);		
		cudaMemcpy(model->pQi_array_next,copy_model->pQi_array_next,model->sizeof_pQi_array*model->rows*model->columns*sizeof(CALint), cudaMemcpyDeviceToHost);
	}
	if(model->sizeof_pQr_array > 0){
		cudaMemcpy(model->pQr_array_current,copy_model->pQr_array_current,model->sizeof_pQr_array*model->rows*model->columns*sizeof(CALreal), cudaMemcpyDeviceToHost);		
		cudaMemcpy(model->pQr_array_next,copy_model->pQr_array_next,model->sizeof_pQr_array*model->rows*model->columns*sizeof(CALreal), cudaMemcpyDeviceToHost);
	}

	calCudaFinalizeModel();

	return result;
}

void printError(cudaError error){
	//if(error != cudaSuccess)
	printf("Error: %s\n", error);
}

void cudaErrorCheck(char* message){	
	cudaError_t error = cudaGetLastError();
	if(error != cudaSuccess)
	{	
		printf("\n******************\n*** Attention! *** \n******************\n"); 
		printf("\nError: %s \nName: %s\n", cudaGetErrorString(error), cudaGetErrorName(error));
		printf("\n******************\n\n"); 
		system("pause");
		exit(-1);
	}else{
		printf("Message: %s\n", message);
	}
}

void cudaErrorCheck(char* message, CALbyte &result){	
	cudaError_t error = cudaGetLastError();
	if(error != cudaSuccess)
	{	
		printf("\n******************\n*** Attention! *** \n******************\n"); 
		printf("\nError: %s \nName: %s\n", cudaGetErrorString(error), cudaGetErrorName(error));
		printf("\n******************\n\n"); 
		result = CAL_FALSE;
		system("pause");
		exit(-1);
	}else{
		printf("Message: %s\n", message);
	}
}

// SCIDDICA_T TEST
#define OUTPUT_PATH "./data/width_final.txt"
#define OUTPUT_PATH_S "./data/width_final_s.txt"
CALbyte calCudaCheckFinalResult2Dr(CALreal* parallel, CALreal* sequential, CALint rows, CALint columns){

	CALint i,j;
	cudaHostAlloc((void**)&parallel, sizeof(CALreal)*rows*columns, cudaHostAllocDefault); 
	cudaHostAlloc((void**)&sequential,	sizeof(CALreal)*rows*columns, cudaHostAllocDefault);

	calCudaLoadMatrix2Dr(parallel, rows, columns, OUTPUT_PATH,0);
	calCudaLoadMatrix2Dr(sequential, rows, columns, OUTPUT_PATH_S,0);

	for(i=0; i<rows; i++)
		for(j=0; j<columns; j++)
		{
			if(parallel[i*columns+j] != sequential[i*columns+j])
				return CAL_FALSE;
		}

		return CAL_TRUE;
}

__device__ CALbyte calCudaImAlive(struct CudaCALModel2D* ca2D, CALint offset){
	return calCudaGetMatrixElement(ca2D->activecell_flags, offset, ca2D->rows, ca2D->columns, 0); 
}

void calCudaPerformGridAndBlockForStreamCompaction2D(struct CudaCALRun2D* simulation, dim3 grid, dim3 block){
	simulation->ca2D->activecell_size_current = simulation->h_device_ca2D->activecell_size_next;
	CALint num_blocks = simulation->ca2D->activecell_size_current / ((block.x) * (block.y));
	grid.x = (num_blocks+2);
	grid.y = 1;
}
