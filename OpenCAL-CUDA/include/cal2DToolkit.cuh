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

#ifndef cal2DToolkit_h
#define cal2DToolkit_h

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/fill.h>

#include "cuda_profiler_api.h"
#include "cal2D.cuh"

/*****************************************************************************
						DEFINITIONS OF NEW DATA TYPES

 *****************************************************************************/

/*! \brief 
	Object that help to copy data from CPU to GPU and viceversa. 
*/
struct Predicate
{
	__host__ __device__
		bool operator()(unsigned int x) const
	{
		return (x != -1);
	}
};



/******************************************************************************
					DEFINITIONS OF FUNCTIONS PROTOTYPES

*******************************************************************************/

/*! \brief Allocate CUDACALModel2D on device.
*/
struct CudaCALModel2D* calCudaAlloc();

/*! \brief Support function for the library: if you're final user please don't use this one.
*/
struct CudaCALModel2D* calCudaHostAlloc(struct CudaCALModel2D *model);

/*! \brief Support function for the library: if you're final user please don't use this one.
*/
struct CudaCALModel2D* calCudaAllocatorModel(struct CudaCALModel2D *model);

/*! \brief Support function for the library: if you're final user please don't use this one.
*/
void calCudaFinalizeModel();

/*! \brief Deallocate memory occupied by model.
*/
void calCudaFreeModel2D(struct CudaCALModel2D *copy_model);

/*! \brief Perform the offset in grid.
*/
__device__ CALint calCudaGetIndex(CudaCALModel2D* model);

/*! \brief Perform the offset in grid.
*/
__device__ CALint calCudaGetSimpleOffset();

/*! \brief Perform the offset in grid.
*/
__device__ CALint calCudaGetIndexRow(CudaCALModel2D* model, CALint offset);

/*! \brief Perform the offset in grid.
*/
__device__ CALint calCudaGetIndexColumn(CudaCALModel2D* model, CALint offset);


/*! \brief initialize CudaCALModel2D on GPU.
	If you use CUDA version of this library, you should copy the model in device.
	With this function all of the variable and data structure of model will be copied to device (d_model)
*/
CALbyte calInitializeInGPU2D(struct CudaCALModel2D* model, struct CudaCALModel2D *d_model);

/*! \brief Copy the CudaCALModel2D final state in CPU.
	If you use CUDA version of this library, you should copy the final state of model in CPU at the end of your simulation.
	With this function all of the variable and data structure of model will be copied to host (model)
*/
CALbyte calSendDataGPUtoCPU(struct CudaCALModel2D* model, struct CudaCALModel2D *d_model);

/*! \brief return error with name and code error 
			if something for cuda function went wrong.
*/
void cudaErrorCheck(char* message, CALbyte &result);
/*! \brief return error with name and code error 
			if something for cuda function went wrong, without return statement
*/
void cudaErrorCheck(char* message);

/*! \brief Return false if final real(floating point) results are not the same. True otherwise.
*/
CALbyte calCudaCheckFinalResult2Dr(CALreal* parallel, CALreal* sequential, CALint rows, CALint columns);


/**  
** Active cells function
**/

__device__ CALbyte calCudaImAlive(struct CudaCALModel2D* ca2D ,CALint offset);

void calCudaPerformGridAndBlockForStreamCompaction2D(struct CudaCALRun2D* simulation, dim3 grid, dim3 block);


#endif
