#include <stdlib.h>
#include <stdio.h>
#include ".\..\include\cal2D.cuh"
#include ".\..\include\cal2DBuffer.cuh"

/******************************************************************************
PRIVATE FUNCIONS

*******************************************************************************/

void calDefineVonNeumannNeighborhood2D(struct CudaCALModel2D* ca2D	//!< Pointer to the cellular automaton structure.
									   ) 
{
	/*
	| 1 |  
	---|---|---
	2 | 0 | 3
	---|---|---
	| 4 |  
	*/

	calCudaAddNeighbor2D(ca2D,   0,   0);
	calCudaAddNeighbor2D(ca2D, - 1,   0);
	calCudaAddNeighbor2D(ca2D,   0, - 1);
	calCudaAddNeighbor2D(ca2D,   0, + 1);
	calCudaAddNeighbor2D(ca2D, + 1,   0);
}

void calDefineMooreNeighborhood2D(struct CudaCALModel2D* ca2D		//!< Pointer to the cellular automaton structure.
								  )
{
	/*
	5 | 1 | 8
	---|---|---
	2 | 0 | 3
	---|---|---
	6 | 4 | 7
	*/

	calCudaAddNeighbor2D(ca2D,   0,   0);
	calCudaAddNeighbor2D(ca2D, - 1,   0);
	calCudaAddNeighbor2D(ca2D,   0, - 1);
	calCudaAddNeighbor2D(ca2D,   0, + 1);
	calCudaAddNeighbor2D(ca2D, + 1,   0);
	calCudaAddNeighbor2D(ca2D, - 1, - 1);
	calCudaAddNeighbor2D(ca2D, + 1, - 1);
	calCudaAddNeighbor2D(ca2D, + 1, + 1);
	calCudaAddNeighbor2D(ca2D, - 1, + 1);
}

void calDefineHexagonalNeighborhood2D(struct CudaCALModel2D* ca2D		//!< Pointer to the cellular automaton structure.
									  )
{
	/*
	cell orientation
	__	
	/  \
	\__/
	*/
	/*
	3 | 2 | 1
	---|---|---
	4 | 0 | 6		if (j%2 == 0), i.e. even columns
	---|---|---
	| 5 |  
	*/

	calCudaAddNeighbor2D(ca2D,   0,   0);
	calCudaAddNeighbor2D(ca2D, - 1, + 1);
	calCudaAddNeighbor2D(ca2D, - 1,   0);
	calCudaAddNeighbor2D(ca2D, - 1, - 1);
	calCudaAddNeighbor2D(ca2D,   0, - 1);
	calCudaAddNeighbor2D(ca2D, + 1,   0);
	calCudaAddNeighbor2D(ca2D,   0, + 1);

	/*
	| 2 |  
	---|---|---
	3 | 0 | 1		if (j%2 == 1), i.e. odd columns
	---|---|---
	4 | 5 | 6
	*/

	calCudaAddNeighbor2D(ca2D,   0,   0);
	calCudaAddNeighbor2D(ca2D,   0, + 1);
	calCudaAddNeighbor2D(ca2D, - 1,   0);
	calCudaAddNeighbor2D(ca2D,   0, - 1);
	calCudaAddNeighbor2D(ca2D, + 1, - 1);
	calCudaAddNeighbor2D(ca2D, + 1,   0);
	calCudaAddNeighbor2D(ca2D, + 1, + 1);

	ca2D->sizeof_X = 7;
}

void calDefineAlternativeHexagonalNeighborhood2D(struct CudaCALModel2D* ca2D	//!< Pointer to the cellular automaton structure.
												 )
{
	/*
	cell orientation

	/\
	/  \
	|  |
	\  /
	\/
	*/
	/*
	2 | 1 |  
	---|---|---
	3 | 0 | 6		if (i%2 == 0), i.e. even rows
	---|---|---
	4 | 5 |  
	*/

	calCudaAddNeighbor2D(ca2D,   0,   0);
	calCudaAddNeighbor2D(ca2D, - 1,   0);
	calCudaAddNeighbor2D(ca2D, - 1, - 1);
	calCudaAddNeighbor2D(ca2D,   0, - 1);
	calCudaAddNeighbor2D(ca2D, + 1, - 1);
	calCudaAddNeighbor2D(ca2D, + 1,   0);
	calCudaAddNeighbor2D(ca2D,   0, + 1);

	/*
	| 2 | 1
	---|---|---
	3 | 0 | 6		if (i%2 == 1), i.e. odd rows
	---|---|---
	| 4 | 5
	*/

	calCudaAddNeighbor2D(ca2D,   0,   0);
	calCudaAddNeighbor2D(ca2D, - 1, + 1);
	calCudaAddNeighbor2D(ca2D, - 1,   0);
	calCudaAddNeighbor2D(ca2D,   0, - 1);
	calCudaAddNeighbor2D(ca2D, + 1,   0);
	calCudaAddNeighbor2D(ca2D, + 1, + 1);
	calCudaAddNeighbor2D(ca2D,   0, + 1);

	ca2D->sizeof_X = 7;
}


/******************************************************************************
PUBLIC FUNCIONS

*******************************************************************************/

struct CudaCALModel2D* calCudaCADef2D(int rows,
	int columns,
	enum CALNeighborhood2D CAL_NEIGHBORHOOD_2D,
	enum CALSpaceBoundaryCondition CAL_TOROIDALITY,
	enum CALOptimization CAL_OPTIMIZATION
	)
{
	struct CudaCALModel2D *ca2D = 0;
	cudaMallocHost((void **)&ca2D, sizeof(CudaCALModel2D), cudaHostAllocPortable);

	if (!ca2D)
		return NULL;

	ca2D->rows = rows;
	ca2D->columns = columns;

	ca2D->T = CAL_TOROIDALITY;

	ca2D->OPTIMIZATION = CAL_OPTIMIZATION;
	if (ca2D->OPTIMIZATION == CAL_OPT_ACTIVE_CELLS) {
		ca2D->activecell_flags = calAllocBuffer2Db(ca2D->rows, ca2D->columns);
		calSetBuffer2Db(ca2D->activecell_flags, ca2D->rows, ca2D->columns, CAL_FALSE);

		ca2D->activecell_index = (unsigned int*) calAllocBuffer2Di(ca2D->rows, ca2D->columns);
		calSetBuffer2Di((CALint*)ca2D->activecell_index, ca2D->rows, ca2D->columns, 0);

		ca2D->array_of_index_result = (unsigned int*) calAllocBuffer2Di(ca2D->rows, ca2D->columns);
		calSetBuffer2Di((CALint*)ca2D->array_of_index_result, ca2D->rows, ca2D->columns, 0);
	}
	else{
		ca2D->activecell_flags = NULL;
		ca2D->activecell_index = NULL;
	}
	ca2D->activecell_size_next = 0;
	//	ca2D->i_activecell = NULL;
	//  ca2D->j_activecell = 0;

	ca2D->stop = CAL_FALSE;

	ca2D->i = NULL;
	ca2D->j = NULL;
	ca2D->sizeof_X = 0;

	ca2D->X_id = CAL_NEIGHBORHOOD_2D;
	switch (CAL_NEIGHBORHOOD_2D) {	
	case CAL_VON_NEUMANN_NEIGHBORHOOD_2D:
		calDefineVonNeumannNeighborhood2D(ca2D);
		break;
	case CAL_MOORE_NEIGHBORHOOD_2D:
		calDefineMooreNeighborhood2D(ca2D);
		break;
	case CAL_HEXAGONAL_NEIGHBORHOOD_2D:
		calDefineHexagonalNeighborhood2D(ca2D);
		break;
	case CAL_HEXAGONAL_NEIGHBORHOOD_ALT_2D:
		calDefineAlternativeHexagonalNeighborhood2D(ca2D);
		break;
	}

	ca2D->pQb_array_current = NULL;
	ca2D->pQb_array_next = NULL;
	ca2D->pQi_array_current = NULL;
	ca2D->pQi_array_next = NULL;
	ca2D->pQr_array_current = NULL;
	ca2D->pQr_array_next = NULL;

	ca2D->sizeof_pQb_array = 0;
	ca2D->sizeof_pQi_array = 0;
	ca2D->sizeof_pQr_array = 0;

	ca2D->elementary_processes = NULL;
	ca2D->num_of_elementary_processes = 0;

	return ca2D;
}

__device__
	void calCudaAddActiveCell2D(struct CudaCALModel2D* ca2D, int offset)
{
	if (!calCudaGetMatrixElement(ca2D->activecell_flags, offset, ca2D->rows, ca2D->columns, 0))
	{
		calCudaSetMatrixElement(ca2D->activecell_flags, offset, CAL_TRUE, ca2D->rows, ca2D->columns, 0);

		atomicAdd(&ca2D->activecell_size_next, 1);

	}
}

__device__
	void calCudaAddActiveCellX2D(struct CudaCALModel2D* ca2D, int offset, int n)
{
	CALint i = calCudaGetIndexRow(ca2D, offset), j = calCudaGetIndexColumn(ca2D, offset);

	if ((ca2D->X_id == CAL_HEXAGONAL_NEIGHBORHOOD_2D && j%2 ==1) || (ca2D->X_id == CAL_HEXAGONAL_NEIGHBORHOOD_ALT_2D && i%2 ==1))
		n += CAL_HEXAGONAL_SHIFT;

	if (ca2D->T == CAL_SPACE_FLAT)
	{
		int index = calGetLinearIndex(offset, ca2D->columns, ca2D->rows, ca2D->i[n],ca2D->j[n], 0);
		if (!calCudaGetMatrixElement_(ca2D->activecell_flags, index, ca2D->rows, ca2D->columns, 0))
		{
			calCudaSetMatrixElement(ca2D->activecell_flags, index, CAL_TRUE,ca2D->rows, ca2D->columns, 0);
			//ca2D->activecell_size_next++;
			atomicAdd(&ca2D->activecell_size_next, 1);
		}
	}
	else
	{
		int index = calGetToroidalLinearIndex(offset, ca2D->columns, ca2D->rows, ca2D->i[n],ca2D->j[n], 0);
		if (!calCudaGetMatrixElement_(ca2D->activecell_flags, index, ca2D->rows, ca2D->columns, 0))
		{
			calCudaSetMatrixElement(ca2D->activecell_flags, index, CAL_TRUE,ca2D->rows, ca2D->columns, 0);
			//ca2D->activecell_size_next++;
			atomicAdd(&ca2D->activecell_size_next, 1);
		}
	}
}

__device__
	void calCudaRemoveActiveCell2D(struct CudaCALModel2D* ca2D, int offset)
{
	if (calCudaGetMatrixElement(ca2D->activecell_flags, offset, ca2D->rows, ca2D->columns, 0))
	{
		calCudaSetMatrixElement(ca2D->activecell_flags, offset, CAL_FALSE, ca2D->rows, ca2D->columns, 0);
		//ca2D->activecell_size_next--;
		atomicSub(&ca2D->activecell_size_next, 1);
	}
}

__global__ void generateSetOfIndex(CudaCALModel2D *device_ca2D){
	//algoritmo che trasforma la matrice di flag in matrice di index dove è diverso da 0 il valore del flag.

	CALint index = calCudaGetSimpleOffset(), no_value = -1; 
	if(calCudaGetMatrixElement_(device_ca2D->activecell_flags, index, device_ca2D->rows, device_ca2D->columns, 0) == CAL_TRUE){
		calCudaSetMatrixElement(device_ca2D->activecell_index, index, index,device_ca2D->rows, device_ca2D->columns, 0);
	}
	else{
		calCudaSetMatrixElement(device_ca2D->activecell_index, index, no_value,device_ca2D->rows, device_ca2D->columns, 0);
	}

	device_ca2D->array_of_index_result[index] = no_value;
}

void calCudaApplyStreamCompaction(struct CudaCALRun2D* simulation, dim3 grid, dim3 block){

	CALint SIZE = (simulation->ca2D->rows*simulation->ca2D->columns);

	if(simulation->ca2D->activecell_size_current != simulation->h_device_ca2D->activecell_size_next){
		generateSetOfIndex<<<grid,block>>>(simulation->device_ca2D);

		cudaMemcpy(simulation->h_device_ca2D, simulation->device_ca2D, sizeof(struct CudaCALModel2D), cudaMemcpyDeviceToHost);

		pp::compact( 
			simulation->h_device_ca2D->activecell_index,              /* Input start pointer */
			simulation->h_device_ca2D->activecell_index+SIZE,     /* Input end pointer */
			simulation->h_device_ca2D->array_of_index_result,              /* Output start pointer */
			simulation->device_array_of_index_dim,            /* Storage for valid element count */
			Predicate()             /* Predicate */
			);

		cudaMemcpy(simulation->device_ca2D, simulation->h_device_ca2D, sizeof(struct CudaCALModel2D), cudaMemcpyHostToDevice);

	}


	//qui devo avviare solo N threads per N celle attive
	//In cal2DToolkit devo ritornare invece dell'offset quello nella string compaction
	//calCudaPerformGridAndBlockForStreamCompaction2D(simulation, grid, block);
	simulation->ca2D->activecell_size_current = simulation->h_device_ca2D->activecell_size_next;
	CALint num_blocks = simulation->ca2D->activecell_size_current / ((block.x) * (block.y));
	grid.x = (num_blocks+2);
	grid.y = 1;

}

void calCudaUpdateActiveCells2D(struct CudaCALRun2D* simulation)
{

	cudaMemcpy(&simulation->h_device_ca2D->activecell_size_next, &simulation->device_ca2D->activecell_size_next, sizeof(int), cudaMemcpyDeviceToHost);
	simulation->ca2D->activecell_size_current = simulation->h_device_ca2D->activecell_size_next;	
}



void calCudaAddNeighbor2D(struct CudaCALModel2D* ca2D, int i, int j) {
	//struct CALCell2D* X_tmp = ca2D->X;
	int *i_tmp = ca2D->i;
	int *j_tmp = ca2D->j;

	//struct CALCell2D* X_new;
	int *i_new;
	int *j_new;

	int n;

	i_new = (int*)malloc(sizeof(int)*(ca2D->sizeof_X + 1));
	j_new = (int*)malloc(sizeof(int)*(ca2D->sizeof_X + 1));

	if (!i_new || !j_new)
		return;

	for (n = 0; n < ca2D->sizeof_X; n++) {
		i_new[n] = ca2D->i[n];
		j_new[n] = ca2D->j[n];
	}
	i_new[ca2D->sizeof_X] = i;
	j_new[ca2D->sizeof_X] = j;

	ca2D->i = i_new;
	ca2D->j = j_new;

	free(i_tmp);
	free(j_tmp);

	ca2D->sizeof_X++;

	//return ca2D->X;
}

cudaError_t calCudaAddSubstate2Db(struct CudaCALModel2D* ca2D, CALint NUMBER_OF_SUBSTATE){

	cudaError_t check;

	check = cudaMallocHost((void **)&ca2D->pQb_array_current, sizeof(CALbyte)*(NUMBER_OF_SUBSTATE * ca2D->rows * ca2D->columns), cudaHostAllocPortable);
	check = cudaMallocHost((void **)&ca2D->pQb_array_next, sizeof(CALbyte)*(NUMBER_OF_SUBSTATE * ca2D->rows * ca2D->columns), cudaHostAllocPortable);

	ca2D->sizeof_pQb_array += NUMBER_OF_SUBSTATE;

	return check;
}

cudaError_t calCudaAddSubstate2Di(struct CudaCALModel2D* ca2D, CALint NUMBER_OF_SUBSTATE){

	cudaError_t check;

	check = cudaMallocHost((void **)&ca2D->pQi_array_current, sizeof(CALint)*(NUMBER_OF_SUBSTATE * ca2D->rows * ca2D->columns), cudaHostAllocPortable);
	check = cudaMallocHost((void **)&ca2D->pQi_array_next, sizeof(CALint)*(NUMBER_OF_SUBSTATE * ca2D->rows * ca2D->columns), cudaHostAllocPortable);

	ca2D->sizeof_pQi_array += NUMBER_OF_SUBSTATE;

	return check;
}

cudaError_t calCudaAddSubstate2Dr(struct CudaCALModel2D* ca2D, CALint NUMBER_OF_SUBSTATE){

	cudaError_t check;

	check = cudaMallocHost((void **)&ca2D->pQr_array_current, sizeof(CALreal)*(NUMBER_OF_SUBSTATE * ca2D->rows * ca2D->columns), cudaHostAllocPortable);
	check = cudaMallocHost((void **)&ca2D->pQr_array_next, sizeof(CALreal)*(NUMBER_OF_SUBSTATE * ca2D->rows * ca2D->columns), cudaHostAllocPortable);

	ca2D->sizeof_pQr_array += NUMBER_OF_SUBSTATE;

	return check;
}
/*
struct CALSubstate2Db* calAddSingleLayerSubstate2Db(struct CALModel2D* ca2D){

struct CALSubstate2Db* Q;
Q = (struct CALSubstate2Db*)malloc(sizeof(struct CALSubstate2Db));
if (!Q)
return NULL;
Q->current = calAllocBuffer2Db(ca2D->rows, ca2D->columns);
if (!Q->current)
return NULL;
Q->next = NULL;

return Q;
}

struct CALSubstate2Di* calAddSingleLayerSubstate2Di(struct CALModel2D* ca2D){

struct CALSubstate2Di* Q;
Q = (struct CALSubstate2Di*)malloc(sizeof(struct CALSubstate2Di));
if (!Q)
return NULL;
Q->current = calAllocBuffer2Di(ca2D->rows, ca2D->columns);
if (!Q->current)
return NULL;
Q->next = NULL;

return Q;
}

struct CALSubstate2Dr* calAddSingleLayerSubstate2Dr(struct CALModel2D* ca2D){

struct CALSubstate2Dr* Q;
Q = (struct CALSubstate2Dr*)malloc(sizeof(struct CALSubstate2Dr));
if (!Q)
return NULL;
Q->current = calAllocBuffer2Dr(ca2D->rows, ca2D->columns);
if (!Q->current)
return NULL;
Q->next = NULL;

return Q;
}

*/

CALCudaCallbackFunc2D* calCudaAddElementaryProcess2D(struct CudaCALModel2D* ca2D,	//!< Pointer to the cellular automaton structure.
													 void (* elementary_process)(struct CudaCALModel2D* ca2D)
													 )
{
	void(** callbacks_temp)(struct CudaCALModel2D* ca2D) = ca2D->elementary_processes;
	void(** callbacks_new)(struct CudaCALModel2D* ca2D) = (void(**)(struct CudaCALModel2D* ca2D))malloc(sizeof(void (*)(struct CudaCALModel2D* ca2D))*(ca2D->num_of_elementary_processes + 1));
	int n;

	if (!callbacks_new)
		return NULL;

	for (n = 0; n < ca2D->num_of_elementary_processes; n++)
		callbacks_new[n] = ca2D->elementary_processes[n];
	callbacks_new[ca2D->num_of_elementary_processes] = elementary_process;

	ca2D->elementary_processes = callbacks_new;
	free(callbacks_temp);

	ca2D->num_of_elementary_processes++;

	return ca2D->elementary_processes;
}

void calCudaApplyElementaryProcess2D(struct CudaCALRun2D* simulation,	//!< Pointer to the cellular automaton structure.
									 void (* elementary_process)(struct CudaCALModel2D* ca2D), //!< Pointer to a transition function's elementary process.
									 dim3 grid, dim3 block
									 )
{

	if (simulation->ca2D->OPTIMIZATION == CAL_OPT_ACTIVE_CELLS){

		//calCudaApplyStreamCompaction(simulation, grid, block);

		CALint SIZE = (simulation->ca2D->rows*simulation->ca2D->columns);

		if(simulation->ca2D->activecell_size_current != simulation->h_device_ca2D->activecell_size_next){
			generateSetOfIndex<<<grid,block>>>(simulation->device_ca2D);

			cudaMemcpy(simulation->h_device_ca2D, simulation->device_ca2D, sizeof(struct CudaCALModel2D), cudaMemcpyDeviceToHost);

			pp::compact( 
				simulation->h_device_ca2D->activecell_index,              /* Input start pointer */
				simulation->h_device_ca2D->activecell_index+SIZE,     /* Input end pointer */
				simulation->h_device_ca2D->array_of_index_result,              /* Output start pointer */
				simulation->device_array_of_index_dim,            /* Storage for valid element count */
				Predicate()             /* Predicate */
				);

			cudaMemcpy(simulation->device_ca2D, simulation->h_device_ca2D, sizeof(struct CudaCALModel2D), cudaMemcpyHostToDevice);

		}


		//qui devo avviare solo N threads per N celle attive
		//In cal2DToolkit devo ritornare invece dell'offset quello nella string compaction
		//calCudaPerformGridAndBlockForStreamCompaction2D(simulation, grid, block);
		simulation->ca2D->activecell_size_current = simulation->h_device_ca2D->activecell_size_next;
		CALint num_blocks = simulation->ca2D->activecell_size_current / ((block.x) * (block.y));
		grid.x = (num_blocks+2);
		grid.y = 1;

		//lancio il processo elementare
		elementary_process<<<grid,block>>>(simulation->device_ca2D);
	}else
		//Standart cicle of the transition function
		elementary_process<<<grid,block>>>(simulation->device_ca2D);
}

void calCudaGlobalTransitionFunction2D(struct CudaCALRun2D* simulation, dim3 grid, dim3 block)
{
	//The global transition function.
	//It applies transition function elementary processes sequentially.
	//Note that a substates' update is performed after each elementary process.

	CALint b;

	for (b=0; b<simulation->ca2D->num_of_elementary_processes; b++)
	{

		//applying the b-th elementary process
		calCudaApplyElementaryProcess2D(simulation, simulation->ca2D->elementary_processes[b], grid, block);

		//updating substates
		calCudaUpdate2D(simulation);
	}   
}

//updating substates
void calCudaUpdate2D(struct CudaCALRun2D* simulation)
{

	//	updating active cells
	//if (simulation->ca2D->OPTIMIZATION == CAL_OPT_ACTIVE_CELLS)
	//calCudaUpdateActiveCells2D(simulation);

	if(simulation->ca2D->sizeof_pQb_array > 0){
		cudaMemcpy(simulation->h_device_ca2D, simulation->device_ca2D, sizeof(struct CudaCALModel2D), cudaMemcpyDeviceToHost);
		cudaMemcpy(simulation->h_device_ca2D->pQb_array_current,simulation->h_device_ca2D->pQb_array_next, simulation->ca2D->sizeof_pQb_array*simulation->ca2D->columns*simulation->ca2D->rows*sizeof(CALbyte),cudaMemcpyDeviceToDevice);
		cudaMemcpy(simulation->device_ca2D, simulation->h_device_ca2D, sizeof(struct CudaCALModel2D), cudaMemcpyHostToDevice);
	}
	if(simulation->ca2D->sizeof_pQi_array > 0){
		cudaMemcpy(simulation->h_device_ca2D, simulation->device_ca2D, sizeof(struct CudaCALModel2D), cudaMemcpyDeviceToHost);
		cudaMemcpy(simulation->h_device_ca2D->pQi_array_current,simulation->h_device_ca2D->pQi_array_next, simulation->ca2D->sizeof_pQi_array*simulation->ca2D->columns*simulation->ca2D->rows*sizeof(CALint),cudaMemcpyDeviceToDevice);
		cudaMemcpy(simulation->device_ca2D, simulation->h_device_ca2D, sizeof(struct CudaCALModel2D), cudaMemcpyHostToDevice);
	}
	if(simulation->ca2D->sizeof_pQr_array > 0){
		cudaMemcpy(simulation->h_device_ca2D, simulation->device_ca2D, sizeof(struct CudaCALModel2D), cudaMemcpyDeviceToHost);
		cudaMemcpy(simulation->h_device_ca2D->pQr_array_current,simulation->h_device_ca2D->pQr_array_next, simulation->ca2D->sizeof_pQr_array*simulation->ca2D->columns*simulation->ca2D->rows*sizeof(CALreal),cudaMemcpyDeviceToDevice);
		cudaMemcpy(simulation->device_ca2D, simulation->h_device_ca2D, sizeof(struct CudaCALModel2D), cudaMemcpyHostToDevice);
	}
}

__device__ 
	void calCudaInit2Db(struct CudaCALModel2D* ca2D, int offset, CALbyte value, CALint substate_index) {
		calCudaSetMatrixElement(ca2D->pQb_array_current, offset, value, ca2D->rows, ca2D->columns, substate_index);
		calCudaSetMatrixElement(ca2D->pQb_array_next, offset, value, ca2D->rows, ca2D->columns, substate_index);
}

__device__ 
	void calCudaInit2Di(struct CudaCALModel2D* ca2D, int offset, CALint value, CALint substate_index) {
		calCudaSetMatrixElement(ca2D->pQi_array_current, offset, value, ca2D->rows, ca2D->columns, substate_index);
		calCudaSetMatrixElement(ca2D->pQi_array_next, offset, value, ca2D->rows, ca2D->columns, substate_index);
}

__device__ 
	void calCudaInit2Dr(struct CudaCALModel2D* ca2D, int offset, CALreal value, CALint substate_index) {
		calCudaSetMatrixElement(ca2D->pQr_array_current, offset, value, ca2D->rows, ca2D->columns, substate_index);
		calCudaSetMatrixElement(ca2D->pQr_array_next, offset, value, ca2D->rows, ca2D->columns, substate_index);
}

__device__
	CALbyte calCudaGet2Db(CudaCALModel2D* ca2D, int index, CALint substate_index) {
		return calCudaGetMatrixElement(ca2D->pQb_array_current, index, ca2D->rows, ca2D->columns, substate_index);
}

__device__
	CALint calCudaGet2Di(CudaCALModel2D* ca2D, int index, CALint substate_index) {
		return calCudaGetMatrixElement(ca2D->pQi_array_current, index, ca2D->rows, ca2D->columns, substate_index);
}

__device__
	CALreal calCudaGet2Dr(CudaCALModel2D* ca2D, int index, CALint substate_index) {
		return calCudaGetMatrixElement(ca2D->pQr_array_current, index, ca2D->rows, ca2D->columns, substate_index);
}

__device__
	CALint calGetLinearIndex(int offset, CALint columns, CALint rows, CALint in, CALint jn, CALint substate_index){
		if(offset % columns == 0 && jn < 0)
			return -1;

		if(offset % columns == columns - 1 && jn > 0)
			return -1;

		CALint i;
		if(in < 0)
			i = offset - columns;
		else if(in > 0)
			i = offset + columns;
		else
			i = offset;

		if(i < 0 || i >= (rows * columns))
			return -1;

		return (i + jn) + rows*columns*substate_index;
}

__device__
	CALint calGetToroidalLinearIndex(int offset, CALint columns, CALint rows, CALint in, CALint jn, CALint substate_index){

		CALint irow = offset / columns;
		CALint jcolumn = offset % columns;//)%columns;

		CALint toroidal_i = calGetToroidalX(irow + in, rows);
		CALint toroidal_j = calGetToroidalX(jcolumn + jn, columns);

		return (toroidal_i * columns + toroidal_j) + (rows*columns*substate_index);
}

__device__
	CALbyte calCudaGetX2Db(struct CudaCALModel2D* ca2D, int offset, int n, CALint substate_index)
{
	CALint i = calCudaGetIndexRow(ca2D, offset),j = calCudaGetIndexColumn(ca2D, offset);

	if ((ca2D->X_id == CAL_HEXAGONAL_NEIGHBORHOOD_2D && j%2 ==1) || (ca2D->X_id == CAL_HEXAGONAL_NEIGHBORHOOD_ALT_2D && i%2 ==1))
		n += CAL_HEXAGONAL_SHIFT;

	if (ca2D->T == CAL_SPACE_FLAT){
		int index = calGetLinearIndex(offset, ca2D->columns, ca2D->rows,ca2D->i[n],ca2D->j[n], substate_index);
		return calCudaGetMatrixElement_(ca2D->pQb_array_current, index, ca2D->rows, ca2D->columns, substate_index);
	}
	else{ 
		//è toroidale
		int index = calGetToroidalLinearIndex(offset, ca2D->columns, ca2D->rows,ca2D->i[n],ca2D->j[n], substate_index);
		return calCudaGetMatrixElement_(ca2D->pQb_array_current, index, ca2D->rows, ca2D->columns, substate_index);
	}

}

__device__
	CALint calCudaGetX2Di(struct CudaCALModel2D* ca2D, int offset, int n, CALint substate_index)
{
	CALint i = calCudaGetIndexRow(ca2D, offset),j = calCudaGetIndexColumn(ca2D, offset);

	if ((ca2D->X_id == CAL_HEXAGONAL_NEIGHBORHOOD_2D && j%2 ==1) || (ca2D->X_id == CAL_HEXAGONAL_NEIGHBORHOOD_ALT_2D && i%2 ==1))
		n += CAL_HEXAGONAL_SHIFT;

	if (ca2D->T == CAL_SPACE_FLAT){
		int index = calGetLinearIndex(offset, ca2D->columns, ca2D->rows,ca2D->i[n],ca2D->j[n], substate_index);
		return calCudaGetMatrixElement_(ca2D->pQi_array_current, index, ca2D->rows, ca2D->columns, substate_index);
	}
	else{ 
		//è toroidale
		int index = calGetToroidalLinearIndex(offset, ca2D->columns, ca2D->rows,ca2D->i[n],ca2D->j[n], substate_index);
		return calCudaGetMatrixElement_(ca2D->pQi_array_current, index, ca2D->rows, ca2D->columns, substate_index);
	}

}

__device__
	CALreal calCudaGetX2Dr(struct CudaCALModel2D* ca2D, int offset, int n, CALint substate_index)
{
	CALint i = calCudaGetIndexRow(ca2D, offset),j = calCudaGetIndexColumn(ca2D, offset);

	if ((ca2D->X_id == CAL_HEXAGONAL_NEIGHBORHOOD_2D && j%2 ==1) || (ca2D->X_id == CAL_HEXAGONAL_NEIGHBORHOOD_ALT_2D && i%2 ==1))
		n += CAL_HEXAGONAL_SHIFT;

	if (ca2D->T == CAL_SPACE_FLAT){
		int index = calGetLinearIndex(offset, ca2D->columns, ca2D->rows,ca2D->i[n],ca2D->j[n], substate_index);
		return calCudaGetMatrixElement_(ca2D->pQr_array_current, index, ca2D->rows, ca2D->columns, substate_index);
	}
	else{ 
		//è toroidale
		int index = calGetToroidalLinearIndex(offset, ca2D->columns, ca2D->rows,ca2D->i[n],ca2D->j[n], substate_index);
		return calCudaGetMatrixElement_(ca2D->pQr_array_current, index, ca2D->rows, ca2D->columns, substate_index);
	}

}

__device__
	void calCudaSet2Db(struct CudaCALModel2D* ca2D, int index, CALbyte value, CALint substate_index){
		calCudaSetMatrixElement(ca2D->pQb_array_next, index, value, ca2D->rows, ca2D->columns, substate_index);
}

__device__
	void calCudaSet2Di(struct CudaCALModel2D* ca2D, int index, CALint value, CALint substate_index ){
		calCudaSetMatrixElement(ca2D->pQi_array_next, index, value, ca2D->rows, ca2D->columns, substate_index);
}

__device__
	void calCudaSet2Dr(struct CudaCALModel2D* ca2D, int index, CALreal value, CALint substate_index ){
		calCudaSetMatrixElement(ca2D->pQr_array_next, index, value, ca2D->rows, ca2D->columns, substate_index);
}

__device__
	void calCudaSetCurrent2Db(struct CudaCALModel2D* ca2D, int index, CALbyte value, CALint substate_index){
		calCudaSetMatrixElement(ca2D->pQb_array_current, index, value, ca2D->rows, ca2D->columns,substate_index);
}

__device__
	void calCudaSetCurrent2Di(struct CudaCALModel2D* ca2D, int index, CALint value, CALint substate_index){
		calCudaSetMatrixElement(ca2D->pQi_array_current, index, value, ca2D->rows, ca2D->columns,substate_index);
}

__device__
	void calCudaSetCurrent2Dr(struct CudaCALModel2D* ca2D, int index, CALreal value, CALint substate_index){
		calCudaSetMatrixElement(ca2D->pQr_array_current, index, value, ca2D->rows, ca2D->columns,substate_index);
}

void calCudaFinalize2D(struct CudaCALModel2D* ca2D, struct CudaCALModel2D* device_ca2D)
{
	cudaFreeHost(ca2D->activecell_flags);
	cudaFreeHost(ca2D->activecell_index);
	//	cudaFreeHost(ca2D->j_activecell);

	cudaFreeHost(ca2D->i);
	cudaFreeHost(ca2D->j);

	cudaFreeHost(ca2D->pQb_array_current);
	cudaFreeHost(ca2D->pQb_array_next);

	cudaFreeHost(ca2D->pQi_array_current);
	cudaFreeHost(ca2D->pQi_array_next);

	cudaFreeHost(ca2D->pQr_array_current);
	cudaFreeHost(ca2D->pQr_array_next);

	cudaFreeHost(ca2D->elementary_processes);

	cudaFreeHost(ca2D);
	cudaFree(device_ca2D);

	ca2D = NULL;
}


__device__
	void calCudaStop(struct CudaCALModel2D* ca2D){
		ca2D->stop = CAL_TRUE;
}


__device__
	void calCudaSetStop(struct CudaCALModel2D* ca2D, CALbyte flag){
		ca2D->stop = flag;
}
