#include ".\include\cal2D.cuh"
#include ".\include\cal2DIO.cuh"
#include ".\include\cal2DRun.cuh"
#include ".\include\cal2DToolkit.cuh"
#include ".\include\cal2DBuffer.cuh"

#include <stdlib.h>
#include <time.h>

#include "cuda_profiler_api.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

//-----------------------------------------------------------------------
//   THE sciddicaT (Toy model) CELLULAR AUTOMATON
//-----------------------------------------------------------------------

#define ROWS 610
#define COLS 496
#define P_R 0.5
#define P_EPSILON 0.001
#define STEPS 4000
#define DEM_PATH "./data/dem.txt"
#define SOURCE_PATH "./data/source.txt"
#define OUTPUT_PATH "./data/width_final.txt"
#define OUTPUT_PATH_S "./data/width_final_s.txt"

#define ACTIVE_CELLS
#define NUMBER_OF_OUTFLOWS 4
#define NUMBER_OF_SUBSTATES_REAL 6

enum SUBSTATES_NAME{
	DEM = 0, SOURCE, OUTFLOWS_0, OUTFLOWS_1, OUTFLOWS_2, OUTFLOWS_3
};

CALint N = 16;
CALint M = 61;
dim3 block(N,M);
dim3 grid(COLS/block.x, ROWS/block.y);

struct CudaCALRun2D* sciddicaT_simulation;

__global__ void sciddicaT_flows_computation_parallel(struct CudaCALModel2D* sciddicaT)
{
	CALbyte eliminated_cells[5]={CAL_FALSE,CAL_FALSE,CAL_FALSE,CAL_FALSE,CAL_FALSE};
	CALbyte again;
	CALint cells_count;
	CALreal average;
	CALreal m;
	CALreal u[5];
	CALint n;
	CALreal z, h;
	CALint offset = calCudaGetIndex(sciddicaT);
	
	//if(offset == 636 || offset == 635 || offset == 637 || offset == 2500)
		//printf("%d %d \n", offset, sciddicaT->activecell_index[offset]);

#ifdef ACTIVE_CELLS
	//if(!calCudaImAlive(sciddicaT, offset))return;
#endif // ACTIVE_CELLS



	if (calCudaGet2Dr(sciddicaT, offset, SOURCE) <= P_EPSILON)
		return;

	m = calCudaGet2Dr(sciddicaT, offset, SOURCE) - P_EPSILON;
	u[0] = calCudaGet2Dr(sciddicaT, offset, DEM) + P_EPSILON;



	for (n=1; n<sciddicaT->sizeof_X; n++)
	{
		z = calCudaGetX2Dr(sciddicaT, offset, n, DEM);
		h = calCudaGetX2Dr(sciddicaT, offset, n, SOURCE);
		//if(offset == 42590)
		//{
		//printf("M: %3.1f\t U[0]: %3.1f\n",m,u[0]);
		//	printf("%d: z= %f h= %f \n",n,z,h);
		//}

		u[n] = z + h;
	}


	//computes outflows
	do{
		again = CAL_FALSE;
		average = m;
		cells_count = 0;

		for (n=0; n<sciddicaT->sizeof_X; n++)
			if (!eliminated_cells[n]){
				average += u[n];
				cells_count++;
			}

			if (cells_count != 0)
				average /= cells_count;

			for (n=0; n<sciddicaT->sizeof_X; n++)
				if( (average<=u[n]) && (!eliminated_cells[n]) ){
					eliminated_cells[n]=CAL_TRUE;
					again=CAL_TRUE;
				}

	}while (again); 


	if(eliminated_cells[1])
		calCudaSet2Dr(sciddicaT, offset, 0.0, OUTFLOWS_0);
	else{
		calCudaSet2Dr(sciddicaT, offset, (average-u[1])*P_R, OUTFLOWS_0);

#ifdef ACTIVE_CELLS
		//adds the cell (i, j, n) to the set of active ones
		calCudaAddActiveCellX2D(sciddicaT, offset, 1);
#endif
	}
	if(eliminated_cells[2])
		calCudaSet2Dr(sciddicaT, offset, 0.0, OUTFLOWS_1);
	else{
		calCudaSet2Dr(sciddicaT, offset, (average-u[2])*P_R, OUTFLOWS_1);
#ifdef ACTIVE_CELLS
		//adds the cell (i, j, n) to the set of active ones
		calCudaAddActiveCellX2D(sciddicaT, offset, 2);
#endif
	}

	if(eliminated_cells[3])
		calCudaSet2Dr(sciddicaT, offset, 0.0, OUTFLOWS_2);
	else{
		calCudaSet2Dr(sciddicaT, offset, (average-u[3])*P_R, OUTFLOWS_2);
#ifdef ACTIVE_CELLS
		//adds the cell (i, j, n) to the set of active ones
		calCudaAddActiveCellX2D(sciddicaT, offset, 3);
#endif
	}

	if(eliminated_cells[4])
		calCudaSet2Dr(sciddicaT, offset, 0.0, OUTFLOWS_3);
	else{
		calCudaSet2Dr(sciddicaT, offset, (average-u[4])*P_R, OUTFLOWS_3);
#ifdef ACTIVE_CELLS
		//adds the cell (i, j, n) to the set of active ones
		calCudaAddActiveCellX2D(sciddicaT, offset, 4);
#endif
	}

}

__global__ void sciddicaT_width_update_parallel(struct CudaCALModel2D* sciddicaT)
{
	CALreal h_next;
	CALint n, offset = calCudaGetIndex(sciddicaT);
#ifdef ACTIVE_CELLS
	//if(!calCudaImAlive(sciddicaT, offset))return;
#endif // ACTIVE_CELLS
	h_next = calCudaGet2Dr(sciddicaT, offset, SOURCE);

	h_next += calCudaGetX2Dr(sciddicaT, offset,1, OUTFLOWS_3) 
		- calCudaGet2Dr(sciddicaT, offset, OUTFLOWS_0);
	h_next += calCudaGetX2Dr(sciddicaT, offset,2, OUTFLOWS_2)
		- calCudaGet2Dr(sciddicaT, offset, OUTFLOWS_1);
	h_next += calCudaGetX2Dr(sciddicaT, offset,3, OUTFLOWS_1)
		- calCudaGet2Dr(sciddicaT, offset, OUTFLOWS_2);
	h_next += calCudaGetX2Dr(sciddicaT, offset,4, OUTFLOWS_0)
		- calCudaGet2Dr(sciddicaT, offset, OUTFLOWS_3);

	calCudaSet2Dr(sciddicaT, offset, h_next, SOURCE);
}

__global__ void sciddicaT_remove_inactive_cells(struct CudaCALModel2D* sciddicaT)
{
	CALint offset = calCudaGetIndex(sciddicaT);
#ifdef ACTIVE_CELLS
	if (calCudaGet2Dr(sciddicaT, offset, SOURCE) <= P_EPSILON)
		calCudaRemoveActiveCell2D(sciddicaT,offset);	
#endif
}

__global__ void sciddicaT_simulation_init_parallel(struct CudaCALModel2D* sciddicaT)
{
	CALreal z, h;
	CALint i, j, offset = calCudaGetSimpleOffset();

	//initializing substates to 0
	calCudaInit2Dr(sciddicaT,offset,0,OUTFLOWS_0);
	calCudaInit2Dr(sciddicaT,offset,0,OUTFLOWS_1);
	calCudaInit2Dr(sciddicaT,offset,0,OUTFLOWS_2);
	calCudaInit2Dr(sciddicaT,offset,0,OUTFLOWS_3);

	//sciddicaT parameters setting
	//P.r = P_R;
	//P.epsilon = P_EPSILON;

	//sciddicaT source initialization
	h = calCudaGet2Dr(sciddicaT, offset, SOURCE);
	if ( h > 0.0 ) {
		z = calCudaGet2Dr(sciddicaT, offset, DEM);
		calCudaSet2Dr(sciddicaT, offset, z-h, DEM);

#ifdef ACTIVE_CELLS
		//adds the cell (i, j) to the set of active ones
		calCudaAddActiveCell2D(sciddicaT, offset);
#endif
	}
}

__global__ void sciddicaTSteering_parallel(struct CudaCALModel2D* sciddicaT)
{
	CALint offset = calCudaGetIndex(sciddicaT);
#ifdef ACTIVE_CELLS
	//if(!calCudaImAlive(sciddicaT, offset))return;
#endif // ACTIVE_CELLS
	//initializing substates to 0
	calCudaInit2Dr(sciddicaT,offset,0,OUTFLOWS_0);
	calCudaInit2Dr(sciddicaT,offset,0,OUTFLOWS_1);
	calCudaInit2Dr(sciddicaT,offset,0,OUTFLOWS_2);
	calCudaInit2Dr(sciddicaT,offset,0,OUTFLOWS_3);
}

__global__ void sciddicaTStop_parallel(struct CudaCALModel2D* sciddicaT)
{
	CALint offset = calCudaGetIndex(sciddicaT);
#ifdef ACTIVE_CELLS
	//if(!calCudaImAlive(sciddicaT, offset))return;
#endif // ACTIVE_CELLS
		
	if (calCudaGet2Dr(sciddicaT, offset, SOURCE) > P_EPSILON)
		//calCudaStop(sciddicaT);
	//else{
		calCudaSetStop(sciddicaT, CAL_FALSE);
	//}
}

int main()
{
	time_t start_time, end_time;
	cudaProfilerStart();
	//parallel
	//cadef and rundef
	struct CudaCALModel2D* sciddicaT;
#ifdef ACTIVE_CELLS
	sciddicaT = calCudaCADef2D (ROWS, COLS, CAL_VON_NEUMANN_NEIGHBORHOOD_2D, CAL_SPACE_TOROIDAL, CAL_OPT_ACTIVE_CELLS);
#else  
	sciddicaT = calCudaCADef2D (ROWS, COLS, CAL_VON_NEUMANN_NEIGHBORHOOD_2D, CAL_SPACE_TOROIDAL, CAL_NO_OPT);
#endif
	struct CudaCALModel2D* device_sciddicaT = calCudaAlloc();

	//add transition function's elementary processes
	calCudaAddElementaryProcess2D(sciddicaT, sciddicaT_flows_computation_parallel);
	calCudaAddElementaryProcess2D(sciddicaT, sciddicaT_width_update_parallel);
#ifdef ACTIVE_CELLS
	calCudaAddElementaryProcess2D(sciddicaT, sciddicaT_remove_inactive_cells);
#endif
	//add substates
	calCudaAddSubstate2Dr(sciddicaT,NUMBER_OF_SUBSTATES_REAL);

	//load configuration
	calCudaLoadSubstate2Dr(sciddicaT, DEM_PATH, DEM);
	calCudaLoadSubstate2Dr(sciddicaT, SOURCE_PATH, SOURCE);

	calInitializeInGPU2D(sciddicaT,device_sciddicaT);
	
	cudaErrorCheck("Data initialized on device\n");

	sciddicaT_simulation = calCudaRunDef2D(device_sciddicaT,sciddicaT, 1, STEPS, CAL_UPDATE_IMPLICIT);
	
	//simulation run
	calCudaRunAddInitFunc2D(sciddicaT_simulation, sciddicaT_simulation_init_parallel);
	calCudaRunAddSteeringFunc2D(sciddicaT_simulation, sciddicaTSteering_parallel);
	//calCudaRunAddStopConditionFunc2D(sciddicaT_simulation, sciddicaTStop_parallel);

	printf ("Starting simulation...\n");
	start_time = time(NULL);
	calCudaRun2D(sciddicaT_simulation, grid, block);
	//send data to CPU
	calSendDataGPUtoCPU(sciddicaT,device_sciddicaT);

	cudaErrorCheck("Final configuration sent to CPU\n");
	end_time = time(NULL);
	printf ("Simulation terminated.\nElapsed time: %d\n", end_time-start_time);

	//saving configuration
	calCudaSaveSubstate2Dr(sciddicaT, OUTPUT_PATH, SOURCE);

	cudaErrorCheck("Data saved on output file\n");

	//finalizations
	calCudaRunFinalize2D(sciddicaT_simulation);
	calCudaFinalize2D(sciddicaT, device_sciddicaT);
	cudaProfilerStop();
	system("pause");
	return 0;
}