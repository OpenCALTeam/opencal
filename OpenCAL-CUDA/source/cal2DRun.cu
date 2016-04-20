/*
 * Copyright (c) 2016 OpenCALTeam (https://github.com/OpenCALTeam),
 * University of Calabria, Italy.
 *
 * This file is part of OpenCAL (Open Computing Abstraction Layer).
 *
 * OpenCAL is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3 of
 * the License, or (at your option) any later version.
 *
 * OpenCAL is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with OpenCAL. If not, see <http://www.gnu.org/licenses/>.
 */

#include ".\..\include\cal2DRun.cuh"
#include <stdlib.h>
#include <stdio.h>

struct CudaCALRun2D* calCudaRunDef2D(
struct CudaCALModel2D* device_ca2D,
struct CudaCALModel2D* ca2D,
	int initial_step,
	int final_step,
	enum CALUpdateMode UPDATE_MODE)
{
	struct CudaCALRun2D* simulation = 0;//(struct CudaCALRun2D*)malloc(sizeof(struct CudaCALRun2D));
	cudaMallocHost((void**)&simulation, sizeof(struct CudaCALRun2D), cudaHostAllocDefault);
	if (!simulation)
		return NULL;

	simulation->ca2D = ca2D;
	simulation->device_ca2D = device_ca2D;
	simulation->h_device_ca2D = calCudaAllocatorModel(ca2D);

	cudaMalloc((void**)&simulation->device_array_of_index_dim,sizeof(unsigned int)*ca2D->rows*ca2D->columns);

	simulation->step = 0;
	simulation->initial_step = initial_step;
	simulation->final_step = final_step;

	simulation->UPDATE_MODE = UPDATE_MODE;

	simulation->init = NULL;
	simulation->globalTransition = NULL;
	simulation->steering = NULL;
	simulation->stopCondition = NULL;
	simulation->finalize = NULL;

	return simulation;
}

void calCudaRunAddInitFunc2D(struct CudaCALRun2D* simulation, void (*init)(struct CudaCALModel2D*))
{
	simulation->init = init;
}

void calCudaRunAddGlobalTransitionFunc2D(struct CudaCALRun2D* simulation, void (*globalTransition)(struct CudaCALModel2D*))
{
	simulation->globalTransition = globalTransition;
}

void calCudaRunAddSteeringFunc2D(struct CudaCALRun2D* simulation, void (*steering)(struct CudaCALModel2D*))
{
	simulation->steering = steering;
}

void calCudaRunAddStopConditionFunc2D(struct CudaCALRun2D* simulation, void (*stopCondition)(struct CudaCALModel2D*))
{
	simulation->stopCondition = stopCondition;
}

void calCudaRunAddFinalizeFunc2D(struct CudaCALRun2D* simulation, void (*finalize)(struct CudaCALModel2D*))
{
	simulation->finalize = finalize;
}


void calCudaRunInitSimulation2D(struct CudaCALRun2D* simulation, dim3 grid, dim3 block)
{
	if (simulation->init)
	{
		simulation->init<<<grid,block>>>(simulation->device_ca2D);
		if (simulation->UPDATE_MODE == CAL_UPDATE_IMPLICIT)
			calCudaUpdate2D(simulation);
	}
}

CALbyte calCudaRunCAStep2D(struct CudaCALRun2D* simulation, dim3 grid, dim3 block)
{
	if (simulation->globalTransition)
	{
		//aggiungere celle attive
		simulation->globalTransition<<<grid,block>>>(simulation->device_ca2D);
		if (simulation->UPDATE_MODE == CAL_UPDATE_IMPLICIT)
			calCudaUpdate2D(simulation);
	}
	else
		calCudaGlobalTransitionFunction2D(simulation, grid, block);
	//No explicit substates and active cells updates are needed in this case

	if (simulation->steering)
	{
		if (simulation->ca2D->OPTIMIZATION == CAL_OPT_ACTIVE_CELLS){
			simulation->ca2D->activecell_size_current = simulation->h_device_ca2D->activecell_size_next;
			CALint num_blocks = simulation->ca2D->activecell_size_current / ((block.x) * (block.y));
			grid.x = (num_blocks+2);
			grid.y = 1;
			//calCudaPerformGridAndBlockForStreamCompaction2D(simulation, grid, block);
			simulation->steering<<<grid,block>>>(simulation->device_ca2D);
		}else{
			simulation->steering<<<grid,block>>>(simulation->device_ca2D);
		}
		if (simulation->UPDATE_MODE == CAL_UPDATE_IMPLICIT)
			calCudaUpdate2D(simulation);
	}

	if (simulation->stopCondition){

		if (simulation->ca2D->OPTIMIZATION == CAL_OPT_ACTIVE_CELLS){
			simulation->ca2D->activecell_size_current = simulation->h_device_ca2D->activecell_size_next;
			CALint num_blocks = simulation->ca2D->activecell_size_current / ((block.x) * (block.y));
			grid.x = (num_blocks+2);
			grid.y = 1;
			//calCudaPerformGridAndBlockForStreamCompaction2D(simulation, grid, block);
			simulation->stopCondition<<<grid,block>>>(simulation->device_ca2D);
		}else{
			simulation->stopCondition<<<grid,block>>>(simulation->device_ca2D);
		}

		cudaMemcpy(simulation->h_device_ca2D, simulation->device_ca2D, sizeof(struct CudaCALModel2D), cudaMemcpyDeviceToHost);
		if(simulation->h_device_ca2D->stop)
			return CAL_FALSE;

	}
	return CAL_TRUE;
}

void calCudaRunFinalizeSimulation2D(struct CudaCALRun2D* simulation, dim3 grid, dim3 block)
{
	if (simulation->finalize)
	{
		simulation->finalize<<<grid,block>>>(simulation->device_ca2D);
		if (simulation->UPDATE_MODE == CAL_UPDATE_IMPLICIT)
			calCudaUpdate2D(simulation);
	}
}

void calCudaRun2D(struct CudaCALRun2D* simulation, dim3 grid, dim3 block)
{
	CALbyte again;

	calCudaRunInitSimulation2D(simulation,grid,block);

	for (simulation->step = simulation->initial_step; (simulation->step <= simulation->final_step || simulation->final_step == CAL_RUN_LOOP); simulation->step++)
	{
		again = calCudaRunCAStep2D(simulation, grid, block);
		if (!again)
			break;
	}

	calCudaRunFinalizeSimulation2D(simulation, grid, block);
}

void calCudaRunFinalize2D(struct CudaCALRun2D* cal2DRun)
{
	//Note that cal2DRun->ca2D and cal2DRun->device_ca2D MUST NOT BE DEALLOCATED as it is not allocated within cal2DRun.
	calCudaFreeModel2D(cal2DRun->h_device_ca2D);
	cudaFree(cal2DRun);
	cal2DRun = NULL;
}
