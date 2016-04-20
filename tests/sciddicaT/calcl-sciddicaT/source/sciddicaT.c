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

#include <OpenCAL-CL/calcl2D.h>
#include <OpenCAL/cal2DIO.h>

#include <stdlib.h>
#include <time.h>

//-----------------------------------------------------------------------
//   THE sciddicaT (Toy model) CELLULAR AUTOMATON
//-----------------------------------------------------------------------

#define ROWS 610
#define COLS 496
#define P_R 0.5
#define P_EPSILON 0.001
#define STEPS 200
#define DEM_PATH "./testData/sciddicaT-data/dem.txt"
#define SOURCE_PATH "./testData/sciddicaT-data/source.txt"
#define KERNEL_SRC "./sciddicaT/calcl-sciddicaT/kernel/source/"
#define KERNEL_INC "./sciddicaT/calcl-sciddicaT/kernel/include/"
#define KERNEL_SRC_AC "./sciddicaT/calcl-sciddicaT/kernelActive/source/"
#define KERNEL_INC_AC "./sciddicaT/calcl-sciddicaT/kernelActive/include/"

#define ACTIVE_CELLS

#define NUMBER_OF_OUTFLOWS 4
struct sciddicaTSubstates {
	struct CALSubstate2Dr *f[NUMBER_OF_OUTFLOWS];
	struct CALSubstate2Dr *z;
	struct CALSubstate2Dr *h;
};

struct sciddicaTParameters {
	CALParameterr epsilon;
	CALParameterr r;
};

//cadef
struct CALModel2D* sciddicaT;						//the cellular automaton
struct sciddicaTSubstates Q;						//the substates
struct sciddicaTParameters P;						//the parameters

//defining kernels' names
#define KERNEL_ELEM_PROC_FLOW_COMPUTATION "sciddicaT_flows_computation"
#define KERNEL_ELEM_PROC_WIDTH_UPDATE "sciddicaT_width_update"
#define KERNEL_STEERING  "sciddicaTSteering"
#ifdef ACTIVE_CELLS
#define KERNEL_ELEM_PROC_RM_ACT_CELLS "sciddicaT_remove_inactive_cells"
#endif



void sciddicaTSimulationInit(struct CALModel2D* sciddicaT) {
	CALreal z, h;
	CALint i, j;

	//initializing substates to 0
	calInitSubstate2Dr(sciddicaT, Q.f[0], 0);
	calInitSubstate2Dr(sciddicaT, Q.f[1], 0);
	calInitSubstate2Dr(sciddicaT, Q.f[2], 0);
	calInitSubstate2Dr(sciddicaT, Q.f[3], 0);

	//sciddicaT parameters setting
	P.r = P_R;
	P.epsilon = P_EPSILON;

	//sciddicaT source initialization
	for (i = 0; i < sciddicaT->rows; i++)
		for (j = 0; j < sciddicaT->columns; j++) {
			h = calGet2Dr(sciddicaT, Q.h, i, j);

			if (h > 0.0) {
				z = calGet2Dr(sciddicaT, Q.z, i, j);
				calSet2Dr(sciddicaT, Q.z, i, j, z - h);

#ifdef ACTIVE_CELLS
				//adds the cell (i, j) to the set of active ones
				calAddActiveCell2D(sciddicaT, i, j);
#endif
			}
		}
}


#define PREFIX_PATH(version,name,pathVarName) \
	if(version==0)\
		 pathVarName="./testsout/serial/"name;\
	 else if(version>0)\
		 pathVarName="./testsout/other/"name;

int main(int argc, char** argv) {

    time_t start_time, end_time;

	int platformNum = 0;
	int deviceNum = 0;

	CALOpenCL * calOpenCL;
	CALCLcontext context;
	CALCLdevice device;
	CALCLprogram program;
	CALCLToolkit2D * sciddicaToolkit;
#ifdef ACTIVE_CELLS
	char * kernelSrc = KERNEL_SRC_AC;
	char * kernelInc = KERNEL_INC_AC;
#else
	char * kernelSrc = KERNEL_SRC;
	char * kernelInc = KERNEL_INC;
#endif
	CALCLkernel kernel_elem_proc_flow_computation;
	CALCLkernel kernel_elem_proc_width_update;
	CALCLkernel kernel_elem_proc_rm_act_cells;
	CALCLkernel kernel_steering;
	CALCLmem * buffersKernelFlowComp;
	CALCLmem bufferEpsilonParameter;
	CALCLmem bufferRParameter;



	calOpenCL = calclCreateCALOpenCL();
	calclInitializePlatforms(calOpenCL);
	calclInitializeDevices(calOpenCL);
	calclPrintAllPlatformAndDevices(calOpenCL);

	device = calclGetDevice(calOpenCL, platformNum, deviceNum);
	context = calclcreateContext(&device, 1);
	program = calclLoadProgramLib2D(context, device, kernelSrc, kernelInc);


	//cadef
#ifdef ACTIVE_CELLS
	sciddicaT = calCADef2D(ROWS, COLS, CAL_VON_NEUMANN_NEIGHBORHOOD_2D, CAL_SPACE_TOROIDAL, CAL_OPT_ACTIVE_CELLS);
#else
	sciddicaT = calCADef2D(ROWS, COLS, CAL_VON_NEUMANN_NEIGHBORHOOD_2D, CAL_SPACE_TOROIDAL, CAL_NO_OPT);
#endif

	//add substates
	Q.f[0] = calAddSubstate2Dr(sciddicaT);
	Q.f[1] = calAddSubstate2Dr(sciddicaT);
	Q.f[2] = calAddSubstate2Dr(sciddicaT);
	Q.f[3] = calAddSubstate2Dr(sciddicaT);
	Q.z = calAddSubstate2Dr(sciddicaT);
	Q.h = calAddSubstate2Dr(sciddicaT);

	//load configuration
	calLoadSubstate2Dr(sciddicaT, Q.z, DEM_PATH);
	calLoadSubstate2Dr(sciddicaT, Q.h, SOURCE_PATH);

	//initialization
	sciddicaTSimulationInit(sciddicaT);
	calUpdate2D(sciddicaT);

	//calcl toolkit
#ifdef ACTIVE_CELLS
	sciddicaToolkit = calclCreateToolkit2D(sciddicaT, context, program, device, CAL_OPT_ACTIVE_CELLS);
#else
	sciddicaToolkit = calclCreateToolkit2D(sciddicaT, context, program, device, CAL_NO_OPT);
#endif


	//calcl kernels
	kernel_elem_proc_flow_computation = calclGetKernelFromProgram(&program, KERNEL_ELEM_PROC_FLOW_COMPUTATION);
	kernel_elem_proc_width_update = calclGetKernelFromProgram(&program, KERNEL_ELEM_PROC_WIDTH_UPDATE);
#ifdef ACTIVE_CELLS
	kernel_elem_proc_rm_act_cells = calclGetKernelFromProgram(&program, KERNEL_ELEM_PROC_RM_ACT_CELLS);
#endif
	kernel_steering = calclGetKernelFromProgram(&program, KERNEL_STEERING);


	buffersKernelFlowComp = (CALCLmem *) malloc(sizeof(CALCLmem) * 2);
	bufferEpsilonParameter = calclCreateBuffer(context, &P.epsilon, sizeof(CALParameterr));
	bufferRParameter = calclCreateBuffer(context, &P.r, sizeof(CALParameterr));
	buffersKernelFlowComp[0] = bufferEpsilonParameter;
	buffersKernelFlowComp[1] = bufferRParameter;

	calclSetCALKernelArgs2D(&kernel_elem_proc_flow_computation, buffersKernelFlowComp, 2);
	calclAddElementaryProcessKernel2D(sciddicaToolkit, sciddicaT, &kernel_elem_proc_flow_computation);
	calclAddElementaryProcessKernel2D(sciddicaToolkit, sciddicaT, &kernel_elem_proc_width_update);
	calclSetSteeringKernel2D(sciddicaToolkit, sciddicaT, &kernel_steering);
#ifdef ACTIVE_CELLS
	calclSetCALKernelArgs2D(&kernel_elem_proc_rm_act_cells, &bufferEpsilonParameter, 1);
	calclAddElementaryProcessKernel2D(sciddicaToolkit, sciddicaT, &kernel_elem_proc_rm_act_cells);
#endif

	//simulation execution
	start_time = time(NULL);
	calclRun2D(sciddicaToolkit, sciddicaT, STEPS);
	end_time = time(NULL);
	printf("%d", end_time - start_time);



	//saving configuration
	calSaveSubstate2Dr(sciddicaT, Q.h,"./testsout/other/1.txt");

	//finalizations
	calFinalize2D(sciddicaT);
	calclFinalizeCALOpencl(calOpenCL);
	calclFinalizeToolkit2D(sciddicaToolkit);

	return 0;
}
