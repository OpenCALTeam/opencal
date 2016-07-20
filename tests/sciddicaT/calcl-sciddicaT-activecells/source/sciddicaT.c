/*
 * Copyright (c) 2016 OpenCALTeam (https://github.com/OpenCALTeam),
 * Telesio Research Group,
 * Department of Mathematics and Computer Science,
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
#include <OpenCALTime.h>
#include <stdlib.h>
#include <time.h>

//-----------------------------------------------------------------------
//   THE sciddicaT (Toy model) CELLULAR AUTOMATON
//-----------------------------------------------------------------------
#define ACTIVE_CELLS
#define ROWS 610
#define COLS 496
#define P_R 0.5
#define P_EPSILON 0.001
#define STEPS 4000
#define DEM_PATH "./testData/sciddicaT-data/dem.txt"
#define SOURCE_PATH "./testData/sciddicaT-data/source.txt"
#define KERNEL_SRC "./sciddicaT/calcl-sciddicaT-activecells/kernel/source/"
#define KERNEL_INC "./sciddicaT/calcl-sciddicaT-activecells/kernel/include/"
#define KERNEL_SRC_AC "./sciddicaT/calcl-sciddicaT-activecells/kernelActive/source/"
#define KERNEL_INC_AC "./sciddicaT/calcl-sciddicaT-activecells/kernelActive/include/"
#define NUMBER_OF_OUTFLOWS 4
#define KERNEL_ELEM_PROC_FLOW_COMPUTATION "flowsComputation"
#define KERNEL_ELEM_PROC_WIDTH_UPDATE "widthUpdate"
#define KERNEL_STEERING  "steering"
#ifdef ACTIVE_CELLS
#define KERNEL_ELEM_PROC_RM_ACT_CELLS "removeInactiveCells"
#endif

int numberOfLoops;


// Declare XCA model (host_CA), substates (Q), parameters (P)
struct CALModel2D* host_CA;

struct sciddicaTSubstates {
	struct CALSubstate2Dr *f[NUMBER_OF_OUTFLOWS];
	struct CALSubstate2Dr *z;
	struct CALSubstate2Dr *h;
} Q;

struct sciddicaTParameters {
	CALParameterr epsilon;
	CALParameterr r;
} P;

// SciddicaT simulation init function
void sciddicaTSimulationInit(struct CALModel2D* host_CA) {
	CALreal z, h;
	CALint i, j;

	//initializing substates to 0
	calInitSubstate2Dr(host_CA, Q.f[0], 0);
	calInitSubstate2Dr(host_CA, Q.f[1], 0);
	calInitSubstate2Dr(host_CA, Q.f[2], 0);
	calInitSubstate2Dr(host_CA, Q.f[3], 0);

	//sciddicaT parameters setting
	P.r = P_R;
	P.epsilon = P_EPSILON;

	//sciddicaT source initialization
	for (i = 0; i < host_CA->rows; i++)
		for (j = 0; j < host_CA->columns; j++) {
			h = calGet2Dr(host_CA, Q.h, i, j);

			if (h > 0.0) {
				z = calGet2Dr(host_CA, Q.z, i, j);
				calSet2Dr(host_CA, Q.z, i, j, z - h);

#ifdef ACTIVE_CELLS
				//adds the cell (i, j) to the set of active ones
				calAddActiveCell2D(host_CA, i, j);
#endif
			}
		}
}

int main(int argc, char** argv)
{
    // read from argv the number of steps
    int steps;
    if (sscanf (argv[2], "%i", &steps)!=1 && steps >=0) {
        printf ("number of steps is not an integer");
        exit(-1);
    }

		// read from argv the number of steps
    if (sscanf (argv[3], "%i", &numberOfLoops)!=1 && numberOfLoops >=0) {
        printf ("number of loops is not an integer");
        exit(-1);
    }

    int platform;
    if (sscanf (argv[4], "%i", &platform)!=1 && platform >=0) {
        printf ("platform number is not an integer");
        exit(-1);
    }
    int deviceNumber;
    if (sscanf (argv[5], "%i", &deviceNumber)!=1 && deviceNumber >=0) {
        printf ("device number is not an integer");
        exit(-1);
    }


	struct CALCLDeviceManager * calcl_device_manager;
	CALCLcontext context;
	CALCLdevice device;
	CALCLprogram program;
	struct CALCLModel2D * device_CA;
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

	calcl_device_manager = calclCreateManager();
	//calclGetPlatformAndDeviceFromStdIn(calcl_device_manager, &device);
    device = calclGetDevice(calcl_device_manager, platform, deviceNumber);
	context = calclCreateContext(&device);
	program = calclLoadProgram2D(context, device, kernelSrc, kernelInc);


	// Define of the host-side CA objects
#ifdef ACTIVE_CELLS
	host_CA = calCADef2D(ROWS, COLS, CAL_VON_NEUMANN_NEIGHBORHOOD_2D, CAL_SPACE_TOROIDAL, CAL_OPT_ACTIVE_CELLS);
#else
	host_CA = calCADef2D(ROWS, COLS, CAL_VON_NEUMANN_NEIGHBORHOOD_2D, CAL_SPACE_TOROIDAL, CAL_NO_OPT);
#endif

	// Add substates
	Q.f[0] = calAddSubstate2Dr(host_CA);
	Q.f[1] = calAddSubstate2Dr(host_CA);
	Q.f[2] = calAddSubstate2Dr(host_CA);
	Q.f[3] = calAddSubstate2Dr(host_CA);
	Q.z = calAddSubstate2Dr(host_CA);
	Q.h = calAddSubstate2Dr(host_CA);

	// Load configuration
	calLoadSubstate2Dr(host_CA, Q.z, DEM_PATH);
	calLoadSubstate2Dr(host_CA, Q.h, SOURCE_PATH);

	// Initialization
	sciddicaTSimulationInit(host_CA);
	calUpdate2D(host_CA);

	// Define a device-side CA
	device_CA = calclCADef2D(host_CA, context, program, device);

	// Extract kernels from program
	kernel_elem_proc_flow_computation = calclGetKernelFromProgram(&program, KERNEL_ELEM_PROC_FLOW_COMPUTATION);
	kernel_elem_proc_width_update = calclGetKernelFromProgram(&program, KERNEL_ELEM_PROC_WIDTH_UPDATE);
#ifdef ACTIVE_CELLS
	kernel_elem_proc_rm_act_cells = calclGetKernelFromProgram(&program, KERNEL_ELEM_PROC_RM_ACT_CELLS);
#endif
	kernel_steering = calclGetKernelFromProgram(&program, KERNEL_STEERING);

	bufferEpsilonParameter = calclCreateBuffer(context, &P.epsilon, sizeof(CALParameterr));
	bufferRParameter = calclCreateBuffer(context, &P.r, sizeof(CALParameterr));

	calclSetKernelArg2D(&kernel_elem_proc_flow_computation, 0, sizeof(CALCLmem), &bufferEpsilonParameter);
	calclSetKernelArg2D(&kernel_elem_proc_flow_computation, 1, sizeof(CALCLmem), &bufferRParameter);
	calclSetKernelArg2D(&kernel_elem_proc_flow_computation, 2, sizeof(int), &numberOfLoops);

	calclSetKernelArg2D(&kernel_elem_proc_width_update, 0, sizeof(int), &numberOfLoops);

  // Register transition function's elementary processes kernels
	calclAddElementaryProcess2D(device_CA, &kernel_elem_proc_flow_computation);
	calclAddElementaryProcess2D(device_CA, &kernel_elem_proc_width_update);
	calclAddSteeringFunc2D(device_CA, &kernel_steering);
#ifdef ACTIVE_CELLS
	calclSetKernelArg2D(&kernel_elem_proc_rm_act_cells, 0, sizeof(CALCLmem), &bufferEpsilonParameter);
	calclAddElementaryProcess2D(device_CA, &kernel_elem_proc_rm_act_cells);
#endif

	// Simulation run
    struct OpenCALTime * opencalTime= (struct OpenCALTime *)malloc(sizeof(struct OpenCALTime));
    startTime(opencalTime);
    calclRun2D(device_CA, 1, steps);
    endTime(opencalTime);

	// Saving results
	calSaveSubstate2Dr(host_CA, Q.h,"./testsout/other/1.txt");

	// Finalizations
	calclFinalizeManager(calcl_device_manager);
	calclFinalize2D(device_CA);
	calFinalize2D(host_CA);

	return 0;
}
