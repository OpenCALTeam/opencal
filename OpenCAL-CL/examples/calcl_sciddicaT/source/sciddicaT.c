// The SciddicaT debris flows XCA simulation model

#include <OpenCAL-CL/calcl2D.h>
#include <OpenCAL/cal2DIO.h>
#include <stdlib.h>
#include <time.h>

// Some definitions...
#define ROWS 610
#define COLS 496
#define P_R 0.5
#define P_EPSILON 0.001
#define STEPS 4000
#define DEM_PATH "./data/dem.txt"
#define SOURCE_PATH "./data/source.txt"
#define KERNEL_SRC "./kernel/source/"
#define KERNEL_INC "./kernel/include/"
#define KERNEL_SRC_AC "./kernelActive/source/"
#define KERNEL_INC_AC "./kernelActive/include/"
#define OUTPUT_PATH "./data/width_final.txt"
#define NUMBER_OF_OUTFLOWS 4
#define ACTIVE_CELLS
// Defining kernels' names
#define KERNEL_ELEM_PROC_FLOW_COMPUTATION "flowsComputation"
#define KERNEL_ELEM_PROC_WIDTH_UPDATE "widthUpdate"
#define KERNEL_STEERING  "steering"
#ifdef ACTIVE_CELLS
#define KERNEL_ELEM_PROC_RM_ACT_CELLS "removeInactiveCells"
#endif


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

int main()
{
	time_t start_time, end_time;

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
	calclGetPlatformAndDeviceFromStdIn(calcl_device_manager, &device);
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

  // Register transition function's elementary processes kernels
	calclAddElementaryProcess2D(device_CA, &kernel_elem_proc_flow_computation);
	calclAddElementaryProcess2D(device_CA, &kernel_elem_proc_width_update);
	calclAddSteeringFunc2D(device_CA, &kernel_steering);
#ifdef ACTIVE_CELLS
	calclSetKernelArg2D(&kernel_elem_proc_rm_act_cells, 0, sizeof(CALCLmem), &bufferEpsilonParameter);
	calclAddElementaryProcess2D(device_CA, &kernel_elem_proc_rm_act_cells);
#endif

	// Simulation run
	start_time = time(NULL);
	calclRun2D(device_CA, 1, STEPS);
	end_time = time(NULL);
	printf("%lds", end_time - start_time);

	// Saving results
	calSaveSubstate2Dr(host_CA, Q.h, OUTPUT_PATH);

	// Finalizations
	calclFinalizeManager(calcl_device_manager);
	calclFinalize2D(device_CA);
	calFinalize2D(host_CA);

	return 0;
}
