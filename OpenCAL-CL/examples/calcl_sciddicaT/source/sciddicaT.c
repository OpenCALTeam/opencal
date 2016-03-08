#include <OpenCAL/cal2DIO.h>
#include <stdlib.h>
#include <time.h>
#include <OpenCAL-CL/calcl2D.h>

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
#define KERNEL_SRC "./kernel/source/"
#define KERNEL_INC "./kernel/include/"
#define KERNEL_SRC_AC "./kernelActive/source/"
#define KERNEL_INC_AC "./kernelActive/include/"
#define OUTPUT_PATH "./data/width_final.txt"

//#define ACTIVE_CELLS

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
struct CALModel2D* hostCA;						//the cellular automaton
struct sciddicaTSubstates Q;						//the substates
struct sciddicaTParameters P;						//the parameters

//defining kernels' names
#define KERNEL_ELEM_PROC_FLOW_COMPUTATION "sciddicaT_flows_computation"
#define KERNEL_ELEM_PROC_WIDTH_UPDATE "sciddicaT_width_update"
#define KERNEL_STEERING  "sciddicaTSteering"
#ifdef ACTIVE_CELLS
#define KERNEL_ELEM_PROC_RM_ACT_CELLS "sciddicaT_remove_inactive_cells"
#endif



void sciddicaTSimulationInit(struct CALModel2D* hostCA) {
	CALreal z, h;
	CALint i, j;

	//initializing substates to 0
	calInitSubstate2Dr(hostCA, Q.f[0], 0);
	calInitSubstate2Dr(hostCA, Q.f[1], 0);
	calInitSubstate2Dr(hostCA, Q.f[2], 0);
	calInitSubstate2Dr(hostCA, Q.f[3], 0);

	//sciddicaT parameters setting
	P.r = P_R;
	P.epsilon = P_EPSILON;

	//sciddicaT source initialization
	for (i = 0; i < hostCA->rows; i++)
		for (j = 0; j < hostCA->columns; j++) {
			h = calGet2Dr(hostCA, Q.h, i, j);

			if (h > 0.0) {
				z = calGet2Dr(hostCA, Q.z, i, j);
				calSet2Dr(hostCA, Q.z, i, j, z - h);

#ifdef ACTIVE_CELLS
				//adds the cell (i, j) to the set of active ones
				calAddActiveCell2D(hostCA, i, j);
#endif
			}
		}
}



int main(int argc, char** argv) {

	time_t start_time, end_time;

	int platformNum = 0;
	int deviceNum = 0;

	CALCLManager * calOpenCL;
	CALCLcontext context;
	CALCLdevice device;
	CALCLprogram program;
	CALCLModel2D * deviceCA;
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



	calOpenCL = calclCreateManager();
	calclInitializePlatforms(calOpenCL);
	calclInitializeDevices(calOpenCL);
	calclPrintPlatformsAndDevices(calOpenCL);

	device = calclGetDevice(calOpenCL, platformNum, deviceNum);
	context = calclCreateContext(&device, 1);
	program = calclLoadProgram2D(context, device, kernelSrc, kernelInc);


	//cadef
#ifdef ACTIVE_CELLS
	hostCA = calCADef2D(ROWS, COLS, CAL_VON_NEUMANN_NEIGHBORHOOD_2D, CAL_SPACE_TOROIDAL, CAL_OPT_ACTIVE_CELLS);
#else
	hostCA = calCADef2D(ROWS, COLS, CAL_VON_NEUMANN_NEIGHBORHOOD_2D, CAL_SPACE_TOROIDAL, CAL_NO_OPT);
#endif

	//add substates
	Q.f[0] = calAddSubstate2Dr(hostCA);
	Q.f[1] = calAddSubstate2Dr(hostCA);
	Q.f[2] = calAddSubstate2Dr(hostCA);
	Q.f[3] = calAddSubstate2Dr(hostCA);
	Q.z = calAddSubstate2Dr(hostCA);
	Q.h = calAddSubstate2Dr(hostCA);

	//load configuration
	calLoadSubstate2Dr(hostCA, Q.z, DEM_PATH);
	calLoadSubstate2Dr(hostCA, Q.h, SOURCE_PATH);

	//initialization
	sciddicaTSimulationInit(hostCA);
	calUpdate2D(hostCA);

	//calcl toolkit
	deviceCA = calclCADef2D(hostCA, context, program, device);

	//calcl kernels
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

//	calclSetKernelArg2D(&kernel_elem_proc_flow_computation, 0, sizeof(CALParameterr), &P.epsilon);
//	calclSetKernelArg2D(&kernel_elem_proc_flow_computation, 1, sizeof(CALParameterr), &P.r);

	calclAddElementaryProcess2D(deviceCA, hostCA, &kernel_elem_proc_flow_computation);
	calclAddElementaryProcess2D(deviceCA, hostCA, &kernel_elem_proc_width_update);
	calclAddSteeringFunc2D(deviceCA, hostCA, &kernel_steering);
#ifdef ACTIVE_CELLS
	calclSetKernelArg2D(&kernel_elem_proc_rm_act_cells, 0, sizeof(CALCLmem), &bufferEpsilonParameter);
	//calclSetKernelArg2D(&kernel_elem_proc_rm_act_cells, 0, sizeof(CALParameterr), &P.epsilon);
	calclAddElementaryProcess2D(deviceCA, hostCA, &kernel_elem_proc_rm_act_cells);
#endif

	//simulation execution
	start_time = time(NULL);
	calclRun2D(deviceCA, hostCA, 1, STEPS);
	end_time = time(NULL);
	printf("%d", end_time - start_time);

	//saving results
	calSaveSubstate2Dr(hostCA, Q.h, OUTPUT_PATH);

	//finalizations
	calFinalize2D(hostCA);
	calclFinalizeCALOpencl(calOpenCL);
	calclFinalizeToolkit2D(deviceCA);

	return 0;
}
