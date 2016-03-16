	// The SciddicaT debris flows CCA simulation model width_final
// a 3D graphic visualizer in OpenCAL-GL

#include <OpenCAL/cal2DIO.h>
#include <OpenCAL-GL/calgl2D.h>
#include <OpenCAL-GL/calgl2DWindow.h>
#include <OpenCAL-CL/calgl2DUpdaterCL.h>
#include <stdlib.h>
#include <OpenCAL-CL/calcl2D.h>

// Some definitions...
#define P_R 0.5
#define P_EPSILON 0.001
#define NUMBER_OF_OUTFLOWS 4
#define DEM "./data/dem.txt"
#define SOURCE "./data/source.txt"
#define FINAL "./data/width_final.txt"
#define ROWS 610
#define COLUMNS 496
#define STEPS 4000
#define DEM_PATH "./data/dem.txt"
#define SOURCE_PATH "./data/source.txt"
#define KERNEL_SRC "./kernel/source/"
#define KERNEL_INC "./kernel/include/"
#define KERNEL_SRC_AC "./kernelActive/source/"
#define KERNEL_INC_AC "./kernelActive/include/"
#define OUTPUT_PATH "./data/width_final.txt"

// declare CCA model (sciddicaT), substates (Q), parameters (P),
// and simulation object (sciddicaT_simulation)
struct sciddicaTSubstates {
	struct CALSubstate2Dr *z;
	struct CALSubstate2Dr *h;
	struct CALSubstate2Dr *f[NUMBER_OF_OUTFLOWS];
};

struct sciddicaTParameters {
	CALParameterr epsilon;
	CALParameterr r;
};

struct CALModel2D* hostCA;						//the cellular automaton
struct sciddicaTSubstates Q;						//the substates
struct sciddicaTParameters P;						//the parameters
struct CALCLManager * calOpenCL;
struct CALCLModel2D * deviceCA;

//defining kernels' names
#define KERNEL_ELEM_PROC_FLOW_COMPUTATION "sciddicaT_flows_computation"
#define KERNEL_ELEM_PROC_WIDTH_UPDATE "sciddicaT_width_update"
#define KERNEL_STEERING  "sciddicaTSteering"
#ifdef ACTIVE_CELLS
#define KERNEL_ELEM_PROC_RM_ACT_CELLS "sciddicaT_remove_inactive_cells"
#endif

// SciddicaT exit function
void exitFunction(void)
{
	// saving configuration
	calSaveSubstate2Dr (hostCA, Q.h, FINAL);

	// finalizations
	//calRunFinalize2D (sciddicaTsimulation);
	calclFinalizeCALOpencl(calOpenCL);
	calclFinalizeToolkit2D(deviceCA);
	calFinalize2D (hostCA);
}


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
void callback(struct CALModel2D* hostCA) {
	system("clear");
printf("********************************************************************");
}



int main(int argc, char** argv)
{
	//OpenCL definition
	int platformNum = 0;
	int deviceNum = 0;

	CALCLcontext context;
	CALCLdevice device;
	CALCLprogram program;

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
	//OpenCL definition end


	struct CALGLDrawModel2D* draw_model3D = NULL;
	struct CALGLDrawModel2D* draw_model2D;

	atexit(exitFunction);

	calglInitViewer("SciddicaT OpenCAL-GL visualizer", 5, 800, 600, 10, 10, CAL_TRUE, 0);

	//cadef and rundef
	//cadef
#ifdef ACTIVE_CELLS
	hostCA = calCADef2D(ROWS, COLUMNS, CAL_VON_NEUMANN_NEIGHBORHOOD_2D, CAL_SPACE_TOROIDAL, CAL_OPT_ACTIVE_CELLS);
#else
	hostCA = calCADef2D(ROWS, COLUMNS, CAL_VON_NEUMANN_NEIGHBORHOOD_2D, CAL_SPACE_TOROIDAL, CAL_NO_OPT);
#endif

	//add substates
	Q.f[0] = calAddSubstate2Dr(hostCA);
	Q.f[1] = calAddSubstate2Dr(hostCA);
	Q.f[2] = calAddSubstate2Dr(hostCA);
	Q.f[3] = calAddSubstate2Dr(hostCA);
	Q.z = calAddSubstate2Dr(hostCA);
	Q.h = calAddSubstate2Dr(hostCA);

	//load configuration
	calLoadSubstate2Dr(hostCA, Q.z, DEM);
	calLoadSubstate2Dr(hostCA, Q.h, SOURCE);

	//initialization
	sciddicaTSimulationInit(hostCA);
	calUpdate2D(hostCA);



	//calcl device CA
	deviceCA = calclCADef2D(hostCA, context, program, device);


  //calclBackToHostFunc2D(deviceCA,callback,1000);
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


	// draw_model3D definition
	struct CALUpdater2D * calUpdater= calglCreateUpdater2DCL(deviceCA,hostCA,100,1,4000);
	draw_model3D = calglDefDrawModel2DCL(CALGL_DRAW_MODE_SURFACE, "SciddicaT 3D view", hostCA, calUpdater);
	// Add nodes
	calglAddToDrawModel2Dr(draw_model3D, NULL, &Q.z, CALGL_TYPE_INFO_VERTEX_DATA, CALGL_TYPE_INFO_USE_NO_COLOR, CALGL_DATA_TYPE_STATIC);
	calglColor2D(draw_model3D, 0.5, 0.5, 0.5, 1.0);
	calglAddToDrawModel2Dr(draw_model3D, Q.z, &Q.z, CALGL_TYPE_INFO_COLOR_DATA, CALGL_TYPE_INFO_USE_CURRENT_COLOR, CALGL_DATA_TYPE_DYNAMIC);
	calglAddToDrawModel2Dr(draw_model3D, Q.z, &Q.z, CALGL_TYPE_INFO_NORMAL_DATA, CALGL_TYPE_INFO_USE_NO_COLOR, CALGL_DATA_TYPE_DYNAMIC);
	calglAddToDrawModel2Dr(draw_model3D, Q.z, &Q.h, CALGL_TYPE_INFO_VERTEX_DATA, CALGL_TYPE_INFO_USE_NO_COLOR, CALGL_DATA_TYPE_DYNAMIC);
	calglAddToDrawModel2Dr(draw_model3D, Q.h, &Q.h, CALGL_TYPE_INFO_COLOR_DATA, CALGL_TYPE_INFO_USE_RED_YELLOW_SCALE, CALGL_DATA_TYPE_DYNAMIC);
	calglAddToDrawModel2Dr(draw_model3D, Q.h, &Q.h, CALGL_TYPE_INFO_NORMAL_DATA, CALGL_TYPE_INFO_USE_NO_COLOR, CALGL_DATA_TYPE_DYNAMIC);
	// InfoBar
	//calglRelativeInfoBar2Dr(draw_model3D, Q.h, "Debris thickness", CALGL_TYPE_INFO_USE_RED_SCALE, CALGL_INFO_BAR_ORIENTATION_VERTICAL);
	calglInfoBar2Dr(draw_model3D, Q.h, "Debris thickness", CALGL_TYPE_INFO_USE_RED_SCALE, 20, 120, 300, 40);

	// Hide/display intervals of cells
//	calglHideDrawJBound2D(draw_model3D, 0, draw_model3D->calModel->columns);
//	calglDisplayDrawJBound2D(draw_model3D, 300, draw_model3D->calModel->columns);
//	calglHideDrawIBound2D(draw_model3D, 100, 150);

	//struct CALUpdater2D * calUpdater = calglCreateUpdater2DCL(sciddicaTsimulation);
	draw_model2D = calglDefDrawModel2DCL(CALGL_DRAW_MODE_FLAT, "SciddicaT 2D view", hostCA,calUpdater);
	draw_model2D->realModel = draw_model3D->realModel;
	calglInfoBar2Dr(draw_model2D, Q.h, "Debris thickness", CALGL_TYPE_INFO_USE_RED_SCALE, 20, 200, 50, 150);

	calglSetLayoutOrientation2D(CALGL_LAYOUT_ORIENTATION_VERTICAL);

	calglSetDisplayStep(100);

	calglMainLoop2D(argc, argv);

	return 0;
}
