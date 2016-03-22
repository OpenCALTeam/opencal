// The SciddicaT debris flows XCA simulation model with
// a 3D graphic viewer in OpenCAL-GL

#include <OpenCAL/cal2DIO.h>
#include <OpenCAL-CL/calcl2D.h>
#include <OpenCAL-CL/calgl2DRunCL.h>
#include <OpenCAL-GL/calgl2D.h>
#include <OpenCAL-GL/calgl2DWindow.h>
#include <stdlib.h>

// Some definitions...
#define ROWS 610
#define COLUMNS 496
#define P_R 0.5
#define P_EPSILON 0.001
#define NUMBER_OF_OUTFLOWS 4
#define STEPS 4000
#define DEM_PATH "./data/dem.txt"
#define SOURCE_PATH "./data/source.txt"
#define OUTPUT_PATH "./data/width_final.txt"
#define GRAPHIC_UPDATE_INTERVAL 100

// kernels' names definitions
#define ACTIVE_CELLS
#define KERNEL_SRC "./kernel/source/"
#define KERNEL_INC "./kernel/include/"
#define KERNEL_SRC_AC "./kernelActive/source/"
#define KERNEL_INC_AC "./kernelActive/include/"
#define KERNEL_ELEM_PROC_FLOW_COMPUTATION "flowsComputation"
#define KERNEL_ELEM_PROC_WIDTH_UPDATE "widthUpdate"
#define KERNEL_STEERING  "steering"
#ifdef ACTIVE_CELLS
#define KERNEL_ELEM_PROC_RM_ACT_CELLS "removeInactiveCells"
#endif

// The set of CA substates
struct sciddicaTSubstates {
	struct CALSubstate2Dr *z;
	struct CALSubstate2Dr *h;
	struct CALSubstate2Dr *f[NUMBER_OF_OUTFLOWS];
};

// The set of CA parameters
struct sciddicaTParameters {
	CALParameterr epsilon;
	CALParameterr r;
};

// Objects declaration
struct CALCLDeviceManager * calcl_device_manager; //the device manager object
struct CALModel2D* host_CA;						 						//the host-side CA
struct sciddicaTSubstates Q;											//the CA substates object
struct sciddicaTParameters P;											//the CA parameters object
struct CALCLModel2D * device_CA;									//the device-side CA


// SciddicaT exit function
void exitFunction(void)
{
	// saving configuration
	calSaveSubstate2Dr (host_CA, Q.h, OUTPUT_PATH);

	// finalizations
	//calRunFinalize2D (sciddicaTsimulation);
	calclFinalizeManager(calcl_device_manager);
	calclFinalize2D(device_CA);
	calFinalize2D (host_CA);
}

// SciddicaT init function
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
	// OpenCL device, context and program declaration
	CALCLdevice device;
	CALCLcontext context;
	CALCLprogram program;

	// kernels paths, names and buffers (for kernel parameters)
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
	CALCLmem bufferEpsilonParameter;
	CALCLmem bufferRParameter;

	//OpenCL device selection from stdin and context definition
	calcl_device_manager = calclCreateManager();
	calclGetPlatformAndDeviceFromStdIn(calcl_device_manager, &device);
	context = calclCreateContext(&device);

	// Load kernels and return a compiled program
	program = calclLoadProgram2D(context, device, kernelSrc, kernelInc);

	// host-side CA definition
#ifdef ACTIVE_CELLS
	host_CA = calCADef2D(ROWS, COLUMNS, CAL_VON_NEUMANN_NEIGHBORHOOD_2D, CAL_SPACE_TOROIDAL, CAL_OPT_ACTIVE_CELLS);
#else
	host_CA = calCADef2D(ROWS, COLUMNS, CAL_VON_NEUMANN_NEIGHBORHOOD_2D, CAL_SPACE_TOROIDAL, CAL_NO_OPT);
#endif

	// Add substates to the host-side CA
	Q.f[0] = calAddSubstate2Dr(host_CA);
	Q.f[1] = calAddSubstate2Dr(host_CA);
	Q.f[2] = calAddSubstate2Dr(host_CA);
	Q.f[3] = calAddSubstate2Dr(host_CA);
	Q.z = calAddSubstate2Dr(host_CA);
	Q.h = calAddSubstate2Dr(host_CA);

	// Load data from file
	calLoadSubstate2Dr(host_CA, Q.z, DEM_PATH);
	calLoadSubstate2Dr(host_CA, Q.h, SOURCE_PATH);
	// Host-side CA initialization
	sciddicaTSimulationInit(host_CA);
	calUpdate2D(host_CA);

	//device-side CA definition
	device_CA = calclCADef2D(host_CA, context, program, device);

	// Extract kernels from program
	kernel_elem_proc_flow_computation = calclGetKernelFromProgram(&program, KERNEL_ELEM_PROC_FLOW_COMPUTATION);
	kernel_elem_proc_width_update = calclGetKernelFromProgram(&program, KERNEL_ELEM_PROC_WIDTH_UPDATE);
#ifdef ACTIVE_CELLS
	kernel_elem_proc_rm_act_cells = calclGetKernelFromProgram(&program, KERNEL_ELEM_PROC_RM_ACT_CELLS);
#endif
	kernel_steering = calclGetKernelFromProgram(&program, KERNEL_STEERING);

	// Setting kernel parameters
	bufferEpsilonParameter = calclCreateBuffer(context, &P.epsilon, sizeof(CALParameterr));
	bufferRParameter = calclCreateBuffer(context, &P.r, sizeof(CALParameterr));
 	calclSetKernelArg2D(&kernel_elem_proc_flow_computation, 0, sizeof(CALCLmem), &bufferEpsilonParameter);
	calclSetKernelArg2D(&kernel_elem_proc_flow_computation, 1, sizeof(CALCLmem), &bufferRParameter);

	// Register transition function’s elementary processes to the device-side CA
	calclAddElementaryProcess2D(device_CA, &kernel_elem_proc_flow_computation);
	calclAddElementaryProcess2D(device_CA, &kernel_elem_proc_width_update);
	#ifdef ACTIVE_CELLS
		calclSetKernelArg2D(&kernel_elem_proc_rm_act_cells, 0, sizeof(CALCLmem), &bufferEpsilonParameter);
		calclAddElementaryProcess2D(device_CA, &kernel_elem_proc_rm_act_cells);
	#endif
	// Register a steering function to the device-side CA
	calclAddSteeringFunc2D(device_CA, &kernel_steering);

	// Register a function to be executed before program termination
	atexit(exitFunction);


	// Graphic viewer initialization
	calglInitViewer("SciddicaT OpenCAL-GL visualizer", 5, 800, 600, 10, 10, CAL_TRUE, 0);
	//calglSetLayoutOrientation2D(CALGL_LAYOUT_ORIENTATION_VERTICAL);

	// Rendering objects declaration
	struct CALGLDrawModel2D* render_3D;
	struct CALGLDrawModel2D* render_2D;

	// render_3D definition
	struct CALGLRun2D * calgl_run= calglRunCLDef2D(device_CA, GRAPHIC_UPDATE_INTERVAL, 1, 4000);
	calglSetDisplayStep(GRAPHIC_UPDATE_INTERVAL);

	// 3D view rendering object
	render_3D = calglDefDrawModelCL2D(CALGL_DRAW_MODE_SURFACE, "SciddicaT 3D view", host_CA, calgl_run);
	// Add nodes
	calglAdd2Dr(render_3D, NULL, &Q.z, CALGL_TYPE_INFO_VERTEX_DATA, CALGL_TYPE_INFO_USE_NO_COLOR, CALGL_DATA_TYPE_STATIC);
	calglColor2D(render_3D, 0.5, 0.5, 0.5, 1.0);
	calglAdd2Dr(render_3D, Q.z, &Q.z, CALGL_TYPE_INFO_COLOR_DATA, CALGL_TYPE_INFO_USE_CURRENT_COLOR, CALGL_DATA_TYPE_DYNAMIC);
	calglAdd2Dr(render_3D, Q.z, &Q.z, CALGL_TYPE_INFO_NORMAL_DATA, CALGL_TYPE_INFO_USE_NO_COLOR, CALGL_DATA_TYPE_DYNAMIC);
	calglAdd2Dr(render_3D, Q.z, &Q.h, CALGL_TYPE_INFO_VERTEX_DATA, CALGL_TYPE_INFO_USE_NO_COLOR, CALGL_DATA_TYPE_DYNAMIC);
	calglAdd2Dr(render_3D, Q.h, &Q.h, CALGL_TYPE_INFO_COLOR_DATA, CALGL_TYPE_INFO_USE_RED_YELLOW_SCALE, CALGL_DATA_TYPE_DYNAMIC);
	calglAdd2Dr(render_3D, Q.h, &Q.h, CALGL_TYPE_INFO_NORMAL_DATA, CALGL_TYPE_INFO_USE_NO_COLOR, CALGL_DATA_TYPE_DYNAMIC);
	calglSetHeightOffset2D(render_3D,100);

	// Scalar bar
	calglInfoBar2Dr(render_3D, Q.h, "Debris thickness", CALGL_TYPE_INFO_USE_RED_SCALE, 20, 120, 300, 40);

	// 2D view rendering object
	render_2D = calglDefDrawModelCL2D(CALGL_DRAW_MODE_FLAT, "SciddicaT 2D view", host_CA, calgl_run);
	render_2D->realModel = render_3D->realModel;
	calglInfoBar2Dr(render_2D, Q.h, "Debris thickness", CALGL_TYPE_INFO_USE_RED_SCALE, 20, 200, 50, 150);

	// calgl main loop
	calglMainLoop2D(argc, argv);

	return 0;
}


// void callback(struct CALModel2D* host_CA) {
// 	system("clear");
// printf("********************************************************************");
// }

//calclBackToHostFunc2D(device_CA,callback,1000);



//	calclSetKernelArg2D(&kernel_elem_proc_flow_computation, 0, sizeof(CALParameterr), &P.epsilon);
//	calclSetKernelArg2D(&kernel_elem_proc_flow_computation, 1, sizeof(CALParameterr), &P.r);
		//calclSetKernelArg2D(&kernel_elem_proc_rm_act_cells, 0, sizeof(CALParameterr), &P.epsilon);
