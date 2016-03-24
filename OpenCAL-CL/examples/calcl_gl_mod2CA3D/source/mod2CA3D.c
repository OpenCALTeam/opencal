// mod2 3D Cellular Automaton

#include <OpenCAL-CL/calcl3D.h>
#include <OpenCAL-CL/calgl3DRunCL.h>
#include <OpenCAL-GL/calgl3D.h>
#include <OpenCAL-GL/calgl3DWindow.h>
#include <OpenCAL/cal3DIO.h>

// Some definitions...
#define ROWS 25
#define COLS 25
#define LAYERS 25
#define KERNEL_SRC "./kernel/source/"
#define KERNEL_LIFE_TRANSITION_FUNCTION "mod2TransitionFunction"
#define PLATFORM_NUM 0
#define DEVICE_NUM 0

struct CALModel3D* host_CA;					//the cellular automaton
struct CALSubstate3Db *Q;						//the substate Q
struct CALCLModel3D* device_CA;			//the simulartion run
struct CALCLDeviceManager * calcl_device_manager;

// Callback unction called just before program termination
void exitFunction(void)
{
	// Finalizations
	calclFinalizeManager(calcl_device_manager);
	calclFinalize3D(device_CA);
	calFinalize3D(host_CA);
}

// Simulation init callback function used to set a seed at position (24, 0, 0)
void mod2SimulationInit(struct CALModel3D* ca)
{
	//initializing substate to 0
	calInitSubstate3Db(ca, Q, 0);
	//setting a specific cell
	calSet3Db(ca, Q, 24, 0, 0, 1);
}


int main(int argc, char** argv)
{
	// Declare a viewer object
	struct CALGLDrawModel3D* drawModel;

	atexit(exitFunction);

	// Select a compliant device
	calcl_device_manager = calclCreateManager();
	calclPrintPlatformsAndDevices(calcl_device_manager);
	CALCLdevice device = calclGetDevice(calcl_device_manager, PLATFORM_NUM, DEVICE_NUM);
	CALCLcontext context = calclCreateContext(&device);

	// Load kernels and return a compiled program
	CALCLprogram program = calclLoadProgram3D(context, device, KERNEL_SRC, NULL);

	// Define of the mod2 CA object and declare a substate
	host_CA = calCADef3D(ROWS, COLS, LAYERS, CAL_MOORE_NEIGHBORHOOD_3D, CAL_SPACE_TOROIDAL, CAL_NO_OPT);

	// Add the Q substate to the host_CA CA
	Q = calAddSubstate3Db(host_CA);

	// Set the whole substate to 0
	calInitSubstate3Db(host_CA, Q, 0);

	//setting a specific cell
	calInit3Db(host_CA, Q, 24, 0, 0, 1);

	// Save the Q substate to file
	calSaveSubstate3Db(host_CA, Q, "./mod2_0000.txt");

	// Define a device-side CA
  device_CA = calclCADef3D(host_CA, context, program, device);

  // Register a transition function's elementary process kernel
	CALCLkernel kernel_transition_function = calclGetKernelFromProgram(&program, KERNEL_LIFE_TRANSITION_FUNCTION);

	// Add transition function's elementary process
	calclAddElementaryProcess3D(device_CA, &kernel_transition_function);


	// Initialize the viewer
	calglInitViewer("mod2 3D CA viewer", 1.0f, 400, 400, 40, 40, CAL_TRUE, 100);

	//drawModel definition
	struct CALGLRun3D * calUpdater = calglRunCLDef3D(device_CA,100,1,4000);
	drawModel = calglDefDrawModelCL3D(CALGL_DRAW_MODE_FLAT, "3D view", host_CA, calUpdater);
	calglAdd3Db(drawModel, NULL, &Q, CALGL_TYPE_INFO_VERTEX_DATA, CALGL_TYPE_INFO_USE_NO_COLOR, CALGL_DATA_TYPE_DYNAMIC);
	calglColor3D(drawModel, 0.5f, 0.5f, 0.5f, 1.0f);
	calglAdd3Db(drawModel, Q, &Q, CALGL_TYPE_INFO_COLOR_DATA, CALGL_TYPE_INFO_USE_CURRENT_COLOR, CALGL_DATA_TYPE_DYNAMIC);
	calglAdd3Db(drawModel, Q, &Q, CALGL_TYPE_INFO_NORMAL_DATA, CALGL_TYPE_INFO_USE_NO_COLOR, CALGL_DATA_TYPE_DYNAMIC);

	calglMainLoop3D(argc, argv);

	return 0;
}
