// Conway's game of Life Cellular Automaton

#include <OpenCAL-CL/calcl2D.h>
#include <OpenCAL/cal2DIO.h>

// Paths and kernel name definition
#define KERNEL_SRC "./kernel/source/"
#define KERNEL_LIFE_TRANSITION_FUNCTION "lifeTransitionFunction"
#define PLATFORM_NUM 0
#define DEVICE_NUM 0

int main()
{
	// Select a compliant device
	struct CALCLDeviceManager * calcl_device_manager = calclCreateManager();
	calclPrintPlatformsAndDevices(calcl_device_manager);
	CALCLdevice device = calclGetDevice(calcl_device_manager, PLATFORM_NUM, DEVICE_NUM);
	CALCLcontext context = calclCreateContext(&device);

	// Load kernels and return a compiled program
  CALCLprogram program = calclLoadProgram2D(context, device, KERNEL_SRC, NULL);

	// Define a host-side CA and declare a substate
	struct CALModel2D* host_CA = calCADef2D(8, 16, CAL_MOORE_NEIGHBORHOOD_2D, CAL_SPACE_TOROIDAL, CAL_NO_OPT);
	struct CALSubstate2Di* Q;

	// Register the substate to the host CA
	Q = calAddSubstate2Di(host_CA);

	// Initialize the substate to 0 everywhere
	calInitSubstate2Di(host_CA, Q, 0);

	// Set a glider
	calInit2Di(host_CA, Q, 0, 2, 1);
	calInit2Di(host_CA, Q, 1, 0, 1);
	calInit2Di(host_CA, Q, 1, 2, 1);
	calInit2Di(host_CA, Q, 2, 1, 1);
	calInit2Di(host_CA, Q, 2, 2, 1);

	// Define a device-side CA
  struct CALCLModel2D * device_CA = calclCADef2D(host_CA, context, program, device);

	// Extract a kernel from program
	CALCLkernel kernel_life_transition_function = calclGetKernelFromProgram(&program, KERNEL_LIFE_TRANSITION_FUNCTION);

	// Register a transition function's elementary process kernel
	calclAddElementaryProcess2D(device_CA, &kernel_life_transition_function);

	// Save the substate to file
	calSaveSubstate2Di(host_CA, Q, "./life_0000.txt");

	// Run the simulation (actually, only one computational step)
	calclRun2D(device_CA, 1, 1);

	// Save the substate to file
	calSaveSubstate2Di(host_CA, Q, "./life_LAST.txt");

	// Finalizations
	calclFinalizeManager(calcl_device_manager);
	calclFinalize2D(device_CA);
	calFinalize2D(host_CA);

	return 0;
}
