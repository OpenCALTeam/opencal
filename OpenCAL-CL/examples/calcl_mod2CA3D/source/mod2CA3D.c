// mod2 3D Cellular Automaton

#include <OpenCAL-CL/calcl3D.h>
#include <OpenCAL/cal3DIO.h>

// Some definitions...
#define ROWS 5
#define COLS 7
#define LAYERS 3
#define KERNEL_SRC "./kernel/source/"
#define KERNEL_LIFE_TRANSITION_FUNCTION "mod2TransitionFunction"
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
	CALCLprogram program = calclLoadProgram3D(context, device, KERNEL_SRC, NULL);

	// Define of the mod2 CA object and declare a substate
	struct CALModel3D* host_CA = calCADef3D(ROWS, COLS, LAYERS, CAL_MOORE_NEIGHBORHOOD_3D, CAL_SPACE_TOROIDAL, CAL_NO_OPT);
	struct CALSubstate3Db* Q;

	// Add the Q substate to the host_CA CA
	Q = calAddSubstate3Db(host_CA);

	// Set the whole substate to 0
	calInitSubstate3Db(host_CA, Q, 0);

	// Set a seed at position (2, 3, 1)
	calInit3Db(host_CA, Q, 2, 3, 1, 1);

	// Save the Q substate to file
	calSaveSubstate3Db(host_CA, Q, "./mod2_0000.txt");

	// Define a device-side CA
	struct CALCLModel3D * device_CA = calclCADef3D(host_CA, context, program, device);

  // Register a transition function's elementary process kernel
	CALCLkernel kernel_transition_function = calclGetKernelFromProgram(&program, KERNEL_LIFE_TRANSITION_FUNCTION);

	// Add transition function's elementary process
	calclAddElementaryProcess3D(device_CA, &kernel_transition_function);

	// Simulation run
	calclRun3D(device_CA, 1, 1);

	// Save the Q substate to file
	calSaveSubstate3Db(host_CA, Q, "./mod2_LAST.txt");

	// Finalizations
	calclFinalizeManager(calcl_device_manager);
	calclFinalize3D(device_CA);
	calFinalize3D(host_CA);

	return 0;
}
