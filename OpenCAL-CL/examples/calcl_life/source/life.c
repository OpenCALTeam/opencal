// Conway's game of Life Cellular Automaton

#include <OpenCAL-CL/calcl2D.h>
#include <OpenCAL/cal2DIO.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>

// Paths and kernel name definition
#define KERNEL_SRC "./kernel/source/"
#define KERNEL_LIFE_TRANSITION_FUNCTION "life_transition_function"
#define PLATFORM_NUM 0
#define DEVICE_NUM 0

int main()
{
	// Compliant device selection and kernel loading
	CALCLManager * calcl_manager = calclCreateManager();
	calclInitializePlatforms(calcl_manager);
	calclInitializeDevices(calcl_manager);
	calclPrintPlatformsAndDevices(calcl_manager);
	CALCLdevice device = calclGetDevice(calcl_manager, PLATFORM_NUM, DEVICE_NUM);
	CALCLcontext context = calclCreateContext(&device, 1);
	// Kernel loading and program compilstion
  CALCLprogram program = calclLoadProgram2D(context, device, KERNEL_SRC, NULL);

	// Host side CA definition and substate declaration
	struct CALModel2D* hostCA = calCADef2D(8, 16, CAL_MOORE_NEIGHBORHOOD_2D, CAL_SPACE_TOROIDAL, CAL_NO_OPT);
	struct CALSubstate2Di* Q;

	// Substate registration to the host CA
	Q = calAddSubstate2Di(hostCA);

	// Substate initialization to 0 everywhere
	calInitSubstate2Di(hostCA, Q, 0);

	// Setting of a glider
	calInit2Di(hostCA, Q, 0, 2, 1);
	calInit2Di(hostCA, Q, 1, 0, 1);
	calInit2Di(hostCA, Q, 1, 2, 1);
	calInit2Di(hostCA, Q, 2, 1, 1);
	calInit2Di(hostCA, Q, 2, 2, 1);

	// Device CA object definition
  CALCLModel2D * deviceCA = calclCADef2D(hostCA, context, program, device);

	//Kernel extraction from program
	CALCLkernel kernel_life_transition_function = calclGetKernelFromProgram(&program, KERNEL_LIFE_TRANSITION_FUNCTION);

	// save the Q substate to file
	calSaveSubstate2Di(hostCA, Q, "./life_0000.txt");

	// add transition function's elementary process
	calclAddElementaryProcess2D(deviceCA, hostCA, &kernel_life_transition_function);

	// simulation run
	calclRun2D(deviceCA, hostCA, 1, 1);

	// save the Q substate to file
	calSaveSubstate2Di(hostCA, Q, "./life_LAST.txt");

	// finalizations
	calFinalize2D(hostCA);
	calclFinalize2D(deviceCA, calcl_manager);

	return 0;
}
