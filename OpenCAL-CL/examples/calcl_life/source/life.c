// Conway's game of Life Cellular Automaton

#include <OpenCAL/cal2D.h>
#include <OpenCAL/cal2DIO.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>
#include <OpenCAL-CL/calcl2D.h>

// declare CA, substate and simulation objects
struct CALModel2D* hostCA;
struct CALSubstate2Di* Q;
struct CALRun2D* life_simulation;

#define KERNEL_SRC "./kernel/source/"
#define KERNEL_INC "./kernel/include/"
#define KERNEL_LIFE_TRANSITION_FUNCTION "life_transition_function"

int main()
{

	time_t start_time, end_time;

	int platformNum = 0;
	int deviceNum = 0;

	CALCLManager * calOpenCL;
	CALCLcontext context;
	CALCLdevice device;
	CALCLprogram program;
	CALCLModel2D * deviceCA;
	char * kernelSrc = KERNEL_SRC;
	char * kernelInc = KERNEL_INC;
	CALCLkernel kernel_life_transition_function;

	calOpenCL = calclCreateManager();
	calclInitializePlatforms(calOpenCL);
	calclInitializeDevices(calOpenCL);
	calclPrintPlatformsAndDevices(calOpenCL);
	device = calclGetDevice(calOpenCL, platformNum, deviceNum);
	context = calclCreateContext(&device, 1);
	program = calclLoadProgram2D(context, device, kernelSrc, NULL);


	// define of the life CA and life_simulation simulation objects
	hostCA = calCADef2D(8, 16, CAL_MOORE_NEIGHBORHOOD_2D, CAL_SPACE_TOROIDAL, CAL_NO_OPT);

	// add the Q substate to the life CA
	Q = calAddSubstate2Di(hostCA);

	// set the whole substate to 0
	calInitSubstate2Di(hostCA, Q, 0);

	// set a glider
	calInit2Di(hostCA, Q, 0, 2, 1);
	calInit2Di(hostCA, Q, 1, 0, 1);
	calInit2Di(hostCA, Q, 1, 2, 1);
	calInit2Di(hostCA, Q, 2, 1, 1);
	calInit2Di(hostCA, Q, 2, 2, 1);

	// define Toolkit object
    deviceCA = calclCADef2D(hostCA, context, program, device);

	//create kernel
	kernel_life_transition_function = calclGetKernelFromProgram(&program, KERNEL_LIFE_TRANSITION_FUNCTION);

	// save the Q substate to file
	calSaveSubstate2Di(hostCA, Q, "./hostCA_0000.txt");

	// add transition function's elementary process
	calclAddElementaryProcess2D(deviceCA, hostCA, &kernel_life_transition_function);

	start_time = time(NULL);
	// simulation run
	calclRun2D(deviceCA, hostCA, 1, 1);
	end_time = time(NULL);
	printf("%d", end_time - start_time);

	// save the Q substate to file
	calSaveSubstate2Di(hostCA, Q, "./hostCA_LAST.txt");

	// finalize simulation and CA objects
	calFinalize2D(hostCA);

	return 0;
}
