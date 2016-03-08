// mod2 3D Cellular Automaton

#include <OpenCAL/cal3D.h>
#include <OpenCAL/cal3DIO.h>
#include <OpenCAL-CL/calcl3D.h>
#define ROWS 5
#define COLS 7
#define LAYERS 3

// declare CA, substate and simulation objects
struct CALModel3D* hostCA;
struct CALSubstate3Db* Q;

#define KERNEL_SRC "./kernel/source/"
#define KERNEL_INC "./kernel/include/"
#define KERNEL_LIFE_TRANSITION_FUNCTION "mod2_transition_function"


int main()
{
	time_t start_time, end_time;

	int platformNum = 0;
	int deviceNum = 0;

	CALCLManager * calOpenCL;
	CALCLcontext context;
	CALCLdevice device;
	CALCLprogram program;
	CALCLModel3D * deviceCA;
	char * kernelSrc = KERNEL_SRC;
	char * kernelInc = KERNEL_INC;
	CALCLkernel kernel_transition_function;

	calOpenCL = calclCreateManager();
	calclInitializePlatforms(calOpenCL);
	calclInitializeDevices(calOpenCL);
	calclPrintPlatformsAndDevices(calOpenCL);
	device = calclGetDevice(calOpenCL, platformNum, deviceNum);
	context = calclCreateContext(&device, 1);

	program = calclLoadProgram3D(context, device, kernelSrc, NULL);


	// define of the mod2 CA object
	hostCA = calCADef3D(ROWS, COLS, LAYERS, CAL_MOORE_NEIGHBORHOOD_3D, CAL_SPACE_TOROIDAL, CAL_NO_OPT);

	// add the Q substate to the hostCA CA
	Q = calAddSubstate3Db(hostCA);

	// set the whole substate to 0
	calInitSubstate3Db(hostCA, Q, 0);

	// set a seed at position (2, 3, 1)
	calInit3Db(hostCA, Q, 2, 3, 1, 1);

	// save the Q substate to file
	calSaveSubstate3Db(hostCA, Q, "./mod2_0000.txt");

	// define Toolkit object
	deviceCA = calclCADef3D(hostCA, context, program, device);

    //create kernel
	kernel_transition_function = calclGetKernelFromProgram(&program, KERNEL_LIFE_TRANSITION_FUNCTION);

	// add transition function's elementary process
	calclAddElementaryProcess3D(deviceCA, hostCA, &kernel_transition_function);

	start_time = time(NULL);
	// simulation run
	calclRun3D(deviceCA, hostCA, 1, 1);
	end_time = time(NULL);
	printf("%d", end_time - start_time);

	// save the Q substate to file
	calSaveSubstate3Db(hostCA, Q, "./mod2_LAST.txt");

	// finalize simulation and CA objects
	calFinalize3D(hostCA);

	return 0;
}
