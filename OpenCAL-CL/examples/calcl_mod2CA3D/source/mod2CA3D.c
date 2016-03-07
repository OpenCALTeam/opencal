// mod2 3D Cellular Automaton

#include <OpenCAL/cal3D.h>
#include <OpenCAL/cal3DIO.h>
#include <OpenCAL-CL/calcl3D.h>
#define ROWS 5
#define COLS 7
#define LAYERS 3

// declare CA, substate and simulation objects
struct CALModel3D* mod2;
struct CALSubstate3Db* Q;

#define KERNEL_SRC "./kernel/source/"
#define KERNEL_INC "./kernel/include/"
#define KERNEL_LIFE_TRANSITION_FUNCTION "mod2_transition_function"


int main()
{
	time_t start_time, end_time;

	int platformNum = 0;
	int deviceNum = 0;

	CALOpenCL * calOpenCL;
	CALCLcontext context;
	CALCLdevice device;
	CALCLprogram program;
	CALCLToolkit3D * mod2Toolkit;
	char * kernelSrc = KERNEL_SRC;
	char * kernelInc = KERNEL_INC;
	CALCLkernel kernel_transition_function;

	calOpenCL = calclCreateCALOpenCL();
	calclInitializePlatforms(calOpenCL);
	calclInitializeDevices(calOpenCL);
	calclPrintAllPlatformAndDevices(calOpenCL);
	device = calclGetDevice(calOpenCL, platformNum, deviceNum);
	context = calclCreateContext(&device, 1);
	
	program = calclLoadProgram3D(context, device, kernelSrc, NULL);


	// define of the mod2 CA object
	mod2 = calCADef3D(ROWS, COLS, LAYERS, CAL_MOORE_NEIGHBORHOOD_3D, CAL_SPACE_TOROIDAL, CAL_NO_OPT);

	// add the Q substate to the mod2 CA
	Q = calAddSubstate3Db(mod2);

	// set the whole substate to 0
	calInitSubstate3Db(mod2, Q, 0);

	// set a seed at position (2, 3, 1)
	calInit3Db(mod2, Q, 2, 3, 1, 1);

	// save the Q substate to file
	calSaveSubstate3Db(mod2, Q, "./mod2_0000.txt");

	// define Toolkit object
	mod2Toolkit = calclCreateToolkit3D(mod2, context, program, device);

    //create kernel
	kernel_transition_function = calclGetKernelFromProgram(&program, KERNEL_LIFE_TRANSITION_FUNCTION);

	// add transition function's elementary process
	calclAddElementaryProcessKernel3D(mod2Toolkit, mod2, &kernel_transition_function);

	start_time = time(NULL);
	// simulation run
	calclRun3D(mod2Toolkit, mod2, 1, 1);
	end_time = time(NULL);
	printf("%d", end_time - start_time);

	// save the Q substate to file
	calSaveSubstate3Db(mod2, Q, "./mod2_LAST.txt");

	// finalize simulation and CA objects
	calFinalize3D(mod2);

	return 0;
}
