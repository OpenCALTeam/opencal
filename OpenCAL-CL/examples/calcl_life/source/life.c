// Conway's game of Life Cellular Automaton

#include <OpenCAL/cal2D.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>
#include <OpenCAL-CL/calcl2D.h>

// declare CA, substate and simulation objects
struct CALModel2D* life;
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

	CALOpenCL * calOpenCL;
	CALCLcontext context;
	CALCLdevice device;
	CALCLprogram program;
	CALCLToolkit2D * lifeToolkit;
	char * kernelSrc = KERNEL_SRC;
		char * kernelInc = KERNEL_INC;
		CALCLkernel kernel_life_transition_function;

		calOpenCL = calclCreateCALOpenCL();
		calclInitializePlatforms(calOpenCL);
		calclInitializeDevices(calOpenCL);
		calclPrintAllPlatformAndDevices(calOpenCL);
		device = calclGetDevice(calOpenCL, platformNum, deviceNum);
		context = calclcreateContext(&device, 1);
		program = calclLoadProgramLib2D(context, device, kernelSrc, kernelInc);


	// define of the life CA and life_simulation simulation objects
	life = calCADef2D(8, 16, CAL_MOORE_NEIGHBORHOOD_2D, CAL_SPACE_TOROIDAL, CAL_NO_OPT);

	// add the Q substate to the life CA
	Q = calAddSubstate2Di(life);

	// add transition function's elementary process
//	calAddElementaryProcess2D(life, life_transition_function);

	// set the whole substate to 0
	calInitSubstate2Di(life, Q, 0);

	// set a glider
	calInit2Di(life, Q, 0, 2, 1);
	calInit2Di(life, Q, 1, 0, 1);
	calInit2Di(life, Q, 1, 2, 1);
	calInit2Di(life, Q, 2, 1, 1);
	calInit2Di(life, Q, 2, 2, 1);

  lifeToolkit = calclCreateToolkit2D(life, context, program, device, CAL_NO_OPT);

	kernel_life_transition_function = calclGetKernelFromProgram(&program, KERNEL_LIFE_TRANSITION_FUNCTION);
	calclAddElementaryProcessKernel2D(lifeToolkit, life, &kernel_life_transition_function);
	start_time = time(NULL);
	calclRun2D(lifeToolkit, life, 1, 1);
	end_time = time(NULL);
	printf("%d", end_time - start_time);

	// save the Q substate to file
//	calSaveSubstate2Di(life, Q, "./life_0000.txt");

	// simulation run
//	calRun2D(life_simulation);

	// save the Q substate to file
	calSaveSubstate2Di(life, Q, "./life_LAST.txt");

	// finalize simulation and CA objects
	calRunFinalize2D(life_simulation);
	calFinalize2D(life);

	return 0;
}
