/*
 ============================================================================
 Name        : cal-heattransfer.c
 Author      : Davide Spataro
 Version     :
 Copyright   :
 Description : Hello World in C, Ansi-style
 ============================================================================
 */

#include <stdio.h>
#include <stdlib.h>

#include <OpenCAL-CL/calcl3D.h>
#include <OpenCAL-CL/calgl3DRunCL.h>
#include <OpenCAL/cal3DIO.h>
#include <OpenCALTime.h>

#define SIZE (100)
#define ROWS (SIZE)
#define COLS (SIZE)
#define LAYERS (SIZE)
#define STEPS (1000)
#define EPSILON (0.01)

//model&materials parameters
#define DELTA_X (0.001)
#define DELTA_Y (0.001)
#define DELTA_Z (0.001)
#define DELTA_T (0.001)
#define THERMAL_CONDUCTIVITY (1)
#define MASS_DENSITY (1)
#define SPECIFIC_HEAT_CAPACITY (1)
#define THERMAL_DIFFUSIVITY ( (THERMAL_CONDUCTIVITY)/(SPECIFIC_HEAT_CAPACITY)*(MASS_DENSITY) )
#define THERMAL_DIFFUSIVITY_WATER (1.4563e-4) //C/m^2
#define INIT_TEMP (1200)

//OpenCAL-CL
#define KERNEL_SRC "./heattransfer/calcl-heattransfer/kernel/source/"
#define KERNEL_HEAT_TRANSITION_FUNCTION "heatModelTransitionFunction"

// declare CA, substate and simulation objects
struct CALModel3D* host_CA;							//the cellular automaton
struct CALSubstate3Dr *Q_temperature;							//the substate Q
struct CALSubstate3Db *Q_heat_source;							//the substate Q
struct CALCLModel3D* device_CA;			//the simulartion run
struct CALCLDeviceManager * calcl_device_manager;



// Simulation init callback function used to set a seed at position (24, 0, 0)
void heatModel_SimulationInit(struct CALModel3D* host_CA)
{

	calInitSubstate3Dr(host_CA, Q_temperature, (CALreal)0);
	calInitSubstate3Db(host_CA, Q_heat_source, CAL_FALSE);


	//for(int i=1 ; i < ROWS ; ++i){
	int j=0;
	int z=0;
		for ( j = 1; j < COLS; ++j) {
			for ( z = 1; z < LAYERS; ++z) {

				CALreal _i, _j,_z;
				CALreal chunk = ROWS/2;
				/*for(int l =2 ; l < 4; l++){
					_i = i -(ROWS/l);
					_j = i -(COLS/l);
					_z = z -(LAYERS/l);
					if(_i *_i + _j*_j + _z * _z <= radius){*/
						calInit3Dr(host_CA, Q_temperature, chunk, j, z, INIT_TEMP);
						calInit3Dr(host_CA, Q_temperature, chunk+1, j, z, INIT_TEMP);
						calInit3Dr(host_CA, Q_temperature, chunk-1, j, z, INIT_TEMP);
						//calSet3Dr(host_CA, Q_temperature, chunk*2, j, z, INIT_TEMP);
						//calSet3Dr(host_CA, Q_temperature, chunk*3, j, z, INIT_TEMP);
						//calSet3Db(host_CA, Q_heat_source, i, j, z, 1);


				}
			}
		}
//}

// Callback unction called just before program termination
void exitFunction(void)
{

}

int main(int argc, char** argv) {

    int platform;
    if (sscanf (argv[3], "%i", &platform)!=1 && platform >=0) {
        printf ("platform number is not an integer");
        exit(-1);
    }

    int deviceNumber;
    if (sscanf (argv[4], "%i", &deviceNumber)!=1 && deviceNumber >=0) {
        printf ("device number is not an integer");
        exit(-1);
    }

	// Declare a viewer object
	struct CALGLDrawModel3D* drawModel;

	atexit(exitFunction);

	// Select a compliant device
	calcl_device_manager = calclCreateManager();
	CALCLdevice device;
    device = calclGetDevice(calcl_device_manager, platform, deviceNumber);
	CALCLcontext context = calclCreateContext(&device);

	// Load kernels and return a compiled program
	CALCLprogram program = calclLoadProgram3D(context, device, KERNEL_SRC, NULL);

	//cadef and rundef
	host_CA = calCADef3D(ROWS, COLS, LAYERS, CAL_MOORE_NEIGHBORHOOD_3D, CAL_SPACE_FLAT, CAL_NO_OPT);
	//add substates
	Q_temperature = calAddSubstate3Dr(host_CA);
	Q_heat_source = calAddSubstate3Db(host_CA);

	// Set the whole substate to 0
	heatModel_SimulationInit(host_CA);

	device_CA = calclCADef3D(host_CA, context, program, device);

	// Save the Q substate to file
	//calSaveSubstate3Dr(host_CA, Q_temperature, "./mod2_0000.txt");

	// Register a transition function's elementary process kernel
  CALCLkernel kernel_transition_function = calclGetKernelFromProgram(&program, KERNEL_HEAT_TRANSITION_FUNCTION);

	// Add transition function's elementary process
	calclAddElementaryProcess3D(device_CA, &kernel_transition_function);

    struct OpenCALTime * opencalTime= (struct OpenCALTime *)malloc(sizeof(struct OpenCALTime));
    startTime(opencalTime);
    calclRun3D(device_CA,1,STEPS);
    endTime(opencalTime);

	// Save the substate to file
	calSaveSubstate3Dr(host_CA, Q_temperature, "./testsout/other/1.txt");

	// Finalizations
	calclFinalizeManager(calcl_device_manager);
	calclFinalize3D(device_CA);
	calFinalize3D(host_CA);
	return 0;
}
