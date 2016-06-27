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

#include <OpenCAL/cal3D.h>
#include <OpenCAL/cal3DRun.h>
#include <OpenCAL/cal3DIO.h>
#include <time.h>
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

// declare CA, substate and simulation objects
struct CALModel3D* heatModel;							//the cellular automaton
struct CALSubstate3Dr *Q_temperature;							//the substate Q
struct CALSubstate3Db *Q_heat_source;							//the substate Q
struct CALRun3D* heat_simulation;

const CALreal radius = 5;

// The cell's transition function (first and only elementary process)
void heatModel_TransitionFunction(struct CALModel3D* heatModel, int i, int j, int k)
{


	if(i > 1 && i < ROWS-1 && j > 1 && j < COLS-1 && k > 1 && k < LAYERS-1){
		CALreal currValue =calGet3Dr(heatModel, Q_temperature , i , j , k );

		CALreal dx2 = (calGet3Dr(heatModel, Q_temperature , i+1,j,k) + calGet3Dr(heatModel,Q_temperature ,i-1,j,k) - (2*currValue))/(DELTA_X*DELTA_X);


		CALreal dy2 = (calGet3Dr(heatModel,Q_temperature ,i,j+1,k) + calGet3Dr(heatModel,Q_temperature ,i,j-1,k) - (2*currValue))/(DELTA_Y*DELTA_Y);


		CALreal dz2 = (calGet3Dr(heatModel,Q_temperature ,i,j,k+1) + calGet3Dr(heatModel,Q_temperature ,i,j,k-1) - (2*currValue))/(DELTA_Z*DELTA_Z);

		CALreal newValue = currValue + DELTA_T*THERMAL_DIFFUSIVITY_WATER * (dx2 + dy2 +dz2);

		//||
		if(newValue > EPSILON  && newValue < 10000){
			//xprintf("newVal i,j,k = %i, %i, %i -> dx2=%.15f , dy2=%.15f , dz2=%.15f , val =%.15f \n" ,i,j,k, dx2,dy2,dz2,newValue);
			calSet3Dr(heatModel, Q_temperature, i, j, k, newValue);
			newValue = currValue;

		}

		CALint _i = i -(ROWS/2);
		CALint	_j = j -(COLS/2);
		CALint	_z = k -(LAYERS/2);
		if(_i *_i + _j*_j + _z * _z <= radius){
			//printf("Temp at source is %f  ->>>>>",newValue);
			//printf("newVal i,j,k = %i, %i, %i -> dx2=%.15f , dy2=%.15f , dz2=%.15f , val =%.15f \n" ,i,j,k, dx2,dy2,dz2,newValue);
			//calSet3Dr(heatModel, Q_temperature, i, j, k, newValue-0.1);
			//calSet3Dr(heatModel, Q_temperature, i, j, k, newValue+0.00001);
		}

	}else{
		calSet3Dr(heatModel, Q_temperature, i, j, k, 0);
	}


}

// Simulation init callback function used to set a seed at position (24, 0, 0)
void heatModel_SimulationInit(struct CALModel3D* heatModel)
{

	calInitSubstate3Dr(heatModel, Q_temperature, (CALreal)0);
	calInitSubstate3Db(heatModel, Q_heat_source, CAL_FALSE);


	//for(int i=1 ; i < ROWS ; ++i){
		int j=0;
		int z=0;
		for (j = 1; j < COLS; ++j) {
			for (z = 1; z < LAYERS; ++z) {

				CALreal _i, _j,_z;
				CALreal chunk = ROWS/2;
				/*for(int l =2 ; l < 4; l++){
					_i = i -(ROWS/l);
					_j = i -(COLS/l);
					_z = z -(LAYERS/l);
					if(_i *_i + _j*_j + _z * _z <= radius){*/
						calSet3Dr(heatModel, Q_temperature, chunk, j, z, INIT_TEMP);
						calSet3Dr(heatModel, Q_temperature, chunk+1, j, z, INIT_TEMP);
						calSet3Dr(heatModel, Q_temperature, chunk-1, j, z, INIT_TEMP);
						//calSet3Dr(heatModel, Q_temperature, chunk*2, j, z, INIT_TEMP);
						//calSet3Dr(heatModel, Q_temperature, chunk*3, j, z, INIT_TEMP);
						//calSet3Db(heatModel, Q_heat_source, i, j, z, 1);


				}
			}
		}
//}


int main(int argc, char** argv) {

	//cadef and rundef
	heatModel = calCADef3D(ROWS, COLS, LAYERS, CAL_MOORE_NEIGHBORHOOD_3D, CAL_SPACE_FLAT, CAL_NO_OPT);
	heat_simulation = calRunDef3D(heatModel, 1, STEPS, CAL_UPDATE_IMPLICIT);
	//add substates
	Q_temperature = calAddSubstate3Dr(heatModel);
	Q_heat_source = calAddSubstate3Db(heatModel);
	//add transition function's elementary processes
	calAddElementaryProcess3D(heatModel, heatModel_TransitionFunction);

	//simulation run setup
	calRunAddInitFunc3D(heat_simulation, heatModel_SimulationInit);
	calRunInitSimulation3D(heat_simulation);	//It is required in the case the simulation main loop is explicitated; similarly for calRunFinalizeSimulation3D
//	calRunAddStopConditionFunc3D(heat_simulation, heatModel_SimulationStopCondition);

    struct OpenCALTime * opencalTime= (struct OpenCALTime *)malloc(sizeof(struct OpenCALTime));
    startTime(opencalTime);
    calRun3D(heat_simulation);
    endTime(opencalTime);

	// Save the substate to file
	calSaveSubstate3Dr(heatModel, Q_temperature, "./testsout/serial/1.txt");

	calRunFinalize3D(heat_simulation);
	calFinalize3D(heatModel);
	return 0;
}
