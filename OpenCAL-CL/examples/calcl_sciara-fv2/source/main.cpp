/*
 * main.cpp
 *
 *  Created on: 14/ott/2014
 *      Author: Maurizio
 */

#include "io.h"
#include "Sciara.h"
extern "C"{
#include "../../../include/OpenCAL-CL/calcl2D.h"
}
#include <time.h>

#define CONFIG_PATH "./data/2006/2006"
#define SAVE_PATH "./data/2006_SAVE/2006"
#define DEM_PATH CONFIG_PATH"_000000000000_Morphology.stt"

#define KER_SCIARA_ELEMENTARY_PROCESS_ONE "updateVentsEmission"
#define KER_SCIARA_ELEMENTARY_PROCESS_TWO "empiricalFlows"
#define KER_SCIARA_ELEMENTARY_PROCESS_THREE "width_update"
#define KER_SCIARA_ELEMENTARY_PROCESS_FOUR "updateTemperature"
#define KER_SCIARA_ELEMENTARY_PROCESS_FIVE  "removeActiveCells"
#define KER_SCIARA_STOP_CONDITION "stopCondition"
#define KER_SCIARA_STEERING  "steering"

Sciara *sciara;
int active;
time_t start_time, end_time;

void ventsMapper(CALCLToolkit2D * sciddicaToolkit, CALCLcontext &context, cl_kernel &updateVentsEmissionKernel) {
	typedef struct {
		int x;
		int y;
	} Vent;

	int ventsSize = sciara->vent.size();
	int emissionRateSize = sciara->vent[0].size() * sciara->vent.size();
	int singleEmissionRateSize = sciara->vent[0].size();
	Vent * vents = (Vent *) malloc(sizeof(Vent) * ventsSize);
	CALreal * emissionRates = (CALreal *) malloc(sizeof(CALreal) * emissionRateSize);

	int count = 0;
	for (int i = 0; i < sciara->vent.size(); i++) {
		vents[i].x = sciara->vent[i].x();
		vents[i].y = sciara->vent[i].y();
		for (int j = 0; j < sciara->vent[i].size(); j++) {
			emissionRates[count] = sciara->vent[i][j];
			count++;
		}
	}
	CALCLmem ventsBuffer = calclCreateBuffer(context, vents, sizeof(Vent) * ventsSize);
	CALCLmem emissionRateBuffer = calclCreateBuffer(context, emissionRates, sizeof(CALreal) * emissionRateSize);
	clSetKernelArg(updateVentsEmissionKernel, MODEL_ARGS_NUM, sizeof(CALCLmem), &ventsBuffer);
	clSetKernelArg(updateVentsEmissionKernel, MODEL_ARGS_NUM + 1, sizeof(CALCLmem), &emissionRateBuffer);

	//Private
	clSetKernelArg(updateVentsEmissionKernel, MODEL_ARGS_NUM + 3, sizeof(int), &ventsSize);
	clSetKernelArg(updateVentsEmissionKernel, MODEL_ARGS_NUM + 4, sizeof(int), &singleEmissionRateSize);

}

int main(int argc, char** argv) {

	start_time = time(NULL);

	setenv("CUDA_CACHE_DISABLE", "1", 1);
	const char * outputPath = SAVE_PATH;
	int steps =1000;
	int platformNum = 0;
	int deviceNum = 0;
	active = 0;
	const char * kernelSrc;
	const char * kernelInc;
	if (active == 0) {
		kernelSrc = KERNEL_SRC;
		kernelInc = KERNEL_INC;
	} else {
		kernelSrc = KERNEL_SRC_AC;
		kernelInc = KERNEL_INC_AC;

	}

	//---------------------------------OPENCL INIT----------------------------------/

	CALOpenCL * calOpenCL = calclCreateCALOpenCL();
	calclInitializePlatforms(calOpenCL);
	calclInitializeDevices(calOpenCL);

	CALCLdevice device = calclGetDevice(calOpenCL, platformNum, deviceNum);

	CALCLcontext context = calclCreateContext(&device, 1);

	CALCLprogram program = calclLoadProgramLib2D(context, device,(char*) kernelSrc,(char*) kernelInc);

	char path[1024] = CONFIG_PATH;
	initSciara((char*)DEM_PATH);
	int err = loadConfiguration(path, sciara);
	if (err == 0) {
		perror("cannot load configuration\n");
		exit(EXIT_FAILURE);
	}
	simulationInitialize(sciara->model);
	CALCLToolkit2D * sciaraToolkit = NULL;

	if (active == 0)
		sciaraToolkit = calclCreateToolkit2D(sciara->model, context, program, device, CAL_NO_OPT);
	else
		sciaraToolkit = calclCreateToolkit2D(sciara->model, context, program, device, CAL_OPT_ACTIVE_CELLS);

	CALCLkernel kernel_elementary_process_one = calclGetKernelFromProgram(&program,(char*) KER_SCIARA_ELEMENTARY_PROCESS_ONE);
	CALCLkernel kernel_elementary_process_two = calclGetKernelFromProgram(&program,(char*) KER_SCIARA_ELEMENTARY_PROCESS_TWO);
	CALCLkernel kernel_elementary_process_three = calclGetKernelFromProgram(&program,(char*) KER_SCIARA_ELEMENTARY_PROCESS_THREE);
	CALCLkernel kernel_elementary_process_four = calclGetKernelFromProgram(&program,(char*) KER_SCIARA_ELEMENTARY_PROCESS_FOUR);
	CALCLkernel kernel_stop_condition = calclGetKernelFromProgram(&program,(char*) KER_SCIARA_STOP_CONDITION);
	CALCLkernel kernel_steering = calclGetKernelFromProgram(&program,(char*) KER_SCIARA_STEERING);

	CALCLmem elapsed_timeBuffer = calclCreateBuffer(context, &sciara->elapsed_time, sizeof(CALreal));

	CALCLmem mbBuffer = calclCreateBuffer(context, sciara->substates->Mb->current, sizeof(CALbyte) * (sciara->rows * sciara->cols));
	CALCLmem mslBuffer = calclCreateBuffer(context, sciara->substates->Msl->current, sizeof(CALreal) * (sciara->rows * sciara->cols));

	// first kernel
	ventsMapper(sciaraToolkit, context, kernel_elementary_process_one);

	clSetKernelArg(kernel_elementary_process_one, MODEL_ARGS_NUM + 2, sizeof(CALCLmem), &elapsed_timeBuffer);
	clSetKernelArg(kernel_elementary_process_one, MODEL_ARGS_NUM + 5, sizeof(Parameters), &sciara->parameters);

	// second kernel
	clSetKernelArg(kernel_elementary_process_two, MODEL_ARGS_NUM, sizeof(Parameters), &sciara->parameters);

	// four kernel
	clSetKernelArg(kernel_elementary_process_four, MODEL_ARGS_NUM, sizeof(CALCLmem), &mbBuffer);
	clSetKernelArg(kernel_elementary_process_four, MODEL_ARGS_NUM + 1, sizeof(CALCLmem), &mslBuffer);
	clSetKernelArg(kernel_elementary_process_four, MODEL_ARGS_NUM + 2, sizeof(Parameters), &sciara->parameters);

	//steering
	clSetKernelArg(kernel_steering, MODEL_ARGS_NUM, sizeof(CALCLmem), &mbBuffer);
	clSetKernelArg(kernel_steering, MODEL_ARGS_NUM + 1, sizeof(Parameters), &sciara->parameters);
	clSetKernelArg(kernel_steering, MODEL_ARGS_NUM + 2, sizeof(CALCLmem), &elapsed_timeBuffer);

	//stop condition
	clSetKernelArg(kernel_stop_condition, MODEL_ARGS_NUM, sizeof(Parameters), &sciara->parameters);
	clSetKernelArg(kernel_stop_condition, MODEL_ARGS_NUM + 1, sizeof(CALCLmem), &elapsed_timeBuffer);

	calclAddElementaryProcessKernel2D(sciaraToolkit, sciara->model, &kernel_elementary_process_one);
	calclAddElementaryProcessKernel2D(sciaraToolkit, sciara->model, &kernel_elementary_process_two);
	calclAddElementaryProcessKernel2D(sciaraToolkit, sciara->model, &kernel_elementary_process_three);
	calclAddElementaryProcessKernel2D(sciaraToolkit, sciara->model, &kernel_elementary_process_four);

	if (active) {
		CALCLkernel kernel_elementary_process_five = calclGetKernelFromProgram(&program,(char*) KER_SCIARA_ELEMENTARY_PROCESS_FIVE);
		clSetKernelArg(kernel_elementary_process_five, MODEL_ARGS_NUM, sizeof(CALCLmem), &mbBuffer);
		clSetKernelArg(kernel_elementary_process_five, MODEL_ARGS_NUM + 1, sizeof(Parameters), &sciara->parameters);
		calclAddElementaryProcessKernel2D(sciaraToolkit, sciara->model, &kernel_elementary_process_five);
	}

	calclSetSteeringKernel2D(sciaraToolkit, sciara->model, &kernel_steering);
	calclSetStopConditionKernel2D(sciaraToolkit, sciara->model, &kernel_stop_condition);

	calclRun2D(sciaraToolkit, sciara->model, 1, steps);

	clEnqueueReadBuffer(sciaraToolkit->queue,mslBuffer,CAL_TRUE,0,sizeof(CALreal)*sciara->rows*sciara->cols,sciara->substates->Msl->current,0,NULL,NULL);

	err = saveConfiguration((char*)outputPath, sciara);

	if (err == 0) {
		perror("cannot save configuration\n");
		exit(EXIT_FAILURE);
	}

	calclFinalizeCALOpencl(calOpenCL);
	calclFinalizeToolkit2D(sciaraToolkit);
	exitSciara();

	end_time = time(NULL);
	printf("%d", end_time - start_time);

	return 0;

}
