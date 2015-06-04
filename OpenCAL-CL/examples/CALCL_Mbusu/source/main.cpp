/*
 * main.c
 *
 *  Created on: 16/apr/2015
 *      Author: alessio
 */




/*
 * main.cpp
 *
 *  Created on: 14/ott/2014
 *      Author: Maurizio
 */

#include "MbusuCL.h"
extern "C"{
#include "calcl3D.h"
}
#define STEPS 10
#include <time.h>

#define KER_SCIARA_ELEMENTARY_PROCESS_ONE "mbusuTransitionFunction"
#define KER_SCIARA_STOP_CONDITION "stopCondition"
#define KER_SCIARA_STEERING  "steering"

FILE *stream;
Mbusu *mbusu;
int active;
time_t start_time, end_time;
int main(int argc, char** argv) {

	start_time = time(NULL);
//	int steps = atoi(argv[1]);
//	char path[1024];
//	strcpy(path, argv[2]);
//	char * demPath = (char*)malloc(sizeof(char)*(strlen(path)+strlen("_000000000000_Morphology.stt")+1));
//	strcpy(demPath, path);
//	strcat(demPath, "_000000000000_Morphology.stt\0");
//	char * outputPath = argv[3];
//	active = atoi(argv[4]);
//	int platformNum = atoi(argv[5]);
//	int deviceNum = atoi(argv[6]);

//	setenv("CUDA_CACHE_DISABLE", "1", 1);

	int steps = STEPS;
	int platformNum = 0;
	int deviceNum = 0;
	active = 0;
	char * kernelSrc;
	char * kernelInc;
	char * openCALCLPath;

	kernelSrc = "./kernel/source/";
	kernelInc = "./kernel/include/";
	//---------------------------------OPENCL INIT----------------------------------/

	CALOpenCL * calOpenCL = calclCreateCALOpenCL();
	calclInitializePlatforms(calOpenCL);
	calclInitializeDevices(calOpenCL);
	calclPrintAllPlatformAndDevices(calOpenCL);
	
	CALCLdevice device = calclGetDevice(calOpenCL, platformNum, deviceNum);

	CALCLcontext context = calclcreateContext(&device, 1);

	CALCLprogram program = calclLoadProgramLib3D(context, device, kernelSrc, kernelInc);
	initMbusu();
	simulationInitialize();
	CALCLToolkit3D * mbusuToolkit = NULL;

	mbusuToolkit = calclCreateToolkit3D(mbusu->model,context,program,device,CAL_NO_OPT);

	CALCLkernel kernel_elementary_process_one = calclGetKernelFromProgram(&program, KER_SCIARA_ELEMENTARY_PROCESS_ONE);
	CALCLkernel kernel_stop_condition = calclGetKernelFromProgram(&program, KER_SCIARA_STOP_CONDITION);
	CALCLkernel kernel_steering = calclGetKernelFromProgram(&program, KER_SCIARA_STEERING);

	CALCLmem parametersBuff = calclCreateBuffer(context, &mbusu->parameters, sizeof(Parameters));
	clSetKernelArg(kernel_elementary_process_one, MODEL_ARGS_NUM, sizeof(CALCLmem), &parametersBuff);
	clSetKernelArg(kernel_stop_condition, MODEL_ARGS_NUM, sizeof(CALCLmem), &parametersBuff);

	//steering
	clSetKernelArg(kernel_steering, MODEL_ARGS_NUM, sizeof(CALCLmem), &parametersBuff);

	calclAddElementaryProcessKernel3D(mbusuToolkit,mbusu->model,&kernel_elementary_process_one);

	calclSetSteeringKernel3D(mbusuToolkit, mbusu->model, &kernel_steering);
	calclSetStopConditionKernel3D(mbusuToolkit, mbusu->model, &kernel_stop_condition);
	calclRun3D(mbusuToolkit, mbusu->model,STEPS);

	CALreal moist_print;
	CALint k_inv;
	int j = YOUT/2;
	for (int k = 0; k < mbusu->layers; k++)
		for (int i = 0; i < mbusu->rows; i++)
			{
				k_inv = (mbusu->layers - 1) - k;
				moist_print = calGet3Dr(mbusu->model, mbusu->Q->moist_cont, i, j, k);
				if (i == XW && k_inv == ZSUP)
				{
					stream = fopen(SAVE_PATH, "w+");
					fprintf(stream, "%f\t", moist_print);
				}
				else if (i == XE && k_inv == ZFONDO)
				{
					fprintf(stream, "%f\n", moist_print);
					fclose(stream);
				}
				else if (i == XE)
					fprintf(stream, "%f\n", moist_print);
				else
					fprintf(stream, "%f\t", moist_print);
			}

	calclFinalizeCALOpencl(calOpenCL);
	calclFinalizeToolkit3D(mbusuToolkit);
	exit();

	end_time = time(NULL);
	printf("%d", end_time - start_time);

	return 0;

}

