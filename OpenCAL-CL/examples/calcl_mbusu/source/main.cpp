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
#include "../../../include/OpenCAL-CL/calcl3D.h"
}
#define STEPS 0
#include <time.h>

#define KER_SCIARA_ELEMENTARY_PROCESS_ONE "mbusuTransitionFunction"
#define KER_SCIARA_STOP_CONDITION "stopCondition"
#define KER_SCIARA_STEERING  "steering"

FILE *stream;
Mbusu *mbusu;
int active;
time_t start_time, end_time;

void setParameters(){
	mbusu->parameters.ascii_output_time_step = 8;				//[s] in seconds
	mbusu->parameters.lato = 5.0;
	mbusu->parameters.delta_t = 10.0;
	mbusu->parameters.delta_t_cum = 0.0;
	mbusu->parameters.delta_t_cum_prec = 0.0;
	mbusu->parameters.tetas1 = 0.368, mbusu->parameters.tetas2 = 0.351, mbusu->parameters.tetas3 = 0.325, mbusu->parameters.tetas4 = 0.325;
	mbusu->parameters.tetar1 = 0.102, mbusu->parameters.tetar2 = 0.0985, mbusu->parameters.tetar3 = 0.0859, mbusu->parameters.tetar4 = 0.0859;
	mbusu->parameters.alfa1 = 0.0334, mbusu->parameters.alfa2 = 0.0363, mbusu->parameters.alfa3 = 0.0345, mbusu->parameters.alfa4 = 0.0345;
	mbusu->parameters.n1 = 1.982, mbusu->parameters.n2 = 1.632, mbusu->parameters.n3 = 1.573, mbusu->parameters.n4 = 1.573;
	mbusu->parameters.ks1 = 0.009154, mbusu->parameters.ks2 = 0.005439, mbusu->parameters.ks3 = 0.004803, mbusu->parameters.ks4 = 0.048032;
	mbusu->parameters.rain = 0.000023148148;
}


int main(int argc, char** argv) {


	int steps = STEPS;
	int platformNum = 0;
	int deviceNum = 0;
	active = 0;
	const char * kernelSrc = "./kernel/source/";
	const char * kernelInc = "./kernel/include/";


	//---------------------------------OPENCL INIT----------------------------------/

	struct CALCLDeviceManager * calcl_device_manager = calclCreateManager();
	calclPrintPlatformsAndDevices(calcl_device_manager);

	CALCLdevice device = calclGetDevice(calcl_device_manager, platformNum, deviceNum);

	CALCLcontext context = calclCreateContext(&device);

	CALCLprogram program = calclLoadProgram3D(context, device, (char *)kernelSrc, (char *)kernelInc);
	initMbusu();
	setParameters();
	simulationInitialize();
	CALCLModel3D * device_CA = NULL;

	device_CA = calclCADef3D(mbusu->host_CA,context,program,device);

	CALCLkernel kernel_elementary_process_one = calclGetKernelFromProgram(&program, (char *)KER_SCIARA_ELEMENTARY_PROCESS_ONE);
	CALCLkernel kernel_stop_condition = calclGetKernelFromProgram(&program, (char *)KER_SCIARA_STOP_CONDITION);
	CALCLkernel kernel_steering = calclGetKernelFromProgram(&program, (char *)KER_SCIARA_STEERING);

	CALCLmem parametersBuff = calclCreateBuffer(context, &mbusu->parameters, sizeof(Parameters));
	clSetKernelArg(kernel_elementary_process_one, MODEL_ARGS_NUM, sizeof(CALCLmem), &parametersBuff);
	clSetKernelArg(kernel_stop_condition, MODEL_ARGS_NUM, sizeof(CALCLmem), &parametersBuff);

	//steering
	clSetKernelArg(kernel_steering, MODEL_ARGS_NUM, sizeof(CALCLmem), &parametersBuff);

	calclAddElementaryProcess3D(device_CA, &kernel_elementary_process_one);

	calclAddSteeringFunc3D(device_CA, &kernel_steering);
	calclAddStopConditionFunc3D(device_CA, &kernel_stop_condition);

	start_time = time(NULL);
	calclRun3D(device_CA, 1, STEPS);
	end_time = time(NULL);

	CALreal moist_print;
	CALint k_inv;
	int j = YOUT/2;

	for (int k = 0; k < mbusu->layers; k++)
		for (int i = 0; i < mbusu->rows; i++)
			{

				k_inv = (mbusu->layers - 1) - k;
				moist_print = calGet3Dr(mbusu->host_CA, mbusu->Q->moist_cont, i, j, k);
				//printf("%.3Lf\n",moist_print);
				if (i == XW && k_inv == ZSUP)
				{
					stream = fopen("ris10g.txt", "w+");
					fprintf(stream, "%f\t", moist_print);
				}
				else if (i == XE && k_inv == ZFONDO)
				{
					fprintf(stream, "%f\n",moist_print);
					fclose(stream);
				}
				else if (i == XE)
					fprintf(stream, "%f\n", moist_print);
				else
					fprintf(stream, "%f\t", moist_print);

			}

	calclFinalizeManager(calcl_device_manager);
	calclFinalize3D(device_CA);
	exit();


	printf("%lds", end_time - start_time);

	return 0;

}
