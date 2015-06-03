
#include "sciddicaT.h"
#include <calcl2D.h>
#ifdef __APPLE__
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif
#include <time.h>

#ifdef _WIN32
int setenv(const char *name, const char *value, int overwrite)
{
	int errcode = 0;
	if(!overwrite) {
		size_t envsize = 0;
		errcode = getenv_s(&envsize, NULL, 0, name);
		if(errcode || envsize) return errcode;
	}
	return _putenv_s(name, value);
}
#endif

//#define KERNEL_SOURCE_USER_DIR ROOT_DIR"/sciddicaTglut_CL/kernel/source/"
//#define KERNEL_INCLUDE_USER_DIR ROOT_DIR"/sciddicaTglut_CL/kernel/include"

#define KER_SCIDDICA_ELEMENTARY_PROCESS_ONE "sciddicaT_flows_computation"
#define KER_SCIDDICA_ELEMENTARY_PROCESS_TWO "sciddicaT_width_update"
#define KER_SCIDDICA_ELEMENTARY_PROCESS_THREE "sciddicaT_remove_inactive_cells"
#define KER_SCIDDICA_INITSUBSTATES  "sciddicaTInitSubstates"
#define KER_SCIDDICA_STEERING  "sciddicaTSteering"

size_t * dimSize;

time_t start_time, end_time;

CALreal z_min, z_Max, h_min, h_Max;

void sciddicaTComputeExtremes(struct CALModel2D* ca2D, struct CALSubstate2Dr* Q, CALreal* m, CALreal* M) {
	int i, j;

	//computing min and max z
	for (i = 0; i < ca2D->rows; i++)
		for (j = 0; j < ca2D->columns; j++)
			if (calGet2Dr(ca2D, Q, i, j) > 0) {
				*m = calGet2Dr(ca2D, Q, i, j);
				*M = calGet2Dr(ca2D, Q, i, j);
			}
	for (i = 0; i < ca2D->rows; i++)
		for (j = 0; j < ca2D->columns; j++) {
			if (*M < calGet2Dr(ca2D, Q, i, j) && calGet2Dr(ca2D, Q, i, j) > 0)
				*M = calGet2Dr(ca2D, Q, i, j);
			if (*m > calGet2Dr(ca2D, Q, i, j) && calGet2Dr(ca2D, Q, i, j) > 0)
				*m = calGet2Dr(ca2D, Q, i, j);
		}
}

int main(int argc, char** argv) {

	start_time = time(NULL);
//	int steps = atoi(argv[1]);
//	char * outputPath = argv[2];
//	int active = atoi(argv[3]);
//	int platformNum = atoi(argv[4]);
//	int deviceNum = atoi(argv[5]);
	int steps = 4000;
	char * outputPath = "./CALCL_Sciddica/result/result";
	int active = 1;
	int platformNum = 0;
	int deviceNum = 0;

	char * kernelSrc;
	char * kernelInc;
	if (active == 0) {
		kernelSrc = KERNEL_SRC;
		kernelInc = KERNEL_INC;
	} else {
		kernelSrc = KERNEL_SRC_AC;
		kernelInc = KERNEL_INC_AC;

	}

	setenv("CUDA_CACHE_DISABLE", "1", 1);
	//---------------------------------OPENCL INIT----------------------------------/

	CALOpenCL * calOpenCL = calclCreateCALOpenCL();
	calclInitializePlatforms(calOpenCL);
	calclInitializeDevices(calOpenCL);

	CALCLdevice device = calclGetDevice(calOpenCL, platformNum, deviceNum);

	CALCLcontext context = calclcreateContext(&device, 1);

	CALCLprogram program = calclLoadProgramLib2D(context, device, kernelSrc, kernelInc);

	//---------------------------------Parallel CA DEF & INIT----------------------------------/

	sciddicaTCADef();
	sciddicaTLoadConfig();
	sciddicaTComputeExtremes(sciddicaT, Q.z, &z_min, &z_Max);
	sciddicaTsimulation->init(sciddicaT);
	calUpdate2D(sciddicaT);

	CALCLToolkit2D * sciddicaToolkit = NULL;

	if (active == 0)
		sciddicaToolkit = calclCreateToolkit2D(sciddicaT, context, program, device, CAL_NO_OPT);
	else
		sciddicaToolkit = calclCreateToolkit2D(sciddicaT, context, program, device, CAL_OPT_ACTIVE_CELLS);

	CALCLkernel kernel_elementary_process_one = calclGetKernelFromProgram(&program, KER_SCIDDICA_ELEMENTARY_PROCESS_ONE);
	CALCLkernel kernel_elementary_process_two = calclGetKernelFromProgram(&program, KER_SCIDDICA_ELEMENTARY_PROCESS_TWO);
	CALCLkernel kernel_steering = calclGetKernelFromProgram(&program, KER_SCIDDICA_STEERING);

	CALCLmem * buffersKernelOne = (CALCLmem *) malloc(sizeof(CALCLmem) * 2);
	CALCLmem bufferEpsilonParameter = calclCreateBuffer(context, &P.epsilon, sizeof(CALParameterr));
	CALCLmem bufferRParameter = calclCreateBuffer(context, &P.r, sizeof(CALParameterr));
	buffersKernelOne[0] = bufferEpsilonParameter;
	buffersKernelOne[1] = bufferRParameter;
	calclSetCALKernelArgs2D(&kernel_elementary_process_one, buffersKernelOne, 2);

	calclAddElementaryProcessKernel2D(sciddicaToolkit, sciddicaT, &kernel_elementary_process_one);
	calclAddElementaryProcessKernel2D(sciddicaToolkit, sciddicaT, &kernel_elementary_process_two);

	calclSetSteeringKernel2D(sciddicaToolkit, sciddicaT, &kernel_steering);

	calclRun2D(sciddicaToolkit, sciddicaT, steps);

	sciddicaTSaveConfig(outputPath);
	sciddicaTExit();
	calclFinalizeCALOpencl(calOpenCL);
	calclFinalizeToolkit2D(sciddicaToolkit);

	end_time = time(NULL);
	printf("%d", end_time - start_time);

	return 0;
}
