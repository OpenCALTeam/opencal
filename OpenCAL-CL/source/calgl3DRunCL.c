// (C) Copyright University of Calabria and others.
// All rights reserved. This program and the accompanying materials
// are made available under the terms of the GNU Lesser General Public License
// (LGPL) version 2.1 which accompanies this distribution, and is available at
// http://www.gnu.org/licenses/lgpl-2.1.html
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
// Lesser General Public License for more details.

#include <OpenCAL/cal3DIO.h>
#include <OpenCAL-CL/calgl3DRunCL.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __APPLE__
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif

// char* calglGetString3DCL(GLdouble number){
// 	char* toReturn = NULL;
// 	GLint tmp = (GLint)(number * 100);
// 	GLint tmpSave = tmp;
// 	GLint dimension = 0;
// 	GLint i = 0;
//
// 	while (tmp > 0){
// 		tmp /= 10;
// 		dimension++;
// 	}
// 	dimension += 2;
// 	tmp = tmpSave;
//
// 	toReturn = (char*)malloc(sizeof(char)*dimension);
//
// 	toReturn[dimension - 1] = '\0';
// 	for (i = dimension - 2; i >= 0; i--){
// 		if (i == dimension - 4){
// 			toReturn[i] = ',';
// 		}
// 		else {
// 			switch (tmp % 10){
// 			case 0: toReturn[i] = '0'; break;
// 			case 1: toReturn[i] = '1'; break;
// 			case 2: toReturn[i] = '2'; break;
// 			case 3: toReturn[i] = '3'; break;
// 			case 4: toReturn[i] = '4'; break;
// 			case 5: toReturn[i] = '5'; break;
// 			case 6: toReturn[i] = '6'; break;
// 			case 7: toReturn[i] = '7'; break;
// 			case 8: toReturn[i] = '8'; break;
// 			case 9: toReturn[i] = '9'; break;
// 			default: toReturn[i] = ' '; break;
// 			}
// 			tmp /= 10;
// 		}
// 	}
//
// 	return toReturn;
// }


// void calglSaveStateUpdater3DCL(struct CALGLRun3D* calglRun){
// 	int i = 0;
// 	char tmpString[50];
// 	struct CALModel3D* calModel = calglRun->device_CA->host_CA;
//
// 	printf("Saving final state to folder \"./data/\"\n");
//
// 	for (i = 0; i < calModel->sizeof_pQb_array; i++){
// 		strcpy(tmpString, "./data/byteSubstate");
// 		strcat(tmpString, calglGetString3DCL(i));
// 		strcat(tmpString, ".txt");
// 		calSaveSubstate3Db(calModel, calModel->pQb_array[i], tmpString);
// 	}
//
// 	for (i = 0; i < calModel->sizeof_pQi_array; i++){
// 		strcpy(tmpString, "./data/intSubstate");
// 		strcat(tmpString, calglGetString3DCL(i));
// 		strcat(tmpString, ".txt");
// 		calSaveSubstate3Di(calModel, calModel->pQi_array[i], tmpString);
// 	}
//
// 	for (i = 0; i < calModel->sizeof_pQr_array; i++){
// 		strcpy(tmpString, "./data/realSubstate");
// 		strcat(tmpString, calglGetString3DCL(i));
// 		strcat(tmpString, ".txt");
// 		calSaveSubstate3Dr(calModel, calModel->pQr_array[i], tmpString);
// 	}
// }

struct CALGLRun3D* calglRunCLDef3D(struct CALCLModel3D* device_CA, CALint fixedStep, CALint initial_step, CALint final_step){
	struct CALGLRun3D* calglRun = (struct CALGLRun3D*) malloc(sizeof(struct CALGLRun3D));

	calglRun->firstRun = CAL_TRUE;
	calglRun->active = CAL_FALSE;
	calglRun->terminated = CAL_FALSE;
	calglRun->stop = CAL_FALSE;
	calglRun->device_CA = device_CA;
	calglRun->onlyOneTime=CAL_FALSE;
	calglRun->fixedStep=fixedStep;
	calglRun->device_CA->steps=initial_step;
	calglRun->step = initial_step;
	calglRun->final_step=final_step;

	calglStartThread3DCL(calglRun);

	return calglRun;
}

void calglDestroyUpdater3DCL(struct CALGLRun3D* calglRun){
	if (calglRun){
		free(calglRun);
	}
}

void* calglFuncThreadUpdate3DCL(void* arg){
	struct CALGLRun3D* calglRun = (struct CALGLRun3D*) arg;

	while (!calglRun->stop){
		calglUpdate3DCL(calglRun);
		//Sleep(10);
	}

	return (void *)0;
}

void calglStartThread3DCL(struct CALGLRun3D* calglRun){
		pthread_create(&calglRun->thread, NULL, calglFuncThreadUpdate3DCL, (void *)calglRun);
}

void calglUpdate3DCL(struct CALGLRun3D* calglRun){
	if (calglRun->active){
		  //	cl_int err;
			CALbyte stop;
			size_t * threadNumMax = (size_t*) malloc(sizeof(size_t) * 3);
			threadNumMax[0] = calglRun->device_CA->host_CA->rows;
			threadNumMax[1] = calglRun->device_CA->host_CA->columns;
			threadNumMax[2] = calglRun->device_CA->host_CA->slices;
			size_t * singleStepThreadNum;
			int dimNum;

			if (calglRun->device_CA->opt == CAL_NO_OPT) {
				singleStepThreadNum = (size_t*) malloc(sizeof(size_t) * 3);
				singleStepThreadNum[0] = threadNumMax[0];
				singleStepThreadNum[1] = threadNumMax[1];
				singleStepThreadNum[2] = threadNumMax[2];
				dimNum = 3;
			} else {
				singleStepThreadNum = (size_t*) malloc(sizeof(size_t));
				singleStepThreadNum[0] = calglRun->device_CA->host_CA->A.size_current;
				dimNum = 1;
			}
		//	calclRoundThreadsNum(singleStepThreadNum, dimNum);
		if (calglRun->firstRun){
			calglRun->firstRun = CAL_FALSE;
			calglRun->start_time = time(NULL);
			if (calglRun->device_CA->kernelInitSubstates != NULL)
				calclKernelCall3D(calglRun->device_CA, calglRun->device_CA->kernelInitSubstates, dimNum, threadNumMax, NULL);
		}
		//simulation main loop
		calglRun->step=calglRun->device_CA->steps;
		//exectutes the global transition function, the steering function and check for the stop condition.
		calglRun->terminated = calclSingleStep3D(calglRun->device_CA, singleStepThreadNum, dimNum);
		//graphic rendering
		//#ifdef WIN32
		//		system("cls");
		//#else
		//		system("clear");
		//#endif
		//		printf("*----------------  Cellular Automata  ----------------*\n");
		//		printf(" Rows: %d, Columns: %d\n", calglGetGlobalSettings()->rows, calglGetGlobalSettings()->columns);
		//		printf(" Current Step: %d/%d; Active cells: %d\n", calglRun->calRun->step, calglGetGlobalSettings()->step, calglRun->calRun->ca3D->A.size_current);
		printf ("Cellular Automata: Current Step: %d\r", calglRun->device_CA->steps);
		//		printf("*-----------------------------------------------------*\n");
		//check for the stop condition
		if (calglRun->step%calglRun->fixedStep==0){
			calclGetSubstatesDeviceToHost3D(calglRun->device_CA);
		}
		if (calglRun->terminated || calglRun->step%calglRun->final_step==0)
		{
			calclGetSubstatesDeviceToHost3D(calglRun->device_CA);
			calglRun->active = CAL_FALSE;
			//breaking the simulation
			calglRun->end_time = time(NULL);
			printf("\nSimulation terminated\n");
			printf(" Elapsed time: %d\n",(int)( calglRun->end_time - calglRun->start_time));
			printf("*-----------------------------------------------------*\n");
			//saving configuration
			//calglSaveStateUpdater3DCL(calglRun);
		}
	}else{
		if(calglRun->onlyOneTime ){
			calglRun->onlyOneTime=CAL_FALSE;
			printf("\nSimulation Pause\n");
				calclGetSubstatesDeviceToHost3D(calglRun->device_CA);
				//calglSaveStateUpdater3DCL(calglRun);
		}

	}
}
