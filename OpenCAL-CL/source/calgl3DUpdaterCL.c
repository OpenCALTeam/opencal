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
#include <OpenCAL-CL/calgl3DUpdaterCL.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __APPLE__
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif

char* calglGetString3DCL(GLdouble number){
	char* toReturn = NULL;
	GLint tmp = (GLint)(number * 100);
	GLint tmpSave = tmp;
	GLint dimension = 0;
	GLint i = 0;

	while (tmp > 0){
		tmp /= 10;
		dimension++;
	}
	dimension += 2;
	tmp = tmpSave;

	toReturn = (char*)malloc(sizeof(char)*dimension);

	toReturn[dimension - 1] = '\0';
	for (i = dimension - 2; i >= 0; i--){
		if (i == dimension - 4){
			toReturn[i] = ',';
		}
		else {
			switch (tmp % 10){
			case 0: toReturn[i] = '0'; break;
			case 1: toReturn[i] = '1'; break;
			case 2: toReturn[i] = '2'; break;
			case 3: toReturn[i] = '3'; break;
			case 4: toReturn[i] = '4'; break;
			case 5: toReturn[i] = '5'; break;
			case 6: toReturn[i] = '6'; break;
			case 7: toReturn[i] = '7'; break;
			case 8: toReturn[i] = '8'; break;
			case 9: toReturn[i] = '9'; break;
			default: toReturn[i] = ' '; break;
			}
			tmp /= 10;
		}
	}

	return toReturn;
}


void calglSaveStateUpdater3DCL(struct CALUpdater3D* calUpdater){
	int i = 0;
	char tmpString[50];
	struct CALModel3D* calModel = calUpdater->device_CA->host_CA;

	printf("Saving final state to folder \"./data/\"\n");

	for (i = 0; i < calModel->sizeof_pQb_array; i++){
		strcpy(tmpString, "./data/byteSubstate");
		strcat(tmpString, calglGetString3DCL(i));
		strcat(tmpString, ".txt");
		calSaveSubstate3Db(calModel, calModel->pQb_array[i], tmpString);
	}

	for (i = 0; i < calModel->sizeof_pQi_array; i++){
		strcpy(tmpString, "./data/intSubstate");
		strcat(tmpString, calglGetString3DCL(i));
		strcat(tmpString, ".txt");
		calSaveSubstate3Di(calModel, calModel->pQi_array[i], tmpString);
	}

	for (i = 0; i < calModel->sizeof_pQr_array; i++){
		strcpy(tmpString, "./data/realSubstate");
		strcat(tmpString, calglGetString3DCL(i));
		strcat(tmpString, ".txt");
		calSaveSubstate3Dr(calModel, calModel->pQr_array[i], tmpString);
	}
}

struct CALUpdater3D* calglCreateUpdater3DCL(struct CALCLModel3D* device_CA, CALint fixedStep, CALint initial_step, CALint final_step){
	struct CALUpdater3D* calUpdater = (struct CALUpdater3D*) malloc(sizeof(struct CALUpdater3D));

	calUpdater->firstRun = CAL_TRUE;
	calUpdater->active = CAL_FALSE;
	calUpdater->terminated = CAL_FALSE;
	calUpdater->stop = CAL_FALSE;
	calUpdater->device_CA = device_CA;
	calUpdater->onlyOneTime=CAL_FALSE;
	calUpdater->fixedStep=fixedStep;
	calUpdater->device_CA->steps=initial_step;
	calUpdater->step = initial_step;
	calUpdater->final_step=final_step;

	calglStartThread3DCL(calUpdater);

	return calUpdater;
}

void calglDestroyUpdater3DCL(struct CALUpdater3D* calUpdater){
	if (calUpdater){
		free(calUpdater);
	}
}

void* calglFuncThreadUpdate3DCL(void* arg){
	struct CALUpdater3D* calUpdater = (struct CALUpdater3D*) arg;

	while (!calUpdater->stop){
		calglUpdate3DCL(calUpdater);
		//Sleep(10);
	}

	return (void *)0;
}

void calglStartThread3DCL(struct CALUpdater3D* calUpdater){
		pthread_create(&calUpdater->thread, NULL, calglFuncThreadUpdate3DCL, (void *)calUpdater);
}

void calglUpdate3DCL(struct CALUpdater3D* calUpdater){
	if (calUpdater->active){
		  //	cl_int err;
			CALbyte stop;
			size_t * threadNumMax = (size_t*) malloc(sizeof(size_t) * 3);
			threadNumMax[0] = calUpdater->device_CA->host_CA->rows;
			threadNumMax[1] = calUpdater->device_CA->host_CA->columns;
			threadNumMax[2] = calUpdater->device_CA->host_CA->slices;
			size_t * singleStepThreadNum;
			int dimNum;

			if (calUpdater->device_CA->opt == CAL_NO_OPT) {
				singleStepThreadNum = (size_t*) malloc(sizeof(size_t) * 3);
				singleStepThreadNum[0] = threadNumMax[0];
				singleStepThreadNum[1] = threadNumMax[1];
				singleStepThreadNum[2] = threadNumMax[2];
				dimNum = 3;
			} else {
				singleStepThreadNum = (size_t*) malloc(sizeof(size_t));
				singleStepThreadNum[0] = calUpdater->device_CA->host_CA->A.size_current;
				dimNum = 1;
			}
		//	calclRoundThreadsNum(singleStepThreadNum, dimNum);
		if (calUpdater->firstRun){
			calUpdater->firstRun = CAL_FALSE;
			calUpdater->start_time = time(NULL);
			if (calUpdater->device_CA->kernelInitSubstates != NULL)
				calclKernelCall3D(calUpdater->device_CA, calUpdater->device_CA->kernelInitSubstates, dimNum, threadNumMax, NULL);
		}
		//simulation main loop
		calUpdater->step=calUpdater->device_CA->steps;
		//exectutes the global transition function, the steering function and check for the stop condition.
		calUpdater->terminated = calclSingleStep3D(calUpdater->device_CA, singleStepThreadNum, dimNum);
		//graphic rendering
		//#ifdef WIN32
		//		system("cls");
		//#else
		//		system("clear");
		//#endif
		//		printf("*----------------  Cellular Automata  ----------------*\n");
		//		printf(" Rows: %d, Columns: %d\n", calglGetGlobalSettings()->rows, calglGetGlobalSettings()->columns);
		//		printf(" Current Step: %d/%d; Active cells: %d\n", calUpdater->calRun->step, calglGetGlobalSettings()->step, calUpdater->calRun->ca3D->A.size_current);
		printf ("Cellular Automata: Current Step: %d\r", calUpdater->device_CA->steps);
		//		printf("*-----------------------------------------------------*\n");
		//check for the stop condition
		if (calUpdater->step%calUpdater->fixedStep==0){
			calclGetSubstatesDeviceToHost3D(calUpdater->device_CA);
		}
		if (calUpdater->terminated || calUpdater->step%calUpdater->final_step==0)
		{
			calclGetSubstatesDeviceToHost3D(calUpdater->device_CA);
			calUpdater->active = CAL_FALSE;
			//breaking the simulation
			calUpdater->end_time = time(NULL);
			printf("\nSimulation terminated\n");
			printf(" Elapsed time: %d\n",(int)( calUpdater->end_time - calUpdater->start_time));
			printf("*-----------------------------------------------------*\n");
			//saving configuration
			calglSaveStateUpdater3DCL(calUpdater);
		}
	}else{
		if(calUpdater->onlyOneTime ){
			calUpdater->onlyOneTime=CAL_FALSE;
			printf("\nSimulation Pause\n");
				calclGetSubstatesDeviceToHost3D(calUpdater->device_CA);
				calglSaveStateUpdater3DCL(calUpdater);
		}

	}
}
