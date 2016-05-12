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

#include <OpenCAL/cal2DIO.h>
#include <OpenCAL-CL/calgl2DRunCL.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __APPLE__
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif

// char* GetString2D(GLdouble number) {
// 	char* toReturn = NULL;
// 	GLint tmp = (GLint) (number*100);
// 	GLint tmpSave = tmp;
// 	GLint dimension = 0;
// 	GLint i = 0;
//
// 	while(tmp>0) {
// 		tmp /= 10;
// 		dimension++;
// 	}
// 	dimension += 2;
// 	tmp = tmpSave;
//
// 	toReturn = (char*) malloc(sizeof(char)*dimension);
//
// 	toReturn[dimension-1] = '\0';
// 	for(i = dimension-2; i>=0; i--) {
// 		if(i==dimension-4) {
// 			toReturn[i] = ',';
// 		} else {
// 			switch(tmp%10) {
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

struct CALGLRun2D* calglRunCLDef2D(struct CALCLModel2D* deviceCA,CALint fixedStep,CALint initial_step,CALint final_step){
	struct CALGLRun2D* calglRun = (struct CALGLRun2D*) malloc(sizeof(struct CALGLRun2D));

	calglRun->firstRun = CAL_TRUE;
	calglRun->active = CAL_FALSE;
	calglRun->terminated = CAL_FALSE;
	calglRun->deviceCA = deviceCA;
	calglRun->stop = CAL_FALSE;
	calglRun->onlyOneTime=CAL_FALSE;
	calglRun->fixedStep=fixedStep;
	calglRun->deviceCA->steps=initial_step;
	calglRun->final_step=final_step;
	calglStartThread2DCL(calglRun);

	return calglRun;
}

void calglDestroyUpdater2DCL(struct CALGLRun2D* calglRun){
	if (calglRun){
		free(calglRun);
	}
}

void* calglFuncThreadUpdate2DCL(void* arg){
	struct CALGLRun2D* calglRun = (struct CALGLRun2D*) arg;

	while (!calglRun->stop){
		calglUpdate2DCL(calglRun);
		//Sleep(10);
	}

	return (void *)0;
}

void calglStartThread2DCL(struct CALGLRun2D* calglRun){
	pthread_create(&calglRun->thread, NULL, calglFuncThreadUpdate2DCL, (void *)calglRun);
}

void calglUpdate2DCL(struct CALGLRun2D* calglRun){

	if (calglRun->active){
		calglRun->onlyOneTime=CAL_TRUE;
		size_t * threadNumMax = (size_t*) malloc(sizeof(size_t) * 2);
		threadNumMax[0] = calglRun->deviceCA->host_CA->rows;
		threadNumMax[1] = calglRun->deviceCA->host_CA->columns;
		size_t * singleStepThreadNum;
		int dimNum;

		if (calglRun->deviceCA->opt == CAL_NO_OPT) {
			singleStepThreadNum = (size_t*) malloc(sizeof(size_t) * 2);
			singleStepThreadNum[0] = threadNumMax[0];
			singleStepThreadNum[1] = threadNumMax[1];
			dimNum = 2;
		} else {
			singleStepThreadNum = (size_t*) malloc(sizeof(size_t));
			singleStepThreadNum[0] = calglRun->deviceCA->host_CA->A->size_current;
			dimNum = 1;
		}
		//system("clear");


		if (calglRun->firstRun){
			calglRun->firstRun = CAL_FALSE;
			calglRun->start_time = time(NULL);
			if (calglRun->deviceCA->kernelInitSubstates != NULL)
				calclKernelCall2D(calglRun->deviceCA, calglRun->deviceCA->kernelInitSubstates, 1, threadNumMax, NULL);
		}

		//calglRun->deviceCA->steps++;
		calglRun->step=calglRun->deviceCA->steps;
		//printf("%d\n",calglRun->step);
		//printf ("VEROOOOOOOOOOO \n");
		calglRun->terminated = calclSingleStep2D(calglRun->deviceCA, singleStepThreadNum, dimNum);
		printf ("Cellular Automata: Current Step: %d \r", calglRun->step);



    //calclGetSubstateKernel2D(calglRun->deviceCA, calglRun->host_CA);
		//printf("calglRun->step = %d\n",calglRun->step);
		//printf("calglRun->fixedStep = %d\n",calglRun->fixedStep);
		//printf("calglRun->step%calglRun->fixedStep==0  %d\n",calglRun->step%calglRun->fixedStep);
		if (calglRun->step%calglRun->fixedStep==0){
			calclGetSubstatesDeviceToHost2D(calglRun->deviceCA);
		}
			if (calglRun->terminated || calglRun->step%calglRun->final_step==0)
			{
				calclGetSubstatesDeviceToHost2D(calglRun->deviceCA);

				calglRun->active = CAL_FALSE;
				//breaking the simulation
				calglRun->end_time = time(NULL);
				printf("\nSimulation terminated\n");
				printf(" Elapsed time: %d\n", (int)(calglRun->end_time - calglRun->start_time));
				printf("*-----------------------------------------------------*\n");
				//saving configuration
				//calglSaveStateUpdater2DCL(calglRun);
				calglRun->stop = CAL_TRUE;
			}


	}else{
		if(calglRun->onlyOneTime ){
			calglRun->onlyOneTime=CAL_FALSE;
			printf("\nSimulation Pause\n");
				calclGetSubstatesDeviceToHost2D(calglRun->deviceCA);
				//calglSaveStateUpdater2DCL(calglRun);
		}

	}


}

// void calglSaveStateUpdater2DCL(struct CALGLRun2D* calglRun){
// 	int i = 0;
// 	char tmpString[50];
// 	struct CALModel2D* calModel = calglRun->deviceCA->host_CA;
//
// 	printf("Saving final state to folder \"./data/\"\n");
//
// 	for (i = 0; i < calModel->sizeof_pQb_array; i++){
//
// 		strcpy(tmpString, "./data/byteSubstate");
// 		strcat(tmpString, GetString2D(i));
// 		strcat(tmpString, ".txt");
//
// 		calSaveSubstate2Db(calModel, calModel->pQb_array[i], tmpString);
// 	}
//
// 	for (i = 0; i < calModel->sizeof_pQi_array; i++){
// 		strcpy(tmpString, "./data/intSubstate");
// 		strcat(tmpString, GetString2D(i));
// 		strcat(tmpString, ".txt");
// 		calSaveSubstate2Di(calModel, calModel->pQi_array[i], tmpString);
// 	}
//
// 	for (i = 0; i < calModel->sizeof_pQr_array; i++){
// 		strcpy(tmpString, "./data/realSubstate");
// 		strcat(tmpString, GetString2D(i));
// 		strcat(tmpString, ".txt");
// 		calSaveSubstate2Dr(calModel, calModel->pQr_array[i], tmpString);
// 	}
// }
