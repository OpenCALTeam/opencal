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
#include <OpenCAL-CL/calgl2DUpdaterCL.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __APPLE__
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif

char* GetString2D(GLdouble number) {
	char* toReturn = NULL;
	GLint tmp = (GLint) (number*100);
	GLint tmpSave = tmp;
	GLint dimension = 0;
	GLint i = 0;

	while(tmp>0) {
		tmp /= 10;
		dimension++;
	}
	dimension += 2;
	tmp = tmpSave;

	toReturn = (char*) malloc(sizeof(char)*dimension);

	toReturn[dimension-1] = '\0';
	for(i = dimension-2; i>=0; i--) {
		if(i==dimension-4) {
			toReturn[i] = ',';
		} else {
			switch(tmp%10) {
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

struct CALUpdater2D* calglCreateUpdater2DCL(struct CALCLModel2D* deviceCA,struct CALModel2D* hostCA,CALint fixedStep,CALint initial_step,CALint final_step){
	struct CALUpdater2D* calUpdater = (struct CALUpdater2D*) malloc(sizeof(struct CALUpdater2D));

	calUpdater->firstRun = CAL_TRUE;
	calUpdater->active = CAL_FALSE;
	calUpdater->terminated = CAL_FALSE;
	calUpdater->deviceCA = deviceCA;
	calUpdater->hostCA = hostCA;
	calUpdater->stop = CAL_FALSE;
	calUpdater->onlyOneTime=CAL_FALSE;
	calUpdater->fixedStep=fixedStep;
	calUpdater->deviceCA->steps=initial_step;
	calUpdater->final_step=final_step;
	//printf("CIAO\n" );
	calglStartThread2DCL(calUpdater);

	return calUpdater;
}

void calglDestroyUpdater2DCL(struct CALUpdater2D* calUpdater){
	if (calUpdater){
		free(calUpdater);
	}
}

void* calglFuncThreadUpdate2DCL(void* arg){
	struct CALUpdater2D* calUpdater = (struct CALUpdater2D*) arg;

	while (!calUpdater->stop){
		calglUpdate2DCL(calUpdater);
		//Sleep(10);
	}

	return (void *)0;
}

void calglStartThread2DCL(struct CALUpdater2D* calUpdater){
	pthread_create(&calUpdater->thread, NULL, calglFuncThreadUpdate2DCL, (void *)calUpdater);
}

void calglUpdate2DCL(struct CALUpdater2D* calUpdater){

	if (calUpdater->active){
		calUpdater->onlyOneTime=CAL_TRUE;
		size_t * threadNumMax = (size_t*) malloc(sizeof(size_t) * 2);
		threadNumMax[0] = calUpdater->hostCA->rows;
		threadNumMax[1] = calUpdater->hostCA->columns;
		size_t * singleStepThreadNum;
		int dimNum;

		if (calUpdater->deviceCA->opt == CAL_NO_OPT) {
			singleStepThreadNum = (size_t*) malloc(sizeof(size_t) * 2);
			singleStepThreadNum[0] = threadNumMax[0];
			singleStepThreadNum[1] = threadNumMax[1];
			dimNum = 2;
		} else {
			singleStepThreadNum = (size_t*) malloc(sizeof(size_t));
			singleStepThreadNum[0] = calUpdater->hostCA->A.size_current;
			dimNum = 1;
		}
		//system("clear");


		if (calUpdater->firstRun){
			calUpdater->firstRun = CAL_FALSE;
			calUpdater->start_time = time(NULL);
			if (calUpdater->deviceCA->kernelInitSubstates != NULL)
				calclKernelCall2D(calUpdater->deviceCA, calUpdater->deviceCA->kernelInitSubstates, 1, threadNumMax, NULL);
		}

		//calUpdater->deviceCA->steps++;
		calUpdater->step=calUpdater->deviceCA->steps;
		//printf("%d\n",calUpdater->step);
		//printf ("VEROOOOOOOOOOO \n");
		calUpdater->terminated = calclSingleStep2D(calUpdater->deviceCA, calUpdater->hostCA, singleStepThreadNum, dimNum);
		printf ("Cellular Automata: Current Step: %d \r", calUpdater->step);



    //calclGetSubstateKernel2D(calUpdater->deviceCA, calUpdater->hostCA);
		//printf("calUpdater->step = %d\n",calUpdater->step);
		//printf("calUpdater->fixedStep = %d\n",calUpdater->fixedStep);
		//printf("calUpdater->step%calUpdater->fixedStep==0  %d\n",calUpdater->step%calUpdater->fixedStep);
		if (calUpdater->step%calUpdater->fixedStep==0){
			calclGetSubstateKernel2D(calUpdater->deviceCA, calUpdater->hostCA);
		}
			if (calUpdater->terminated || calUpdater->step%calUpdater->final_step==0)
			{
				calclGetSubstateKernel2D(calUpdater->deviceCA, calUpdater->hostCA);

				calUpdater->active = CAL_FALSE;
				//breaking the simulation
				calUpdater->end_time = time(NULL);
				printf("\nSimulation terminated\n");
				printf(" Elapsed time: %ds\n", calUpdater->end_time - calUpdater->start_time);
				printf("*-----------------------------------------------------*\n");
				//saving configuration
				calglSaveStateUpdater2DCL(calUpdater);
				calUpdater->stop = CAL_TRUE;
			}


	}else{
		if(calUpdater->onlyOneTime ){
			calUpdater->onlyOneTime=CAL_FALSE;
			printf("\nSimulation Pause\n");
				calclGetSubstateKernel2D(calUpdater->deviceCA, calUpdater->hostCA);
				calglSaveStateUpdater2DCL(calUpdater);
		}

	}


}

void calglSaveStateUpdater2DCL(struct CALUpdater2D* calUpdater){
	int i = 0;
	char tmpString[50];
	struct CALModel2D* calModel = calUpdater->hostCA;

	printf("Saving final state to folder \"./data/\"\n");

	for (i = 0; i < calModel->sizeof_pQb_array; i++){

		strcpy(tmpString, "./data/byteSubstate");
		strcat(tmpString, GetString2D(i));
		strcat(tmpString, ".txt");

		calSaveSubstate2Db(calModel, calModel->pQb_array[i], tmpString);
	}

	for (i = 0; i < calModel->sizeof_pQi_array; i++){
		strcpy(tmpString, "./data/intSubstate");
		strcat(tmpString, GetString2D(i));
		strcat(tmpString, ".txt");
		calSaveSubstate2Di(calModel, calModel->pQi_array[i], tmpString);
	}

	for (i = 0; i < calModel->sizeof_pQr_array; i++){
		strcpy(tmpString, "./data/realSubstate");
		strcat(tmpString, GetString2D(i));
		strcat(tmpString, ".txt");
		calSaveSubstate2Dr(calModel, calModel->pQr_array[i], tmpString);
	}
}
