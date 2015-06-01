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

#include <calgl2DUpdater.h>
#include <cal2DIO.h>
#include <calglGlobalSettings.h>
#include <calgl2DWindow.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __APPLE__
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif

struct CALUpdater2D* calglCreateUpdater2D(struct CALRun2D* calRun){
	struct CALUpdater2D* calUpdater = (struct CALUpdater2D*) malloc(sizeof(struct CALUpdater2D));

	calUpdater->firstRun = CAL_TRUE;
	calUpdater->active = CAL_FALSE;
	calUpdater->terminated = CAL_FALSE;
	calUpdater->calRun = calRun;
	calUpdater->stop = CAL_FALSE;

	calglStartThread2D(calUpdater);

	return calUpdater;
}

void calglDestroyUpdater2D(struct CALUpdater2D* calUpdater){
	if(calUpdater){
		free(calUpdater);
	}
}

void* calglFuncThreadUpdate2D(void* arg){
	struct CALUpdater2D* calUpdater = (struct CALUpdater2D*) arg;

	while(!calUpdater->stop){
		calglUpdate2D(calUpdater);
		Sleep(10);	
	}

	return (void *) 0; 
}

void calglStartThread2D(struct CALUpdater2D* calUpdater){
	pthread_create(&calUpdater->thread, NULL, calglFuncThreadUpdate2D, (void *)calUpdater);
}

void calglUpdate2D(struct CALUpdater2D* calUpdater){
	if(calUpdater->active){
		if(calUpdater->firstRun){
			calUpdater->firstRun = CAL_FALSE;
			calUpdater->start_time = time(NULL);
		}
		//simulation main loop
		calUpdater->calRun->step++;
		//exectutes the global transition function, the steering function and check for the stop condition.
		calUpdater->terminated = calRunCAStep2D(calUpdater->calRun);
		//graphic rendering
#ifdef WIN32
		system("cls");
#else
		system("clear");
#endif
		printf("*----------------  Cellular Automata  ----------------*\n");
		printf(" Rows: %d, Columns: %d\n", calglGetGlobalSettings()->rows, calglGetGlobalSettings()->columns);
		printf(" Current Step: %d/%d; Active cells: %d\n", calUpdater->calRun->step, calglGetGlobalSettings()->step, calUpdater->calRun->ca2D->A.size_current);
		printf("*-----------------------------------------------------*\n");
		//check for the stop condition
		if (!calUpdater->terminated)
		{
			calUpdater->active = CAL_FALSE;
			//breaking the simulation
			calUpdater->end_time = time(NULL);
			printf("\nSimulation terminated\n");
			printf(" Elapsed time: %ds\n", calUpdater->end_time - calUpdater->start_time); 
			printf("*-----------------------------------------------------*\n");
			//saving configuration
			calglSaveStateUpdater2D(calUpdater);
			calUpdater->stop = CAL_TRUE;
		}
	}
}

void calglSaveStateUpdater2D(struct CALUpdater2D* calUpdater){
	int i=0;
	char tmpString[50];
	struct CALModel2D* calModel = calUpdater->calRun->ca2D;

	printf("Saving final state to folder \"./data/\"\n");

	for(i=0; i<calModel->sizeof_pQb_array; i++){
		strcpy(tmpString, "./data/byteSubstate");
		strcat(tmpString, calglGetString2D(i));
		strcat(tmpString, ".txt");		
		calSaveSubstate2Db(calModel, calModel->pQb_array[i], tmpString);
	}

	for(i=0; i<calModel->sizeof_pQi_array; i++){
		strcpy(tmpString, "./data/intSubstate");
		strcat(tmpString, calglGetString2D(i));
		strcat(tmpString, ".txt");		
		calSaveSubstate2Di(calModel, calModel->pQi_array[i], tmpString);
	}

	for(i=0; i<calModel->sizeof_pQr_array; i++){
		strcpy(tmpString, "./data/realSubstate");
		strcat(tmpString, calglGetString2D(i));
		strcat(tmpString, ".txt");		
		calSaveSubstate2Dr(calModel, calModel->pQr_array[i], tmpString);
	}
}


