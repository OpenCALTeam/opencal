/*
 * Copyright (c) 2016 OpenCALTeam (https://github.com/OpenCALTeam),
 * Telesio Research Group,
 * Department of Mathematics and Computer Science,
 * University of Calabria, Italy.
 *
 * This file is part of OpenCAL (Open Computing Abstraction Layer).
 *
 * OpenCAL is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3 of
 * the License, or (at your option) any later version.
 *
 * OpenCAL is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with OpenCAL. If not, see <http://www.gnu.org/licenses/>.
 */

#include <OpenCAL/cal2DIO.h>
#include <OpenCAL-GL/calgl2DRun.h>
#include <OpenCAL-GL/calgl2DWindow.h>
#include <OpenCAL-GL/calglGlobalSettings.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __APPLE__
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif

struct CALGLRun2D* calglRunDef2D(struct CALRun2D* calRun){
	struct CALGLRun2D* calglRun = (struct CALGLRun2D*) malloc(sizeof(struct CALGLRun2D));

	calglRun->firstRun = CAL_TRUE;
	calglRun->active = CAL_FALSE;
	calglRun->terminated = CAL_FALSE;
	calglRun->calRun = calRun;
	calglRun->stop = CAL_FALSE;

	calglStartThread2D(calglRun);

	return calglRun;
}

void calglDestroyUpdater2D(struct CALGLRun2D* calglRun){
	if (calglRun){
		free(calglRun);
	}
}

void* calglFuncThreadUpdate2D(void* arg){
	struct CALGLRun2D* calglRun = (struct CALGLRun2D*) arg;

	while (!calglRun->stop){
		calglUpdate2D(calglRun);
		//Sleep(10);
	}

	return (void *)0;
}

void calglStartThread2D(struct CALGLRun2D* calglRun){
	pthread_create(&calglRun->thread, NULL, calglFuncThreadUpdate2D, (void *)calglRun);
}

void calglUpdate2D(struct CALGLRun2D* calglRun){
	if (calglRun->active){
		if (calglRun->firstRun){
			calglRun->firstRun = CAL_FALSE;
			calglRun->start_time = time(NULL);
			if (calglRun->calRun->init)
				calRunInitSimulation2D (calglRun->calRun);
		}
		//simulation main loop
		calglRun->calRun->step++;
		calglRun->step=calglRun->calRun->step;
		//exectutes the global transition function, the steering function and check for the stop condition.
		calglRun->terminated = calRunCAStep2D(calglRun->calRun);
		//graphic rendering
		//#ifdef WIN32
		//		system("cls");
		//#else
		//		system("clear");
		//#endif
		//printf("*----------------  Cellular Automata  ----------------*\n");
		//printf(" Rows: %d, Columns: %d\n", calglGetGlobalSettings()->rows, calglGetGlobalSettings()->columns);
		//printf(" Current Step: %d/%d; Active cells: %d\n", calglRun->calRun->step, calglGetGlobalSettings()->step, calglRun->calRun->ca2D->A.size_current);
	        printf ("Cellular Automata: Current Step: %d/%d;", calglRun->calRun->step, calglRun->calRun->final_step);	
		if(calglRun->calRun->ca2D->A){
		
			printf ("Active cells: %d\r", calglRun->calRun->ca2D->A->size_current);
		}else
			printf("\r");
		//printf ("*-----------------------------------------------------*\n");
		//check for the stop condition
		if (!calglRun->terminated)
		{
			calglRun->active = CAL_FALSE;
			//breaking the simulation
			calglRun->end_time = time(NULL);
			printf("\nSimulation terminated\n");
			printf(" Elapsed time: %d\n", (int)(calglRun->end_time - calglRun->start_time));
			printf("*-----------------------------------------------------*\n");
			//saving configuration
			//calglSaveStateUpdater2D(calglRun);
			calglRun->stop = CAL_TRUE;
		}
	}
}

// void calglSaveStateUpdater2D(struct CALGLRun2D* calglRun){
// 	int i = 0;
// 	char tmpString[50];
// 	struct CALModel2D* calModel = calglRun->calRun->ca2D;
//
// 	printf("Saving final state to folder \"./data/\"\n");
//
// 	for (i = 0; i < calModel->sizeof_pQb_array; i++){
// 		strcpy(tmpString, "./data/byteSubstate");
// 		strcat(tmpString, calglGetString2D(i));
// 		strcat(tmpString, ".txt");
// 		calSaveSubstate2Db(calModel, calModel->pQb_array[i], tmpString);
// 	}
//
// 	for (i = 0; i < calModel->sizeof_pQi_array; i++){
// 		strcpy(tmpString, "./data/intSubstate");
// 		strcat(tmpString, calglGetString2D(i));
// 		strcat(tmpString, ".txt");
// 		calSaveSubstate2Di(calModel, calModel->pQi_array[i], tmpString);
// 	}
//
// 	for (i = 0; i < calModel->sizeof_pQr_array; i++){
// 		strcpy(tmpString, "./data/realSubstate");
// 		strcat(tmpString, calglGetString2D(i));
// 		strcat(tmpString, ".txt");
// 		calSaveSubstate2Dr(calModel, calModel->pQr_array[i], tmpString);
// 	}
// }
