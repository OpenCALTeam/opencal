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
		int err;
		int sizeCA = calglRun->deviceCA->host_CA->rows * calglRun->deviceCA->host_CA->columns;
		//TODO eliminare bufferFlags Urgent

		calglRun->deviceCA->bufferMinimab = clCreateBuffer(calglRun->deviceCA->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(CALreal) * (calglRun->deviceCA->host_CA->sizeof_pQb_array + 1),
				calglRun->deviceCA->minimab, &err);
		calclHandleError(err);
		calglRun->deviceCA->bufferMiximab = clCreateBuffer(calglRun->deviceCA->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(CALreal) * (calglRun->deviceCA->host_CA->sizeof_pQb_array + 1),
				calglRun->deviceCA->maximab, &err);
		calclHandleError(err);
		calglRun->deviceCA->bufferSumb = clCreateBuffer(calglRun->deviceCA->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(CALreal) * (calglRun->deviceCA->host_CA->sizeof_pQb_array + 1), calglRun->deviceCA->sumsb,
				&err);
		calclHandleError(err);
		calglRun->deviceCA->bufferLogicalAndsb = clCreateBuffer(calglRun->deviceCA->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(CALint) * (calglRun->deviceCA->host_CA->sizeof_pQb_array + 1),
				calglRun->deviceCA->logicalAndsb, &err);
		calclHandleError(err);
		calglRun->deviceCA->bufferLogicalOrsb = clCreateBuffer(calglRun->deviceCA->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(CALint) * (calglRun->deviceCA->host_CA->sizeof_pQb_array + 1),
				calglRun->deviceCA->logicalOrsb, &err);
		calclHandleError(err);
		calglRun->deviceCA->bufferLogicalXOrsb = clCreateBuffer(calglRun->deviceCA->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(CALint) * (calglRun->deviceCA->host_CA->sizeof_pQb_array + 1),
				calglRun->deviceCA->logicalXOrsb, &err);
		calclHandleError(err);
		calglRun->deviceCA->bufferBinaryAndsb = clCreateBuffer(calglRun->deviceCA->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(CALint) * (calglRun->deviceCA->host_CA->sizeof_pQb_array + 1),
				calglRun->deviceCA->binaryAndsb, &err);
		calclHandleError(err);
		calglRun->deviceCA->bufferBinaryOrsb = clCreateBuffer(calglRun->deviceCA->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(CALint) * (calglRun->deviceCA->host_CA->sizeof_pQb_array + 1),
				calglRun->deviceCA->binaryOrsb, &err);
		calclHandleError(err);
		calglRun->deviceCA->bufferBinaryXOrsb = clCreateBuffer(calglRun->deviceCA->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(CALint) * (calglRun->deviceCA->host_CA->sizeof_pQb_array + 1),
				calglRun->deviceCA->binaryXOrsb, &err);
		calclHandleError(err);

		clSetKernelArg(calglRun->deviceCA->kernelMinReductionb, 0, sizeof(CALCLmem), &calglRun->deviceCA->bufferMinimab);
		clSetKernelArg(calglRun->deviceCA->kernelMinReductionb, 2, sizeof(CALCLmem), &calglRun->deviceCA->bufferPartialMinb);
		clSetKernelArg(calglRun->deviceCA->kernelMinReductionb, 4, sizeof(int), &sizeCA);

		clSetKernelArg(calglRun->deviceCA->kernelMaxReductionb, 0, sizeof(CALCLmem), &calglRun->deviceCA->bufferMiximab);
		clSetKernelArg(calglRun->deviceCA->kernelMaxReductionb, 2, sizeof(CALCLmem), &calglRun->deviceCA->bufferPartialMaxr);
		clSetKernelArg(calglRun->deviceCA->kernelMaxReductionb, 4, sizeof(int), &sizeCA);

		clSetKernelArg(calglRun->deviceCA->kernelSumReductionb, 0, sizeof(CALCLmem), &calglRun->deviceCA->bufferSumb);
		clSetKernelArg(calglRun->deviceCA->kernelSumReductionb, 2, sizeof(CALCLmem), &calglRun->deviceCA->bufferPartialSumb);
		clSetKernelArg(calglRun->deviceCA->kernelSumReductionb, 4, sizeof(int), &sizeCA);

		clSetKernelArg(calglRun->deviceCA->kernelLogicalAndReductionb, 0, sizeof(CALCLmem), &calglRun->deviceCA->bufferLogicalAndsb);
		clSetKernelArg(calglRun->deviceCA->kernelLogicalAndReductionb, 2, sizeof(CALCLmem), &calglRun->deviceCA->bufferPartialLogicalAndb);
		clSetKernelArg(calglRun->deviceCA->kernelLogicalAndReductionb, 4, sizeof(int), &sizeCA);

		clSetKernelArg(calglRun->deviceCA->kernelLogicalOrReductionb, 0, sizeof(CALCLmem), &calglRun->deviceCA->bufferLogicalOrsb);
		clSetKernelArg(calglRun->deviceCA->kernelLogicalOrReductionb, 2, sizeof(CALCLmem), &calglRun->deviceCA->bufferPartialLogicalOrb);
		clSetKernelArg(calglRun->deviceCA->kernelLogicalOrReductionb, 4, sizeof(int), &sizeCA);

		clSetKernelArg(calglRun->deviceCA->kernelLogicalXOrReductionb, 0, sizeof(CALCLmem), &calglRun->deviceCA->bufferLogicalXOrsb);
		clSetKernelArg(calglRun->deviceCA->kernelLogicalXOrReductionb, 2, sizeof(CALCLmem), &calglRun->deviceCA->bufferPartialLogicalXOrb);
		clSetKernelArg(calglRun->deviceCA->kernelLogicalXOrReductionb, 4, sizeof(int), &sizeCA);

		clSetKernelArg(calglRun->deviceCA->kernelBinaryAndReductionb, 0, sizeof(CALCLmem), &calglRun->deviceCA->bufferBinaryAndsb);
		clSetKernelArg(calglRun->deviceCA->kernelBinaryAndReductionb, 2, sizeof(CALCLmem), &calglRun->deviceCA->bufferPartialBinaryAndb);
		clSetKernelArg(calglRun->deviceCA->kernelBinaryAndReductionb, 4, sizeof(int), &sizeCA);

		clSetKernelArg(calglRun->deviceCA->kernelBinaryOrReductionb, 0, sizeof(CALCLmem), &calglRun->deviceCA->bufferBinaryOrsb);
		clSetKernelArg(calglRun->deviceCA->kernelBinaryOrReductionb, 2, sizeof(CALCLmem), &calglRun->deviceCA->bufferBinaryOrsb);
		clSetKernelArg(calglRun->deviceCA->kernelBinaryOrReductionb, 4, sizeof(int), &sizeCA);

		clSetKernelArg(calglRun->deviceCA->kernelBinaryXorReductionb, 0, sizeof(CALCLmem), &calglRun->deviceCA->bufferBinaryXOrsb);
		clSetKernelArg(calglRun->deviceCA->kernelBinaryXorReductionb, 2, sizeof(CALCLmem), &calglRun->deviceCA->bufferBinaryXOrsb);
		clSetKernelArg(calglRun->deviceCA->kernelBinaryXorReductionb, 4, sizeof(int), &sizeCA);

		calglRun->deviceCA->bufferMinimai = clCreateBuffer(calglRun->deviceCA->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(CALreal) * (calglRun->deviceCA->host_CA->sizeof_pQi_array + 1),
				calglRun->deviceCA->minimai, &err);
		calclHandleError(err);
		calglRun->deviceCA->bufferMiximai = clCreateBuffer(calglRun->deviceCA->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(CALreal) * (calglRun->deviceCA->host_CA->sizeof_pQi_array + 1),
				calglRun->deviceCA->maximai, &err);
		calclHandleError(err);
		calglRun->deviceCA->bufferSumi = clCreateBuffer(calglRun->deviceCA->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(CALreal) * (calglRun->deviceCA->host_CA->sizeof_pQi_array + 1), calglRun->deviceCA->sumsi,
				&err);
		calclHandleError(err);
		calglRun->deviceCA->bufferLogicalAndsi = clCreateBuffer(calglRun->deviceCA->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(CALint) * (calglRun->deviceCA->host_CA->sizeof_pQi_array + 1),
				calglRun->deviceCA->logicalAndsi, &err);
		calclHandleError(err);
		calglRun->deviceCA->bufferLogicalOrsi = clCreateBuffer(calglRun->deviceCA->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(CALint) * (calglRun->deviceCA->host_CA->sizeof_pQi_array + 1),
				calglRun->deviceCA->logicalOrsi, &err);
		calclHandleError(err);
		calglRun->deviceCA->bufferLogicalXOrsi = clCreateBuffer(calglRun->deviceCA->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(CALint) * (calglRun->deviceCA->host_CA->sizeof_pQi_array + 1),
				calglRun->deviceCA->logicalXOrsi, &err);
		calclHandleError(err);
		calglRun->deviceCA->bufferBinaryAndsi = clCreateBuffer(calglRun->deviceCA->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(CALint) * (calglRun->deviceCA->host_CA->sizeof_pQi_array + 1),
				calglRun->deviceCA->binaryAndsi, &err);
		calclHandleError(err);
		calglRun->deviceCA->bufferBinaryOrsi = clCreateBuffer(calglRun->deviceCA->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(CALint) * (calglRun->deviceCA->host_CA->sizeof_pQi_array + 1),
				calglRun->deviceCA->binaryOrsi, &err);
		calclHandleError(err);
		calglRun->deviceCA->bufferBinaryXOrsi = clCreateBuffer(calglRun->deviceCA->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(CALint) * (calglRun->deviceCA->host_CA->sizeof_pQi_array + 1),
				calglRun->deviceCA->binaryXOrsi, &err);
		calclHandleError(err);

		clSetKernelArg(calglRun->deviceCA->kernelMinReductioni, 0, sizeof(CALCLmem), &calglRun->deviceCA->bufferMinimai);
		clSetKernelArg(calglRun->deviceCA->kernelMinReductioni, 2, sizeof(CALCLmem), &calglRun->deviceCA->bufferPartialMini);
		clSetKernelArg(calglRun->deviceCA->kernelMinReductioni, 4, sizeof(int), &sizeCA);

		clSetKernelArg(calglRun->deviceCA->kernelMaxReductioni, 0, sizeof(CALCLmem), &calglRun->deviceCA->bufferMiximai);
		clSetKernelArg(calglRun->deviceCA->kernelMaxReductioni, 2, sizeof(CALCLmem), &calglRun->deviceCA->bufferPartialMaxi);
		clSetKernelArg(calglRun->deviceCA->kernelMaxReductioni, 4, sizeof(int), &sizeCA);

		clSetKernelArg(calglRun->deviceCA->kernelSumReductioni, 0, sizeof(CALCLmem), &calglRun->deviceCA->bufferSumi);
		clSetKernelArg(calglRun->deviceCA->kernelSumReductioni, 2, sizeof(CALCLmem), &calglRun->deviceCA->bufferPartialSumi);
		clSetKernelArg(calglRun->deviceCA->kernelSumReductioni, 4, sizeof(int), &sizeCA);

		clSetKernelArg(calglRun->deviceCA->kernelLogicalAndReductioni, 0, sizeof(CALCLmem), &calglRun->deviceCA->bufferLogicalAndsi);
		clSetKernelArg(calglRun->deviceCA->kernelLogicalAndReductioni, 2, sizeof(CALCLmem), &calglRun->deviceCA->bufferPartialLogicalAndi);
		clSetKernelArg(calglRun->deviceCA->kernelLogicalAndReductioni, 4, sizeof(int), &sizeCA);

		clSetKernelArg(calglRun->deviceCA->kernelLogicalOrReductioni, 0, sizeof(CALCLmem), &calglRun->deviceCA->bufferLogicalOrsi);
		clSetKernelArg(calglRun->deviceCA->kernelLogicalOrReductioni, 2, sizeof(CALCLmem), &calglRun->deviceCA->bufferPartialLogicalOri);
		clSetKernelArg(calglRun->deviceCA->kernelLogicalOrReductioni, 4, sizeof(int), &sizeCA);

		clSetKernelArg(calglRun->deviceCA->kernelLogicalXOrReductioni, 0, sizeof(CALCLmem), &calglRun->deviceCA->bufferLogicalXOrsi);
		clSetKernelArg(calglRun->deviceCA->kernelLogicalXOrReductioni, 2, sizeof(CALCLmem), &calglRun->deviceCA->bufferPartialLogicalXOri);
		clSetKernelArg(calglRun->deviceCA->kernelLogicalXOrReductioni, 4, sizeof(int), &sizeCA);

		clSetKernelArg(calglRun->deviceCA->kernelBinaryAndReductioni, 0, sizeof(CALCLmem), &calglRun->deviceCA->bufferBinaryAndsi);
		clSetKernelArg(calglRun->deviceCA->kernelBinaryAndReductioni, 2, sizeof(CALCLmem), &calglRun->deviceCA->bufferPartialBinaryAndi);
		clSetKernelArg(calglRun->deviceCA->kernelBinaryAndReductioni, 4, sizeof(int), &sizeCA);

		clSetKernelArg(calglRun->deviceCA->kernelBinaryOrReductioni, 0, sizeof(CALCLmem), &calglRun->deviceCA->bufferBinaryOrsi);
		clSetKernelArg(calglRun->deviceCA->kernelBinaryOrReductioni, 2, sizeof(CALCLmem), &calglRun->deviceCA->bufferPartialBinaryOri);
		clSetKernelArg(calglRun->deviceCA->kernelBinaryOrReductioni, 4, sizeof(int), &sizeCA);

		clSetKernelArg(calglRun->deviceCA->kernelBinaryXorReductioni, 0, sizeof(CALCLmem), &calglRun->deviceCA->bufferBinaryXOrsi);
		clSetKernelArg(calglRun->deviceCA->kernelBinaryXorReductioni, 2, sizeof(CALCLmem), &calglRun->deviceCA->bufferPartialBinaryXOri);
		clSetKernelArg(calglRun->deviceCA->kernelBinaryXorReductioni, 4, sizeof(int), &sizeCA);

		calglRun->deviceCA->bufferMinimar = clCreateBuffer(calglRun->deviceCA->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(CALreal) * (calglRun->deviceCA->host_CA->sizeof_pQr_array + 1),
				calglRun->deviceCA->minimar, &err);
		calclHandleError(err);
		calglRun->deviceCA->bufferMiximar = clCreateBuffer(calglRun->deviceCA->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(CALreal) * (calglRun->deviceCA->host_CA->sizeof_pQr_array + 1),
				calglRun->deviceCA->maximar, &err);
		calclHandleError(err);
		calglRun->deviceCA->bufferSumr = clCreateBuffer(calglRun->deviceCA->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(CALreal) * (calglRun->deviceCA->host_CA->sizeof_pQr_array + 1), calglRun->deviceCA->sumsr,
				&err);
		calclHandleError(err);
		calglRun->deviceCA->bufferLogicalAndsr = clCreateBuffer(calglRun->deviceCA->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(CALint) * (calglRun->deviceCA->host_CA->sizeof_pQr_array + 1),
				calglRun->deviceCA->logicalAndsr, &err);
		calclHandleError(err);
		calglRun->deviceCA->bufferLogicalOrsr = clCreateBuffer(calglRun->deviceCA->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(CALint) * (calglRun->deviceCA->host_CA->sizeof_pQr_array + 1),
				calglRun->deviceCA->logicalOrsr, &err);
		calclHandleError(err);
		calglRun->deviceCA->bufferLogicalXOrsr = clCreateBuffer(calglRun->deviceCA->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(CALint) * (calglRun->deviceCA->host_CA->sizeof_pQr_array + 1),
				calglRun->deviceCA->logicalXOrsr, &err);
		calclHandleError(err);

		calglRun->deviceCA->bufferBinaryAndsr = clCreateBuffer(calglRun->deviceCA->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(CALint) * (calglRun->deviceCA->host_CA->sizeof_pQr_array + 1),
				calglRun->deviceCA->binaryAndsr, &err);
		calclHandleError(err);

		calglRun->deviceCA->bufferBinaryOrsr = clCreateBuffer(calglRun->deviceCA->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(CALint) * (calglRun->deviceCA->host_CA->sizeof_pQr_array + 1),
				calglRun->deviceCA->binaryOrsr, &err);
		calclHandleError(err);

		calglRun->deviceCA->bufferBinaryXOrsr = clCreateBuffer(calglRun->deviceCA->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(CALint) * (calglRun->deviceCA->host_CA->sizeof_pQr_array + 1),
				calglRun->deviceCA->binaryXOrsr, &err);
		calclHandleError(err);

		clSetKernelArg(calglRun->deviceCA->kernelMinReductionr, 0, sizeof(CALCLmem), &calglRun->deviceCA->bufferMinimar);
		clSetKernelArg(calglRun->deviceCA->kernelMinReductionr, 2, sizeof(CALCLmem), &calglRun->deviceCA->bufferPartialMinr);
		clSetKernelArg(calglRun->deviceCA->kernelMinReductionr, 4, sizeof(int), &sizeCA);

		clSetKernelArg(calglRun->deviceCA->kernelMaxReductionr, 0, sizeof(CALCLmem), &calglRun->deviceCA->bufferMiximar);
		clSetKernelArg(calglRun->deviceCA->kernelMaxReductionr, 2, sizeof(CALCLmem), &calglRun->deviceCA->bufferPartialMaxr);
		clSetKernelArg(calglRun->deviceCA->kernelMaxReductionr, 4, sizeof(int), &sizeCA);

		clSetKernelArg(calglRun->deviceCA->kernelSumReductionr, 0, sizeof(CALCLmem), &calglRun->deviceCA->bufferSumr);
		clSetKernelArg(calglRun->deviceCA->kernelSumReductionr, 2, sizeof(CALCLmem), &calglRun->deviceCA->bufferPartialSumr);
		clSetKernelArg(calglRun->deviceCA->kernelSumReductionr, 4, sizeof(int), &sizeCA);

		clSetKernelArg(calglRun->deviceCA->kernelLogicalAndReductionr, 0, sizeof(CALCLmem), &calglRun->deviceCA->bufferLogicalAndsr);
		clSetKernelArg(calglRun->deviceCA->kernelLogicalAndReductionr, 2, sizeof(CALCLmem), &calglRun->deviceCA->bufferPartialLogicalAndr);
		clSetKernelArg(calglRun->deviceCA->kernelLogicalAndReductionr, 4, sizeof(int), &sizeCA);

		clSetKernelArg(calglRun->deviceCA->kernelLogicalOrReductionr, 0, sizeof(CALCLmem), &calglRun->deviceCA->bufferLogicalOrsr);
		clSetKernelArg(calglRun->deviceCA->kernelLogicalOrReductionr, 2, sizeof(CALCLmem), &calglRun->deviceCA->bufferPartialLogicalOrr);
		clSetKernelArg(calglRun->deviceCA->kernelLogicalOrReductionr, 4, sizeof(int), &sizeCA);

		clSetKernelArg(calglRun->deviceCA->kernelLogicalXOrReductionr, 0, sizeof(CALCLmem), &calglRun->deviceCA->bufferLogicalXOrsr);
		clSetKernelArg(calglRun->deviceCA->kernelLogicalXOrReductionr, 2, sizeof(CALCLmem), &calglRun->deviceCA->bufferPartialLogicalXOrr);
		clSetKernelArg(calglRun->deviceCA->kernelLogicalXOrReductionr, 4, sizeof(int), &sizeCA);

		clSetKernelArg(calglRun->deviceCA->kernelBinaryAndReductionr, 0, sizeof(CALCLmem), &calglRun->deviceCA->bufferBinaryAndsr);
		clSetKernelArg(calglRun->deviceCA->kernelBinaryAndReductionr, 2, sizeof(CALCLmem), &calglRun->deviceCA->bufferPartialBinaryAndr);
		clSetKernelArg(calglRun->deviceCA->kernelBinaryAndReductionr, 4, sizeof(int), &sizeCA);

		clSetKernelArg(calglRun->deviceCA->kernelBinaryOrReductionr, 0, sizeof(CALCLmem), &calglRun->deviceCA->bufferBinaryOrsr);
		clSetKernelArg(calglRun->deviceCA->kernelBinaryOrReductionr, 2, sizeof(CALCLmem), &calglRun->deviceCA->bufferPartialBinaryOrr);
		clSetKernelArg(calglRun->deviceCA->kernelBinaryOrReductionr, 4, sizeof(int), &sizeCA);

		clSetKernelArg(calglRun->deviceCA->kernelBinaryXorReductionr, 0, sizeof(CALCLmem), &calglRun->deviceCA->bufferBinaryXOrsr);
		clSetKernelArg(calglRun->deviceCA->kernelBinaryXorReductionr, 2, sizeof(CALCLmem), &calglRun->deviceCA->bufferPartialBinaryXOrr);
		clSetKernelArg(calglRun->deviceCA->kernelBinaryXorReductionr, 4, sizeof(int), &sizeCA);

		if (calglRun->deviceCA->kernelInitSubstates != NULL)
			calclSetReductionParameters2D(calglRun->deviceCA, &calglRun->deviceCA->kernelInitSubstates);
		if (calglRun->deviceCA->kernelStopCondition != NULL)
			calclSetReductionParameters2D(calglRun->deviceCA, &calglRun->deviceCA->kernelStopCondition);
		if (calglRun->deviceCA->kernelSteering != NULL)
			calclSetReductionParameters2D(calglRun->deviceCA, &calglRun->deviceCA->kernelSteering);

		int i = 0;

		for (i = 0; i < calglRun->deviceCA->elementaryProcessesNum; i++) {
			calclSetReductionParameters2D(calglRun->deviceCA, &calglRun->deviceCA->elementaryProcesses[i]);
		}

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
			singleStepThreadNum[0] = calglRun->deviceCA->host_CA->A.size_current;
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
