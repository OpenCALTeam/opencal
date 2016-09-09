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

struct CALGLRun3D* calglRunCLDef3D(struct CALCLModel3D* device_CA, CALint fixedStep, CALint initial_step, CALint final_step) {
	struct CALGLRun3D* calglRun = (struct CALGLRun3D*) malloc(sizeof(struct CALGLRun3D));

	calglRun->firstRun = CAL_TRUE;
	calglRun->active = CAL_FALSE;
	calglRun->terminated = CAL_FALSE;
	calglRun->stop = CAL_FALSE;
	calglRun->device_CA = device_CA;
	calglRun->onlyOneTime = CAL_FALSE;
	calglRun->fixedStep = fixedStep;
	calglRun->device_CA->steps = initial_step;
	calglRun->step = initial_step;
	calglRun->final_step = final_step;
	cl_int err;

	int sizeCA = calglRun->device_CA->host_CA->rows * calglRun->device_CA->host_CA->columns * calglRun->device_CA->host_CA->slices;
	//TODO eliminare bufferFlags Urgent

	calglRun->device_CA->bufferMinimab = clCreateBuffer(calglRun->device_CA->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(CALreal) * (calglRun->device_CA->host_CA->sizeof_pQb_array + 1),
			calglRun->device_CA->minimab, &err);
	calclHandleError(err);
	calglRun->device_CA->bufferMaximab = clCreateBuffer(calglRun->device_CA->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(CALreal) * (calglRun->device_CA->host_CA->sizeof_pQb_array + 1),
			calglRun->device_CA->maximab, &err);
	calclHandleError(err);
	calglRun->device_CA->bufferSumb = clCreateBuffer(calglRun->device_CA->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(CALreal) * (calglRun->device_CA->host_CA->sizeof_pQb_array + 1),
			calglRun->device_CA->sumsb, &err);
	calclHandleError(err);
	calglRun->device_CA->bufferProdb = clCreateBuffer(calglRun->device_CA->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(CALreal) * (calglRun->device_CA->host_CA->sizeof_pQb_array + 1),
			calglRun->device_CA->prodsb, &err);
	calclHandleError(err);
	calglRun->device_CA->bufferLogicalAndsb = clCreateBuffer(calglRun->device_CA->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
			sizeof(CALint) * (calglRun->device_CA->host_CA->sizeof_pQb_array + 1), calglRun->device_CA->logicalAndsb, &err);
	calclHandleError(err);
	calglRun->device_CA->bufferLogicalOrsb = clCreateBuffer(calglRun->device_CA->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
			sizeof(CALint) * (calglRun->device_CA->host_CA->sizeof_pQb_array + 1), calglRun->device_CA->logicalOrsb, &err);
	calclHandleError(err);
	calglRun->device_CA->bufferLogicalXOrsb = clCreateBuffer(calglRun->device_CA->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
			sizeof(CALint) * (calglRun->device_CA->host_CA->sizeof_pQb_array + 1), calglRun->device_CA->logicalXOrsb, &err);
	calclHandleError(err);
	calglRun->device_CA->bufferBinaryAndsb = clCreateBuffer(calglRun->device_CA->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
			sizeof(CALint) * (calglRun->device_CA->host_CA->sizeof_pQb_array + 1), calglRun->device_CA->binaryAndsb, &err);
	calclHandleError(err);
	calglRun->device_CA->bufferBinaryOrsb = clCreateBuffer(calglRun->device_CA->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
			sizeof(CALint) * (calglRun->device_CA->host_CA->sizeof_pQb_array + 1), calglRun->device_CA->binaryOrsb, &err);
	calclHandleError(err);
	calglRun->device_CA->bufferBinaryXOrsb = clCreateBuffer(calglRun->device_CA->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
			sizeof(CALint) * (calglRun->device_CA->host_CA->sizeof_pQb_array + 1), calglRun->device_CA->binaryXOrsb, &err);
	calclHandleError(err);

	clSetKernelArg(calglRun->device_CA->kernelMinReductionb, 0, sizeof(CALCLmem), &calglRun->device_CA->bufferMinimab);
	clSetKernelArg(calglRun->device_CA->kernelMinReductionb, 2, sizeof(CALCLmem), &calglRun->device_CA->bufferPartialMinb);
	clSetKernelArg(calglRun->device_CA->kernelMinReductionb, 4, sizeof(int), &sizeCA);

	clSetKernelArg(calglRun->device_CA->kernelMaxReductionb, 0, sizeof(CALCLmem), &calglRun->device_CA->bufferMaximab);
	clSetKernelArg(calglRun->device_CA->kernelMaxReductionb, 2, sizeof(CALCLmem), &calglRun->device_CA->bufferPartialMaxr);
	clSetKernelArg(calglRun->device_CA->kernelMaxReductionb, 4, sizeof(int), &sizeCA);

	clSetKernelArg(calglRun->device_CA->kernelSumReductionb, 0, sizeof(CALCLmem), &calglRun->device_CA->bufferSumb);
	clSetKernelArg(calglRun->device_CA->kernelSumReductionb, 2, sizeof(CALCLmem), &calglRun->device_CA->bufferPartialSumb);
	clSetKernelArg(calglRun->device_CA->kernelSumReductionb, 4, sizeof(int), &sizeCA);

	clSetKernelArg(calglRun->device_CA->kernelProdReductionb, 0, sizeof(CALCLmem), &calglRun->device_CA->bufferProdb);
	clSetKernelArg(calglRun->device_CA->kernelProdReductionb, 2, sizeof(CALCLmem), &calglRun->device_CA->bufferPartialProdb);
	clSetKernelArg(calglRun->device_CA->kernelProdReductionb, 4, sizeof(int), &sizeCA);

	clSetKernelArg(calglRun->device_CA->kernelLogicalAndReductionb, 0, sizeof(CALCLmem), &calglRun->device_CA->bufferLogicalAndsb);
	clSetKernelArg(calglRun->device_CA->kernelLogicalAndReductionb, 2, sizeof(CALCLmem), &calglRun->device_CA->bufferPartialLogicalAndb);
	clSetKernelArg(calglRun->device_CA->kernelLogicalAndReductionb, 4, sizeof(int), &sizeCA);

	clSetKernelArg(calglRun->device_CA->kernelLogicalOrReductionb, 0, sizeof(CALCLmem), &calglRun->device_CA->bufferLogicalOrsb);
	clSetKernelArg(calglRun->device_CA->kernelLogicalOrReductionb, 2, sizeof(CALCLmem), &calglRun->device_CA->bufferPartialLogicalOrb);
	clSetKernelArg(calglRun->device_CA->kernelLogicalOrReductionb, 4, sizeof(int), &sizeCA);

	clSetKernelArg(calglRun->device_CA->kernelLogicalXOrReductionb, 0, sizeof(CALCLmem), &calglRun->device_CA->bufferLogicalXOrsb);
	clSetKernelArg(calglRun->device_CA->kernelLogicalXOrReductionb, 2, sizeof(CALCLmem), &calglRun->device_CA->bufferPartialLogicalXOrb);
	clSetKernelArg(calglRun->device_CA->kernelLogicalXOrReductionb, 4, sizeof(int), &sizeCA);

	clSetKernelArg(calglRun->device_CA->kernelBinaryAndReductionb, 0, sizeof(CALCLmem), &calglRun->device_CA->bufferBinaryAndsb);
	clSetKernelArg(calglRun->device_CA->kernelBinaryAndReductionb, 2, sizeof(CALCLmem), &calglRun->device_CA->bufferPartialBinaryAndb);
	clSetKernelArg(calglRun->device_CA->kernelBinaryAndReductionb, 4, sizeof(int), &sizeCA);

	clSetKernelArg(calglRun->device_CA->kernelBinaryOrReductionb, 0, sizeof(CALCLmem), &calglRun->device_CA->bufferBinaryOrsb);
	clSetKernelArg(calglRun->device_CA->kernelBinaryOrReductionb, 2, sizeof(CALCLmem), &calglRun->device_CA->bufferBinaryOrsb);
	clSetKernelArg(calglRun->device_CA->kernelBinaryOrReductionb, 4, sizeof(int), &sizeCA);

	clSetKernelArg(calglRun->device_CA->kernelBinaryXorReductionb, 0, sizeof(CALCLmem), &calglRun->device_CA->bufferBinaryXOrsb);
	clSetKernelArg(calglRun->device_CA->kernelBinaryXorReductionb, 2, sizeof(CALCLmem), &calglRun->device_CA->bufferBinaryXOrsb);
	clSetKernelArg(calglRun->device_CA->kernelBinaryXorReductionb, 4, sizeof(int), &sizeCA);

	calglRun->device_CA->bufferMinimai = clCreateBuffer(calglRun->device_CA->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(CALreal) * (calglRun->device_CA->host_CA->sizeof_pQi_array + 1),
			calglRun->device_CA->minimai, &err);
	calclHandleError(err);
	calglRun->device_CA->bufferMaximai = clCreateBuffer(calglRun->device_CA->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(CALreal) * (calglRun->device_CA->host_CA->sizeof_pQi_array + 1),
			calglRun->device_CA->maximai, &err);
	calclHandleError(err);
	calglRun->device_CA->bufferSumi = clCreateBuffer(calglRun->device_CA->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(CALreal) * (calglRun->device_CA->host_CA->sizeof_pQi_array + 1),
			calglRun->device_CA->sumsi, &err);
	calclHandleError(err);
	calglRun->device_CA->bufferProdi = clCreateBuffer(calglRun->device_CA->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(CALreal) * (calglRun->device_CA->host_CA->sizeof_pQi_array + 1),
			calglRun->device_CA->prodsi, &err);
	calclHandleError(err);
	calglRun->device_CA->bufferLogicalAndsi = clCreateBuffer(calglRun->device_CA->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
			sizeof(CALint) * (calglRun->device_CA->host_CA->sizeof_pQi_array + 1), calglRun->device_CA->logicalAndsi, &err);
	calclHandleError(err);
	calglRun->device_CA->bufferLogicalOrsi = clCreateBuffer(calglRun->device_CA->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
			sizeof(CALint) * (calglRun->device_CA->host_CA->sizeof_pQi_array + 1), calglRun->device_CA->logicalOrsi, &err);
	calclHandleError(err);
	calglRun->device_CA->bufferLogicalXOrsi = clCreateBuffer(calglRun->device_CA->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
			sizeof(CALint) * (calglRun->device_CA->host_CA->sizeof_pQi_array + 1), calglRun->device_CA->logicalXOrsi, &err);
	calclHandleError(err);
	calglRun->device_CA->bufferBinaryAndsi = clCreateBuffer(calglRun->device_CA->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
			sizeof(CALint) * (calglRun->device_CA->host_CA->sizeof_pQi_array + 1), calglRun->device_CA->binaryAndsi, &err);
	calclHandleError(err);
	calglRun->device_CA->bufferBinaryOrsi = clCreateBuffer(calglRun->device_CA->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
			sizeof(CALint) * (calglRun->device_CA->host_CA->sizeof_pQi_array + 1), calglRun->device_CA->binaryOrsi, &err);
	calclHandleError(err);
	calglRun->device_CA->bufferBinaryXOrsi = clCreateBuffer(calglRun->device_CA->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
			sizeof(CALint) * (calglRun->device_CA->host_CA->sizeof_pQi_array + 1), calglRun->device_CA->binaryXOrsi, &err);
	calclHandleError(err);

	clSetKernelArg(calglRun->device_CA->kernelMinReductioni, 0, sizeof(CALCLmem), &calglRun->device_CA->bufferMinimai);
	clSetKernelArg(calglRun->device_CA->kernelMinReductioni, 2, sizeof(CALCLmem), &calglRun->device_CA->bufferPartialMini);
	clSetKernelArg(calglRun->device_CA->kernelMinReductioni, 4, sizeof(int), &sizeCA);

	clSetKernelArg(calglRun->device_CA->kernelMaxReductioni, 0, sizeof(CALCLmem), &calglRun->device_CA->bufferMaximai);
	clSetKernelArg(calglRun->device_CA->kernelMaxReductioni, 2, sizeof(CALCLmem), &calglRun->device_CA->bufferPartialMaxi);
	clSetKernelArg(calglRun->device_CA->kernelMaxReductioni, 4, sizeof(int), &sizeCA);

	clSetKernelArg(calglRun->device_CA->kernelSumReductioni, 0, sizeof(CALCLmem), &calglRun->device_CA->bufferSumi);
	clSetKernelArg(calglRun->device_CA->kernelSumReductioni, 2, sizeof(CALCLmem), &calglRun->device_CA->bufferPartialSumi);
	clSetKernelArg(calglRun->device_CA->kernelSumReductioni, 4, sizeof(int), &sizeCA);

	clSetKernelArg(calglRun->device_CA->kernelProdReductioni, 0, sizeof(CALCLmem), &calglRun->device_CA->bufferProdi);
	clSetKernelArg(calglRun->device_CA->kernelProdReductioni, 2, sizeof(CALCLmem), &calglRun->device_CA->bufferPartialProdi);
	clSetKernelArg(calglRun->device_CA->kernelProdReductioni, 4, sizeof(int), &sizeCA);

	clSetKernelArg(calglRun->device_CA->kernelLogicalAndReductioni, 0, sizeof(CALCLmem), &calglRun->device_CA->bufferLogicalAndsi);
	clSetKernelArg(calglRun->device_CA->kernelLogicalAndReductioni, 2, sizeof(CALCLmem), &calglRun->device_CA->bufferPartialLogicalAndi);
	clSetKernelArg(calglRun->device_CA->kernelLogicalAndReductioni, 4, sizeof(int), &sizeCA);

	clSetKernelArg(calglRun->device_CA->kernelLogicalOrReductioni, 0, sizeof(CALCLmem), &calglRun->device_CA->bufferLogicalOrsi);
	clSetKernelArg(calglRun->device_CA->kernelLogicalOrReductioni, 2, sizeof(CALCLmem), &calglRun->device_CA->bufferPartialLogicalOri);
	clSetKernelArg(calglRun->device_CA->kernelLogicalOrReductioni, 4, sizeof(int), &sizeCA);

	clSetKernelArg(calglRun->device_CA->kernelLogicalXOrReductioni, 0, sizeof(CALCLmem), &calglRun->device_CA->bufferLogicalXOrsi);
	clSetKernelArg(calglRun->device_CA->kernelLogicalXOrReductioni, 2, sizeof(CALCLmem), &calglRun->device_CA->bufferPartialLogicalXOri);
	clSetKernelArg(calglRun->device_CA->kernelLogicalXOrReductioni, 4, sizeof(int), &sizeCA);

	clSetKernelArg(calglRun->device_CA->kernelBinaryAndReductioni, 0, sizeof(CALCLmem), &calglRun->device_CA->bufferBinaryAndsi);
	clSetKernelArg(calglRun->device_CA->kernelBinaryAndReductioni, 2, sizeof(CALCLmem), &calglRun->device_CA->bufferPartialBinaryAndi);
	clSetKernelArg(calglRun->device_CA->kernelBinaryAndReductioni, 4, sizeof(int), &sizeCA);

	clSetKernelArg(calglRun->device_CA->kernelBinaryOrReductioni, 0, sizeof(CALCLmem), &calglRun->device_CA->bufferBinaryOrsi);
	clSetKernelArg(calglRun->device_CA->kernelBinaryOrReductioni, 2, sizeof(CALCLmem), &calglRun->device_CA->bufferPartialBinaryOri);
	clSetKernelArg(calglRun->device_CA->kernelBinaryOrReductioni, 4, sizeof(int), &sizeCA);

	clSetKernelArg(calglRun->device_CA->kernelBinaryXorReductioni, 0, sizeof(CALCLmem), &calglRun->device_CA->bufferBinaryXOrsi);
	clSetKernelArg(calglRun->device_CA->kernelBinaryXorReductioni, 2, sizeof(CALCLmem), &calglRun->device_CA->bufferPartialBinaryXOri);
	clSetKernelArg(calglRun->device_CA->kernelBinaryXorReductioni, 4, sizeof(int), &sizeCA);

	calglRun->device_CA->bufferMinimar = clCreateBuffer(calglRun->device_CA->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(CALreal) * (calglRun->device_CA->host_CA->sizeof_pQr_array + 1),
			calglRun->device_CA->minimar, &err);
	calclHandleError(err);
	calglRun->device_CA->bufferMaximar = clCreateBuffer(calglRun->device_CA->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(CALreal) * (calglRun->device_CA->host_CA->sizeof_pQr_array + 1),
			calglRun->device_CA->maximar, &err);
	calclHandleError(err);
	calglRun->device_CA->bufferSumr = clCreateBuffer(calglRun->device_CA->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(CALreal) * (calglRun->device_CA->host_CA->sizeof_pQr_array + 1),
			calglRun->device_CA->sumsr, &err);
	calclHandleError(err);
	calglRun->device_CA->bufferProdr = clCreateBuffer(calglRun->device_CA->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(CALreal) * (calglRun->device_CA->host_CA->sizeof_pQr_array + 1),
			calglRun->device_CA->prodsr, &err);
	calclHandleError(err);
	calglRun->device_CA->bufferLogicalAndsr = clCreateBuffer(calglRun->device_CA->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
			sizeof(CALint) * (calglRun->device_CA->host_CA->sizeof_pQr_array + 1), calglRun->device_CA->logicalAndsr, &err);
	calclHandleError(err);
	calglRun->device_CA->bufferLogicalOrsr = clCreateBuffer(calglRun->device_CA->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
			sizeof(CALint) * (calglRun->device_CA->host_CA->sizeof_pQr_array + 1), calglRun->device_CA->logicalOrsr, &err);
	calclHandleError(err);
	calglRun->device_CA->bufferLogicalXOrsr = clCreateBuffer(calglRun->device_CA->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
			sizeof(CALint) * (calglRun->device_CA->host_CA->sizeof_pQr_array + 1), calglRun->device_CA->logicalXOrsr, &err);
	calclHandleError(err);

	calglRun->device_CA->bufferBinaryAndsr = clCreateBuffer(calglRun->device_CA->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
			sizeof(CALint) * (calglRun->device_CA->host_CA->sizeof_pQr_array + 1), calglRun->device_CA->binaryAndsr, &err);
	calclHandleError(err);

	calglRun->device_CA->bufferBinaryOrsr = clCreateBuffer(calglRun->device_CA->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
			sizeof(CALint) * (calglRun->device_CA->host_CA->sizeof_pQr_array + 1), calglRun->device_CA->binaryOrsr, &err);
	calclHandleError(err);

	calglRun->device_CA->bufferBinaryXOrsr = clCreateBuffer(calglRun->device_CA->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
			sizeof(CALint) * (calglRun->device_CA->host_CA->sizeof_pQr_array + 1), calglRun->device_CA->binaryXOrsr, &err);
	calclHandleError(err);

	clSetKernelArg(calglRun->device_CA->kernelMinReductionr, 0, sizeof(CALCLmem), &calglRun->device_CA->bufferMinimar);
	clSetKernelArg(calglRun->device_CA->kernelMinReductionr, 2, sizeof(CALCLmem), &calglRun->device_CA->bufferPartialMinr);
	clSetKernelArg(calglRun->device_CA->kernelMinReductionr, 4, sizeof(int), &sizeCA);

	clSetKernelArg(calglRun->device_CA->kernelMaxReductionr, 0, sizeof(CALCLmem), &calglRun->device_CA->bufferMaximar);
	clSetKernelArg(calglRun->device_CA->kernelMaxReductionr, 2, sizeof(CALCLmem), &calglRun->device_CA->bufferPartialMaxr);
	clSetKernelArg(calglRun->device_CA->kernelMaxReductionr, 4, sizeof(int), &sizeCA);

	clSetKernelArg(calglRun->device_CA->kernelSumReductionr, 0, sizeof(CALCLmem), &calglRun->device_CA->bufferSumr);
	clSetKernelArg(calglRun->device_CA->kernelSumReductionr, 2, sizeof(CALCLmem), &calglRun->device_CA->bufferPartialSumr);
	clSetKernelArg(calglRun->device_CA->kernelSumReductionr, 4, sizeof(int), &sizeCA);

	clSetKernelArg(calglRun->device_CA->kernelProdReductionr, 0, sizeof(CALCLmem), &calglRun->device_CA->bufferProdr);
	clSetKernelArg(calglRun->device_CA->kernelProdReductionr, 2, sizeof(CALCLmem), &calglRun->device_CA->bufferPartialProdr);
	clSetKernelArg(calglRun->device_CA->kernelProdReductionr, 4, sizeof(int), &sizeCA);

	clSetKernelArg(calglRun->device_CA->kernelLogicalAndReductionr, 0, sizeof(CALCLmem), &calglRun->device_CA->bufferLogicalAndsr);
	clSetKernelArg(calglRun->device_CA->kernelLogicalAndReductionr, 2, sizeof(CALCLmem), &calglRun->device_CA->bufferPartialLogicalAndr);
	clSetKernelArg(calglRun->device_CA->kernelLogicalAndReductionr, 4, sizeof(int), &sizeCA);

	clSetKernelArg(calglRun->device_CA->kernelLogicalOrReductionr, 0, sizeof(CALCLmem), &calglRun->device_CA->bufferLogicalOrsr);
	clSetKernelArg(calglRun->device_CA->kernelLogicalOrReductionr, 2, sizeof(CALCLmem), &calglRun->device_CA->bufferPartialLogicalOrr);
	clSetKernelArg(calglRun->device_CA->kernelLogicalOrReductionr, 4, sizeof(int), &sizeCA);

	clSetKernelArg(calglRun->device_CA->kernelLogicalXOrReductionr, 0, sizeof(CALCLmem), &calglRun->device_CA->bufferLogicalXOrsr);
	clSetKernelArg(calglRun->device_CA->kernelLogicalXOrReductionr, 2, sizeof(CALCLmem), &calglRun->device_CA->bufferPartialLogicalXOrr);
	clSetKernelArg(calglRun->device_CA->kernelLogicalXOrReductionr, 4, sizeof(int), &sizeCA);

	clSetKernelArg(calglRun->device_CA->kernelBinaryAndReductionr, 0, sizeof(CALCLmem), &calglRun->device_CA->bufferBinaryAndsr);
	clSetKernelArg(calglRun->device_CA->kernelBinaryAndReductionr, 2, sizeof(CALCLmem), &calglRun->device_CA->bufferPartialBinaryAndr);
	clSetKernelArg(calglRun->device_CA->kernelBinaryAndReductionr, 4, sizeof(int), &sizeCA);

	clSetKernelArg(calglRun->device_CA->kernelBinaryOrReductionr, 0, sizeof(CALCLmem), &calglRun->device_CA->bufferBinaryOrsr);
	clSetKernelArg(calglRun->device_CA->kernelBinaryOrReductionr, 2, sizeof(CALCLmem), &calglRun->device_CA->bufferPartialBinaryOrr);
	clSetKernelArg(calglRun->device_CA->kernelBinaryOrReductionr, 4, sizeof(int), &sizeCA);

	clSetKernelArg(calglRun->device_CA->kernelBinaryXorReductionr, 0, sizeof(CALCLmem), &calglRun->device_CA->bufferBinaryXOrsr);
	clSetKernelArg(calglRun->device_CA->kernelBinaryXorReductionr, 2, sizeof(CALCLmem), &calglRun->device_CA->bufferPartialBinaryXOrr);
	clSetKernelArg(calglRun->device_CA->kernelBinaryXorReductionr, 4, sizeof(int), &sizeCA);

	if (calglRun->device_CA->kernelInitSubstates != NULL)
		calclSetReductionParameters3D(calglRun->device_CA, &calglRun->device_CA->kernelInitSubstates);
	if (calglRun->device_CA->kernelStopCondition != NULL)
		calclSetReductionParameters3D(calglRun->device_CA, &calglRun->device_CA->kernelStopCondition);
	if (calglRun->device_CA->kernelSteering != NULL)
		calclSetReductionParameters3D(calglRun->device_CA, &calglRun->device_CA->kernelSteering);

	int i = 0;

	for (i = 0; i < calglRun->device_CA->elementaryProcessesNum; i++) {
		calclSetReductionParameters3D(calglRun->device_CA, &calglRun->device_CA->elementaryProcesses[i]);
	}

	calglRun->threadNumMax = (size_t*) malloc(sizeof(size_t) * 3);
	calglRun->threadNumMax[0] = calglRun->device_CA->host_CA->rows;
	calglRun->threadNumMax[1] = calglRun->device_CA->host_CA->columns;
	calglRun->threadNumMax[2] = calglRun->device_CA->host_CA->slices;


	if (calglRun->device_CA->opt == CAL_NO_OPT) {
		calglRun->singleStepThreadNum = (size_t*) malloc(sizeof(size_t) * 3);
		calglRun->singleStepThreadNum[0] = calglRun->threadNumMax[0];
		calglRun->singleStepThreadNum[1] = calglRun->threadNumMax[1];
		calglRun->singleStepThreadNum[2] = calglRun->threadNumMax[2];
		calglRun->dimNum = 3;
	} else {
		calglRun->singleStepThreadNum = (size_t*) malloc(sizeof(size_t));
        calglRun->singleStepThreadNum[0] = calglRun->device_CA->host_CA->A->size_current;
		calglRun->dimNum = 1;
	}

	calglStartThread3DCL(calglRun);

	return calglRun;
}

void calglDestroyUpdater3DCL(struct CALGLRun3D* calglRun) {
	if (calglRun) {
		free(calglRun);
	}
}

void* calglFuncThreadUpdate3DCL(void* arg) {
	struct CALGLRun3D* calglRun = (struct CALGLRun3D*) arg;

	while (!calglRun->stop) {
		calglUpdate3DCL(calglRun);
		//Sleep(10);
	}

	return (void *) 0;
}

void calglStartThread3DCL(struct CALGLRun3D* calglRun) {
	pthread_create(&calglRun->thread, NULL, calglFuncThreadUpdate3DCL, (void *) calglRun);
}

void calglUpdate3DCL(struct CALGLRun3D* calglRun) {
	if (calglRun->active) {

		//	calclRoundThreadsNum(singleStepThreadNum, dimNum);
		if (calglRun->firstRun) {
			calglRun->firstRun = CAL_FALSE;
			calglRun->start_time = time(NULL);
			if (calglRun->device_CA->kernelInitSubstates != NULL)
				calclKernelCall3D(calglRun->device_CA, calglRun->device_CA->kernelInitSubstates, calglRun->dimNum, calglRun->threadNumMax, NULL);
		}
		//simulation main loop
		calglRun->step = calglRun->device_CA->steps;
		printf("Cellular Automata: Current Step: %d\r", calglRun->device_CA->steps);
		//exectutes the global transition function, the steering function and check for the stop condition.
		calglRun->terminated = calclSingleStep3D(calglRun->device_CA, calglRun->singleStepThreadNum,calglRun->dimNum);
		//graphic rendering
		//#ifdef WIN32
		//		system("cls");
		//#else
		//		system("clear");
		//#endif
		//		printf("*----------------  Cellular Automata  ----------------*\n");
		//		printf(" Rows: %d, Columns: %d\n", calglGetGlobalSettings()->rows, calglGetGlobalSettings()->columns);
		//		printf(" Current Step: %d/%d; Active cells: %d\n", calglRun->calRun->step, calglGetGlobalSettings()->step, calglRun->calRun->ca3D->A.size_current);

		//		printf("*-----------------------------------------------------*\n");
		//check for the stop condition
		if (calglRun->step % calglRun->fixedStep == 0) {
			calclGetSubstatesDeviceToHost3D(calglRun->device_CA);
		}
		if (calglRun->terminated || calglRun->step == calglRun->final_step) {
			calclGetSubstatesDeviceToHost3D(calglRun->device_CA);
			calglRun->active = CAL_FALSE;
			//breaking the simulation
			calglRun->end_time = time(NULL);
			printf("\nSimulation terminated\n");
			printf(" Elapsed time: %d\n", (int) (calglRun->end_time - calglRun->start_time));
			printf("*-----------------------------------------------------*\n");
			//saving configuration
			//calglSaveStateUpdater3DCL(calglRun);
		}
	} else {
		if (calglRun->onlyOneTime) {
			calglRun->onlyOneTime = CAL_FALSE;
			printf("\nSimulation Pause\n");
			calclGetSubstatesDeviceToHost3D(calglRun->device_CA);
			//calglSaveStateUpdater3DCL(calglRun);
		}

	}
}
