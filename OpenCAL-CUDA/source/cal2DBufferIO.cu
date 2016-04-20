/*
 * Copyright (c) 2016 OpenCALTeam (https://github.com/OpenCALTeam),
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

#include ".\..\include\cal2DBuffer.cuh"
#include ".\..\include\cal2DBufferIO.cuh"


#include <stdlib.h>

#define STRLEN 256


void calfCudaLoadMatrix2Db(CALbyte* M, int rows, int columns, FILE* f, int i_substate)
{
	char str[STRLEN];
	int i, j;
	for (i=0; i<rows; i++)
		for (j=0; j<columns; j++){
			fscanf(f, "%s", str);
			calCuSetMatrixElement(M, columns,rows, i, j, atof(str), i_substate);
	}
}

void calfCudaLoadMatrix2Di(CALint* M, int rows, int columns, FILE* f, int i_substate)
{
	char str[STRLEN];
	int i, j;
	for (i=0; i<rows; i++)
		for (j=0; j<columns; j++){
			fscanf(f, "%s", str);
			calCuSetMatrixElement(M, columns,rows, i, j, atof(str), i_substate);
	}
}

void calfCudaLoadMatrix2Dr(CALreal* M, int rows, int columns, FILE* f, int i_substate)
{
	char str[STRLEN];
	int i, j;

	for (i=0; i<rows; i++)
		for (j=0; j<columns; j++){
			fscanf(f, "%s", str);
			calCuSetMatrixElement(M, columns,rows, i, j, atof(str), i_substate);
	}
}

CALbyte calCudaLoadMatrix2Db(CALbyte* M, int rows, int columns, char* path, int i_substate)
{
	FILE *f = NULL;
	f = fopen(path, "r");

	if ( !f )
		return CAL_FALSE;

	calfCudaLoadMatrix2Db(M, rows, columns, f, i_substate);

	fclose(f);

	return CAL_TRUE;
}

CALbyte calCudaLoadMatrix2Di(CALint* M, int rows, int columns, char* path, int i_substate)
{
	FILE *f = NULL;
	f = fopen(path, "r");

	if ( !f )
		return CAL_FALSE;

	calfCudaLoadMatrix2Di(M, rows, columns, f, i_substate);

	fclose(f);

	return CAL_TRUE;
}

CALbyte calCudaLoadMatrix2Dr(CALreal* M, int rows, int columns, char* path, int i_substate)
{
	FILE *f = NULL;
	f = fopen(path, "r");

	if ( !f )
		return CAL_FALSE;

	calfCudaLoadMatrix2Dr(M, rows, columns, f, i_substate);

	fclose(f);

	return CAL_TRUE;
}

void calCudafSaveMatrix2Db(CALbyte* M, int rows, int columns, FILE* f, CALint index_substate)
{
	char str[STRLEN];
	int i, j;

	for (i=0; i<rows; i++) {
		for (j=0; j<columns; j++) {
			sprintf(str, "%d ", calCuGetMatrixElement(M, columns, rows, i, j, index_substate));
			fprintf(f,"%s ",str);
		}
		fprintf(f,"\n");
 	}
}

void calCudafSaveMatrix2Di(CALint* M, int rows, int columns, FILE* f, CALint index_substate)
{
	char str[STRLEN];
	int i, j;

	for (i=0; i<rows; i++) {
		for (j=0; j<columns; j++) {
			sprintf(str, "%d ", calCuGetMatrixElement(M, columns, rows, i, j, index_substate));
			fprintf(f,"%s ",str);
		}
		fprintf(f,"\n");
 	}
}

void calCudafSaveMatrix2Dr(CALreal* M, int rows, int columns, FILE* f, CALint index_substate)
{
	char str[STRLEN];
	int i, j;

	for (i=0; i<rows; i++) {
		for (j=0; j<columns; j++) {
			sprintf(str, "%f ", calCuGetMatrixElement(M, columns, rows, i, j, index_substate));
			fprintf(f,"%s ",str);
		}
		fprintf(f,"\n");
 	}
}

CALbyte calCudaSaveMatrix2Db(CALbyte* M, int rows, int columns, char* path, CALint index_substate)
{
	FILE *f;
	f = fopen(path, "w");

	if ( !f )
		return CAL_FALSE;

	calCudafSaveMatrix2Db(M, rows, columns, f, index_substate);

	fclose(f);

	return CAL_TRUE;
}

CALbyte calCudaSaveMatrix2Di(CALint* M, int rows, int columns, char* path, CALint index_substate)
{
	FILE *f;
	f = fopen(path, "w");

	if ( !f )
		return CAL_FALSE;

	calCudafSaveMatrix2Di(M, rows, columns, f, index_substate);

	fclose(f);

	return CAL_TRUE;
}

CALbyte calCudaSaveMatrix2Dr(CALreal* M, int rows, int columns, char* path, CALint index_substate)
{
	FILE *f;
	f = fopen(path, "w");

	if ( !f )
		return CAL_FALSE;

	calCudafSaveMatrix2Dr(M, rows, columns, f, index_substate);

	fclose(f);

	return CAL_TRUE;
}
