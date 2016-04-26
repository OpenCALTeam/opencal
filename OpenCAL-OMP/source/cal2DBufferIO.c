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

#include <OpenCAL-OMP/cal2DBuffer.h>
#include <OpenCAL-OMP/cal2DBufferIO.h>


#include <stdlib.h>

#define STRLEN 256



void calfLoadMatrix2Db(CALbyte* M, int rows, int columns, FILE* f)
{
	char str[STRLEN];
	int i, j;

	for (i=0; i<rows; i++)
		for (j=0; j<columns; j++){
			fscanf(f, "%s", str);
			calSetMatrixElement(M, columns, i, j, atoi(str));
	}
}

void calfLoadMatrix2Di(CALint* M, int rows, int columns, FILE* f)
{
	char str[STRLEN];
	int i, j;

	for (i=0; i<rows; i++)
		for (j=0; j<columns; j++){
			fscanf(f, "%s", str);
			calSetMatrixElement(M, columns, i, j, atoi(str));
	}
}

void calfLoadMatrix2Dr(CALreal* M, int rows, int columns, FILE* f)
{
	char str[STRLEN];
	int i, j;

	for (i=0; i<rows; i++)
		for (j=0; j<columns; j++){
			fscanf(f, "%s", str);
			calSetMatrixElement(M, columns, i, j, atof(str));
	}
}



CALbyte calLoadMatrix2Db(CALbyte* M, int rows, int columns, char* path)
{
	FILE *f = NULL;
	f = fopen(path, "r");

	if ( !f )
		return CAL_FALSE;

	calfLoadMatrix2Db(M, rows, columns, f);

	fclose(f);

	return CAL_TRUE;
}

CALbyte calLoadMatrix2Di(CALint* M, int rows, int columns, char* path)
{
	FILE *f = NULL;
	f = fopen(path, "r");

	if ( !f )
		return CAL_FALSE;

	calfLoadMatrix2Di(M, rows, columns, f);

	fclose(f);

	return CAL_TRUE;
}

CALbyte calLoadMatrix2Dr(CALreal* M, int rows, int columns, char* path)
{
	FILE *f = NULL;
	f = fopen(path, "r");

	if ( !f )
		return CAL_FALSE;

	calfLoadMatrix2Dr(M, rows, columns, f);

	fclose(f);

	return CAL_TRUE;
}



void calfSaveMatrix2Db(CALbyte* M, int rows, int columns, FILE* f)
{
	char str[STRLEN];
	int i, j;

	for (i=0; i<rows; i++) {
		for (j=0; j<columns; j++) {
			sprintf(str, "%d ", calGetMatrixElement(M, columns, i, j));
			fprintf(f,"%s ",str);
		}
		fprintf(f,"\n");
 	}
}

void calfSaveMatrix2Di(CALint* M, int rows, int columns, FILE* f)
{
	char str[STRLEN];
	int i, j;

	for (i=0; i<rows; i++) {
		for (j=0; j<columns; j++) {
			sprintf(str, "%d ", calGetMatrixElement(M, columns, i, j));
			fprintf(f,"%s ",str);
		}
		fprintf(f,"\n");
 	}
}

void calfSaveMatrix2Dr(CALreal* M, int rows, int columns, FILE* f)
{
	char str[STRLEN];
	int i, j;

	for (i=0; i<rows; i++) {
		for (j=0; j<columns; j++) {
			sprintf(str, "%f ", calGetMatrixElement(M, columns, i, j));
			fprintf(f,"%s ",str);
		}
		fprintf(f,"\n");
 	}
}



CALbyte calSaveMatrix2Db(CALbyte* M, int rows, int columns, char* path)
{
	FILE *f;
	f = fopen(path, "w");

	if ( !f )
		return CAL_FALSE;

	calfSaveMatrix2Db(M, rows, columns, f);

	fclose(f);

	return CAL_TRUE;
}

CALbyte calSaveMatrix2Di(CALint* M, int rows, int columns, char* path)
{
	FILE *f;
	f = fopen(path, "w");

	if ( !f )
		return CAL_FALSE;

	calfSaveMatrix2Di(M, rows, columns, f);

	fclose(f);

	return CAL_TRUE;
}

CALbyte calSaveMatrix2Dr(CALreal* M, int rows, int columns, char* path)
{
	FILE *f;
	f = fopen(path, "w");

	if ( !f )
		return CAL_FALSE;

	calfSaveMatrix2Dr(M, rows, columns, f);

	fclose(f);

	return CAL_TRUE;
}
