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

#include <OpenCAL-OMP/cal3DBuffer.h>
#include <OpenCAL-OMP/cal3DBufferIO.h>


#include <stdlib.h>

#define STRLEN 256



void calfLoadBuffer3Db(CALbyte* M, int rows, int columns, int slices, FILE* f)
{
	char str[STRLEN];
	int i, j, k;

	for (k=0; k<slices; k++)
		for (i=0; i<rows; i++)
			for (j=0; j<columns; j++){
				fscanf(f, "%s", str);
				calSetBuffer3DElement(M, rows, columns, i, j, k, atoi(str));
			}
}

void calfLoadBuffer3Di(CALint* M, int rows, int columns, int slices, FILE* f)
{
	char str[STRLEN];
	int i, j, k;

	for (k=0; k<slices; k++)
		for (i=0; i<rows; i++)
			for (j=0; j<columns; j++){
				fscanf(f, "%s", str);
				calSetBuffer3DElement(M, rows, columns, i, j, k, atoi(str));
		}
}

void calfLoadBuffer3Dr(CALreal* M, int rows, int columns, int slices, FILE* f)
{
	char str[STRLEN];
	int i, j, k;

	for (k=0; k<slices; k++)
		for (i=0; i<rows; i++)
			for (j=0; j<columns; j++){
				fscanf(f, "%s", str);
				calSetBuffer3DElement(M, rows, columns, i, j, k, atof(str));
		}
}



CALbyte calLoadBuffer3Db(CALbyte* M, int rows, int columns, int slices, char* path)
{
	FILE *f = NULL;
	f = fopen(path, "r");

	if ( !f )
		return CAL_FALSE;

	calfLoadBuffer3Db(M, rows, columns, slices, f);

	fclose(f);

	return CAL_TRUE;
}

CALbyte calLoadBuffer3Di(CALint* M, int rows, int columns, int slices, char* path)
{
	FILE *f = NULL;
	f = fopen(path, "r");

	if ( !f )
		return CAL_FALSE;

	calfLoadBuffer3Di(M, rows, columns, slices, f);

	fclose(f);

	return CAL_TRUE;
}

CALbyte calLoadBuffer3Dr(CALreal* M, int rows, int columns, int slices, char* path)
{
	FILE *f = NULL;
	f = fopen(path, "r");

	if ( !f )
		return CAL_FALSE;

	calfLoadBuffer3Dr(M, rows, columns, slices, f);

	fclose(f);

	return CAL_TRUE;
}



void calfSaveBuffer3Db(CALbyte* M, int rows, int columns, int slices, FILE* f)
{
	char str[STRLEN];
	int i, j, k;

	for (k=0; k<slices; k++) {
		for (i=0; i<rows; i++) {
			for (j=0; j<columns; j++) {
				sprintf(str, "%d ", calGetBuffer3DElement(M, rows, columns, i, j, k));
				fprintf(f,"%s ",str);
			}
			fprintf(f,"\n");
 		}
		fprintf(f,"\n");
	}
}

void calfSaveBuffer3Di(CALint* M, int rows, int columns, int slices, FILE* f)
{
	char str[STRLEN];
	int i, j, k;

	for (k=0; k<slices; k++) {
		for (i=0; i<rows; i++) {
			for (j=0; j<columns; j++) {
				sprintf(str, "%d ", calGetBuffer3DElement(M, rows, columns, i, j, k));
				fprintf(f,"%s ",str);
			}
			fprintf(f,"\n");
 		}
		fprintf(f,"\n");
	}
}

void calfSaveBuffer3Dr(CALreal* M, int rows, int columns, int slices, FILE* f)
{
	char str[STRLEN];
	int i, j, k;

	for (k=0; k<slices; k++) {
		for (i=0; i<rows; i++) {
			for (j=0; j<columns; j++) {
				sprintf(str, "%f ", calGetBuffer3DElement(M, rows, columns, i, j, k));
				fprintf(f,"%s ",str);
			}
			fprintf(f,"\n");
 		}
		fprintf(f,"\n");
	}
}



CALbyte calSaveBuffer3Db(CALbyte* M, int rows, int columns, int slices, char* path)
{
	FILE *f;
	f = fopen(path, "w");

	if ( !f )
		return CAL_FALSE;

	calfSaveBuffer3Db(M, rows, columns, slices, f);

	fclose(f);

	return CAL_TRUE;
}

CALbyte calSaveBuffer3Di(CALint* M, int rows, int columns, int slices, char* path)
{
	FILE *f;
	f = fopen(path, "w");

	if ( !f )
		return CAL_FALSE;

	calfSaveBuffer3Di(M, rows, columns, slices, f);

	fclose(f);

	return CAL_TRUE;
}

CALbyte calSaveBuffer3Dr(CALreal* M, int rows, int columns, int slices, char* path)
{
	FILE *f;
	f = fopen(path, "w");

	if ( !f )
		return CAL_FALSE;

	calfSaveBuffer3Dr(M, rows, columns, slices, f);

	fclose(f);

	return CAL_TRUE;
}
