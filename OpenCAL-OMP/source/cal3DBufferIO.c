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

#include <cal3DBuffer.h>
#include <cal3DBufferIO.h>


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
