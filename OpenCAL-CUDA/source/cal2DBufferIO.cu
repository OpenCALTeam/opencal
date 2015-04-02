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
