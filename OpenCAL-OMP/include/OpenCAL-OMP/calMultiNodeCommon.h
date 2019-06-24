#ifndef CALMULTINODECOMMON_H_
#define CALMULTINODECOMMON_H_
extern "C"
{
#include <OpenCAL-OMP/cal2DIO.h>
#include <OpenCAL-OMP/cal2DBuffer.h>
#include <OpenCAL-OMP/cal2DBufferIO.h>
};

#include <OpenCAL-OMP/calDistributedDomain2D.h>


#define STRLEN 256

void calfLoadMatrix2Dr(CALreal* M, int rows, int columns, FILE* f, int offset)
{
	char str[STRLEN];
	int i, j;

	for (i=offset; i<rows-offset; i++)
		for (j=0; j<columns; j++){
			fscanf(f, "%s", str);
			calSetMatrixElement(M, columns, i, j, atof(str));
	}
}


CALbyte calNodeLoadMatrix2Dr(CALreal* M, const int rows, const int columns, const char* path,  int read_offset = 0, const int write_offset = 0)
{
    FILE *f = NULL;
    f = fopen(path, "r");
    printf("loading Real %s   -->   %d\n",path, write_offset);

    if ( !f )
        return CAL_FALSE;

    //skip #offset rows
    const int _s = 20*2*columns; //assuming 20char per number + spaces
    char tmp[_s];
    while(read_offset--)
        fgets(tmp,sizeof(char)*_s,f);

    calfLoadMatrix2Dr(M, rows, columns, f, write_offset);

    fclose(f);

    return CAL_TRUE;
}



void calfLoadMatrix2Di(CALint* M, int rows, int columns, FILE* f, int offset)
{
	char str[STRLEN];
	int i, j;

	for (i=offset; i<rows-offset; i++)
		for (j=0; j<columns; j++){
			fscanf(f, "%s", str);
			calSetMatrixElement(M, columns, i, j, atoi(str));
	}
}

CALbyte calNodeLoadMatrix2Di(CALint* M, const int rows, const int columns, const char* path,  int read_offset = 0, const int write_offset = 0)
{
    FILE *f = NULL;
    f = fopen(path, "r");

    if ( !f )
        return CAL_FALSE;

    //skip #offset rows
    const int _s = 20*2*columns; //assuming 20char per number + spaces
    char tmp[_s];
    while(read_offset--)
        fgets(tmp,sizeof(char)*_s,f);

    calfLoadMatrix2Di(M, rows, columns, f,write_offset);

    fclose(f);

    return CAL_TRUE;
}

void calfLoadMatrix2Db(CALbyte* M, int rows, int columns, FILE* f, int offset)
{
	char str[STRLEN];
	int i, j;

	for (i=offset; i<rows-offset; i++)
		for (j=0; j<columns; j++){
			fscanf(f, "%s", str);
			calSetMatrixElement(M, columns, i, j, atoi(str));
	}
}

CALbyte calNodeLoadMatrix2Db(CALbyte* M, const int rows, const int columns, const char* path,  int read_offset = 0, const int write_offset = 0)
{
    FILE *f = NULL;
    f = fopen(path, "r");

    if ( !f )
        return CAL_FALSE;

    //skip #offset rows
    const int _s = 20*2*columns; //assuming 20char per number + spaces
    char tmp[_s];
    while(read_offset--)
        fgets(tmp,sizeof(char)*_s,f);

    calfLoadMatrix2Db(M, rows, columns, f,write_offset);

    fclose(f);

    return CAL_TRUE;
}

CALbyte calNodeLoadSubstate2Dr(CALModel2D* ca2D, struct CALSubstate2Dr* Q, char* path,const Node& mynode) {
    CALbyte return_state = calNodeLoadMatrix2Dr(Q->current, ca2D->rows, ca2D->columns, path,mynode.offset,mynode.borderSize);
    if (Q->next)
        calCopyBuffer2Dr(Q->current, Q->next, ca2D->rows, ca2D->columns);
    return return_state;
}

CALbyte calNodeLoadSubstate2Di(CALModel2D* ca2D, struct CALSubstate2Di* Q, char* path,const Node& mynode) {
    CALbyte return_state = calNodeLoadMatrix2Di(Q->current, ca2D->rows, ca2D->columns, path,mynode.offset,mynode.borderSize);
    if (Q->next)
        calCopyBuffer2Di(Q->current, Q->next, ca2D->rows, ca2D->columns);
    return return_state;
}

CALbyte calNodeLoadSubstate2Db(CALModel2D* ca2D, struct CALSubstate2Db* Q, char* path,const Node& mynode) {
    CALbyte return_state = calNodeLoadMatrix2Db(Q->current, ca2D->rows, ca2D->columns, path,mynode.offset,mynode.borderSize);
    if (Q->next)
        calCopyBuffer2Db(Q->current, Q->next, ca2D->rows, ca2D->columns);
    return return_state;
}

void calfSaveMatrix2Db(CALbyte* M, int rows, int columns, FILE* f, int offset)
{
	char str[STRLEN];
	int i, j;

	for (i=offset; i<rows-offset; i++) {
		for (j=0; j<columns; j++) {
			sprintf(str, "%d ", calGetMatrixElement(M, columns, i, j));
			fprintf(f,"%s ",str);
		}
		fprintf(f,"\n");
 	}
}

void calfSaveMatrix2Di(CALint* M, int rows, int columns, FILE* f, int offset)
{
	char str[STRLEN];
	int i, j;

	for (i=offset; i<rows-offset; i++) {
		for (j=0; j<columns; j++) {
			sprintf(str, "%d ", calGetMatrixElement(M, columns, i, j));
			fprintf(f,"%s ",str);
		}
		fprintf(f,"\n");
 	}
}

void calfSaveMatrix2Dr(CALreal* M, int rows, int columns, FILE* f, int offset)
{
	char str[STRLEN];
	int i, j;

	for (i=offset; i<rows-offset; i++) {
		for (j=0; j<columns; j++) {
			sprintf(str, "%f ", calGetMatrixElement(M, columns, i, j));
			fprintf(f,"%s ",str);
		}
		fprintf(f,"\n");
 	}
}

CALbyte calNodeSaveMatrix2Dr(CALreal* M, int rows, int columns, char* path, int readoffset, int writeoffset, int offset)
{

	FILE *f;
	f = fopen(path, "w");
	if ( !f )
		return CAL_FALSE;

	calfSaveMatrix2Dr(M, rows, columns, f,offset);

	fclose(f);

	return CAL_TRUE;
}

CALbyte calNodeSaveSubstate2Dr(CALModel2D* ca2D, struct CALSubstate2Dr* Q, char* path,const Node& mynode) {
	int write_offset=0;
    CALbyte return_state = calNodeSaveMatrix2Dr(Q->current, ca2D->rows, ca2D->columns, path,mynode.offset,write_offset, mynode.borderSize);
    return return_state;
}



CALbyte calNodeSaveMatrix2Di(CALint* M, int rows, int columns, char* path, int readoffset, int writeoffset, int offset)
{

	FILE *f;
	f = fopen(path, "w");
	if ( !f )
		return CAL_FALSE;

	calfSaveMatrix2Di(M, rows, columns, f,offset);

	fclose(f);

	return CAL_TRUE;
}

CALbyte calNodeSaveSubstate2Di(CALModel2D* ca2D, struct CALSubstate2Di* Q, char* path,const Node& mynode) {
	int write_offset=0;
    CALbyte return_state = calNodeSaveMatrix2Di(Q->current, ca2D->rows, ca2D->columns, path,mynode.offset,write_offset, mynode.borderSize);
    return return_state;
}

CALbyte calNodeSaveMatrix2Db(CALbyte* M, int rows, int columns, char* path, int readoffset, int writeoffset, int offset)
{

	FILE *f;
	f = fopen(path, "w");
	if ( !f )
		return CAL_FALSE;

	calfSaveMatrix2Db(M, rows, columns, f,offset);

	fclose(f);

	return CAL_TRUE;
}

CALbyte calNodeSaveSubstate2Db(CALModel2D* ca2D, struct CALSubstate2Db* Q, char* path,const Node& mynode) {
	int write_offset=0;
    CALbyte return_state = calNodeSaveMatrix2Db(Q->current, ca2D->rows, ca2D->columns, path,mynode.offset,write_offset, mynode.borderSize);
    return return_state;
}
#endif

