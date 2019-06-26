#ifndef CALMULTINODECOMMON_H_
#define CALMULTINODECOMMON_H_
extern "C"
{
#include <OpenCAL-OMP/cal3DIO.h>
#include <OpenCAL-OMP/cal3DBuffer.h>
#include <OpenCAL-OMP/cal3DBufferIO.h>
};

#include <OpenCAL-OMP/cal3DDistributedDomain.h>


#define STRLEN 256


// LOAD 

void calfLoadMatrix3Dr(CALreal* M, int rows, int columns, int slices, FILE* f, int offset)
{
	char str[STRLEN];
	int i, j,k;

	for (k=offset; k<slices-offset; k++)
		for (i=0; i<rows; i++)
			for (j=0; j<columns; j++){
			fscanf(f, "%s", str);
			calSetBuffer3DElement(M, rows, columns, i, j, k, atof(str));
	}
}



void calfLoadMatrix3Di(CALint* M, int rows, int columns, int slices, FILE* f, int offset)
{
	char str[STRLEN];
	int i, j, k;

	for (k=offset; k<slices-offset; k++)
		for (i=0; i<rows; i++)
			for (j=0; j<columns; j++){
			fscanf(f, "%s", str);
			calSetBuffer3DElement(M, rows, columns, i, j, k, atoi(str));
	}
}


void calfLoadMatrix3Db(CALbyte* M, int rows, int columns, int slices, FILE* f, int offset)
{
	char str[STRLEN];
	int i, j, k;

	for (k=offset; k<slices-offset; k++)
		for (i=0; i<rows; i++)
			for (j=0; j<columns; j++){
			fscanf(f, "%s", str);
			calSetBuffer3DElement(M, rows, columns, i, j, k, atoi(str));
	}
}

CALbyte calNodeLoadMatrix3Dr(CALreal* M, const int rows, const int columns, const int slices,const char* path,  int read_offset = 0, const int write_offset = 0)
{
    FILE *f = NULL;
    f = fopen(path, "r");

    if ( !f )
        return CAL_FALSE;
    // read_offset number of layers from top
    int numberofrowstoskip = read_offset*rows;
    //skip #offset rows
    const int _s = 20*2*columns; //assuming 20char per number + spaces
    char tmp[_s];
    while(numberofrowstoskip--)
        fgets(tmp,sizeof(char)*_s,f);

    calfLoadMatrix3Dr(M, rows, columns, slices, f, write_offset);

    fclose(f);

    return CAL_TRUE;
}

CALbyte calNodeLoadSubstate3Dr(CALModel3D* ca3D, struct CALSubstate3Dr* Q, char* path,const Node& mynode) {

    CALbyte return_state = calNodeLoadMatrix3Dr(Q->current, ca3D->rows, ca3D->columns, ca3D->slices, path,mynode.offset, mynode.borderSize);
    if (Q->next)
        calCopyBuffer3Dr(Q->current, Q->next, ca3D->rows, ca3D->columns, ca3D->slices);
    return return_state;
}

CALbyte calNodeLoadMatrix3Di(CALint* M, const int rows, const int columns, const int slices, const char* path,  int read_offset = 0, const int write_offset = 0)
{
    FILE *f = NULL;
    f = fopen(path, "r");

    if ( !f )
        return CAL_FALSE;
    // read_offset number of layers from top
    int numberofrowstoskip = read_offset*rows;
    //skip #offset rows
    const int _s = 20*2*columns; //assuming 20char per number + spaces
    char tmp[_s];
    while(numberofrowstoskip--)
        fgets(tmp,sizeof(char)*_s,f);

    calfLoadMatrix3Di(M+write_offset, rows, columns, slices, f, write_offset);

    fclose(f);

    return CAL_TRUE;
}

CALbyte calNodeLoadSubstate3Di(CALModel3D* ca3D, struct CALSubstate3Di* Q, char* path,const Node& mynode) {

    CALbyte return_state = calNodeLoadMatrix3Di(Q->current, ca3D->rows, ca3D->columns, ca3D->slices, path,mynode.offset, mynode.borderSize);
    if (Q->next)
        calCopyBuffer3Di(Q->current, Q->next, ca3D->rows, ca3D->columns, ca3D->slices);
    return return_state;
}

CALbyte calNodeLoadMatrix3Db(CALbyte* M, const int rows, const int columns,  const int slices, const char* path,  int read_offset = 0, const int write_offset = 0)
{
    FILE *f = NULL;
    f = fopen(path, "r");

    if ( !f )
        return CAL_FALSE;
    // read_offset number of layers from top
    int numberofrowstoskip = read_offset*rows;
    //skip #offset rows
    const int _s = 20*2*columns; //assuming 20char per number + spaces
    char tmp[_s];
    while(numberofrowstoskip--)
        fgets(tmp,sizeof(char)*_s,f);

    calfLoadMatrix3Db(M+write_offset, rows, columns, slices, f, write_offset);

    fclose(f);

    return CAL_TRUE;
}

CALbyte calNodeLoadSubstate3Db(CALModel3D* ca3D, struct CALSubstate3Db* Q, char* path,const Node& mynode) {

    CALbyte return_state = calNodeLoadMatrix3Db(Q->current, ca3D->rows, ca3D->columns, ca3D->slices, path,mynode.offset, mynode.borderSize);
    if (Q->next)
        calCopyBuffer3Db(Q->current, Q->next, ca3D->rows, ca3D->columns, ca3D->slices);
    return return_state;
}





// SAVE

void calfSaveMatrix3Db(CALbyte* M, int rows, int columns, int slices, FILE* f, int offset)
{
	char str[STRLEN];
	int i, j, k;

	for (k=offset; k<slices-offset; k++){
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

void calfSaveMatrix3Di(CALint* M, int rows, int columns, int slices, FILE* f, int offset)
{
	char str[STRLEN];
	int i, j, k;

	for (k=offset; k<slices-offset; k++){
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

void calfSaveMatrix3Dr(CALreal* M, int rows, int columns, int slices, FILE* f, int offset)
{
	char str[STRLEN];
	int i, j, k;

	for (k=offset; k<slices-offset; k++){
//		for (k=0; k<slices; k++){
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

CALbyte calNodeSaveMatrix3Dr(CALreal* M, int rows, int columns, int slices, char* path, int readoffset, int writeoffset, int offset)
{

	FILE *f;
	f = fopen(path, "w");
	if ( !f )
		return CAL_FALSE;

	calfSaveMatrix3Dr(M, rows, columns,  slices, f,offset);

	fclose(f);

	return CAL_TRUE;
}

CALbyte calNodeSaveSubstate3Dr(CALModel3D* ca3D, struct CALSubstate3Dr* Q, char* path,const Node& mynode) {
	int write_offset=0;
    CALbyte return_state = calNodeSaveMatrix3Dr(Q->current, ca3D->rows, ca3D->columns, ca3D->slices, path,mynode.offset,write_offset, mynode.borderSize);
    return return_state;
}



CALbyte calNodeSaveMatrix3Di(CALint* M, int rows, int columns, int slices, char* path, int readoffset, int writeoffset, int offset)
{

	FILE *f;
	f = fopen(path, "w");
	if ( !f )
		return CAL_FALSE;

	calfSaveMatrix3Di(M, rows, columns, slices, f,offset);

	fclose(f);

	return CAL_TRUE;
}

CALbyte calNodeSaveSubstate3Di(CALModel3D* ca3D, struct CALSubstate3Di* Q, char* path,const Node& mynode) {
	int write_offset=0;
    CALbyte return_state = calNodeSaveMatrix3Di(Q->current, ca3D->rows, ca3D->columns, ca3D->slices, path,mynode.offset,write_offset, mynode.borderSize);
    return return_state;
}

CALbyte calNodeSaveMatrix3Db(CALbyte* M, int rows, int columns, int slices, char* path, int readoffset, int writeoffset, int offset)
{

	FILE *f;
	f = fopen(path, "w");
	if ( !f )
		return CAL_FALSE;

	calfSaveMatrix3Db(M, rows, columns,  slices, f,offset);

	fclose(f);

	return CAL_TRUE;
}

CALbyte calNodeSaveSubstate3Db(CALModel3D* ca3D, struct CALSubstate3Db* Q, char* path,const Node& mynode) {
	int write_offset=0;
    CALbyte return_state = calNodeSaveMatrix3Db(Q->current, ca3D->rows, ca3D->columns, ca3D->slices, path,mynode.offset,write_offset, mynode.borderSize);
    return return_state;
}
#endif

