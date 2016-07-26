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

#include <OpenCAL-CPU/calBuffer.h>
#include <OpenCAL-CPU/calBufferIO.h>


#include <stdlib.h>

#define STRLEN 256



void calfLoadMatrix_b(CALbyte* M, int cellularSpaceDimension, FILE* f)
{
    char str[STRLEN];
    int i;

    for (i = 0; i < cellularSpaceDimension; i++)
    {
        fscanf(f, "%s", str);
        calSetMatrixElement(M, i, atoi(str));
    }
}

void calfLoadMatrix_i(CALint* M, int cellularSpaceDimension, FILE* f)
{
    char str[STRLEN];
    int i;

    for (i = 0; i < cellularSpaceDimension; i++)
    {
        fscanf(f, "%s", str);
        calSetMatrixElement(M, i, atoi(str));
    }
}

void calfLoadMatrix_r(CALreal* M, int cellularSpaceDimension, FILE* f)
{
    char str[STRLEN];
    int i;

    for (i = 0; i < cellularSpaceDimension; i++)
    {
        fscanf(f, "%s", str);
        calSetMatrixElement(M, i, atof(str));
    }
}



CALbyte calLoadMatrix_b(CALbyte* M, int cellularSpaceDimension, char* path)
{
    FILE *f = NULL;
    f = fopen(path, "r");

    if ( !f )
        return CAL_FALSE;

    calfLoadMatrix_b(M, cellularSpaceDimension, f);

    fclose(f);

    return CAL_TRUE;
}

CALbyte calLoadMatrix_i(CALint* M, int cellularSpaceDimension, char* path)
{
    FILE *f = NULL;
    f = fopen(path, "r");

    if ( !f )
        return CAL_FALSE;

    calfLoadMatrix_i(M, cellularSpaceDimension, f);

    fclose(f);

    return CAL_TRUE;
}

CALbyte calLoadMatrix_r(CALreal* M, int cellularSpaceDimension, char* path)
{
    printf("prima f open \n");
    FILE *f = NULL;
    f = fopen(path, "r");
    printf("dopo f open \n");

    if ( !f )
        return CAL_FALSE;

    calfLoadMatrix_r(M, cellularSpaceDimension, f);
    printf("dopo fload \n");

    fclose(f);

    return CAL_TRUE;
}

//TODO try and find a way to save multidimensional models to files

void calfSaveMatrix_b(CALbyte* M, int cellularSpaceDimension, FILE* f)
{
    char str[STRLEN];
    int i;

    for (i = 0; i < cellularSpaceDimension; i++)
    {

        sprintf(str, "%d ", calGetMatrixElement(M, i));
        fprintf(f,"%s ",str);
        //if( )
    }
    fprintf(f,"\n");
}


void calfSaveMatrix_i(CALint* M, int cellularSpaceDimension, FILE* f)
{
    char str[STRLEN];
    int i;

    for (i = 0; i < cellularSpaceDimension; i++)
    {
        sprintf(str, "%d ", calGetMatrixElement(M, i));
        fprintf(f,"%s ",str);
    }
    fprintf(f,"\n");
}


void calfSaveMatrix_r(CALreal* M, int cellularSpaceDimension, FILE* f)
{
    char str[STRLEN];
    int i, j;

    for (i = 0; i < cellularSpaceDimension; i++) {
        sprintf(str, "%f ", calGetMatrixElement(M,  i));
        fprintf(f,"%s ",str);
    }
    fprintf(f,"\n");
}




CALbyte calSaveMatrix_b(CALbyte* M, int cellularSpaceDimension, char* path)
{
    FILE *f;
    f = fopen(path, "w");

    if ( !f )
        return CAL_FALSE;

    calfSaveMatrix_b(M, cellularSpaceDimension, f);

    fclose(f);

    return CAL_TRUE;
}

CALbyte calSaveMatrix_i(CALint* M, int cellularSpaceDimension, char* path)
{
    FILE *f;
    f = fopen(path, "w");

    if ( !f )
        return CAL_FALSE;

    calfSaveMatrix_i(M, cellularSpaceDimension, f);

    fclose(f);

    return CAL_TRUE;
}

CALbyte calSaveMatrix_r(CALreal* M, int cellularSpaceDimension, char* path)
{
    FILE *f;
    f = fopen(path, "w");

    if ( !f )
        return CAL_FALSE;

    calfSaveMatrix_r(M, cellularSpaceDimension, f);

    fclose(f);

    return CAL_TRUE;
}

