﻿#include <OpenCAL/cal3D.h>
#include <OpenCAL/calCommon.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdint.h>

#ifndef ca3d_contiguous_linked_list
#define ca3d_contiguous_linked_list

typedef struct CALCell3D CALCell3D;
typedef struct CALModel3D CALModel3D;

typedef struct CALBufferElement3D
{
    CALCell3D cell;
    bool isActive;

    struct CALBufferElement3D* previous;
    struct CALBufferElement3D* next;


} CALBufferElement3D;

typedef struct CALContiguousLinkedList3D
{
   int size;
   int columns;
   int rows;
   int slices;
   int size_current;

   CALBufferElement3D* buffer;

   CALBufferElement3D* firstElementAddedAtCurrentIteration;
   CALBufferElement3D* head;
   CALBufferElement3D* tail;

} CALContiguousLinkedList3D;

DllExport
CALContiguousLinkedList3D* calMakeContiguousLinkedList3D( CALModel3D* model );


/*! \brief Links the (i,j,k) cell to the tail of the list
*/
DllExport
void calAddActiveCellCLL3D( CALModel3D* model, int i, int j, int k);

/*! \brief Removes the (i,j,k) cell from the list
*/
DllExport
void calRemoveActiveCellCLL3D( CALModel3D* model, int i, int j, int k);

DllExport
CALBufferElement3D* calGetNextBufferElement3D( CALContiguousLinkedList3D* model, CALBufferElement3D* current );

DllExport
CALBufferElement3D* calGetFirstBufferElement3D( CALContiguousLinkedList3D* buffer );

DllExport
void calSetNewIterationLinkedBuffer3D( CALContiguousLinkedList3D* buffer );

DllExport
void calUpdateContiguousLinkedList3D( CALContiguousLinkedList3D* buffer );

DllExport
void calApplyElementaryProcessActiveCellsCLL3D(CALModel3D* ca3D, CALCallbackFunc3D elementary_process);

DllExport
void calCopyBufferActiveCellsCLL3Db(CALbyte* M_src, CALbyte* M_dest,  struct CALModel3D* ca3D);

DllExport
void calCopyBufferActiveCellsCLL3Di(CALint* M_src, CALint* M_dest,  struct CALModel3D* ca3D);

DllExport
void calCopyBufferActiveCellsCLL3Dr(CALreal* M_src, CALreal* M_dest,  struct CALModel3D* ca3D);


void calSetActiveCellsCLLBuffer3Db(CALbyte* M, CALbyte value, CALModel3D* ca3D);
void calSetActiveCellsCLLBuffer3Di(CALint* M, CALint value, CALModel3D* ca3D);
void calSetActiveCellsCLLBuffer3Dr(CALreal* M, CALreal value, CALModel3D* ca3D);

void calFreeContiguousLinkedList3D(CALContiguousLinkedList3D* cll);

#endif
