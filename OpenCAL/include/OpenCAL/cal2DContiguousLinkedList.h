#include <OpenCAL/cal2D.h>
#include <OpenCAL/calCommon.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdint.h>

#ifndef ca2d_contiguous_linked_list
#define ca2d_contiguous_linked_list

typedef struct CALCell2D CALCell2D;
typedef struct CALModel2D CALModel2D;

typedef struct CALBufferElement2D
{
    CALCell2D cell;
    bool isActive;

    struct CALBufferElement2D* previous;
    struct CALBufferElement2D* next;


} CALBufferElement2D;

typedef struct CALContiguousLinkedList2D
{
   int size;
   int columns;
   int rows;
   int size_current;

   CALBufferElement2D* buffer;

   CALBufferElement2D* firstElementAddedAtCurrentIteration;
   CALBufferElement2D* head;
   CALBufferElement2D* tail;

} CALContiguousLinkedList2D;

DllExport 
CALContiguousLinkedList2D* calMakeContiguousLinkedList2D( CALModel2D* model );


/*! \brief Links the (i,j) cell to the tail of the list
*/
DllExport 
void calAddActiveCellCLL2D( CALModel2D* model, int i, int j );

/*! \brief Removes the (i,j) cell from the list
*/
DllExport 
void calRemoveActiveCellCLL2D( CALModel2D* model, int i, int j );

DllExport 
CALBufferElement2D* calGetNextBufferElement2D( CALContiguousLinkedList2D* model, CALBufferElement2D* current );

DllExport 
CALBufferElement2D* calGetFirstBufferElement2D( CALContiguousLinkedList2D* buffer );

DllExport 
void calSetNewIterationLinkedBuffer2D( CALContiguousLinkedList2D* buffer );

DllExport 
void calUpdateContiguousLinkedList2D( CALContiguousLinkedList2D* buffer );

DllExport 
void calApplyElementaryProcessActiveCellsCLL2D(CALModel2D* ca2D, CALCallbackFunc2D elementary_process);

DllExport 
void calCopyBufferActiveCellsCLL2Db(CALbyte* M_src, CALbyte* M_dest,  struct CALModel2D* ca2D);

DllExport 
void calCopyBufferActiveCellsCLL2Di(CALint* M_src, CALint* M_dest,  struct CALModel2D* ca2D);

DllExport 
void calCopyBufferActiveCellsCLL2Dr(CALreal* M_src, CALreal* M_dest,  struct CALModel2D* ca2D);

DllExport 
void calSetActiveCellsCLLBuffer2Db(CALbyte* M, CALbyte value, CALModel2D* ca2D);

DllExport 
void calSetActiveCellsCLLBuffer2Di(CALint* M, CALint value, CALModel2D* ca2D);

DllExport 
void calSetActiveCellsCLLBuffer2Dr(CALreal* M, CALreal value, CALModel2D* ca2D);

DllExport 
void calFreeContiguousLinkedList2D(CALContiguousLinkedList2D* cll);

DllExport 
int getLinearIndex2D(int columns, int i, int j );

#endif
