#include <OpenCAL-OMP/cal2D.h>
#include <OpenCAL-OMP/calCommon.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdint.h>
#include <omp.h>

#ifndef ca2d_contiguous_linked_list
#define ca2d_contiguous_linked_list

typedef struct CALCell2D CALCell2D;

typedef struct CALQueueElement2D
{
        CALCell2D cell;
        struct CALQueueElement2D* next; //after you (you go first)
        struct CALQueueElement2D* previous; //before you (you go second)

}CALQueueElement2D;

typedef struct CALQueue2D
{
        CALQueueElement2D* first;
        CALQueueElement2D* last;

        omp_lock_t lock;

        int size;

}CALQueue2D;

DllExport
CALQueueElement2D calTakeElement2D( CALQueue2D* queue );


typedef struct CALBufferElement2D
{
        CALCell2D cell;
        bool isActive;

        struct CALBufferElement2D* previous;
        struct CALBufferElement2D* next;

        omp_lock_t* lock;

} CALBufferElement2D;

typedef struct CALContiguousLinkedList2D
{
        int size;
        int columns;
        int rows;
        int size_current;

        int* numberOfActiveCellsPerThread;

        int numberOfThreads;

        CALBufferElement2D* buffer;

        CALBufferElement2D** _heads;
        CALBufferElement2D** _tails;

        CALQueue2D* queuesOfElementsToAdd;

} CALContiguousLinkedList2D;

DllExport
CALContiguousLinkedList2D* calMakeContiguousLinkedList2D( struct CALModel2D* model );


/*! \brief Links the (i,j) cell to the tail of the list
*/
DllExport
void calAddActiveCellCLL2D( struct CALModel2D* model, int i, int j );

/*! \brief Removes the (i,j) cell from the list
*/
DllExport
void calRemoveActiveCellCLL2D( struct CALModel2D* model, int i, int j );

DllExport
CALBufferElement2D* calGetNextBufferElement2D( CALContiguousLinkedList2D* model, CALBufferElement2D* current );

DllExport
CALBufferElement2D* calGetFirstBufferElement2D( CALContiguousLinkedList2D* buffer );

DllExport
void calSetNewIterationLinkedBuffer2D( CALContiguousLinkedList2D* buffer );

DllExport
void calUpdateContiguousLinkedList2D( CALContiguousLinkedList2D* buffer );

DllExport
void calApplyElementaryProcessActiveCellsCLL2D(struct CALModel2D* ca2D, CALCallbackFunc2D elementary_process);

DllExport
void calCopyBufferActiveCellsCLL2Db(CALbyte* M_src, CALbyte* M_dest,  struct CALModel2D* ca2D);

DllExport
void calCopyBufferActiveCellsCLL2Di(CALint* M_src, CALint* M_dest,  struct CALModel2D* ca2D);

DllExport
void calCopyBufferActiveCellsCLL2Dr(CALreal* M_src, CALreal* M_dest,  struct CALModel2D* ca2D);

DllExport
void calSetActiveCellsCLLBuffer2Db(CALbyte* M, CALbyte value, struct CALModel2D* ca2D);

DllExport
void calSetActiveCellsCLLBuffer2Di(CALint* M, CALint value, struct CALModel2D* ca2D);

DllExport
void calSetActiveCellsCLLBuffer2Dr(CALreal* M, CALreal value, struct CALModel2D* ca2D);

DllExport
void calFreeContiguousLinkedList2D(CALContiguousLinkedList2D* cll);

#endif
