#include <OpenCAL-OMP/cal3D.h>
#include <OpenCAL-OMP/calCommon.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdint.h>
#include <omp.h>

#ifndef ca3d_contiguous_linked_list
#define ca3d_contiguous_linked_list

typedef struct CALCell3D CALCell3D;

typedef struct CALQueueElement3D
{
        CALCell3D cell;
        struct CALQueueElement3D* next; //after you (you go first)
        struct CALQueueElement3D* previous; //before you (you go second)

}CALQueueElement3D;

typedef struct CALQueue3D
{
        CALQueueElement3D* first;
        CALQueueElement3D* last;

        omp_lock_t lock;

        int size;

}CALQueue3D;

DllExport
CALQueueElement3D calTakeElement3D( CALQueue3D* queue );


typedef struct CALModel3D CALModel3D;

typedef struct CALBufferElement3D
{
        CALCell3D cell;
        bool isActive;

        struct CALBufferElement3D* previous;
        struct CALBufferElement3D* next;

        omp_lock_t* lock;

} CALBufferElement3D;

typedef struct CALContiguousLinkedList3D
{
        int size;
        int columns;
        int rows;
        int slices;
        int size_current;

        int* numberOfActiveCellsPerThread;

        int numberOfThreads;

        CALBufferElement3D* buffer;

        CALBufferElement3D** _heads;
        CALBufferElement3D** _tails;

        CALQueue3D* queuesOfElementsToAdd;

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

DllExport
void calSetActiveCellsCLLBuffer3Db(CALbyte* M, CALbyte value, CALModel3D* ca3D);

DllExport
void calSetActiveCellsCLLBuffer3Di(CALint* M, CALint value, CALModel3D* ca3D);

DllExport
void calSetActiveCellsCLLBuffer3Dr(CALreal* M, CALreal value, CALModel3D* ca3D);

DllExport
void calFreeContiguousLinkedList3D(CALContiguousLinkedList3D* cll);

#endif
