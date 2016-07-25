#include <OpenCAL-CPU/calModel.h>
#include <OpenCAL-CPU/calCommon.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdint.h>
#include <omp.h>

#ifndef cal_contiguous_linked_list
#define cal_contiguous_linked_list

typedef struct CALQueueElement
{
        CALIndices cell;
        struct CALQueueElement* next; //after you (you go first)
        struct CALQueueElement* previous; //before you (you go second)

}CALQueueElement;

typedef struct CALQueue
{
        CALQueueElement* first;
        CALQueueElement* last;

        omp_lock_t lock;

        int size;

}CALQueue;

typedef struct CALBufferElement
{
        CALIndices cell;
        bool isActive;

        struct CALBufferElement* previous;
        struct CALBufferElement* next;

        omp_lock_t* lock;

} CALBufferElement;

struct CALActiveCellsCLL
{
        struct CALActiveCells* inherited_pointer;

        int size;
        int size_current;

        int* number_of_active_cells_per_thread;

        int number_of_threads;

        CALBufferElement* buffer;

        CALBufferElement** _heads;
        CALBufferElement** _tails;

        CALQueue* queues_of_elements_to_add;
};

void calPutElement(struct CALActiveCellsCLL* A, CALIndices cell);
void calPushBack(struct CALActiveCellsCLL* A, int thread, CALIndices cell );
void calUpdateParallelCLL(struct CALActiveCellsCLL* A);
void calUpdateSerialCLL(struct CALActiveCellsCLL* A);

struct CALActiveCellsCLL* calMakeContiguousLinkedList(struct CALModel* model);


/*! \brief Links the cell to the tail of the list
*/
static void calAddActiveCellCLL(struct CALActiveCellsCLL* A, CALIndices cell)
{
#if CAL_PARALLEL == 1
    calPutElement(A, cell);
#else
    calPushBack(A, 0, cell);
#endif
}

/*! \brief Removes the cell from the list
*/
void calRemoveActiveCellCLL(struct CALActiveCellsCLL* A, CALIndices cell);

#if CAL_PARALLEL == 1
    #define calGetNextBufferElement(A, current)(current->next)
#elif CAL_PARALLEL == 0
    #define calGetNextBufferElement(A, current)(A->_tails[1] == current->next? NULL : current->next)
#endif

/*
 * TODO test if this macros work as they should
#if CAL_PARALLEL == 1
#define calGetNextBufferElement(A, current)(printf("sono nel par == 1 \n\n"))
#elif CAL_PARALLEL == 0
#define calGetNextBufferElement(A, current)(printf("sono nel par == 0 \n\n"))
#endif
*/
static void calUpdateActiveCellsCLL(struct CALActiveCellsCLL* A)
{
#if CAL_PARALLEL == 1
    calUpdateParallelCLL(A);
#else
    calUpdateSerialCLL(A);
#endif
}

void calRemoveInactiveCellsCLL(struct CALActiveCellsCLL* A, CALbyte (*inactive_cells_def)(struct CALModel*, CALIndices, int));

void calApplyElementaryProcessActiveCellsCLL(struct CALActiveCellsCLL* A, CALLocalProcess elementary_process);

void calCopyBufferActiveCellsCLL_b(CALbyte* M_src, CALbyte* M_dest,  struct CALActiveCellsCLL* A);
void calCopyBufferActiveCellsCLL_i(CALint* M_src, CALint* M_dest,  struct CALActiveCellsCLL* A);
void calCopyBufferActiveCellsCLL_r(CALreal* M_src, CALreal* M_dest,  struct CALActiveCellsCLL* A);


void calSetActiveCellsCLLBuffer_b(CALbyte* M, CALbyte value, struct CALActiveCellsCLL* A);
void calSetActiveCellsCLLBuffer_i(CALint* M, CALint value, struct CALActiveCellsCLL* A);
void calSetActiveCellsCLLBuffer_r(CALreal* M, CALreal value, struct CALActiveCellsCLL* A);

void calFreeContiguousLinkedList(struct CALActiveCellsCLL* A);

#endif
