#include <OpenCAL-CPU/calModel.h>
#include <OpenCAL-CPU/calCommon.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdint.h>
#include <omp.h>

#ifndef cal_contiguous_linked_list
#define cal_contiguous_linked_list

#if CAL_PARALLEL == 1
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


CALQueueElement calTakeElement(CALQueue* queue);
#endif

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

#if CAL_PARALLEL == 1
        CALQueue* queues_of_elements_to_add;
#endif
};

struct CALActiveCellsCLL* calMakeContiguousLinkedList(struct CALModel* model);


/*! \brief Links the cell to the tail of the list
*/
void calAddActiveCellCLL(struct CALActiveCellsCLL* A, CALIndices cell);

/*! \brief Removes the cell from the list
*/
void calRemoveActiveCellCLL(struct CALActiveCellsCLL* A, CALIndices cell);


#define calGetNextBufferElement(current)(current->next)

void calUpdateContiguousLinkedList(struct CALActiveCellsCLL* A);

void calApplyElementaryProcessActiveCellsCLL(struct CALActiveCellsCLL* A, CALLocalProcess elementary_process);

void calCopyBufferActiveCellsCLL_b(CALbyte* M_src, CALbyte* M_dest,  struct CALActiveCellsCLL* A);
void calCopyBufferActiveCellsCLL_i(CALint* M_src, CALint* M_dest,  struct CALActiveCellsCLL* A);
void calCopyBufferActiveCellsCLL_r(CALreal* M_src, CALreal* M_dest,  struct CALActiveCellsCLL* A);


void calSetActiveCellsCLLBuffer_b(CALbyte* M, CALbyte value, struct CALActiveCellsCLL* A);
void calSetActiveCellsCLLBuffer_i(CALint* M, CALint value, struct CALActiveCellsCLL* A);
void calSetActiveCellsCLLBuffer_r(CALreal* M, CALreal value, struct CALActiveCellsCLL* A);

void calFreeContiguousLinkedList(struct CALActiveCellsCLL* A);

#endif
