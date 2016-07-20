#include <OpenCAL-CPU/calActiveCellsCLL.h>

static int calGetBuffer(struct CALActiveCellsCLL* A)
{
    int min = 0;
    int index = 0;
    int ie = 0;
    for( ; index < A->number_of_threads; index++ )
    {
        if( index  == 0 )
        {
            min = A->number_of_active_cells_per_thread[index];
        }
        else if( min >  A->number_of_active_cells_per_thread[index] )
        {
            min = A->number_of_active_cells_per_thread[index];
            ie = index;
        }
    }
    return ie;
}

void calPutElement(struct CALActiveCellsCLL* A, CALIndices cell)
{
    int linear_index = getLinearIndex(cell, A->inherited_pointer->calModel->coordinatesDimensions,A->inherited_pointer->calModel->numberOfCoordinates);
    CALBufferElement* element = &A->buffer[linear_index];
    omp_set_lock(element->lock);
    if(element->isActive)
    {
        omp_unset_lock(element->lock );
        return;
    }
    int thread = calGetBuffer(A);
    A->buffer[linear_index].isActive = true;

    A->number_of_active_cells_per_thread[thread]++;
    CALQueue* queue = &A->queues_of_elements_to_add[thread];
    CALQueueElement* newElement = (CALQueueElement*) malloc (sizeof(CALQueueElement));
    newElement->cell = cell;

    omp_set_lock(&queue->lock);

    newElement->next = NULL;

    if(queue->first == NULL)
    {
        newElement->previous = NULL;
        queue->first = newElement;
        queue->last = newElement;
    }
    else
    {
        newElement->previous = queue->last;
        queue->last->next = newElement;
        queue->last = newElement;
    }
    queue->size++;
    omp_unset_lock(&queue->lock);
    omp_unset_lock(A->buffer[linear_index].lock);
}

static CALQueueElement* calTakeElement(CALQueue* queue)
{
    CALQueueElement* element;
    CALQueueElement* cell;
    cell = NULL;

    if(queue->size > 0)
    {
        element = queue->first;
        if(element == queue->last)
        {
            queue->first = NULL;
            queue->last = NULL;
        }
        else
        {
            queue->first = element->next;
            element->next->previous = NULL;
        }

        cell->cell = element->cell;
        queue->size--;
        free( element );
    }

    return cell;
}

void calUpdateParallelCLL(struct CALActiveCellsCLL* A)
{
    int n = 0;
    CALQueueElement* queueElement;

    int number_of_threads = A->number_of_threads;
    int* size_per_t = (int*) malloc (sizeof(int) * number_of_threads);

#pragma omp parallel for private(queueElement) firstprivate(A, number_of_threads)
    for( n = 0; n < A->number_of_threads; n++ )
    {
        size_per_t[n] = 0;
        queueElement = calTakeElement(&A->queues_of_elements_to_add[n] );
        while(queueElement != NULL)
        {
            calPushBack(A, n, queueElement->cell);

            size_per_t[n]++;
            queueElement = calTakeElement(&A->queues_of_elements_to_add[n]);
        }
    }

    for(n = 0; n < A->number_of_threads; n++ )
        A->size_current += size_per_t[n];


    free(size_per_t);
}

void calUpdateSerialCLL(struct CALActiveCellsCLL* A)
{
    A->_tails[1] = NULL;
}

void calPushBack(struct CALActiveCellsCLL* A, int thread, CALIndices cell )
{
    CALBufferElement* tail = A->_tails[thread];
    CALBufferElement* element = &A->buffer[getLinearIndex(cell, A->inherited_pointer->calModel->coordinatesDimensions, A->inherited_pointer->calModel->numberOfCoordinates)];

    if( A->_heads[thread] == NULL )
    {
        A->_heads[thread] = element;
        A->_tails[thread] = element;
        element->next = NULL;
        element->previous = NULL;
        return;
    }
    element->previous = tail;
    tail->next = element;
    A->_tails[thread] = element;
    element->next = NULL;

}


struct CALActiveCellsCLL* calMakeContiguousLinkedList(struct CALModel* calModel)
{
    struct CALActiveCellsCLL* contiguousLinkedList = (struct CALActiveCellsCLL*) malloc (sizeof(struct CALActiveCellsCLL));
    contiguousLinkedList->inherited_pointer = (struct CALActiveCells*) malloc (sizeof(struct CALActiveCells));
    contiguousLinkedList->inherited_pointer->calModel = calModel;
    contiguousLinkedList->size = calModel->cellularSpaceDimension;

#pragma omp parallel
    {
#pragma omp single
        contiguousLinkedList->number_of_threads = CAL_GET_NUM_THREADS();
    }


    if( contiguousLinkedList->size <= 0 )
    {
        free( contiguousLinkedList );
        return NULL;
    }

    contiguousLinkedList->queues_of_elements_to_add = (CALQueue*) malloc(sizeof(CALQueue) * contiguousLinkedList->number_of_threads);

    contiguousLinkedList->_heads = (CALBufferElement**) malloc (sizeof(CALBufferElement*) * contiguousLinkedList->number_of_threads);

    if(contiguousLinkedList->number_of_threads == 1)
        contiguousLinkedList->_tails = (CALBufferElement**) malloc (sizeof(CALBufferElement*) * 2);
    else
        contiguousLinkedList->_tails = (CALBufferElement**) malloc (sizeof(CALBufferElement*) * contiguousLinkedList->number_of_threads);

    contiguousLinkedList->buffer = (CALBufferElement*) malloc (sizeof(CALBufferElement) * contiguousLinkedList->size);
    contiguousLinkedList->number_of_active_cells_per_thread = (int *) malloc (sizeof(int) * contiguousLinkedList->number_of_threads);

    int n;
    for(n = 0; n < contiguousLinkedList->number_of_threads; n++)
    {
        contiguousLinkedList->number_of_active_cells_per_thread[n] = 0;
        contiguousLinkedList->queues_of_elements_to_add[n].first = NULL;
        contiguousLinkedList->queues_of_elements_to_add[n].last = NULL;
        omp_init_lock( &contiguousLinkedList->queues_of_elements_to_add[n].lock);
        contiguousLinkedList->queues_of_elements_to_add[n].size =  0;
        contiguousLinkedList->_heads[n] = NULL;
        contiguousLinkedList->_tails[n] = NULL;
    }


    int number_of_dimensions = calModel->numberOfCoordinates;
    int* coordinates_dimensions = calModel->coordinatesDimensions;
    int i = 0;

    for( ; i < contiguousLinkedList->size; i++ )
    {
        CALBufferElement element;
        int n, k;
        int linearIndex = i;
        int* v = (int*)malloc(sizeof(int) * number_of_dimensions);
        int t = calModel->cellularSpaceDimension;
        for(n = number_of_dimensions - 1; n >= 0; n--)
        {
            if (n ==1)
                k=0;
            else if (n==0)
                k=1;
            else
                k=n;

            t= (int)t/coordinates_dimensions[k];
            v[k] = (int) linearIndex/t;
            linearIndex = linearIndex%t;
        }
        element.cell = v;
        element.isActive = false;
        element.next = NULL;
        element.previous = NULL;

        element.lock = (omp_lock_t*) malloc (sizeof(omp_lock_t));
        omp_init_lock(element.lock);

        contiguousLinkedList->buffer[i] = element;
    }

    contiguousLinkedList->size_current = 0;
    return contiguousLinkedList;
}

void calRemoveActiveCellCLL(struct CALActiveCellsCLL* A, CALIndices cell)
{
    int linear_address = getLinearIndex(cell, A->inherited_pointer->calModel->coordinatesDimensions, A->inherited_pointer->calModel->numberOfCoordinates);
    CALBufferElement* element = &A->buffer[linear_address];

    if(!element->isActive)
        return;

    element->isActive = false;
    int thread = CAL_GET_THREAD_NUM();

    A->number_of_active_cells_per_thread[thread]--;
    CALBufferElement* next = element->next;
    CALBufferElement* previous = element->previous;

    if(next == NULL && previous == NULL)
    {
        A->_heads[thread] = NULL;
        A->_tails[thread] = NULL;
        return;
    }

    if(previous != NULL)
        previous->next = next;
    else
    {
        next->previous = NULL;
        A->_heads[thread] = next;
    }
    if(next != NULL)
        next->previous = previous;
    else
    {
        previous->next = NULL;
        A->_tails[thread] = previous;
    }
    element->next = NULL;
    element->previous = NULL;
}


void calApplyElementaryProcessActiveCellsCLL(struct CALActiveCellsCLL* A, CALLocalProcess elementary_process)
{
    CALBufferElement* current = NULL;
    CALBufferElement* next = NULL;
    struct CALModel* calModel = A->inherited_pointer->calModel;
    int i;
    int num_of_dims = calModel->numberOfCoordinates;
#pragma omp parallel for private(i, current, next) firstprivate(A, calModel, num_of_dims)
    for(i = 0; i < A->number_of_threads; i++ )
    {
        current = A->_heads[i];

        while( current != NULL )
        {
            next = calGetNextBufferElement(A, current);
            elementary_process(calModel, current->cell, num_of_dims);
            current = next;
        }
    }
}

void calCopyBufferActiveCellsCLL_b(CALbyte* M_src, CALbyte* M_dest,  struct CALActiveCellsCLL* A)
{
    int index;
    int c;
    int numb_of_dims = A->inherited_pointer->calModel->numberOfCoordinates;
    int* coords_dims = A->inherited_pointer->calModel->coordinatesDimensions;
#pragma omp parallel for private(c) firstprivate(A, coords_dims, numb_of_dims, M_src, M_dest)
    for(index = 0; index < A->number_of_threads; index++)
    {
        CALBufferElement* current;
        current = A->_heads[index];
        while( current != NULL )
        {
            c = getLinearIndex(current->cell, coords_dims, numb_of_dims);
            if (M_dest[c] != M_src[c])
                M_dest[c] = M_src[c];
            current = calGetNextBufferElement(A, current);
        }
    }
}
void calCopyBufferActiveCellsCLL_i(CALint* M_src, CALint* M_dest,  struct CALActiveCellsCLL* A)
{
    int index;
    int c;
    int numb_of_dims = A->inherited_pointer->calModel->numberOfCoordinates;
    int* coords_dims = A->inherited_pointer->calModel->coordinatesDimensions;
#pragma omp parallel for private(c) firstprivate(A, coords_dims, numb_of_dims, M_src, M_dest)
    for(index = 0; index < A->number_of_threads; index++)
    {
        CALBufferElement* current;
        current = A->_heads[index];
        while( current != NULL )
        {
            c = getLinearIndex(current->cell, coords_dims, numb_of_dims);
            if (M_dest[c] != M_src[c])
                M_dest[c] = M_src[c];
            current = calGetNextBufferElement(A, current);
        }
    }
}
void calCopyBufferActiveCellsCLL_r(CALreal* M_src, CALreal* M_dest,  struct CALActiveCellsCLL* A)
{
    int index;
    int c;
    int numb_of_dims = A->inherited_pointer->calModel->numberOfCoordinates;
    int* coords_dims = A->inherited_pointer->calModel->coordinatesDimensions;
#pragma omp parallel for private(c) firstprivate(A, coords_dims, numb_of_dims, M_src, M_dest)
    for(index = 0; index < A->number_of_threads; index++)
    {
        CALBufferElement* current;
        current = A->_heads[index];
        while( current != NULL )
        {
            c = getLinearIndex(current->cell, coords_dims, numb_of_dims);
            if (M_dest[c] != M_src[c])
                M_dest[c] = M_src[c];
            current = calGetNextBufferElement(A, current);
        }
    }
}


void calSetActiveCellsCLLBuffer_b(CALbyte* M, CALbyte value, struct CALActiveCellsCLL* A)
{
    int n;
    int numb_of_dims = A->inherited_pointer->calModel->numberOfCoordinates;
    int* coords_dims = A->inherited_pointer->calModel->coordinatesDimensions;

#pragma omp parallel for firstprivate(A, value, numb_of_dims, coords_dims)
    for(n = 0; n < A->number_of_threads; n++)
    {
        CALBufferElement* current = A->_heads[n];
        while(current != NULL)
        {
            M[getLinearIndex(current->cell, coords_dims, numb_of_dims)] = value;
            current = current->next;
        }
    }
}
void calSetActiveCellsCLLBuffer_i(CALint* M, CALint value, struct CALActiveCellsCLL* A)
{
    int n;
    int numb_of_dims = A->inherited_pointer->calModel->numberOfCoordinates;
    int* coords_dims = A->inherited_pointer->calModel->coordinatesDimensions;

#pragma omp parallel for firstprivate(A, value, numb_of_dims, coords_dims)
    for(n = 0; n < A->number_of_threads; n++)
    {
        CALBufferElement* current = A->_heads[n];
        while(current != NULL)
        {
            M[getLinearIndex(current->cell, coords_dims, numb_of_dims)] = value;
            current = current->next;
        }
    }
}
void calSetActiveCellsCLLBuffer_r(CALreal* M, CALreal value, struct CALActiveCellsCLL* A)
{
    int n;
    int numb_of_dims = A->inherited_pointer->calModel->numberOfCoordinates;
    int* coords_dims = A->inherited_pointer->calModel->coordinatesDimensions;

#pragma omp parallel for firstprivate(A, value, numb_of_dims, coords_dims)
    for(n = 0; n < A->number_of_threads; n++)
    {
        CALBufferElement* current = A->_heads[n];
        while(current != NULL)
        {
            M[getLinearIndex(current->cell, coords_dims, numb_of_dims)] = value;
            current = current->next;
        }
    }
}

void calFreeContiguousLinkedList(struct CALActiveCellsCLL* A)
{
    free(A->inherited_pointer);
    free(A->number_of_active_cells_per_thread);
    free(A->_heads);
    free(A->_tails);
}

