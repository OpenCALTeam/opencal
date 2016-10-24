#include<OpenCAL-OMP/cal2DContiguousLinkedList.h>


// PRIVATE FUNCTIONS

static int getLinearIndex2D( int columns, int i, int j)
{
    return i * columns + j;
}

static int calGetBuffer2D(CALContiguousLinkedList2D* buffer, int i, int j)
{
    int min = 0;
    int index = 0;
    int ie = 0;
    for( ; index < buffer->numberOfThreads; index++ )
    {
        if( index  == 0 )
        {
            min = buffer->numberOfActiveCellsPerThread[index];
        }
        else if( min >  buffer->numberOfActiveCellsPerThread[index] )
        {
            min = buffer->numberOfActiveCellsPerThread[index];
            ie = index;
        }
    }
    return ie;
}

static void calPutElement2D(CALContiguousLinkedList2D* buffer, int i, int j )
{
    int index = getLinearIndex2D( buffer->columns, i, j );
    omp_set_lock( buffer->buffer[index].lock );
    if( buffer->buffer[index].isActive )
    {
        omp_unset_lock( buffer->buffer[index].lock );
        return;
    }
    int thread = calGetBuffer2D( buffer, i, j );
    buffer->buffer[index].isActive = true;

    omp_set_lock(&buffer->numberOfActiveCellsPerThreadLock[thread]);
    buffer->numberOfActiveCellsPerThread[thread]++;
    omp_unset_lock(&buffer->numberOfActiveCellsPerThreadLock[thread]);

    CALQueue2D* queue = &buffer->queuesOfElementsToAdd[thread];
    CALQueueElement2D* newElement = ( CALQueueElement2D* )malloc( sizeof( CALQueueElement2D ) );
    newElement->cell.i = i;
    newElement->cell.j = j;


    omp_set_lock( &queue->lock );

    newElement->next = NULL;

    if( queue->first == NULL )
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
    omp_unset_lock( &queue->lock );
    omp_unset_lock( buffer->buffer[index].lock );
}

static void pushBack( CALContiguousLinkedList2D* buffer, int thread, CALQueueElement2D queueElement )
{
    CALBufferElement2D* tail = buffer->_tails[thread];
    CALBufferElement2D* element = &buffer->buffer[getLinearIndex2D( buffer->columns, queueElement.cell.i, queueElement.cell.j )];

    if( buffer->_heads[thread] == NULL )
    {
        buffer->_heads[thread] = element;
        buffer->_tails[thread] = element;
        element->next = NULL;
        element->previous = NULL;
        return;
    }
    element->previous = tail;
    tail->next = element;
    buffer->_tails[thread] = element;
    element->next = NULL;

}


// PUBLIC FUNCTIONS


CALContiguousLinkedList2D* calMakeContiguousLinkedList2D( struct CALModel2D* model )
{
    CALContiguousLinkedList2D* contiguousLinkedList = ( CALContiguousLinkedList2D* )malloc( sizeof( CALContiguousLinkedList2D ) );
    contiguousLinkedList->columns = model->columns;
    contiguousLinkedList->rows = model->rows;
    contiguousLinkedList->size = model->columns * model->rows;

#pragma omp parallel
    {
#pragma omp single
        contiguousLinkedList->numberOfThreads = CAL_GET_NUM_THREADS();
    }


    if( contiguousLinkedList->size <= 0 )
    {
        free( contiguousLinkedList );
        return NULL;
    }
    contiguousLinkedList->queuesOfElementsToAdd = ( CALQueue2D* ) malloc( sizeof( CALQueue2D ) * contiguousLinkedList->numberOfThreads );

    contiguousLinkedList->_heads = ( CALBufferElement2D** ) malloc( sizeof( CALBufferElement2D* ) * contiguousLinkedList->numberOfThreads  );
    contiguousLinkedList->_tails = ( CALBufferElement2D** ) malloc( sizeof( CALBufferElement2D* ) * contiguousLinkedList->numberOfThreads  );

    contiguousLinkedList->buffer = ( CALBufferElement2D* ) malloc( sizeof(CALBufferElement2D) * contiguousLinkedList->size );
    contiguousLinkedList->numberOfActiveCellsPerThread = ( int * ) malloc( sizeof( int ) * contiguousLinkedList->numberOfThreads );
    contiguousLinkedList->numberOfActiveCellsPerThreadLock = (omp_lock_t*)malloc(sizeof(omp_lock_t)*contiguousLinkedList->numberOfThreads);

    int n;
    for( n = 0; n < contiguousLinkedList->numberOfThreads; n++ )
    {
        contiguousLinkedList->numberOfActiveCellsPerThread[n] = 0;
        contiguousLinkedList->queuesOfElementsToAdd[n].first = NULL;
        contiguousLinkedList->queuesOfElementsToAdd[n].last = NULL;
        contiguousLinkedList->queuesOfElementsToAdd[n].size =  0;
        contiguousLinkedList->_heads[n] = NULL;
        contiguousLinkedList->_tails[n] = NULL;
        omp_init_lock( &contiguousLinkedList->queuesOfElementsToAdd[n].lock);
        omp_init_lock( &contiguousLinkedList->numberOfActiveCellsPerThreadLock[n]);
    }


    int columnIndex = 0;
    int rowIndex = 0;
    int i = 0;

    for( ; i < contiguousLinkedList->size; i++ )
    {
        CALBufferElement2D element;
        element.cell.i = rowIndex;
        element.cell.j = columnIndex++;

        if( columnIndex == contiguousLinkedList->columns )
        {
            rowIndex++;
            columnIndex = 0;
        }

        element.isActive = false;
        element.next = NULL;
        element.previous = NULL;

        element.lock = ( omp_lock_t* ) malloc( sizeof( omp_lock_t ) );
        omp_init_lock( element.lock );

        contiguousLinkedList->buffer[i] = element;
    }

    contiguousLinkedList->size_current = 0;
    return contiguousLinkedList;
}

CALQueueElement2D calTakeElement2D( CALQueue2D* queue )
{
    CALQueueElement2D* element;
    CALQueueElement2D cell;
    cell.cell.i = -1;

    if( queue->size > 0 )
    {
        element = queue->first;
        if( element == queue->last )
        {
            queue->first = NULL;
            queue->last = NULL;
        }
        else
        {
            queue->first = element->next;
            element->next->previous = NULL;
        }

        cell.cell.i = element->cell.i;
        cell.cell.j = element->cell.j;
        queue->size--;
        free( element );
    }

    return cell;
}

void calAddActiveCellCLL2D( struct CALModel2D* model, int i, int j )
{
    calPutElement2D( model->contiguousLinkedList, i, j );
}

void calRemoveActiveCellCLL2D( struct CALModel2D *model, int i, int j )
{
    int linearAddress = getLinearIndex2D( model->columns, i, j );
    CALContiguousLinkedList2D* buffer = model->contiguousLinkedList;
    CALBufferElement2D* element = &buffer->buffer[linearAddress];

    if( !element->isActive )
        return;

    element->isActive = false;
    int thread = CAL_GET_THREAD_NUM();

    buffer->numberOfActiveCellsPerThread[thread]--;
    CALBufferElement2D* next = element->next;
    CALBufferElement2D* previous = element->previous;

    if( next == NULL && previous == NULL )
    {
        buffer->_heads[thread] = NULL;
        buffer->_tails[thread] = NULL;
        return;
    }


    if( previous != NULL )
        previous->next = next;
    else
    {
        next->previous = NULL;
        buffer->_heads[thread] = next;
    }
    if( next != NULL )
        next->previous = previous;
    else
    {
        previous->next = NULL;
        buffer->_tails[thread] = previous;
    }
    element->next = NULL;
    element->previous = NULL;
}


CALBufferElement2D*calGetNextBufferElement2D(CALContiguousLinkedList2D* model, CALBufferElement2D* current)
{
    return current->next;
}

CALBufferElement2D*calGetFirstBufferElement2D(CALContiguousLinkedList2D* buffer)
{
    return NULL;
}


void calUpdateContiguousLinkedList2D(CALContiguousLinkedList2D* buffer)
{
    int n = 0;
    CALQueueElement2D queueElement;

    int* size_per_t = (int*) malloc( sizeof(int) * buffer->numberOfThreads);

#pragma omp parallel for private( queueElement ) firstprivate(buffer)
    for( n = 0; n < buffer->numberOfThreads; n++ )
    {
        size_per_t[n] = 0;
        queueElement = calTakeElement2D( &buffer->queuesOfElementsToAdd[n] );
        while( queueElement.cell.i != -1 )
        {
            pushBack( buffer, n, queueElement);

            size_per_t[n]++;
            queueElement = calTakeElement2D( &buffer->queuesOfElementsToAdd[n] );
        }
    }

    buffer->size_current=0;
    for(n = 0; n < buffer->numberOfThreads; n++ )
        buffer->size_current += buffer->numberOfActiveCellsPerThread[n];


    free(size_per_t);
}


void calCopyBufferActiveCellsCLL2Db(CALbyte* M_src, CALbyte* M_dest, struct CALModel2D* ca2D)
{
    int index;ALQueueElement2D* first;
    CALQueueElement2D* last;

    omp_lock_t lock;

    int size;

    int c;
    CALBufferElement2D* current;
    CALContiguousLinkedList2D* contiguousLinkedList = ca2D->contiguousLinkedList;
#pragma omp parallel for private( current, c, ca2D ) firstprivate(contiguousLinkedList)
    for( index = 0; index < ca2D->contiguousLinkedList->numberOfThreads; index++ )
    {
        current = contiguousLinkedList->_heads[index];
        while( current != NULL )
        {
            c = ca2D->columns * ( current->cell.i ) + current->cell.j;
            if (M_dest[c] != M_src[c])
                M_dest[c] = M_src[c];
            current = calGetNextBufferElement2D( contiguousLinkedList, current );
        }
    }

}

void calCopyBufferActiveCellsCLL2Di(CALint* M_src, CALint* M_dest, struct CALModel2D* ca2D)
{
    int index;
    int c;
    CALBufferElement2D* current;
    CALContiguousLinkedList2D* contiguousLinkedList = ca2D->contiguousLinkedList;
#pragma omp parallel for private( current, c ) firstprivate(contiguousLinkedList)
    for( index = 0; index < ca2D->contiguousLinkedList->numberOfThreads; index++ )
    {
        current = contiguousLinkedList->_heads[index];
        while( current != NULL )
        {
            c = ca2D->columns * ( current->cell.i ) + current->cell.j;
            if (M_dest[c] != M_src[c])
                M_dest[c] = M_src[c];
            current = calGetNextBufferElement2D( contiguousLinkedList, current );
        }
    }


}

void calCopyBufferActiveCellsCLL2Dr(CALreal* M_src, CALreal* M_dest, struct CALModel2D* ca2D)
{
    int index;
    int c;
    CALBufferElement2D* current;
    CALContiguousLinkedList2D* contiguousLinkedList = ca2D->contiguousLinkedList;
#pragma omp parallel for private( current, c ) firstprivate(contiguousLinkedList)
    for( index = 0; index < ca2D->contiguousLinkedList->numberOfThreads; index++ )
    {
        current = contiguousLinkedList->_heads[index];
        while( current != NULL )
        {
            c = ca2D->columns * ( current->cell.i ) + current->cell.j;
            if (M_dest[c] != M_src[c])
                M_dest[c] = M_src[c];
            current = calGetNextBufferElement2D( contiguousLinkedList, current );
        }
    }

}

void calSetActiveCellsCLLBuffer2Db(CALbyte* M, CALbyte value, struct CALModel2D* ca2D)
{
    int n;

    CALContiguousLinkedList2D* contiguousLinkedList = ca2D->contiguousLinkedList;
#pragma omp parallel for firstprivate(contiguousLinkedList, ca2D, value)
    for( n = 0; n < contiguousLinkedList->numberOfThreads; n++ )
    {
        CALBufferElement2D* current = contiguousLinkedList->_heads[n];
        while( current != NULL )
        {
            M[current->cell.i * ca2D->columns + current->cell.j] = value;
            current = current->next;
        }
    }
}

void calSetActiveCellsCLLBuffer2Di(CALint* M, CALint value, struct CALModel2D* ca2D)
{
    int n;

    CALContiguousLinkedList2D* contiguousLinkedList = ca2D->contiguousLinkedList;
#pragma omp parallel for firstprivate(contiguousLinkedList, ca2D, value)
    for( n = 0; n < contiguousLinkedList->numberOfThreads; n++ )
    {
        CALBufferElement2D* current = contiguousLinkedList->_heads[n];
        while( current != NULL )
        {
            M[current->cell.i * ca2D->columns + current->cell.j] = value;
            current = current->next;
        }
    }
}

void calSetActiveCellsCLLBuffer2Dr(CALreal* M, CALreal value, struct CALModel2D* ca2D)
{
    int n;

    CALContiguousLinkedList2D* contiguousLinkedList = ca2D->contiguousLinkedList;
#pragma omp parallel for firstprivate(contiguousLinkedList, ca2D, value)
    for( n = 0; n < contiguousLinkedList->numberOfThreads; n++ )
    {
        CALBufferElement2D* current = contiguousLinkedList->_heads[n];
        while( current != NULL )
        {
            M[current->cell.i * ca2D->columns + current->cell.j] = value;
            current = current->next;
        }
    }

}

void calApplyElementaryProcessActiveCellsCLL2D(struct CALModel2D* ca2D, CALCallbackFunc2D elementary_process)
{
    CALContiguousLinkedList2D* buffer = ca2D->contiguousLinkedList;
    CALBufferElement2D* current = NULL;
    CALBufferElement2D* next = NULL;
    int i = 0;
#pragma omp parallel for private(i, current, next) firstprivate(buffer)
    for( i = 0; i < ca2D->contiguousLinkedList->numberOfThreads; i++ )
    {
         current = buffer->_heads[i];

        while( current != NULL )
        {
            next =  calGetNextBufferElement2D( buffer, current );
            elementary_process(ca2D, current->cell.i, current->cell.j);
            current = next;
        }
    }
}

void calFreeContiguousLinkedList2D(CALContiguousLinkedList2D* cll)
{
    free( cll->buffer );
    free( cll->numberOfActiveCellsPerThread );
    free( cll->_heads );
    free( cll->_tails );
    free( cll->queuesOfElementsToAdd );
    free( cll->numberOfActiveCellsPerThreadLock);
}
