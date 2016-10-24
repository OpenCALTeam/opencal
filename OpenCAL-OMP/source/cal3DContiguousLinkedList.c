#include<OpenCAL-OMP/cal3DContiguousLinkedList.h>

static int getLinearIndex3D( int columns, int rows, int i, int j, int k)
{
    return k * columns * rows + i * columns + j;
}

static int calGetBuffer3D(CALContiguousLinkedList3D* buffer)
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

static void calPutElement3D(CALContiguousLinkedList3D* buffer, int i, int j, int k )
{
    int index = getLinearIndex3D( buffer->columns, buffer->rows, i, j, k );
    omp_set_lock( buffer->buffer[index].lock );
    if( buffer->buffer[index].isActive )
    {
        omp_unset_lock( buffer->buffer[index].lock );
        return;
    }
    int thread = calGetBuffer3D( buffer );
    buffer->buffer[index].isActive = true;

    omp_set_lock(&buffer->numberOfActiveCellsPerThreadLock[thread]);
    buffer->numberOfActiveCellsPerThread[thread]++;
    omp_unset_lock(&buffer->numberOfActiveCellsPerThreadLock[thread]);

    CALQueue3D* queue = &buffer->queuesOfElementsToAdd[thread];
    CALQueueElement3D* newElement = ( CALQueueElement3D* )malloc( sizeof( CALQueueElement3D ) );
    newElement->cell.i = i;
    newElement->cell.j = j;
    newElement->cell.k = k;


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

static void pushBack( CALContiguousLinkedList3D* buffer, int thread, CALQueueElement3D queueElement )
{
    CALBufferElement3D* tail = buffer->_tails[thread];
    CALBufferElement3D* element = &buffer->buffer[getLinearIndex3D( buffer->columns, buffer->rows, queueElement.cell.i, queueElement.cell.j, queueElement.cell.k )];

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


CALContiguousLinkedList3D* calMakeContiguousLinkedList3D( CALModel3D* model )
{
    CALContiguousLinkedList3D* contiguousLinkedList = ( CALContiguousLinkedList3D* )malloc( sizeof( CALContiguousLinkedList3D ) );
    contiguousLinkedList->columns = model->columns;
    contiguousLinkedList->rows = model->rows;
    contiguousLinkedList->slices = model->slices;
    contiguousLinkedList->size = model->columns * model->rows * model->slices;

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

    contiguousLinkedList->queuesOfElementsToAdd = ( CALQueue3D* ) malloc( sizeof( CALQueue3D ) * contiguousLinkedList->numberOfThreads );

    contiguousLinkedList->_heads = ( CALBufferElement3D** ) malloc( sizeof( CALBufferElement3D* ) * contiguousLinkedList->numberOfThreads  );
    contiguousLinkedList->_tails = ( CALBufferElement3D** ) malloc( sizeof( CALBufferElement3D* ) * contiguousLinkedList->numberOfThreads  );

    contiguousLinkedList->buffer = ( CALBufferElement3D* ) malloc( sizeof(CALBufferElement3D) * contiguousLinkedList->size );

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
    int sliceIndex = 0;
    int i = 0;

    for( ; i < contiguousLinkedList->size; i++ )
    {
        CALBufferElement3D element;
        element.cell.i = rowIndex;
        element.cell.j = columnIndex;
        element.cell.k = sliceIndex++;

        if( sliceIndex == contiguousLinkedList->slices )
        {
            rowIndex++;
            columnIndex++;
            sliceIndex = 0;
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

CALQueueElement3D calTakeElement3D( CALQueue3D* queue )
{
    CALQueueElement3D* element;
    CALQueueElement3D cell;
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
        cell.cell.k = element->cell.k;
        queue->size--;
        free( element );
    }

    return cell;
}

void calAddActiveCellCLL3D( CALModel3D* model, int i, int j, int k )
{
    calPutElement3D( model->contiguousLinkedList, i, j, k );
}

void calRemoveActiveCellCLL3D( CALModel3D *model, int i, int j, int k )
{
    int linearAddress = getLinearIndex3D( model->columns, model->rows, i, j, k );
    CALContiguousLinkedList3D* buffer = model->contiguousLinkedList;
    CALBufferElement3D* element = &buffer->buffer[linearAddress];

    if( !element->isActive )
        return;

    element->isActive = false;
    int thread = CAL_GET_THREAD_NUM();

    buffer->numberOfActiveCellsPerThread[thread]--;
    CALBufferElement3D* next = element->next;
    CALBufferElement3D* previous = element->previous;

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

CALBufferElement3D*calGetNextBufferElement3D(CALContiguousLinkedList3D* model, CALBufferElement3D* current)
{
    return current->next;
}

CALBufferElement3D*calGetFirstBufferElement3D(CALContiguousLinkedList3D* buffer)
{
    return NULL;
}


void calUpdateContiguousLinkedList3D(CALContiguousLinkedList3D* buffer)
{
    int n = 0;
    CALQueueElement3D queueElement;

    int* size_per_t = (int*) malloc( sizeof(int) * buffer->numberOfThreads);

#pragma omp parallel for private( queueElement ) firstprivate(buffer)
    for( n = 0; n < buffer->numberOfThreads; n++ )
    {
        size_per_t[n] = 0;
        queueElement = calTakeElement3D( &buffer->queuesOfElementsToAdd[n] );
        while( queueElement.cell.i != -1 )
        {
            pushBack( buffer, n, queueElement);

            size_per_t[n]++;
            queueElement = calTakeElement3D( &buffer->queuesOfElementsToAdd[n] );
        }
    }

    for(n = 0; n < buffer->numberOfThreads; n++ )
        buffer->size_current += size_per_t[n];


    free(size_per_t);
}


void calCopyBufferActiveCellsCLL3Db(CALbyte* M_src, CALbyte* M_dest, CALModel3D* ca3D)
{
    int index;
    int c;
    CALBufferElement3D* current;
    CALContiguousLinkedList3D* contiguousLinkedList = ca3D->contiguousLinkedList;
#pragma omp parallel for private( current, c, ca3D ) firstprivate(contiguousLinkedList)
    for( index = 0; index < ca3D->contiguousLinkedList->numberOfThreads; index++ )
    {
        current = contiguousLinkedList->_heads[index];
        while( current != NULL )
        {
            c = getLinearIndex3D(ca3D->columns, ca3D->rows, current->cell.i, current->cell.j, current->cell.k);
            if (M_dest[c] != M_src[c])
                M_dest[c] = M_src[c];
            current = calGetNextBufferElement3D(ca3D->contiguousLinkedList, current);
        }
    }

}

void calCopyBufferActiveCellsCLL3Di(CALint* M_src, CALint* M_dest, CALModel3D* ca3D)
{
    int index;
    int c;
    CALBufferElement3D* current;
    CALContiguousLinkedList3D* contiguousLinkedList = ca3D->contiguousLinkedList;
#pragma omp parallel for private( current, c, ca3D ) firstprivate(contiguousLinkedList)
    for( index = 0; index < ca3D->contiguousLinkedList->numberOfThreads; index++ )
    {
        current = contiguousLinkedList->_heads[index];
        while( current != NULL )
        {
            c = getLinearIndex3D(ca3D->columns, ca3D->rows, current->cell.i, current->cell.j, current->cell.k);
            if (M_dest[c] != M_src[c])
                M_dest[c] = M_src[c];
            current = calGetNextBufferElement3D(ca3D->contiguousLinkedList, current);
        }
    }

}

void calCopyBufferActiveCellsCLL3Dr(CALreal* M_src, CALreal* M_dest, CALModel3D* ca3D)
{
    int index;
    int c;
    CALBufferElement3D* current;
    CALContiguousLinkedList3D* contiguousLinkedList = ca3D->contiguousLinkedList;
#pragma omp parallel for private( current, c, ca3D ) firstprivate(contiguousLinkedList)
    for( index = 0; index < ca3D->contiguousLinkedList->numberOfThreads; index++ )
    {
        current = contiguousLinkedList->_heads[index];
        while( current != NULL )
        {
            c = getLinearIndex3D(ca3D->columns, ca3D->rows, current->cell.i, current->cell.j, current->cell.k);
            if (M_dest[c] != M_src[c])
                M_dest[c] = M_src[c];
            current = calGetNextBufferElement3D(ca3D->contiguousLinkedList, current);
        }
    }

}

void calSetActiveCellsCLLBuffer3Db(CALbyte* M, CALbyte value, CALModel3D* ca3D)
{
    CALContiguousLinkedList3D* contiguousLinkedList = ca3D->contiguousLinkedList;
    int n;
#pragma omp parallel for firstprivate(contiguousLinkedList, ca3D, value) private(n)
    for( n = 0; n < contiguousLinkedList->numberOfThreads; n++ )
    {
        CALBufferElement3D* current = contiguousLinkedList->_heads[n];
        while( current != NULL )
        {
            M[getLinearIndex3D(ca3D->columns, ca3D->rows, current->cell.i, current->cell.j, current->cell.k)] = value;
            current = calGetNextBufferElement3D(ca3D->contiguousLinkedList, current);
        }
    }
}

void calSetActiveCellsCLLBuffer3Di(CALint* M, CALint value, CALModel3D* ca3D)
{
    CALContiguousLinkedList3D* contiguousLinkedList = ca3D->contiguousLinkedList;
    int n;
#pragma omp parallel for firstprivate(contiguousLinkedList, ca3D, value) private(n)
    for( n = 0; n < contiguousLinkedList->numberOfThreads; n++ )
    {
        CALBufferElement3D* current = contiguousLinkedList->_heads[n];
        while( current != NULL )
        {
            M[getLinearIndex3D(ca3D->columns, ca3D->rows, current->cell.i, current->cell.j, current->cell.k)] = value;
            current = calGetNextBufferElement3D(ca3D->contiguousLinkedList, current);
        }
    }

}

void calSetActiveCellsCLLBuffer3Dr(CALreal* M, CALreal value, CALModel3D* ca3D)
{
    CALContiguousLinkedList3D* contiguousLinkedList = ca3D->contiguousLinkedList;
    int n;
#pragma omp parallel for firstprivate(contiguousLinkedList, ca3D, value) private(n)
    for( n = 0; n < contiguousLinkedList->numberOfThreads; n++ )
    {
        CALBufferElement3D* current = contiguousLinkedList->_heads[n];
        while( current != NULL )
        {
            M[getLinearIndex3D(ca3D->columns, ca3D->rows, current->cell.i, current->cell.j, current->cell.k)] = value;
            current = calGetNextBufferElement3D(ca3D->contiguousLinkedList, current);
        }
    }
}

void calApplyElementaryProcessActiveCellsCLL3D(CALModel3D* ca3D, CALCallbackFunc3D elementary_process)
{
    CALContiguousLinkedList3D* buffer = ca3D->contiguousLinkedList;
    CALBufferElement3D* current = NULL;
    CALBufferElement3D* next = NULL;
    int i = 0;
#pragma omp parallel for private(i, current, next) firstprivate(buffer)
    for( i = 0; i < ca3D->contiguousLinkedList->numberOfThreads; i++ )
    {
        current = buffer->_heads[i];

        while( current != NULL )
        {
            next =  calGetNextBufferElement3D( buffer, current );
            elementary_process(ca3D, current->cell.i, current->cell.j, current->cell.k);
            current = next;
        }
    }
}

void calFreeContiguousLinkedList3D(CALContiguousLinkedList3D* cll)
{
    free( cll->buffer );
    free( cll->numberOfActiveCellsPerThread );
    free( cll->_heads );
    free( cll->_tails );
    free( cll->queuesOfElementsToAdd );
    free( cll->numberOfActiveCellsPerThreadLock);
}
