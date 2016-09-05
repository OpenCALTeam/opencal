#include<OpenCAL/cal2DContiguousLinkedList.h>

CALContiguousLinkedList2D* calMakeContiguousLinkedList2D( CALModel2D* model )
{
    CALContiguousLinkedList2D* contiguousLinkedList = ( CALContiguousLinkedList2D* )malloc( sizeof( CALContiguousLinkedList2D ) );
    contiguousLinkedList->columns = model->columns;
    contiguousLinkedList->rows = model->rows;
    contiguousLinkedList->size = model->columns * model->rows;

    if( contiguousLinkedList->size <= 0 )
    {
        free( contiguousLinkedList );
        return NULL;
    }

    contiguousLinkedList->buffer = ( CALBufferElement2D* ) malloc( sizeof(CALBufferElement2D) * contiguousLinkedList->size );


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

        contiguousLinkedList->buffer[i] = element;
    }

    contiguousLinkedList->head = NULL;
    contiguousLinkedList->tail = NULL;
    contiguousLinkedList->firstElementAddedAtCurrentIteration = NULL;
    contiguousLinkedList->size_current = 0;

    return contiguousLinkedList;
}


void calAddActiveCellCLL2D( CALModel2D* model, int i, int j )
{
    CALContiguousLinkedList2D* buffer = model->contiguousLinkedList;

    int linearAddress = getLinearIndex2D(  buffer->columns, i, j );
    CALBufferElement2D* element = &buffer->buffer[linearAddress];

    if( element->isActive )
    {
        return;
    }
    element->isActive = true;

    if( buffer->head == NULL && buffer->tail == NULL )
    {
        buffer->head = element;
        buffer->tail = element;
        element->next = NULL;
        element->previous = NULL;
    }
    else if( buffer->head == buffer->tail )
    {
        buffer->tail = element;
        element->previous = buffer->head;
        buffer->head->next = element;
        element->next = NULL;
    }
    else
    {
        buffer->tail->next = element;
        element->previous = buffer->tail;
        buffer->tail = element;
        element->next = NULL;
    }
    if( buffer->firstElementAddedAtCurrentIteration == NULL )
        buffer->firstElementAddedAtCurrentIteration = element;

    buffer->size_current++;
}

void calRemoveActiveCellCLL2D( CALModel2D *model, int i, int j )
{
    CALContiguousLinkedList2D* buffer = model->contiguousLinkedList;

    int linearAddress = getLinearIndex2D( buffer->columns, i, j );
    CALBufferElement2D* element = &buffer->buffer[linearAddress];

    if( !element->isActive )
        return;

    element->isActive = false;
    buffer->size_current--;

    CALBufferElement2D* previous = element->previous;
    CALBufferElement2D* next = element->next;

    if( previous == NULL && next == NULL )
    {
        model->contiguousLinkedList->head = NULL;
        model->contiguousLinkedList->tail = NULL;
        element->next = NULL;
        element->previous = NULL;
        return;
    }


    if( previous != NULL )
        previous->next = next;
    else
    {
        next->previous = NULL;
        model->contiguousLinkedList->head = next;
    }
    if( next != NULL )
        next->previous = previous;
    else
    {
        previous->next = NULL;
        model->contiguousLinkedList->tail = previous;
    }
    element->next = NULL;
    element->previous = NULL;
}


int getLinearIndex2D( int columns, int i, int j)
{
    return i * columns + j;
}

CALBufferElement2D*calGetNextBufferElement2D(CALContiguousLinkedList2D* model, CALBufferElement2D* current)
{
    if( current->next == model->firstElementAddedAtCurrentIteration )
        return NULL;
    return current->next;
}

CALBufferElement2D*calGetFirstBufferElement2D(CALContiguousLinkedList2D* buffer)
{
    return buffer->head;
}


void calUpdateContiguousLinkedList2D(CALContiguousLinkedList2D* buffer)
{
    buffer->firstElementAddedAtCurrentIteration = NULL;
}


void calCopyBufferActiveCellsCLL2Db(CALbyte* M_src, CALbyte* M_dest, CALModel2D* ca2D)
{
    CALBufferElement2D* current = ca2D->contiguousLinkedList->head;

    while( current != NULL )
    {
        int c = current->cell.i * ca2D->columns + current->cell.j;
        if (M_dest[c] != M_src[c])
            M_dest[c] = M_src[c];
        current = calGetNextBufferElement2D(ca2D->contiguousLinkedList, current);
    }

}

void calCopyBufferActiveCellsCLL2Di(CALint* M_src, CALint* M_dest, CALModel2D* ca2D)
{
    CALBufferElement2D* current = ca2D->contiguousLinkedList->head;

    while( current != NULL )
    {
        int c = current->cell.i * ca2D->columns + current->cell.j;
        if (M_dest[c] != M_src[c])
            M_dest[c] = M_src[c];
        current = calGetNextBufferElement2D(ca2D->contiguousLinkedList, current);
    }

}

void calCopyBufferActiveCellsCLL2Dr(CALreal* M_src, CALreal* M_dest, CALModel2D* ca2D)
{
    CALBufferElement2D* current = ca2D->contiguousLinkedList->head;

    while( current != NULL )
    {
        int c = current->cell.i * ca2D->columns + current->cell.j;
        if (M_dest[c] != M_src[c])
            M_dest[c] = M_src[c];
        current = calGetNextBufferElement2D(ca2D->contiguousLinkedList, current);
    }

}

void calSetActiveCellsCLLBuffer2Db(CALbyte* M, CALbyte value, CALModel2D* ca2D)
{
    CALBufferElement2D* current = ca2D->contiguousLinkedList->head;

    while( current != NULL )
    {
        M[current->cell.i * ca2D->columns + current->cell.j] = value;
        current = calGetNextBufferElement2D(ca2D->contiguousLinkedList, current);
    }
}

void calSetActiveCellsCLLBuffer2Di(CALint* M, CALint value, CALModel2D* ca2D)
{
    CALBufferElement2D* current = ca2D->contiguousLinkedList->head;

    while( current != NULL )
    {
        M[current->cell.i * ca2D->columns + current->cell.j] = value;
        current = calGetNextBufferElement2D(ca2D->contiguousLinkedList, current);
    }

}

void calSetActiveCellsCLLBuffer2Dr(CALreal* M, CALreal value, CALModel2D* ca2D)
{
    CALBufferElement2D* current = ca2D->contiguousLinkedList->head;

    while( current != NULL )
    {
        M[current->cell.i * ca2D->columns + current->cell.j] = value;
        current = calGetNextBufferElement2D(ca2D->contiguousLinkedList, current);
    }

}

void calApplyElementaryProcessActiveCellsCLL2D(CALModel2D* ca2D, CALCallbackFunc2D elementary_process)
{
    //printf("%s", "in teoria sto eseguendo \n");
    CALBufferElement2D* current = ca2D->contiguousLinkedList->head;

    while( current != NULL )
    {
        CALBufferElement2D* next = calGetNextBufferElement2D( ca2D->contiguousLinkedList, current );
        elementary_process( ca2D, current->cell.i, current->cell.j );
        current = next;
    }
}

void calFreeContiguousLinkedList2D(CALContiguousLinkedList2D* cll)
{
    free(cll->buffer);
}
