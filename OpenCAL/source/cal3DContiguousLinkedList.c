#include<OpenCAL/cal3DContiguousLinkedList.h>

static int getLinearIndex3D( int columns, int rows, int i, int j, int k)
{
    return k * columns * rows + i * columns + j;
}


CALContiguousLinkedList3D* calMakeContiguousLinkedList3D( CALModel3D* model )
{
    CALContiguousLinkedList3D* contiguousLinkedList = ( CALContiguousLinkedList3D* )malloc( sizeof( CALContiguousLinkedList3D ) );
    contiguousLinkedList->columns = model->columns;
    contiguousLinkedList->rows = model->rows;
    contiguousLinkedList->slices = model->slices;
    contiguousLinkedList->size = model->columns * model->rows * model->slices;

    if( contiguousLinkedList->size <= 0 )
    {
        free( contiguousLinkedList );
        return NULL;
    }

    contiguousLinkedList->buffer = ( CALBufferElement3D* ) malloc( sizeof(CALBufferElement3D) * contiguousLinkedList->size );


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

            contiguousLinkedList->buffer[i] = element;
        }

        contiguousLinkedList->head = NULL;
        contiguousLinkedList->tail = NULL;
        contiguousLinkedList->firstElementAddedAtCurrentIteration = NULL;

        return contiguousLinkedList;
}


void calAddActiveCellCLL3D( CALModel3D* model, int i, int j, int k )
{
    CALContiguousLinkedList3D* buffer = model->contiguousLinkedList;

    int linearAddress = getLinearIndex3D(  buffer->columns, buffer->rows, i, j, k );
    CALBufferElement3D* element = &buffer->buffer[linearAddress];

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

void calRemoveActiveCellCLL3D( CALModel3D *model, int i, int j, int k )
{
    CALContiguousLinkedList3D* buffer = model->contiguousLinkedList;

    int linearAddress = getLinearIndex3D( buffer->columns, buffer->rows, i, j, k );
    CALBufferElement3D* element = &buffer->buffer[linearAddress];

    if( !element->isActive )
        return;

    element->isActive = false;
    buffer->size_current--;

    CALBufferElement3D* previous = element->previous;
    CALBufferElement3D* next = element->next;

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

CALBufferElement3D*calGetNextBufferElement3D(CALContiguousLinkedList3D* model, CALBufferElement3D* current)
{
    if( current->next == model->firstElementAddedAtCurrentIteration )
        return NULL;
    return current->next;
}

CALBufferElement3D*calGetFirstBufferElement3D(CALContiguousLinkedList3D* buffer)
{
    return buffer->head;
}


void calUpdateContiguousLinkedList3D(CALContiguousLinkedList3D* buffer)
{
    buffer->firstElementAddedAtCurrentIteration = NULL;
}


void calCopyBufferActiveCellsCLL3Db(CALbyte* M_src, CALbyte* M_dest, CALModel3D* ca3D)
{
    CALBufferElement3D* current = ca3D->contiguousLinkedList->head;

    while( current != NULL )
    {
        int c = getLinearIndex3D(ca3D->columns, ca3D->rows, current->cell.i, current->cell.j, current->cell.k);
        if (M_dest[c] != M_src[c])
            M_dest[c] = M_src[c];
        current = calGetNextBufferElement3D(ca3D->contiguousLinkedList, current);
    }

}

void calCopyBufferActiveCellsCLL3Di(CALint* M_src, CALint* M_dest, CALModel3D* ca3D)
{
    CALBufferElement3D* current = ca3D->contiguousLinkedList->head;

    while( current != NULL )
    {
        int c = getLinearIndex3D(ca3D->columns, ca3D->rows, current->cell.i, current->cell.j, current->cell.k);
        if (M_dest[c] != M_src[c])
            M_dest[c] = M_src[c];
        current = calGetNextBufferElement3D(ca3D->contiguousLinkedList, current);
    }

}

void calCopyBufferActiveCellsCLL3Dr(CALreal* M_src, CALreal* M_dest, CALModel3D* ca3D)
{
    CALBufferElement3D* current = ca3D->contiguousLinkedList->head;

    while( current != NULL )
    {
        int c = getLinearIndex3D(ca3D->columns, ca3D->rows, current->cell.i, current->cell.j, current->cell.k);
        if (M_dest[c] != M_src[c])
            M_dest[c] = M_src[c];
        current = calGetNextBufferElement3D(ca3D->contiguousLinkedList, current);
    }

}

void calSetActiveCellsCLLBuffer3Db(CALbyte* M, CALbyte value, CALModel3D* ca3D)
{
    CALBufferElement3D* current = ca3D->contiguousLinkedList->head;

    while( current != NULL )
    {
        M[getLinearIndex3D(ca3D->columns, ca3D->rows, current->cell.i, current->cell.j, current->cell.k)] = value;
        current = calGetNextBufferElement3D(ca3D->contiguousLinkedList, current);
    }
}

void calSetActiveCellsCLLBuffer3Di(CALint* M, CALint value, CALModel3D* ca3D)
{
    CALBufferElement3D* current = ca3D->contiguousLinkedList->head;

    while( current != NULL )
    {
        M[getLinearIndex3D(ca3D->columns, ca3D->rows, current->cell.i, current->cell.j, current->cell.k)] = value;
        current = calGetNextBufferElement3D(ca3D->contiguousLinkedList, current);
    }

}

void calSetActiveCellsCLLBuffer3Dr(CALreal* M, CALreal value, CALModel3D* ca3D)
{
    CALBufferElement3D* current = ca3D->contiguousLinkedList->head;

    while( current != NULL )
    {
        M[getLinearIndex3D(ca3D->columns, ca3D->rows, current->cell.i, current->cell.j, current->cell.k)] = value;
        current = calGetNextBufferElement3D(ca3D->contiguousLinkedList, current);
    }
}

void calApplyElementaryProcessActiveCellsCLL3D(CALModel3D* ca3D, CALCallbackFunc3D elementary_process)
{
    CALBufferElement3D* current = ca3D->contiguousLinkedList->head;

    while( current != NULL )
    {
        CALBufferElement3D* next = calGetNextBufferElement3D( ca3D->contiguousLinkedList, current );
        elementary_process( ca3D, current->cell.i, current->cell.j, current->cell.k );
        current = next;
    }
}

void calFreeContiguousLinkedList3D(CALContiguousLinkedList3D* cll)
{
    free(cll->buffer);
}
