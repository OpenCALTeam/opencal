<<<<<<< HEAD
<<<<<<< HEAD
#include <OpenCAL++/calActiveCells.h>
=======
#include <OpenCAL++11/calActiveCells.h>
>>>>>>> e44630b317eeb506eac14bb3076f71487fe5ed2d
=======
#include <OpenCAL++11/calActiveCells.h>
>>>>>>> e44630b317eeb506eac14bb3076f71487fe5ed2d

CALActiveCells :: CALActiveCells ()
{
    size_current = 0;
    size_next = 0;
    cells = NULL;
    flags = NULL;


}

CALActiveCells :: CALActiveCells (CALBuffer<bool>* flags, int size_next)
{
    this->flags = flags;
    size_current = 0;
    this->size_next = size_next;
    cells = NULL;


}

CALActiveCells :: CALActiveCells (CALBuffer <bool>* flags, int size_next, int* cells, int size_current)
{

    this->flags = flags;
    this->size_next= size_next;
    this->cells = cells;
    this->size_current = size_current;
}


CALActiveCells :: ~ CALActiveCells ()
{
    delete [] cells;
    delete flags;
}

CALActiveCells :: CALActiveCells (const CALActiveCells & obj)
{
    this->flags = obj.flags;
    this->size_current = obj.size_current;
    cells = new int [obj.size_next];

    for (int i = 0; i < obj.size_next; ++i)
    {
        cells[i] = obj.cells[i];
    }
    this->size_next = obj.size_next;

}

CALBuffer <bool>* CALActiveCells :: getFlags()
{
    return flags;
}

void CALActiveCells :: setFlags (CALBuffer <bool>* flags)
{
    this-> flags = flags;
}

int CALActiveCells :: getSizeNext()
{
    return size_next;
}

void CALActiveCells :: setSizeNext(int size_next)
{
    this->size_next = size_next;
}

int* CALActiveCells :: getCells()
{
    return cells;
}

void CALActiveCells :: setCells(int* cells)
{
    this-> cells = cells;
}

int CALActiveCells :: getSizeCurrent()
{
    return size_current;
}

void CALActiveCells :: setSizeCurrent(int size_current)
{
    this->size_current = size_current;
}

bool CALActiveCells :: getElementFlags (int *indexes, int *coordinates, int dimension)
{
    return flags->getElement(indexes, coordinates,dimension);
}

void CALActiveCells :: setElementFlags (int *indexes, int *coordinates, int dimension, calCommon :: CALbyte value)
{
    if (value && !flags->getElement(indexes, coordinates,dimension))
    {
        flags->setElement(indexes, coordinates,dimension, value);
        size_next++;
    }
    else if (!value && flags->getElement(indexes, coordinates,dimension))
    {
        flags->setElement(indexes, coordinates,dimension, value);
        size_next--;
    }

}


void CALActiveCells :: update ()
{
    int i, n;
    if(this->cells)
    {
        delete [] this->cells;
    }

    this->size_current = this->size_next;
    if (size_current == 0)
        return;

    this->cells = new int [this->size_current];

    n = 0;
    int flagSize = flags->getSize();
    for(i = 0; i < flagSize; i++)
    {
        if ((*flags)[i])
        {
            cells[n] = i;
            n++;
        }
    }
}

bool CALActiveCells:: getFlag (int linearIndex)
{
    return (*flags)[linearIndex];
}

void CALActiveCells ::setFlag (int linearIndex, bool value)
{
    if (value && !(*flags)[linearIndex])
    {
        (*flags)[linearIndex] = value;
        size_next++;
    }
    else if (!value && (*flags)[linearIndex])
    {
        (*flags)[linearIndex] = value;
        size_next--;
    }

}
