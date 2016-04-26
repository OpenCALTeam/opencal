/*
 * Copyright (c) 2016 OpenCALTeam (https://github.com/OpenCALTeam),
 * Telesio Research Group,
 * Department of Mathematics and Computer Science,
 * University of Calabria, Italy.
 *
 * This file is part of OpenCAL (Open Computing Abstraction Layer).
 *
 * OpenCAL is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3 of
 * the License, or (at your option) any later version.
 *
 * OpenCAL is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with OpenCAL. If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef calSubstate_h
#define calSubstate_h

#include<OpenCAL++/calSubstateWrapper.h>
#include <OpenCAL++/calBuffer.h>
#include<OpenCAL++/calNeighborPool.h>

template <class T>
class CALSubstate : public CALSubstateWrapper
{
private:
    CALBuffer<T>* current;	//!< Current linearised matrix of the substate, used for reading purposes.
    CALBuffer<T>* next;		//!< Next linearised matrix of the substate, used for writing purposes.
public:
    CALSubstate (CALBuffer<T>* current, CALBuffer<T>* next);
    CALSubstate();
    virtual ~ CALSubstate ();
    CALSubstate (const CALSubstate & obj);
    CALBuffer<T>*& getCurrent ();
    CALBuffer<T>*& getNext ();
    void setCurrent (CALBuffer<T>* current);
    void setNext (CALBuffer<T>* next);
    void setActiveCellsBufferCurrent (CALActiveCells* activeCells, T value);
    void setActiveCellsBufferNext (CALActiveCells* activeCells, T value);

    /*! \brief Copies the next 3D buffer of a byte substate to the current one: current = next.
            If the active cells optimization is considered, it only updates the active cells.
        */
    virtual void update (CALActiveCells* activeCells);
    virtual void saveSubstate (int* coordinates, size_t dimension, CALConverterIO* calConverterInputOutput, char* path);
    virtual void loadSubstate (int* coordinates, size_t dimension, CALConverterIO* calConverterInputOutput, char* path);
    void  setElementCurrent (int* indexes, int* coordinates, int dimension, T value);
    void  setElement (int* indexes, int* coordinates, int dimension, T value);
    T  getElement (int* indexes, int* coordinates, int dimension);
    T  getElementNext (int* indexes, int* coordinates, int dimension);
    T getX (int linearIndex, int n);
    T getX (int* indexes, int* coordinates, int dimension, int n);

    T  getElement (int linearIndex);
    T  getElementNext (int linearIndex);
    void  setElementCurrent (int linearIndex, T value);
    void  setElement (int linearIndex, T value);
    void setCurrentBuffer (T value);
    void setNextBuffer (T value);
    CALSubstate<T>& operator=(const CALSubstate<T> & b);




};

//#include <OpenCAL++11/source/calSubstate.cpp>

template <class T>
CALSubstate <T> :: CALSubstate (CALBuffer<T>* current, CALBuffer<T>* next)
{
    this->current = current;
    this->next = next;

}


template <class T>
CALSubstate <T> :: CALSubstate ()
{
    this->current = NULL;
    this->next = NULL;

}
template <class T>
CALSubstate<T> :: ~ CALSubstate ()
{
    delete current;
    delete next;
}

template <class T>
CALSubstate<T> :: CALSubstate (const CALSubstate<T> & obj)
{
    this->current = obj.current;
    this->next = obj.next;
}

template <class T>
CALBuffer<T>*& CALSubstate<T> :: getCurrent()
{
    return current;
}

template <class T>
CALBuffer<T>*& CALSubstate<T> :: getNext()
{
    return next;
}

template <class T>
void CALSubstate<T> :: setCurrent (CALBuffer<T>* current)
{
    if (this->current)
        delete this->current;
    this->current = current;
    //    CALBuffer <T> * tmp = new CALBuffer <T> ();
    //    *tmp = *current;
    //    if (this->current)
    //        delete this->current;
    //    this->current = tmp;

}

template <class T>
void CALSubstate<T> :: setNext (CALBuffer<T>* next)
{
    if (this->next)
        delete this->next;
    this->next = next;
    //    CALBuffer <T> * tmp = new CALBuffer <T> ();
    //    *tmp = *next;
    //    if (this->next)
    //        delete this->next;
    //    this->next = tmp;

}

template <class T>
void CALSubstate<T> ::  setActiveCellsBufferCurrent (CALActiveCells* activeCells, T value)
{
    this->current->setActiveCellsBuffer(activeCells->getCells(), activeCells->getSizeCurrent(), value);
}

template <class T>
void CALSubstate<T> ::  setActiveCellsBufferNext (CALActiveCells* activeCells, T value)
{

    this->next->setActiveCellsBuffer(activeCells->getCells(), activeCells->getSizeCurrent(), value);

}

template <class T>
void CALSubstate<T> :: update (CALActiveCells* activeCells)
{
    if (activeCells)
        this->current->copyActiveCellsBuffer(next, activeCells->getCells(), activeCells->getSizeCurrent());
    else
    {
        *current = *next;
    }

}

template <class T>
void CALSubstate<T> :: saveSubstate (int* coordinates, size_t dimension, CALConverterIO* calConverterInputOutput, char* path)
{
    this->current->saveBuffer(coordinates,dimension,calConverterInputOutput, path);

}

template <class T>
void CALSubstate<T> :: loadSubstate (int* coordinates, size_t dimension, CALConverterIO* calConverterInputOutput, char* path)
{
    delete current;
    this->current = new CALBuffer <T> (coordinates, dimension, path, calConverterInputOutput);
    if (this->next)
        *next = *current;

}

template <class T>
void CALSubstate<T> :: setElementCurrent (int* indexes, int* coordinates, int dimension, T value)
{
    this->current->setElement(indexes, coordinates,dimension, value);
}

template <class T>
void CALSubstate<T> :: setElement (int* indexes, int* coordinates, int dimension, T value)
{
    this->next->setElement(indexes, coordinates, dimension, value);
}

template <class T>
T CALSubstate<T> :: getElement (int* indexes, int* coordinates, int dimension)
{
    return this->current->getElement(indexes, coordinates, dimension);
}

template <class T>
T CALSubstate<T> :: getElementNext (int* indexes, int* coordinates, int dimension)
{
    return this->next->getElement(indexes, coordinates, dimension);
}

template <class T>
T CALSubstate<T> :: getElement (int linearIndex)
{
    return (*this->current)[linearIndex];
}

template <class T>
T CALSubstate<T> :: getElementNext (int linearIndex)
{
    return (*this->next)[linearIndex];
}
template <class T>
T CALSubstate<T> :: getX(int linearIndex, int n)
{
    return (*this->getCurrent())[CALNeighborPool::getInstance()->getNeighborN(linearIndex,n)];
}

template <class T>
T CALSubstate<T> :: getX (int* indexes, int* coordinates, int dimension, int n)
{
    int linearIndex = calCommon::cellLinearIndex(indexes, coordinates,dimension);
    return (*this->getCurrent())[CALNeighborPool::getInstance()->getNeighborN(linearIndex,n)];

}

template <class T>
void CALSubstate<T> :: setElement (int linearIndex, T value)
{
    (*this->next)[linearIndex] = value;
}

template <class T>
void CALSubstate<T> :: setElementCurrent (int linearIndex, T value)
{
    (*this->current)[linearIndex] = value;
}


template<class T>
void CALSubstate <T> :: setCurrentBuffer (T value)
{
    this->current->setBuffer(value);
}

template<class T>
void CALSubstate <T> :: setNextBuffer (T value)
{
    this->next->setBuffer(value);
}

template<class T>
CALSubstate<T>& CALSubstate<T> :: operator=(const CALSubstate<T> & b)
{
    if (this != &b)
    {

        //TODO SISTEMARE
        CALBuffer<T> * currentTmp = new CALBuffer <T> ();
        CALBuffer<T> * nextTmp = new CALBuffer <T> ();

        *currentTmp = *b.current;
        *nextTmp = *b.next;

        //        delete current;
        //        delete next;

        this->current = currentTmp;
        this->next = nextTmp;
    }
    return *this;


}



#endif
