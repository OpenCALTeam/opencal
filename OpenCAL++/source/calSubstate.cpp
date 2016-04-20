/*
 * Copyright (c) 2016 OpenCALTeam (https://github.com/OpenCALTeam),
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


//#ifdef calSubstate_h

//template <class T>
//CALSubstate <T> :: CALSubstate (CALBuffer<T>* current, CALBuffer<T>* next)
//{
//    this->current = current;
//    this->next = next;

//}


//template <class T>
//CALSubstate <T> :: CALSubstate ()
//{
//    this->current = NULL;
//    this->next = NULL;

//}
//template <class T>
//CALSubstate<T> :: ~ CALSubstate ()
//{
//    delete current;
//    delete next;
//}

//template <class T>
//CALSubstate<T> :: CALSubstate (const CALSubstate<T> & obj)
//{
//    this->current = obj.current;
//    this->next = obj.next;
//}

//template <class T>
//CALBuffer<T>*& CALSubstate<T> :: getCurrent()
//{
//    return current;
//}

//template <class T>
//CALBuffer<T>*& CALSubstate<T> :: getNext()
//{
//    return next;
//}

//template <class T>
//void CALSubstate<T> :: setCurrent (CALBuffer<T>* current)
//{
//    if (this->current)
//        delete this->current;
//    this->current = current;
////    CALBuffer <T> * tmp = new CALBuffer <T> ();
////    *tmp = *current;
////    if (this->current)
////        delete this->current;
////    this->current = tmp;

//}

//template <class T>
//void CALSubstate<T> :: setNext (CALBuffer<T>* next)
//{
//    if (this->next)
//        delete this->next;
//    this->next = next;
////    CALBuffer <T> * tmp = new CALBuffer <T> ();
////    *tmp = *next;
////    if (this->next)
////        delete this->next;
////    this->next = tmp;

//}

//template <class T>
//void CALSubstate<T> ::  setActiveCellsBuffer (enum CALSubstateBuffer calSubstateBuffer, CALActiveCells* activeCells, T value)
//{
//    if (calSubstateBuffer == CAL_SUBSTATE_CURRENT)
//    {
//        this->current->setActiveCellsBuffer(activeCells->getCells(), activeCells->getSizeCurrent(), value);

//    }
//    else if (calSubstateBuffer == CAL_SUBSTATE_NEXT)
//    {
//        this->next->setActiveCellsBuffer(activeCells->getCells(), activeCells->getSizeCurrent(), value);
//    }

//}

//template <class T>
//void CALSubstate<T> :: update (CALActiveCells* activeCells)
//{
//    if (activeCells)
//        this->current->copyActiveCellsBuffer(next, activeCells->getCells(), activeCells->getSizeCurrent());
//    else
//    {
////        std::cout<< "sono in update di "<< this <<std::endl;
////        stampa(coordinates,dimension);

//        *current = *next;

//    }

//}

//template <class T>
//void CALSubstate<T> :: saveSubstate (int* coordinates, size_t dimension, CALConverterInputOutput* calConverterInputOutput, char* path)
//{
//    this->current->saveBuffer(coordinates,dimension,calConverterInputOutput, path);

//}

//template <class T>
//void CALSubstate<T> :: loadSubstate (int* coordinates, size_t dimension, CALConverterInputOutput* calConverterInputOutput, char* path)
//{
//    delete current;
//    this->current = new CALBuffer <T> (coordinates, dimension, path, calConverterInputOutput);
//    if (this->next)
//        *next = *current;

//}

//template <class T>
//void CALSubstate<T> :: setElementCurrentBuffer (int* indexes, int* coordinates, int dimension, T value)
//{
//    this->current->setElement(indexes, coordinates,dimension, value);
//}

//template <class T>
//void CALSubstate<T> :: setElementNextBuffer (int* indexes, int* coordinates, int dimension, T value)
//{
//    this->next->setElement(indexes, coordinates, dimension, value);
//}

//template <class T>
//T CALSubstate<T> :: getElementCurrentBuffer (int* indexes, int* coordinates, int dimension)
//{
//    return this->current->getElement(indexes, coordinates, dimension);
//}

//template <class T>
//T CALSubstate<T> :: getElementNextBuffer (int* indexes, int* coordinates, int dimension)
//{
//    return this->next->getElement(indexes, coordinates, dimension);
//}

//template <class T>
//T CALSubstate<T> :: getElementCurrentBuffer (int linearIndex)
//{
//    return (*this->current)[linearIndex];
//}

//template <class T>
//T CALSubstate<T> :: getElementNextBuffer (int linearIndex)
//{
//    return (*this->next)[linearIndex];
//}

//template<class T>
//void CALSubstate <T> :: setCurrentBuffer (T value)
//{
//    this->current->setBuffer(value);
//}

//template<class T>
//void CALSubstate <T> :: setNextBuffer (T value)
//{
//    this->next->setBuffer(value);
//}

//template<class T>
//CALSubstate<T>& CALSubstate<T> :: operator=(const CALSubstate<T> & b)
//{
//    if (this != &b)
//    {

//        //TODO SISTEMARE
//        CALBuffer<T> * currentTmp = new CALBuffer <T> ();
//        CALBuffer<T> * nextTmp = new CALBuffer <T> ();

//        *currentTmp = *b.current;
//        *nextTmp = *b.next;

////        delete current;
////        delete next;

//        this->current = currentTmp;
//        this->next = nextTmp;
//    }
//    return *this;


//}


//template <class T>
//void  CALSubstate <T> :: stampa (int* coordinates, size_t dimension)
//{
//    std::cout<<"CURRENT "<<std::endl;
//    this->current->stampa(coordinates, dimension);
//    std::cout<<"NEXT "<<std::endl;
//    this->next->stampa(coordinates, dimension);
//}

//#endif
