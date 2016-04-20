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

#ifndef calUnsafe_h
#define calUnsafe_h

#include <OpenCAL++/calModel.h>


class CALUnsafe
{
private:
    CALModel* calModel;
public:
    /*! \brief CALUnsafe's constructor
    */
    CALUnsafe (CALModel* calModel);


    /*! \brief Sets the n-th neighbor of the cell, represented by 'indexes' array, of the matrix flags to
        true.
    */
    void calAddActiveCellX(int* indexes,	//!< Coordinates of the central cell.
                           int n	//!< Index of the n-th neighbor to be added.
                             );

    /*! \brief Sets the n-th neighbor of a certain cell of the matrix flags to
        CAL_TRUE.
    */
    void calAddActiveCellX(int linearIndex,	//!< Linear index of the central cell.
                           int n	//!< Index of the n-th neighbor to be added.
                             );


    /*! \brief Inits the n-th neighbour of cell, represented by 'indexes' array, of a certain substate to value;
    it updates both the current and next matrix at the given position.
    This operation is unsafe since it writes value in a neighbor, both on the current and next matrix.
*/
    template <class T>
    void calInitX(CALSubstate<T>*& Q,	//!< Pointer to a substate.
                  int* indexes,				//!< Indexes of the central cell.
                  int n,						//!< Index of the n-th neighbor to be initialized.
                  T value				//!< initializing value.
                  );

    /*! \brief Inits n-th neighbour of a given cell of a substate to value;
    it updates both the current and next matrix at the position of coordinates.
    This operation is unsafe since it writes value in a neighbor, both on the current and next matrix.
*/
    template <class T>
    void calInitX(CALSubstate<T>*& Q,	//!< Pointer to a substate.
                  int linearIndex,				//!< Linear index of the central cell.
                  int n,						//!< Index of the n-th neighbor to be initialized.
                  T value				//!< initializing value.
                  );


    /*! \brief Returns a certain cell value of a substate from the next matrix.
    This operation is unsafe since it reads a value from the next matrix.
*/
    template <class T>
    T calGetNext(CALSubstate<T>*& Q,	//!< Pointer to a substate.
                 int* indexes				//!< Multidimensional coordinates of the cell.
                 );

    /*! \brief Returns a specific cell value of a substate from the next matrix.
    This operation is unsafe since it reads a value from the next matrix.
*/
    template <class T>
    T calGetNext(CALSubstate<T>*& Q,	//!< Pointer to a substate.
                 int linearIndex				//!< Linear index of the cell.
                 );


    /*! \brief Returns a certain cell n-th neighbor value of a substate from the next matrix.
    This operation is unsafe since it reads a value from the next matrix.
*/
    template <class T>
    T calGetNextX(CALSubstate<T>*& Q,	//!< Pointer to a substate.
                  int* indexes,				//!< Multidimensional coordinates of the cell.
                  int n						//!< Index of the n-th neighbor
                  );

    /*! \brief Returns a specific cell n-th neighbor value of a substate from the next matrix.
    This operation is unsafe since it reads a value from the next matrix.
*/
    template <class T>
    T calGetNextX(CALSubstate<T>*& Q,	//!< Pointer to a substate.
                  int linearIndex,				//!< Linear index of the cell.
                  int n						//!< Index of the n-th neighbor
                  );

    /*! \brief Sets the value of the n-th neighbor of a specific cell of a byte substate.
    This operation is unsafe since it writes a value in a neighbor of the next matrix.
*/
    template <class T>
    void calSetX(CALSubstate<T>*& Q,	//!< Pointer to a substate.
                 int* indexes,          //!< Coordinates of the cell
                 int n,						//!< Index of the n-th neighbor to be initialized.
                 T value                     //!< initializing value.
                 );
    /*! \brief Sets the value of the n-th neighbor of a certain cell of a substate.
    This operation is unsafe since it writes a value in a neighbor of the next matrix.
*/
    template <class T>
    void calSetX(CALSubstate<T>*& Q,	//!< Pointer to a substate.
                 int linearIndex,          //!< Linear index of the cell
                 int n,						//!< Index of the n-th neighbor to be initialized.
                 T value                     //!< initializing value.
                 );

    /*! \brief Sets the value of the n-th neighbor of a certain cell of a substate of the CURRENT matri.
    This operation is unsafe since it writes a value directly to the current matrix.
*/
    template <class T>
    void calSetCurrentX(CALSubstate<T>*& Q,	//!< Pointer to a substate.
                        int* indexes,				//!< Row coordinate of the central cell.
                        int n,					//!< Index of the n-th neighbor to be initialized.
                        T value                   //!< initializing value.
                        );

    /*! \brief Sets the value of the n-th neighbor of a certain cell of a substate of the CURRENT matrix.
    This operation is unsafe since it writes a value directly to the current matrix.
*/
    template <class T>
    void calSetCurrentX(CALSubstate<T>*& Q,	//!< Pointer to a substate.
                        int linearIndex,	//!< Coordinates of the central cell.
                        int n,					//!< Index of the n-th neighbor to be initialized.
                        T value                   //!< initializing value.
                        );
};



template <class T>
void CALUnsafe :: calInitX(CALSubstate<T>*& Q, int* indexes, int n, T value)
{
    int linearIndex = calCommon::cellLinearIndex(indexes, this->calModel->getCoordinates(),this->calModel->getDimension());
    int neighboorIndex =  CALNeighborPool::getInstance()->getNeighborN(linearIndex,n);

    (*Q->getCurrent())[neighboorIndex] = value;
    (*Q->getNext())[neighboorIndex] = value;
}



template <class T>
void CALUnsafe :: calInitX(CALSubstate<T>*& Q, int linearIndex, int n, T value)
{
    int neighboorIndex = CALNeighborPool::getInstance()->getNeighborN(linearIndex,n);

    (*Q->getCurrent())[neighboorIndex] = value;
    (*Q->getNext())[neighboorIndex] = value;
}

template <class T>
T CALUnsafe:: calGetNext(CALSubstate<T>*& Q, int* indexes) {
    return Q->getElementNext(indexes, calModel->getCoordinates(), calModel->getDimension());
}

template <class T>
T CALUnsafe:: calGetNext(CALSubstate<T>*& Q, int linearIndex) {
    eturn (*Q->getNext())[linearIndex];
}

template <class T>
T CALUnsafe::calGetNextX(CALSubstate<T>*& Q, int* indexes, int n)
{
    int linearIndex = calCommon::cellLinearIndex(indexes, this->calModel->getCoordinates(),this->calModel->getDimension());
    int neighboorIndex =  CALNeighborPool::getInstance()->getNeighborN(linearIndex,n);

    return (*Q->getNext())[neighboorIndex];
}

template <class T>
T CALUnsafe::calGetNextX(CALSubstate<T>*& Q, int linearIndex, int n)
{
    int neighboorIndex =  CALNeighborPool::getInstance()->getNeighborN(linearIndex,n);

    return (*Q->getNext())[neighboorIndex];
}


template <class T>
void CALUnsafe::calSetX(CALSubstate<T>*& Q, int* indexes, int n, T value)
{
    int linearIndex = calCommon::cellLinearIndex(indexes, this->calModel->getCoordinates(),this->calModel->getDimension());
    int neighboorIndex =  CALNeighborPool::getInstance()->getNeighborN(linearIndex,n);
    (*Q->getNext())[neighboorIndex] = value;
}

template <class T>
void CALUnsafe::calSetX(CALSubstate<T>*& Q, int linearIndex, int n, T value)
{
    int neighboorIndex = CALNeighborPool::getInstance()->getNeighborN(linearIndex,n);
    (*Q->getNext())[neighboorIndex] = value;
}

template <class T>
void CALUnsafe:: calSetCurrentX(CALSubstate<T>*& Q, int* indexes, int n, T value)
{

    int linearIndex = calCommon::cellLinearIndex(indexes, this->calModel->getCoordinates(),this->calModel->getDimension());
    int neighboorIndex = CALNeighborPool::getInstance()->getNeighborN(linearIndex,n);

    (*Q->getCurrent())[neighboorIndex] = value;
}

template <class T>
void CALUnsafe:: calSetCurrentX(CALSubstate<T>*& Q, int linearIndex, int n, T value)
{
    int neighboorIndex =  CALNeighborPool::getInstance()->getNeighborN(linearIndex,n);

    (*Q->getCurrent())[neighboorIndex] = value;
}


#endif
