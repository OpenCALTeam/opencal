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

#ifndef calActiveCells_h
#define calActiveCells_h

#include <OpenCAL++/calCommon.h>
#include <OpenCAL++/calBuffer.h>

/*! \brief Active cells class.
*/
class CALActiveCells
{
private:
    CALBuffer <bool>* flags;			//!< Buffer of flags having the substates' dimension: flag is true if the corresponding cell is active, false otherwise.
    int* cells;	                        //!< Array of computational active cells.
    int size_current;					//!< Number of active cells in the current step.
    int size_next;                      //!< Number of true flags.
public:
    /*! \brief CALActiveCells' constructor with no parameter.
    */
    CALActiveCells ();
    /*! \brief CALActiveCells' constructor.
    */
    CALActiveCells (CALBuffer <bool>* flags, int size_next);
    CALActiveCells (CALBuffer <bool>* flags, int size_next, int* cells, int size_current);

    /*! \brief CALActiveCells' destructor.
    */
    ~ CALActiveCells ();
    CALActiveCells (const CALActiveCells & obj);

    /*! \brief CALActiveCells' setter and getter methods.
    */
    CALBuffer <bool>* getFlags ();
    void setFlags (CALBuffer <bool>* flags);
    int getSizeNext ();
    void setSizeNext (int size_next);
    int* getCells ();
    void setCells (int* cells);
    int getSizeCurrent ();
    void setSizeCurrent (int size_current);

    bool getElementFlags (int *indexes, int *coordinates, int dimension);

    /*! \brief Sets the cell of coordinates indexes of the matrix flags to parameter value. It
     * increments the couter size_current if value is true, it decreases otherwise.
    */
    void setElementFlags (int *indexes, int *coordinates, int dimension, bool value);

    /*! \brief Puts the cells marked as actives in flags into the array of active cells
        cells and sets its dimension, to size_of_actives, i.e. the actual
        number of active cells.
    */
    void update ();
    bool getFlag (int linearIndex);

    /*! \brief Sets the cell [linearIndex] of the matrix flags to parameter value. It
     * increments the couter size_current if value is true, it decreases otherwise.
    */
    void setFlag (int linearIndex, bool value);



};


#endif
