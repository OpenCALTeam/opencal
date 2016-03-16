// (C) Copyright University of Calabria and others.
// All rights reserved. This program and the accompanying materials
// are made available under the terms of the GNU Lesser General Public License
// (LGPL) version 2.1 which accompanies this distribution, and is available at
// http://www.gnu.org/licenses/lgpl-2.1.html
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
// Lesser General Public License for more details.

#ifndef calCommon_h
#define calCommon_h
#include <stdlib.h>
#include <OpenCAL++/calIndexesPool.h>

namespace calCommon
{

#define CAL_FALSE false		//!< Boolean alias for false
#define CAL_TRUE  true		//!< Boolean alias for true


typedef bool CALbyte;	//!< Redefinition of the type char.
typedef int CALint;		//!< Redefinition of the type int.
typedef double CALreal;	//!< Redefinition of the type double.


typedef CALbyte CALParameterb;	//!< Redefinition of the type CALbyte. It is used for the automaton's parameters.
typedef CALint CALParameteri;	//!< Redefinition of the type CALint. It is used for automaton's parameters.
typedef CALreal CALParameterr;	//!< Redefinition of the type CALreal. It is used for automaton's parameters.


/*!	\brief Enumeration used for cellular space toroidality setting.
     */
enum CALSpaceBoundaryCondition{
    CAL_SPACE_FLAT = 0,			//!< Enumerator used for setting non-toroidal cellular space.
    CAL_SPACE_TOROIDAL			//!< Enumerator used for setting toroidal cellular space.
};


/*!	\brief Enumeration used for substate updating settings.
     */
enum CALUpdateMode{
    CAL_UPDATE_EXPLICIT = 0,	//!< Enumerator used for specifying that explicit calls to calUpdateSubstate2D* and calUpdate2D are needed.
    CAL_UPDATE_IMPLICIT			//!< Enumerator used for specifying that explicit calls to calUpdateSubstate2D* and calUpdate2D are NOT needed.
};


/*!	\brief Enumeration used for optimization strategies.
     */
enum CALOptimization{
    CAL_NO_OPT = 0,				//!< Enumerator used for specifying no optimizations.
    CAL_OPT_ACTIVE_CELLS		//!< Enumerator used for specifying the active cells optimization.
};


/*! \brief Macro recomputing the out of bound neighbourhood indexes in case of toroidal cellular space.
     */

#define calGetToroidalX(index, size) (   (index)<0?((size)+(index)):( (index)>((size)-1)?((index)-(size)):(index) )   )


/*! Constant used to set the run final step to 0, correspondig to a loop condition.
        In this case, a stop condition should be defined.
     */
#define CAL_RUN_LOOP 0


/*! \brief Enumeration defining global reduction operations.

    Enumeration defining global reduction operations inside the
    steering function.
     */
enum REDUCTION_OPERATION {
    REDUCTION_NONE = 0,
    REDUCTION_MAX,
    REDUCTION_MIN,
    REDUCTION_SUM,
    REDUCTION_PROD,
    REDUCTION_LOGICAL_AND,
    REDUCTION_BINARY_AND,
    REDUCTION_LOGICAL_OR,
    REDUCTION_BINARY_OR,
    REDUCTION_LOGICAL_XOR,
    REDUCTION_BINARY_XOR
};

/*! \brief Multiply cordinates array's element from startingIndex to dimension.
     */
inline unsigned int multiplier (int * coordinates, int startingIndex, size_t dimension) {
    int n;
    int m = 1;
    for (n=startingIndex; n<dimension; n++)
        m*= coordinates[n];
    return m;
}

/*! \brief Return the linearIndex of the cell with coordinates indexes.
     */
inline unsigned int cellLinearIndex (int* indexes, int* coordinates, size_t dimension) {
    int c = 0;
    int multiplier = 1;
    int n;
    for (int i = 0;i < dimension;i++)
    {
        if (i== 1)
            n=0;
        else if (i==0)
            n=1;
        else
            n=i;
        c += indexes[n] * multiplier;
        multiplier *= coordinates[n];
    }

    return c;
}

/*! \brief Return multidimensional indexes of a certain cell.
     */
inline int* cellMultidimensionalIndexes (int index)
{
    return CALIndexesPool:: getMultidimensionalIndexes(index);
}


/*! \brief Return multidimensional indexes of n^th neighbour of a certain cell.
     */
inline int* getNeighborN (int* indexes, int* neighbor,int* coordinates, size_t dimension, enum CALSpaceBoundaryCondition CAL_TOROIDALITY)
{
    int i;
    int* newIndexes = new int [dimension];
    if (CAL_TOROIDALITY == CAL_SPACE_FLAT)
        for (i = 0; i < dimension; ++i)
        {
            newIndexes[i] = indexes[i] + neighbor[i];
        }
    else
    {
        for (i=0; i< dimension; i++)
            newIndexes[i] = calGetToroidalX(indexes[i] + neighbor[i], coordinates[i]);
    }
    return newIndexes;
}


/*! \brief Return linear index of n^th neighbour of a certain cell.
     */
inline int getNeighborNLinear (int* indexes, int* neighbor,int* coordinates, size_t dimension, enum CALSpaceBoundaryCondition CAL_TOROIDALITY)
{
    int i;
    int c = 0;
    int t = multiplier (coordinates, 0, dimension);
    if (CAL_TOROIDALITY == CAL_SPACE_FLAT)
        for (i = 0; i < dimension; ++i)
        {
            t= t/coordinates[i];
            c+=(indexes[i] + neighbor[i])*t;
        }
    else
    {
        for (i=0; i< dimension; i++)
        {
            t= t/coordinates[i];
            c+=(calGetToroidalX(indexes[i] + neighbor[i], coordinates[i]))*t;

        }

    }
    return c;
}



}

#endif
