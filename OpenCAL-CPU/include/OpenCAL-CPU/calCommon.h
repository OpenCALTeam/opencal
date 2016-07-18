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
#include <stdio.h>
#include <OpenCAL-CPU/calOmpDef.h>


#define CAL_FALSE 0		//!< Boolean alias for false
#define CAL_TRUE  1		//!< Boolean alias for true

#define CAL_PARALLEL 0

#define calDefineParallel() (CAL_PARALLEL=1)

struct CALDimensions
{
    int number_of_dimensions;
    int* coordinates_dimensions;
};

struct CALActiveCells {
        struct CALModel* calModel;
};

struct CALDimensions* calDefDimensions(int n, ... );

enum CALInitMethod { CAL_NO_INIT = 0, CAL_INIT_CURRENT, CAL_INIT_NEXT, CAL_INIT_BOTH };
enum CALExecutionType {SERIAL = 0, PARALLEL};

typedef int* CALIndices;

int getLinearIndex(CALIndices indices, CALIndices coordinates_dimensions, int number_of_dimensions );

typedef char CALbyte;	//!< Redefinition of the type char.
typedef int CALint;		//!< Redefinition of the type int.
typedef double CALreal;	//!< Redefinition of the type double.


typedef CALbyte CALParameterb;	//!< Redefinition of the type CALbyte. It is used for the automaton's parameters.
typedef CALint CALParameteri;	//!< Redefinition of the type CALint. It is used for automaton's parameters.
typedef CALreal CALParameterr;	//!< Redefinition of the type CALreal. It is used for automaton's parameters.


#include <OpenCAL-CPU/calBuffer.h>

struct CALIndexesPool {
        int cellular_space_dimension;
        CALIndices coordinates_dimensions;
        int number_of_dimensions;
        CALIndices* pool;
};

struct CALIndexesPool* calDefIndexesPool(CALIndices coordinates_dimensions, int number_of_dimensions);


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
    CAL_OPT_ACTIVE_CELLS_NAIVE,		//!< Enumerator used for specifying the naive implementation of the active cells optimization.
    CAL_OPT_ACTIVE_CELLS		//!< Enumerator used for specifying the optimal implementation of the active cells optimization (based on the Cntiguous Linked List structure).
};

/*! \brief Macro recomputing the out of bound neighbourhood indexes in case of toroidal cellular space.
*/
#define calGetToroidalX(index, size) (   (index)<0?((size)+(index)):( (index)>((size)-1)?((index)-(size)):(index) )   )



/*! \brief 8 bit (256 values) integer substate; it can also be used for 1 bit boolean substates.

    Structure that defines the abstraction of cellular automaton
    8 bit (256 values) integer substates. It can be also used for
    1 bit (0, 1 or false, true) boolean substates.
    It consists of two linearised matrices: the first, current, represents
    the (linearised) matrix used for reading the substates values;
    the last, next, is used to write the new computed values.
    In this way, implicit parallelism is obtained, since the changes
    to the values of the substates do not affect the current values
    inside the cells.
*/
struct CALSubstate_b {
        CALbyte* current;	//!< Current linearised matrix of the substate, used for reading purposes.
        CALbyte* next;		//!< Next linearised matrix of the substate, used for writing purposes.
};

/*! \brief integer substate.

    Structure that defines the abstraction of cellular automaton
    integer substates.
    It consists of two linearised matrices: the first, current, represents
    the (linearised) matrix used for reading the substates values;
    the last, next, is used to write the new computed values.
    In this way, implicit parallelism is obtained, since the changes
    to the values of the substates do not affect the current values
    inside the cells.
*/
struct CALSubstate_i {
        CALint* current;	//!< Current linearised matrix of the substate, used for reading purposes.
        CALint* next;		//!< Next linearised matrix of the substate, used for writing purposes.
};

/*! \brief real (floating point) substate.

    Structure that defines the abstraction of cellular automaton
    floating point substates.
    It consists of two linearised matrices: the first, current, represents
    the (linearised) matrix used for reading the substates values;
    the last, next, is used to write the new computed values.
    In this way, implicit parallelism is obtained, since the changes
    to the values of the substates do not affect the current values
    inside the cells.
*/
struct CALSubstate_r {
        CALreal* current;	//!< Current linearised matrix of the substate, used for reading purposes.
        CALreal* next;		//!< Next linearised matrix of the substate, used for writing purposes.
};


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


#endif
