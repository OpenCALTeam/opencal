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

#ifndef calCommon_h
#define calCommon_h

#ifdef _MSC_VER
#define DllExport   __declspec( dllexport )
#else
#define DllExport
#endif


#define CAL_FALSE 0		//!< Boolean alias for false
#define CAL_TRUE  1		//!< Boolean alias for true


typedef char CALbyte;	//!< Redefinition of the type char.
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
    CAL_OPT_ACTIVE_CELLS_NAIVE,		//!< Enumerator used for specifying the naive implementation of the active cells optimization.
    CAL_OPT_ACTIVE_CELLS		//!< Enumerator used for specifying the optimal implementation of the active cells optimization (based on the Cntiguous Linked List structure).
};


/*! \brief Macro recomputing the out of bound neighbourhood indexes in case of toroidal cellular space.
*/
#define calGetToroidalX(index, size) (   (index)<0?((size)+(index)):( (index)>((size)-1)?((index)-(size)):(index) )   )


/*! \brief 2D cell's coordinates structure.

    Structure that defines the cell's coordinates for 2D
    cellular automata.
    Here, the first coordinate, i, represents the cell's row coordinate;
    the second coordinate, j, represents the cell's column coordinate.
*/
struct CALCell2D {
    int i;		//!< Cell row coordinate.
    int j;		//!< Cell column coordinate.
};


/*! \brief 3D cell's coordinates structure.

    Structure that defines the cell's coordinates for 2D
    cellular automata.
    Here, the first coordinate, i, represents the cell's row coordinate;
    the second coordinate, j, represents the cell's column coordinate.
*/
struct CALCell3D {
    int i;		//!< Cell row coordinate.
    int j;		//!< Cell column coordinate.
    int k;		//!< Cell slice coordinate.
};



/*! \brief 8 bit (256 values) 2D integer substate; it can also be used for 1 bit boolean substates.

    Structure that defines the abstraction of 2D cellular automaton
    8 bit (256 values) integer substates. It can be also used for
    1 bit (0, 1 or false, true) boolean substates.
    It consists of two linearised matrices: the first, current, represents
    the (linearised) matrix used for reading the substates values;
    the last, next, is used to write the new computed values.
    In this way, implicit parallelism is obtained, since the changes
    to the values of the substates do not affect the current values
    inside the cells.
*/
struct CALSubstate2Db {
    CALbyte* current;	//!< Current linearised matrix of the substate, used for reading purposes.
    CALbyte* next;		//!< Next linearised matrix of the substate, used for writing purposes.
    };

/*! \brief 2D integer substate.

    Structure that defines the abstraction of 2D cellular automaton
    integer substates.
    It consists of two linearised matrices: the first, current, represents
    the (linearised) matrix used for reading the substates values;
    the last, next, is used to write the new computed values.
    In this way, implicit parallelism is obtained, since the changes
    to the values of the substates do not affect the current values
    inside the cells.
*/
struct CALSubstate2Di {
    CALint* current;	//!< Current linearised matrix of the substate, used for reading purposes.
    CALint* next;		//!< Next linearised matrix of the substate, used for writing purposes.
};

/*! \brief 2D real (floating point) substate.

    Structure that defines the abstraction of 2D cellular automaton
    floating point substates.
    It consists of two linearised matrices: the first, current, represents
    the (linearised) matrix used for reading the substates values;
    the last, next, is used to write the new computed values.
    In this way, implicit parallelism is obtained, since the changes
    to the values of the substates do not affect the current values
    inside the cells.
*/
struct CALSubstate2Dr {
    CALreal* current;	//!< Current linearised matrix of the substate, used for reading purposes.
    CALreal* next;		//!< Next linearised matrix of the substate, used for writing purposes.
};



/*! \brief 8 bit (256 values) 3D integer substate; it can also be used for 1 bit boolean substates.

    Structure that defines the abstraction of 3D cellular automaton
    8 bit (256 values) integer substates. It can be also used for
    1 bit (0, 1 or false, true) boolean substates.
    It consists of two linearised 3D buffers: the first, current, represents
    the (linearised) 3D buffer used for reading the substate's values;
    the last, next, is used to write the new computed values.
    In this way, implicit parallelism is obtained, since the changes
    to the values of the substates do not affect the current values
    inside the cells.
*/
struct CALSubstate3Db {
    CALbyte* current;	//!< Current linearised 3D buffer of the substate, used for reading purposes.
    CALbyte* next;		//!< Next linearised 3D buffer of the substate, used for writing purposes.
    };

/*! \brief 3D integer substate.

    Structure that defines the abstraction of 3D cellular automaton
    integer substates.
    It consists of two linearised 3D buffers: the first, current, represents
    the (linearised) buffer used for reading the substates values;
    the last, next, is used to write the new computed values.
    In this way, implicit parallelism is obtained, since the changes
    to the values of the substates do not affect the current values
    inside the cells.
*/
struct CALSubstate3Di {
    CALint* current;	//!< Current linearised 3D buffer of the substate, used for reading purposes.
    CALint* next;		//!< Next linearised 3D buffer of the substate, used for writing purposes.
};

/*! \brief 3D real (floating point) substate.

    Structure that defines the abstraction of 3D cellular automaton
    floating point substates.
    It consists of two linearised 3D buffers: the first, current, represents
    the (linearised) 3D buffer used for reading the substates values;
    the last, next, is used to write the new computed values.
    In this way, implicit parallelism is obtained, since the changes
    to the values of the substates do not affect the current values
    inside the cells.
*/
struct CALSubstate3Dr {
    CALreal* current;	//!< Current linearised 3D buffer of the substate, used for reading purposes.
    CALreal* next;		//!< Next linearised 3D buffer of the substate, used for writing purposes.
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
