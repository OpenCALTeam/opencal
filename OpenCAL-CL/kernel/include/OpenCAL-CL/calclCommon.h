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

#ifndef calclCommon_h
#define calclCommon_h

#ifndef __OPENCL_VERSION__
#define __kernel
#define __global
#define get_global_id (int)
#endif

//#define USEFLOAT

#define NULL ((void*) 0)

#ifndef USEFLOAT
#pragma OPENCL EXTENSION cl_khr_fp64: enable
#endif

#define CAL_FALSE 0		//!< Boolean alias for false
#define CAL_TRUE  1		//!< Boolean alias for true
typedef char CALbyte;	//!< Redefinition of the type char.
typedef int CALint;		//!< Redefinition of the type int.
#ifdef USEFLOAT
typedef float CALreal;	//!< Redefinition of the type double.
#else
typedef double CALreal;	//!< Redefinition of the type double.
#endif

typedef CALbyte CALParameterb;	//!< Redefinition of the type CALbyte. It is used for the automaton's parameters.
typedef CALint CALParameteri;	//!< Redefinition of the type CALint. It is used for automaton's parameters.
typedef CALreal CALParameterr;	//!< Redefinition of the type CALreal. It is used for automaton's parameters.

/*!	\brief Enumeration used for cellular space toroidality setting.
 */
enum CALSpaceBoundaryCondition {
	CAL_SPACE_FLAT = 0,			//!< Enumerator used for setting non-toroidal cellular space.
	CAL_SPACE_TOROIDAL			//!< Enumerator used for setting toroidal cellular space.
};

/*! \brief Macro recomputing the out of bound neighborhood indexes in case of toroidal cellular space.
 */
#define calclGetToroidalX(index, size) (   (index)<0?((size)+(index)):( (index)>((size)-1)?((index)-(size)):(index) )   )

/*! \brief Cell's coordinates structure.

 Structure that defines the cell's coordinates for 2D
 cellular automata.
 Here, the first coordinate, i, represents the cell's row coordinate;
 the second coordinate, j, represents the cell's column coordinate.
 */
struct CALCell2D {
	int i;		//!< Cell row coordinate.
	int j;		//!< Cell column coordinate.
};

/*! \brief Cell's coordinates structure.

 Structure that defines the cell's coordinates for 3D
 cellular automata.
 Here, the first coordinate, i, represents the cell's row coordinate;
 the second coordinate, j, represents the cell's column coordinate;
 the third coordinate, k, represents the cell's slice coordinate
 */
struct CALCell3D {
	int i;		//!< Cell row coordinate.
	int j;		//!< Cell column coordinate.
	int k;		//!< Cell slice coordinate.
};

#endif
