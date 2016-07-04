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

#ifndef cal_bufferIO_h
#define cal_bufferIO_h

#include <OpenCAL-CPU/calCommon.h>
#include <stdio.h>



/*! \brief Loads a byte matrix from file.
*/
void calfLoadMatrix_b(CALbyte* M, int cellularSpaceDimension, FILE* f);

/*! \brief Loads an int matrix from file.
*/
void calfLoadMatrix_i(CALint* M, int cellularSpaceDimension, FILE* f);

/*! \brief Loads a real (floating point) matrix from file.
*/
void calfLoadMatrix_r(CALreal* M, int cellularSpaceDimension, FILE* f);



/*! \brief Loads a byte matrix from file.
*/
CALbyte calLoadMatrix_b(CALbyte* M, int cellularSpaceDimension, char* path);

/*! \brief Loads an int matrix from file.
*/
CALbyte calLoadMatrix_i(CALint* M, int cellularSpaceDimension, char* path);

/*! \brief Loads a real (floating point) matrix from file.
*/
CALbyte calLoadMatrix_r(CALreal* M, int cellularSpaceDimension, char* path);



/*! \brief Saves a byte matrix to file.
*/
void calfSaveMatrix_b(CALbyte* M, int cellularSpaceDimension, FILE* f);

/*! \brief Saves an int matrix to file.
*/
void calfSaveMatrix_i(CALint* M, int cellularSpaceDimension, FILE* f);

/*! \brief Saves a real (floating point) matrix to file.
*/
void calfSaveMatrix_r(CALreal* M, int cellularSpaceDimension, FILE* f);



/*! \brief Saves a byte matrix to file.
*/
CALbyte calSaveMatrix_b(CALbyte* M, int cellularSpaceDimension, char* path);

/*! \brief Saves a int matrix to file.
*/
CALbyte calSaveMatrix_i(CALint* M, int cellularSpaceDimension, char* path);

/*! \brief Saves a real (floating point) matrix to file.
*/
CALbyte calSaveMatrix_r(CALreal* M, int cellularSpaceDimension, char* path);



#endif

