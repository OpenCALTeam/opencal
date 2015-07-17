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

#ifndef cal2DBufferIO_h
#define cal2DBufferIO_h

#include "calCommon.cuh"
#include <stdio.h>

/*! \brief Loads a byte matrix from file. 
*/
void calfCudaLoadMatrix2Db(CALbyte* M, int rows, int columns, FILE* f, int i_substate);

/*! \brief Loads a byte matrix from file. 
*/
void calfCudaLoadMatrix2Di(CALint* M, int rows, int columns, FILE* f, int i_substate);

/*! \brief Loads a real (floating point) matrix from file. 
*/
void calfCudaLoadMatrix2Dr(CALreal* M, int rows, int columns, FILE* f, int i_substate);

/*! \brief Loads a real (floating point) matrix from file. 
*/
CALbyte calCudaLoadMatrix2Db(CALbyte* M, int rows, int columns, char* path, int i_substate);

/*! \brief Loads a real (floating point) matrix from file. 
*/
CALbyte calCudaLoadMatrix2Di(CALint* M, int rows, int columns, char* path, int i_substate);

/*! \brief Loads a real (floating point) matrix from file. 
*/
CALbyte calCudaLoadMatrix2Dr(CALreal* M, int rows, int columns, char* path, int i_substate);


/*! \brief Saves a byte matrix to file. 
*/
void calCudafSaveMatrix2Db(CALbyte* M, int rows, int columns, FILE* f, CALint index_substate);

/*! \brief Saves an int matrix to file. 
*/
void calCudafSaveMatrix2Di(CALint* M, int rows, int columns, FILE* f, CALint index_substate);

/*! \brief Saves a real (floating point) matrix to file. 
*/
void calCudafSaveMatrix2Dr(CALreal* M, int rows, int columns, FILE* f, CALint index_substate);


/*! \brief Saves a byte matrix to file. 
*/
CALbyte calCudaSaveMatrix2Db(CALbyte* M, int rows, int columns, char* path, CALint index_substate);

/*! \brief Saves a int matrix to file. 
*/
CALbyte calCudaSaveMatrix2Di(CALint* M, int rows, int columns, char* path, CALint index_substate);

/*! \brief Saves a real (floating point) matrix to file.
*/
CALbyte calCudaSaveMatrix2Dr(CALreal* M, int rows, int columns, char* path, CALint index_substate);


#endif
