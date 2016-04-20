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
