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

#ifndef cal3DBufferIO_h
#define cal3DBufferIO_h

#include <OpenCAL-OMP/calCommon.h>
#include <stdio.h>



/*! \brief Loads a byte 3D buffer from file.
*/
void calfLoadBuffer3Db(CALbyte* M, int rows, int columns, int slices, FILE* f);

/*! \brief Loads an int 3D buffer from file.
*/
void calfLoadBuffer3Di(CALint* M, int rows, int columns, int slices, FILE* f);

/*! \brief Loads a real (floating point) 3D buffer from file.
*/
void calfLoadBuffer3Dr(CALreal* M, int rows, int columns, int slices, FILE* f);



/*! \brief Loads a byte substate from file.
*/
CALbyte calLoadBuffer3Db(CALbyte* M, int rows, int columns, int slices, char* path);

/*! \brief Loads an int substate from file.
*/
CALbyte calLoadBuffer3Di(CALint* M, int rows, int columns, int slices, char* path);

/*! \brief Loads a real (floating point) substate from file.
*/
CALbyte calLoadBuffer3Dr(CALreal* M, int rows, int columns, int slices, char* path);



/*! \brief Saves a byte 3D buffer to file.
*/
void calfSaveBuffer3Db(CALbyte* M, int rows, int columns, int slices, FILE* f);

/*! \brief Saves an int 3D buffer to file.
*/
void calfSaveBuffer3Di(CALint* M, int rows, int columns, int slices, FILE* f);

/*! \brief Saves a real (floating point) 3D buffer to file.
*/
void calfSaveBuffer3Dr(CALreal* M, int rows, int columns, int slices, FILE* f);



/*! \brief Saves a byte 3D buffer to file.
*/
CALbyte calSaveBuffer3Db(CALbyte* M, int rows, int columns, int slices, char* path);

/*! \brief Saves a int 3D buffer to file.
*/
CALbyte calSaveBuffer3Di(CALint* M, int rows, int columns, int slices, char* path);

/*! \brief Saves a real (floating point) 3D buffer to file.
*/
CALbyte calSaveBuffer3Dr(CALreal* M, int rows, int columns, int slices, char* path);



#endif
