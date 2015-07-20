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
