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

#ifndef cal3DBuffer_h
#define cal3DBuffer_h

#include <OpenCAL/calCommon.h>
#include <OpenCAL/cal3D.h>


/*! \brief Allocates a byte linearized 3D buffer.
*/
DllExport
CALbyte* calAllocBuffer3Db(int rows, int columns, int slices);

/*! \brief Allocates an int linearized 3D buffer.
*/
DllExport
CALint* calAllocBuffer3Di(int rows, int columns, int slices);

/*! \brief Allocates a real (floating point) linearized 3D buffer.
*/
DllExport
CALreal* calAllocBuffer3Dr(int rows, int columns, int slices);



/*! \brief Deletes the memory associated to a byte linearized 3D buffer.
*/
DllExport
void calDeleteBuffer3Db(CALbyte* M);

/*! \brief Deletes the memory associated to an int linearized 3D buffer.
*/
DllExport
void calDeleteBuffer3Di(CALint* M);

/*! \brief Deletes the memory associated to a real (floating point) linearized 3D buffer.
*/
DllExport
void calDeleteBuffer3Dr(CALreal* M);



/*! \brief Byte linearized 3D buffer copy function.
*/
DllExport
void calCopyBuffer3Db(CALbyte* M_src, CALbyte* M_dest, int rows, int columns, int slices);

/*! \brief Int linearized 3D buffer copy function.
*/
DllExport
void calCopyBuffer3Di(CALint* M_src, CALint* M_dest, int rows, int columns, int slices);

/*! \brief Real (floating point) linearized 3D buffer copy function.
*/
DllExport
void calCopyBuffer3Dr(CALreal* M_src, CALreal* M_dest, int rows, int columns, int slices);



/*! \brief Active cells byte linearized 3D buffer copy function.
*/
DllExport
void calCopyBufferActiveCells3Db(CALbyte* M_src, CALbyte* M_dest, struct CALModel3D* ca3D);

/*! \brief Active cells int linearized 3D buffer copy function.
*/
DllExport
void calCopyBufferActiveCells3Di(CALint* M_src, CALint* M_dest, struct CALModel3D* ca3D);

/*! \brief Active cells real (floating point) linearized 3D buffer copy function.
*/
DllExport
void calCopyBufferActiveCells3Dr(CALreal* M_src, CALreal* M_dest, struct CALModel3D* ca3D);


/*! \brief Byte linearized 3D buffer copy function.
*/
DllExport
void calAddBuffer3Db(CALbyte* M_op1, CALbyte* M_op2,  CALbyte* M_dest, int rows, int columns, int slices);

/*! \brief Int linearized 3D buffer copy function.
*/
DllExport
void calAddBuffer3Di(CALint* M_op1, CALint* M_op2,  CALint* M_dest, int rows, int columns, int slices);

/*! \brief Real (floating point) linearized 3D buffer copy function.
*/
DllExport
void calAddBuffer3Dr(CALreal* M_op1, CALreal* M_op2,  CALreal* M_dest, int rows, int columns, int slices);



/*! \brief Byte linearized 3D buffer subtract function.
*/
DllExport
void calSubtractBuffer3Db(CALbyte* M_op1, CALbyte* M_op2,  CALbyte* M_dest, int rows, int columns, int slices);

/*! \brief Int linearized 3D buffer subtract function.
*/
DllExport
void calSubtractBuffer3Di(CALint* M_op1, CALint* M_op2,  CALint* M_dest, int rows, int columns, int slices);

/*! \brief Real (floating point) linearized 3D buffer subtract function.
*/
DllExport
void calSubtractBuffer3Dr(CALreal* M_op1, CALreal* M_op2,  CALreal* M_dest, int rows, int columns, int slices);



/*! \brief Sets a byte 3D buffer to a constant value.
*/
DllExport
void calSetBuffer3Db(CALbyte* M, int rows, int columns, int slices, CALbyte value);

/*! \brief Sets an int 3D buffer to a constant value.
*/
DllExport
void calSetBuffer3Di(CALint* M, int rows, int columns, int slices, CALint value);

/*! \brief Sets a real (floating point) 3D buffer to a constant value.
*/
DllExport
void calSetBuffer3Dr(CALreal* M, int rows, int columns, int slices, CALreal value);



/*! \brief Sets active cells of a byte 3D buffer to a constant value.
*/
DllExport
void calSetActiveCellsBuffer3Db(CALbyte* M, CALbyte value, struct CALModel3D* ca3D);

/*! \brief Sets active cells of an int 3D buffer to a constant value.
*/
DllExport
void calSetActiveCellsBuffer3Di(CALint* M, CALint value, struct CALModel3D* ca3D);

/*! \brief Sets active cells of a real (floating point) 3D buffer to a constant value.
*/
DllExport
void calSetActiveCellsBuffer3Dr(CALreal* M, CALreal value, struct CALModel3D* ca3D);



/*! \brief Sets the value of the cell (i, j) of the matrix M.
*/
#define calSetBuffer3DElement(M, rows, columns, i, j, k, value) ( (M)[( ((k)*(rows)*(columns)) + ((i)*(columns)) + (j) )] = (value) )


/*! \brief Returns the value of the cell (i, j) of the matrix M.
*/
#define calGetBuffer3DElement(M, rows, columns, i, j, k) ( M[( ((k)*(rows)*(columns)) + ((i)*(columns)) + (j) )] )



#endif
