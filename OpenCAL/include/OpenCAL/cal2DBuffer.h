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

#ifndef cal2DBuffer_h
#define cal2DBuffer_h

#include <OpenCAL/calCommon.h>
#include <OpenCAL/cal2D.h>


/*! \brief Allocates a byte linearized matrix.
*/
DllExport
CALbyte* calAllocBuffer2Db(int rows, int columns);

/*! \brief Allocates an int linearized matrix.
*/
DllExport
CALint* calAllocBuffer2Di(int rows, int columns);

/*! \brief Allocates a real (floating point) linearized matrix.
*/
DllExport
CALreal* calAllocBuffer2Dr(int rows, int columns);



/*! \brief Deletes the memory associated to a byte linearized matrix.
*/
DllExport
void calDeleteBuffer2Db(CALbyte* M);

/*! \brief Deletes the memory associated to an int linearized matrix.
*/
DllExport
void calDeleteBuffer2Di(CALint* M);

/*! \brief Deletes the memory associated to a real (floating point) linearized matrix.
*/
DllExport
void calDeleteBuffer2Dr(CALreal* M);



/*! \brief Byte linearized matrix copy function.
*/
DllExport
void calCopyBuffer2Db(CALbyte* M_src, CALbyte* M_dest, int rows, int columns);

/*! \brief Int linearized matrix copy function.
*/
DllExport
void calCopyBuffer2Di(CALint* M_src, CALint* M_dest, int rows, int columns);

/*! \brief Real (floating point) linearized matrix copy function.
*/
DllExport
void calCopyBuffer2Dr(CALreal* M_src, CALreal* M_dest, int rows, int columns);



/*! \brief Active cells byte linearized matrix copy function.
*/
DllExport
void calCopyBufferActiveCells2Db(CALbyte* M_src, CALbyte* M_dest, struct CALModel2D* ca2D);

/*! \brief Active cells int linearized matrix copy function.
*/
DllExport
void calCopyBufferActiveCells2Di(CALint* M_src, CALint* M_dest, struct CALModel2D* ca2D);

/*! \brief Active cells real (floating point) linearized matrix copy function.
*/
DllExport
void calCopyBufferActiveCells2Dr(CALreal* M_src, CALreal* M_dest,  struct CALModel2D* ca2D);


/*! \brief Byte linearized matrix copy function.
*/
DllExport
void calAddBuffer2Db(CALbyte* M_op1, CALbyte* M_op2,  CALbyte* M_dest, int rows, int columns);

/*! \brief Int linearized matrix copy function.
*/
DllExport
void calAddBuffer2Di(CALint* M_op1, CALint* M_op2,  CALint* M_dest, int rows, int columns);

/*! \brief Real (floating point) linearized matrix copy function.
*/
DllExport
void calAddBuffer2Dr(CALreal* M_op1, CALreal* M_op2,  CALreal* M_dest, int rows, int columns);



/*! \brief Byte linearized matrix subtract function.
*/
DllExport
void calSubtractBuffer2Db(CALbyte* M_op1, CALbyte* M_op2,  CALbyte* M_dest, int rows, int columns);

/*! \brief Int linearized matrix subtract function.
*/
DllExport
void calSubtractBuffer2Di(CALint* M_op1, CALint* M_op2,  CALint* M_dest, int rows, int columns);

/*! \brief Real (floating point) linearized matrix subtract function.
*/
DllExport
void calSubtractBuffer2Dr(CALreal* M_op1, CALreal* M_op2,  CALreal* M_dest, int rows, int columns);



/*! \brief Sets a byte matrix to a constant value.
*/
DllExport
void calSetBuffer2Db(CALbyte* M, int rows, int columns, CALbyte value);

/*! \brief Sets an int matrix to a constant value.
*/
DllExport
void calSetBuffer2Di(CALint* M, int rows, int columns, CALint value);

/*! \brief Sets a real (floating point) matrix to a constant value.
*/
DllExport
void calSetBuffer2Dr(CALreal* M, int rows, int columns, CALreal value);



/*! \brief Sets active cells of a byte matrix to a constant value.
*/
DllExport
void calSetActiveCellsBuffer2Db(CALbyte* M, CALbyte value, struct CALModel2D* ca2D);

/*! \brief Sets active cells of an int matrix to a constant value.
*/
DllExport
void calSetActiveCellsBuffer2Di(CALint* M, CALint value, struct CALModel2D* ca2D);

/*! \brief Sets active cells of a real (floating point) matrix to a constant value.
*/
DllExport
void calSetActiveCellsBuffer2Dr(CALreal* M, CALreal value,struct CALModel2D* ca2D);



/*! \brief Sets the value of the cell (i, j) of the matrix M.
*/
#define calSetMatrixElement(M, columns, i, j, value) ( (M)[(((i)*(columns)) + (j))] = (value) )


/*! \brief Returns the value of the cell (i, j) of the matrix M.
*/
#define calGetMatrixElement(M, columns, i, j) ( M[(((i)*(columns)) + (j))] )



#endif
