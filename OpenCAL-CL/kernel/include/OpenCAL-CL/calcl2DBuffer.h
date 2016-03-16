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

/*! \file calcl2DBuffer.h
 *	\brief calcl2DBuffer contains function to perform common operations on matrices
 *
 *	calcl2DBuffer contains function to perform common operations on matrices. Each function works on
 *	a single element of the matrix because functions are thought to be used in a parallel context
 *	using a thread for each cell (or active cell) of the matrix.
 *
 */

#ifndef calcl2DBuffer_h
#define calcl2DBuffer_h

#include "OpenCAL-CL/calclCommon.h"

/*!	\brief Byte linearized matrix copy function.
 *
 *	Byte linearized matrix copy function. It copies the cell i j of the matrix M_src and writes
 *	its content in the cell i j of the matrix M_dest
 *
 */
void calclCopyBuffer2Db(__global CALbyte* M_src, __global CALbyte* M_dest, int columns, int i, int j);

/*!	\brief Integer linearized matrix copy function.
 *
 *	Integer linearized matrix copy function. It copies the cell i j of the matrix M_src and writes
 *	its content in the cell i j of the matrix M_dest
 *
 */
void calclCopyBuffer2Di(__global CALint* M_src, __global CALint* M_dest, int columns, int i, int j);

/*!	\brief Real linearized matrix copy function.
 *
 *	Real linearized matrix copy function. It copies the cell i j of the matrix M_src and writes
 *	its content in the cell i j of the matrix M_dest
 *
 */
void calclCopyBuffer2Dr(__global CALreal* M_src, __global CALreal* M_dest, int columns, int i, int j);

/*!	\brief Active cells byte linearized matrix copy function.
 *
 *	Active cells byte linearized matrix copy function. It copies the active cell n of the matrix M_src and writes
 *	its content in the active cell n of the matrix M_dest
 *
 */
void calclCopyBufferActiveCells2Db(__global CALbyte* M_src, __global CALbyte* M_dest, int columns, __global struct CALCell2D* active_cells, int n);

/*!	\brief Active cells integer linearized matrix copy function.
 *
 *	Active cells integer linearized matrix copy function. It copies the active cell n of the matrix M_src and writes
 *	its content in the active cell n of the matrix M_dest
 *
 */
void calclCopyBufferActiveCells2Di(__global CALint* M_src, __global CALint* M_dest, int columns, __global struct CALCell2D* active_cells, int n);

/*!	\brief Active cells real linearized matrix copy function.
 *
 *	Active cells real linearized matrix copy function. It copies the active cell n of the matrix M_src and writes
 *	its content in the active cell n of the matrix M_dest
 *
 */
void calclCopyBufferActiveCells2Dr(__global CALreal* M_src, __global CALreal* M_dest, int columns, __global struct CALCell2D* active_cells, int n);

/*!	\brief Byte linearized matrix addition function.
 *
 *	Byte linearized matrix addition function. It adds the cell i j of the matrix M_op1 and
 *	the cell i j of the matrix M_op2. The result is put in the cell i j of the matrix M_dest
 *
 */
void calclAddMatrices2Db(__global CALbyte* M_op1, __global CALbyte* M_op2, __global CALbyte* M_dest, int i, int j, int columns);

/*!	\brief Integer linearized matrix addition function.
 *
 *	Integer linearized matrix addition function. It adds the cell i j of the matrix M_op1 and
 *	the cell i j of the matrix M_op2. The result is put in the cell i j of the matrix M_dest
 *
 */
void calclAddMatrices2Di(__global CALint* M_op1, __global CALint* M_op2, __global CALint* M_dest, int i, int j, int columns);

/*!	\brief Real linearized matrix addition function.
 *
 *	Real linearized matrix addition function. It adds the cell i j of the matrix M_op1 and
 *	the cell i j of the matrix M_op2. The result is put in the cell i j of the matrix M_dest
 *
 */
void calclAddMatrices2Dr(__global CALreal* M_op1, __global CALreal* M_op2, __global CALreal* M_dest, int i, int j, int columns);

/*!	\brief Byte linearized matrix subtraction function.
 *
 *	Byte linearized matrix subtraction function. It subtracts the cell i j of the matrix M_op1 and
 *	the cell i j of the matrix M_op2. The result is put in the cell i j of the matrix M_dest
 *
 */
void calclSubtractMatrices2Db(__global CALbyte* M_op1, __global CALbyte* M_op2, __global CALbyte* M_dest, int i, int j, int columns);

/*!	\brief Integer linearized matrix subtraction function.
 *
 *	Integer linearized matrix subtraction function. It subtracts the cell i j of the matrix M_op1 and
 *	the cell i j of the matrix M_op2. The result is put in the cell i j of the matrix M_dest
 *
 */
void calclSubtractMatrices2Di(__global CALint* M_op1, __global CALint* M_op2, __global CALint* M_dest, int i, int j, int columns);

/*!	\brief Real linearized matrix subtraction function.
 *
 *	Real linearized matrix subtraction function. It subtracts the cell i j of the matrix M_op1 and
 *	the cell i j of the matrix M_op2. The result is put in the cell i j of the matrix M_dest
 *
 */
void calclSubtractMatrices2Dr(__global CALreal* M_op1, __global CALreal* M_op2, __global CALreal* M_dest, int i, int j, int columns);

/*! \brief Sets an active cell of a byte matrix to a constant value.*/
void calclSetBufferActiveCells2Db(__global CALbyte* M, int columns, CALbyte value, __global struct CALCell2D* active_cells, int n);

/*! \brief Sets an active cell of a integer matrix to a constant value.*/
void calclSetBufferActiveCells2Di(__global CALint* M, int columns, CALint value, __global struct CALCell2D* active_cells, int n);

/*! \brief Sets an active cell of a real matrix to a constant value.*/
void calclSetBufferActiveCells2Dr(__global CALreal* M, int columns, CALreal value, __global struct CALCell2D* active_cells, int n);


/*! \brief Sets an element of a matrix to a constant value.*/
#define calclSetBufferElement2D(M, columns, i, j, value) ( M[(i)*(columns)+(j)]) = (value)

/*! \brief Gets an element of a matrix.*/
#define calclGetBufferElement2D(M, columns, i, j) ( M[(i)*(columns)+(j)] )

#endif