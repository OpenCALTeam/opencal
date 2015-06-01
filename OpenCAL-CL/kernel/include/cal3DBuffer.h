/*! \file cal3DBuffer.h
 *	\brief cal3DBuffer contains function to perform common operations on matrices
 *
 *	cal3DBuffer contains function to perform common operations on matrices. Each function works on
 *	a single element of the matrix because functions are thought to be used in a parallel context
 *	using a thread for each cell (or active cell) of the matrix.
 *
 */

#ifndef cal3DBuffer_h
#define cal3DBuffer_h

#include "../../../OpenCAL-CL/kernel/include/calCommon.h"

/*!	\brief Byte linearized matrix copy function.
 *
 *	Byte linearized matrix copy function. It copies the cell i j k of the matrix M_src and writes
 *	its content in the cell i j k of the matrix M_dest
 *
 */
void calCopyBuffer3Db(__global CALbyte* M_src, __global CALbyte* M_dest, int columns, int rows, int i, int j, int k);

/*!	\brief Integer linearized matrix copy function.
 *
 *	Integer linearized matrix copy function. It copies the cell i j k of the matrix M_src and writes
 *	its content in the cell i j k of the matrix M_dest
 *
 */
void calCopyBuffer3Di(__global CALint* M_src, __global CALint* M_dest, int columns, int rows, int i, int j, int k);

/*!	\brief Real linearized matrix copy function.
 *
 *	Real linearized matrix copy function. It copies the cell i j k of the matrix M_src and writes
 *	its content in the cell i j k of the matrix M_dest
 *
 */
void calCopyBuffer3Dr(__global CALreal* M_src, __global CALreal* M_dest, int columns, int rows, int i, int j, int k);

/*!	\brief Active cells byte linearized matrix copy function.
 *
 *	Active cells byte linearized matrix copy function. It copies the active cell n of the matrix M_src and writes
 *	its content in the active cell n of the matrix M_dest
 *
 */
void calCopyBufferActiveCells3Db(__global CALbyte* M_src, __global CALbyte* M_dest, int columns, int rows, __global struct CALCell3D* active_cells, int n);

/*!	\brief Active cells integer linearized matrix copy function.
 *
 *	Active cells integer linearized matrix copy function. It copies the active cell n of the matrix M_src and writes
 *	its content in the active cell n of the matrix M_dest
 *
 */
void calCopyBufferActiveCells3Di(__global CALint* M_src, __global CALint* M_dest, int columns, int rows, __global struct CALCell3D* active_cells, int n);

/*!	\brief Active cells real linearized matrix copy function.
 *
 *	Active cells real linearized matrix copy function. It copies the active cell n of the matrix M_src and writes
 *	its content in the active cell n of the matrix M_dest
 *
 */
void calCopyBufferActiveCells3Dr(__global CALreal* M_src, __global CALreal* M_dest, int columns, int rows, __global struct CALCell3D* active_cells, int n);

/*!	\brief Byte linearized matrix addition function.
 *
 *	Byte linearized matrix addition function. It adds the cell i j k of the matrix M_op1 and
 *	the cell i j k of the matrix M_op2. The result is put in the cell i j k of the matrix M_dest
 *
 */
void calAddMatrices3Db(__global CALbyte* M_op1, __global CALbyte* M_op3, __global CALbyte* M_dest, int i, int j, int k, int columns, int rows);

/*!	\brief Integer linearized matrix addition function.
 *
 *	Integer linearized matrix addition function. It adds the cell i j k of the matrix M_op1 and
 *	the cell i j k of the matrix M_op2. The result is put in the cell i j k of the matrix M_dest
 *
 */
void calAddMatrices3Di(__global CALint* M_op1, __global CALint* M_op3, __global CALint* M_dest, int i, int j, int k, int columns, int rows);

/*!	\brief Real linearized matrix addition function.
 *
 *	Real linearized matrix addition function. It adds the cell i j k of the matrix M_op1 and
 *	the cell i j k of the matrix M_op2. The result is put in the cell i j k of the matrix M_dest
 *
 */
void calAddMatrices3Dr(__global CALreal* M_op1, __global CALreal* M_op3, __global CALreal* M_dest, int i, int j, int k, int columns, int rows);

/*!	\brief Byte linearized matrix subtraction function.
 *
 *	Byte linearized matrix subtraction function. It subtracts the cell i j k of the matrix M_op1 and
 *	the cell i j k of the matrix M_op2. The result is put in the cell i j k of the matrix M_dest
 *
 */
void calSubtractMatrices3Db(__global CALbyte* M_op1, __global CALbyte* M_op3, __global CALbyte* M_dest, int i, int j, int k, int columns, int rows);

/*!	\brief Integer linearized matrix subtraction function.
 *
 *	Integer linearized matrix subtraction function. It subtracts the cell i j k of the matrix M_op1 and
 *	the cell i j k of the matrix M_op2. The result is put in the cell i j k of the matrix M_dest
 *
 */
void calSubtractMatrices3Di(__global CALint* M_op1, __global CALint* M_op3, __global CALint* M_dest, int i, int j, int k, int columns, int rows);

/*!	\brief Real linearized matrix subtraction function.
 *
 *	Real linearized matrix subtraction function. It subtracts the cell i j k of the matrix M_op1 and
 *	the cell i j k of the matrix M_op2. The result is put in the cell i j k of the matrix M_dest
 *
 */
void calSubtractMatrices3Dr(__global CALreal* M_op1, __global CALreal* M_op3, __global CALreal* M_dest, int i, int j, int k, int columns, int rows);

/*! \brief Sets an active cell of a byte matrix to a constant value.*/
void calSetBufferActiveCells3Db(__global CALbyte* M, int columns, int rows, CALbyte value, __global struct CALCell3D* active_cells, int n);

/*! \brief Sets an active cell of a integer matrix to a constant value.*/
void calSetBufferActiveCells3Di(__global CALint* M, int columns, int rows, CALint value, __global struct CALCell3D* active_cells, int n);

/*! \brief Sets an active cell of a real matrix to a constant value.*/
void calSetBufferActiveCells3Dr(__global CALreal* M, int columns, int rows, CALreal value, __global struct CALCell3D* active_cells, int n);

/*! \brief Sets an element of a matrix to a constant value.*/
#define calSetBufferElement3D(M, rows, columns, i, j, k, value) ( (M)[( ((k)*(rows)*(columns)) + ((i)*(columns)) + (j) )] = (value) )

/*! \brief Gets an element of a matrix.*/
#define calGetBufferElement3D(M, rows, columns, i, j, k) ( M[( ((k)*(rows)*(columns)) + ((i)*(columns)) + (j) )] )

#endif
