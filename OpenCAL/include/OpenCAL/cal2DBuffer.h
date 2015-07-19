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

#ifndef cal2DBuffer_h
#define cal2DBuffer_h

#include <OpenCAL/calCommon.h>


/*! \brief Allocates a byte linearized matrix.
*/
CALbyte* calAllocBuffer2Db(int rows, int columns);

/*! \brief Allocates an int linearized matrix.
*/
CALint* calAllocBuffer2Di(int rows, int columns);

/*! \brief Allocates a real (floating point) linearized matrix.
*/
CALreal* calAllocBuffer2Dr(int rows, int columns);



/*! \brief Deletes the memory associated to a byte linearized matrix.
*/
void calDeleteBuffer2Db(CALbyte* M);

/*! \brief Deletes the memory associated to an int linearized matrix.
*/
void calDeleteBuffer2Di(CALint* M);

/*! \brief Deletes the memory associated to a real (floating point) linearized matrix.
*/
void calDeleteBuffer2Dr(CALreal* M);



/*! \brief Byte linearized matrix copy function.
*/
void calCopyBuffer2Db(CALbyte* M_src, CALbyte* M_dest, int rows, int columns);

/*! \brief Int linearized matrix copy function.
*/
void calCopyBuffer2Di(CALint* M_src, CALint* M_dest, int rows, int columns);

/*! \brief Real (floating point) linearized matrix copy function.
*/
void calCopyBuffer2Dr(CALreal* M_src, CALreal* M_dest, int rows, int columns);



/*! \brief Active cells byte linearized matrix copy function.
*/
void calCopyActiveCellsBuffer2Db(CALbyte* M_src, CALbyte* M_dest, int rows, int columns, struct CALCell2D* active_cells, int sizeof_active_cells);

/*! \brief Active cells int linearized matrix copy function.
*/
void calCopyActiveCellsBuffer2Di(CALint* M_src, CALint* M_dest, int rows, int columns, struct CALCell2D* active_cells, int sizeof_active_cells);

/*! \brief Active cells real (floating point) linearized matrix copy function.
*/
void calCopyActiveCellsBuffer2Dr(CALreal* M_src, CALreal* M_dest, int rows, int columns, struct CALCell2D* active_cells, int sizeof_active_cells);


/*! \brief Byte linearized matrix copy function.
*/
void calAddBuffer2Db(CALbyte* M_op1, CALbyte* M_op2,  CALbyte* M_dest, int rows, int columns);

/*! \brief Int linearized matrix copy function.
*/
void calAddBuffer2Di(CALint* M_op1, CALint* M_op2,  CALint* M_dest, int rows, int columns);

/*! \brief Real (floating point) linearized matrix copy function.
*/
void calAddBuffer2Dr(CALreal* M_op1, CALreal* M_op2,  CALreal* M_dest, int rows, int columns);



/*! \brief Byte linearized matrix subtract function.
*/
void calSubtractBuffer2Db(CALbyte* M_op1, CALbyte* M_op2,  CALbyte* M_dest, int rows, int columns);

/*! \brief Int linearized matrix subtract function.
*/
void calSubtractBuffer2Di(CALint* M_op1, CALint* M_op2,  CALint* M_dest, int rows, int columns);

/*! \brief Real (floating point) linearized matrix subtract function.
*/
void calSubtractBuffer2Dr(CALreal* M_op1, CALreal* M_op2,  CALreal* M_dest, int rows, int columns);



/*! \brief Sets a byte matrix to a constant value.  
*/
void calSetBuffer2Db(CALbyte* M, int rows, int columns, CALbyte value);

/*! \brief Sets an int matrix to a constant value.  
*/
void calSetBuffer2Di(CALint* M, int rows, int columns, CALint value);

/*! \brief Sets a real (floating point) matrix to a constant value.  
*/
void calSetBuffer2Dr(CALreal* M, int rows, int columns, CALreal value);



/*! \brief Sets active cells of a byte matrix to a constant value.  
*/
void calSetActiveCellsBuffer2Db(CALbyte* M, int rows, int columns, CALbyte value, struct CALCell2D* active_cells, int sizeof_active_cells);

/*! \brief Sets active cells of an int matrix to a constant value.  
*/
void calSetActiveCellsBuffer2Di(CALint* M, int rows, int columns, CALint value, struct CALCell2D* active_cells, int sizeof_active_cells);

/*! \brief Sets active cells of a real (floating point) matrix to a constant value.  
*/
void calSetActiveCellsBuffer2Dr(CALreal* M, int rows, int columns, CALreal value, struct CALCell2D* active_cells, int sizeof_active_cells);



/*! \brief Sets the value of the cell (i, j) of the matrix M.
*/
#define calSetMatrixElement(M, columns, i, j, value) ( (M)[(((i)*(columns)) + (j))] = (value) )


/*! \brief Returns the value of the cell (i, j) of the matrix M.
*/
#define calGetMatrixElement(M, columns, i, j) ( M[(((i)*(columns)) + (j))] )



#endif
