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

#ifndef cal3DBuffer_h
#define cal3DBuffer_h

#include <OpenCAL-OMP/calCommon.h>


/*! \brief Allocates a byte linearized 3D buffer.
*/
CALbyte* calAllocBuffer3Db(int rows, int columns, int slices);

/*! \brief Allocates an int linearized 3D buffer.
*/
CALint* calAllocBuffer3Di(int rows, int columns, int slices);

/*! \brief Allocates a real (floating point) linearized 3D buffer.
*/
CALreal* calAllocBuffer3Dr(int rows, int columns, int slices);



/*! \brief Deletes the memory associated to a byte linearized 3D buffer.
*/
void calDeleteBuffer3Db(CALbyte* M);

/*! \brief Deletes the memory associated to an int linearized 3D buffer.
*/
void calDeleteBuffer3Di(CALint* M);

/*! \brief Deletes the memory associated to a real (floating point) linearized 3D buffer.
*/
void calDeleteBuffer3Dr(CALreal* M);



/*! \brief Byte linearized 3D buffer copy function.
*/
void calCopyBuffer3Db(CALbyte* M_src, CALbyte* M_dest, int rows, int columns, int slices);

/*! \brief Int linearized 3D buffer copy function.
*/
void calCopyBuffer3Di(CALint* M_src, CALint* M_dest, int rows, int columns, int slices);

/*! \brief Real (floating point) linearized 3D buffer copy function.
*/
void calCopyBuffer3Dr(CALreal* M_src, CALreal* M_dest, int rows, int columns, int slices);



/*! \brief Active cells byte linearized 3D buffer copy function.
*/
void calCopyActiveCellsBuffer3Db(CALbyte* M_src, CALbyte* M_dest, int rows, int columns, int slices, struct CALCell3D* active_cells, int sizeof_active_cells);

/*! \brief Active cells int linearized 3D buffer copy function.
*/
void calCopyActiveCellsBuffer3Di(CALint* M_src, CALint* M_dest, int rows, int columns, int slices, struct CALCell3D* active_cells, int sizeof_active_cells);

/*! \brief Active cells real (floating point) linearized 3D buffer copy function.
*/
void calCopyActiveCellsBuffer3Dr(CALreal* M_src, CALreal* M_dest, int rows, int columns, int slices, struct CALCell3D* active_cells, int sizeof_active_cells);


/*! \brief Byte linearized 3D buffer copy function.
*/
void calAddBuffer3Db(CALbyte* M_op1, CALbyte* M_op2,  CALbyte* M_dest, int rows, int columns, int slices);

/*! \brief Int linearized 3D buffer copy function.
*/
void calAddBuffer3Di(CALint* M_op1, CALint* M_op2,  CALint* M_dest, int rows, int columns, int slices);

/*! \brief Real (floating point) linearized 3D buffer copy function.
*/
void calAddBuffer3Dr(CALreal* M_op1, CALreal* M_op2,  CALreal* M_dest, int rows, int columns, int slices);



/*! \brief Byte linearized 3D buffer subtract function.
*/
void calSubtractBuffer3Db(CALbyte* M_op1, CALbyte* M_op2,  CALbyte* M_dest, int rows, int columns, int slices);

/*! \brief Int linearized 3D buffer subtract function.
*/
void calSubtractBuffer3Di(CALint* M_op1, CALint* M_op2,  CALint* M_dest, int rows, int columns, int slices);

/*! \brief Real (floating point) linearized 3D buffer subtract function.
*/
void calSubtractBuffer3Dr(CALreal* M_op1, CALreal* M_op2,  CALreal* M_dest, int rows, int columns, int slices);



/*! \brief Sets a byte 3D buffer to a constant value.  
*/
void calSetBuffer3Db(CALbyte* M, int rows, int columns, int slices, CALbyte value);

/*! \brief Sets an int 3D buffer to a constant value.  
*/
void calSetBuffer3Di(CALint* M, int rows, int columns, int slices, CALint value);

/*! \brief Sets a real (floating point) 3D buffer to a constant value.  
*/
void calSetBuffer3Dr(CALreal* M, int rows, int columns, int slices, CALreal value);



/*! \brief Sets active cells of a byte 3D buffer to a constant value.  
*/
void calSetActiveCellsBuffer3Db(CALbyte* M, int rows, int columns, int slices, CALbyte value, struct CALCell3D* active_cells, int sizeof_active_cells);

/*! \brief Sets active cells of an int 3D buffer to a constant value.  
*/
void calSetActiveCellsBuffer3Di(CALint* M, int rows, int columns, int slices, CALint value, struct CALCell3D* active_cells, int sizeof_active_cells);

/*! \brief Sets active cells of a real (floating point) 3D buffer to a constant value.  
*/
void calSetActiveCellsBuffer3Dr(CALreal* M, int rows, int columns, int slices, CALreal value, struct CALCell3D* active_cells, int sizeof_active_cells);



/*! \brief Sets the value of the cell (i, j) of the matrix M.
*/
#define calSetBuffer3DElement(M, rows, columns, i, j, k, value) ( (M)[( ((k)*(rows)*(columns)) + ((i)*(columns)) + (j) )] = (value) )


/*! \brief Returns the value of the cell (i, j) of the matrix M.
*/
#define calGetBuffer3DElement(M, rows, columns, i, j, k) ( M[( ((k)*(rows)*(columns)) + ((i)*(columns)) + (j) )] )



#endif
