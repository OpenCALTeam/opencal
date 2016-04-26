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

#ifndef cal2DUnsafe_h
#define cal2DUnsafe_h

#include <OpenCAL-OMP/cal2D.h>



/*! \brief Sets the n-th neighbor of the cell (i,j) of the matrix flags to
	CAL_TRUE and increments the couter sizeof_active_flags.
*/
DllExport
void calAddActiveCellX2D(struct CALModel2D* ca2D,	//!< Pointer to the cellular automaton structure.
						 int i,	//!< Row coordinate of the central cell.
						 int j,	//!< Column coordinate of the central cell.
						 int n	//!< Index of the n-th neighbor to be added.
						 );



/*! \brief Inits the cell (i, j) n-th neighbour of a byte substate to value;
	it updates both the current and next matrix at the position (i, j).
	This operation is unsafe since it writes value in a neighbor, both on the current and next matrix.
*/
DllExport
void calInitX2Db(struct CALModel2D* ca2D,	//!< Pointer to the cellular automaton structure.
				 struct CALSubstate2Db* Q,	//!< Pointer to a 2D byte substate.
				 int i,						//!< Row coordinate of the central cell.
				 int j,						//!< Column coordinate of the central cell.
				 int n,						//!< Index of the n-th neighbor to be initialized.
				 CALbyte value				//!< initializing value.
				 );

/*! \brief Inits the cell (i, j) n-th neighbour of a integer substate to value;
	it updates both the current and next matrix at the position (i, j).
	This operation is unsafe since it writes value in a neighbor, both on the current and next matrix.
*/
DllExport
void calInitX2Di(struct CALModel2D* ca2D,	//!< Pointer to the cellular automaton structure.
				 struct CALSubstate2Di* Q,	//!< Pointer to a 2D int substate.
				 int i,						//!< Row coordinate of the central cell.
				 int j,						//!< Column coordinate of the central cell.
				 int n,						//!< Index of the n-th neighbor to be initialized.
				 CALint value				//!< initializing value.
				 );

/*! \brief Inits the cell (i, j) n-th neighbour of a real (floating point) substate to value;
	it updates both the current and next matrix at the position (i, j).
	This operation is unsafe since it writes value in a neighbor, both on the current and next matrix.
*/
DllExport
void calInitX2Dr(struct CALModel2D* ca2D,	//!< Pointer to the cellular automaton structure.
				 struct CALSubstate2Dr* Q,	//!< Pointer to a 2D real (floating point) substate.
				 int i,						//!< Row coordinate of the central cell.
				 int j,						//!< Column coordinate of the central cell.
				 int n,						//!< Index of the n-th neighbor to be initialized.
				 CALreal value				//!< initializing value.
				 );



/*! \brief Returns the cell (i, j) value of a byte substate from the next matrix.
	This operation is unsafe since it reads a value from the next matrix.
*/
DllExport
CALbyte calGetNext2Db(struct CALModel2D* ca2D,		//!< Pointer to the cellular automaton structure.
						  struct CALSubstate2Db* Q,	//!< Pointer to a 2D byte substate.
						  int i,					//!< Row coordinate of the cell.
						  int j						//!< Column coordinate of the cell.
						  );

/*! \brief Returns the cell (i, j) value of an integer substate from the next matrix.
	This operation is unsafe since it reads a value from the next matrix.
*/
DllExport
CALint calGetNext2Di(struct CALModel2D* ca2D,		//!< Pointer to the cellular automaton structure.
						 struct CALSubstate2Di* Q,	//!< Pointer to a 2D int substate.
						 int i,						//!< Row coordinate of the cell.
						 int j						//!< Column coordinate of the cell.
						 );

/*! \brief Returns the cell (i, j) value of a real (floating point) substate from the next matrix.
	This operation is unsafe since it read a value from the next matrix.
*/
DllExport
CALreal calGetNext2Dr(struct CALModel2D* ca2D,		//!< Pointer to the cellular automaton structure.
						  struct CALSubstate2Dr* Q,	//!< Pointer to a 2D real (floating point) substate.
						  int i,					//!< Row coordinate of the cell.
						  int j						//!< Column coordinate of the cell.
						  );



/*! \brief Returns the cell (i, j) n-th neighbor value of a byte substate from the next matrix.
	This operation is unsafe since it reads a value from the next matrix.
*/
DllExport
CALbyte calGetNextX2Db(struct CALModel2D* ca2D,		//!< Pointer to the cellular automaton structure.
					   struct CALSubstate2Db* Q,	//!< Pointer to a 2D real (floating point) substate.
					   int i,						//!< Row coordinate of the cell.
					   int j,						//!< Column coordinate of the cell.
					   int n						//!< Index of the n-th neighbor
					   );

/*! \brief Returns the cell (i, j) n-th neighbor value of an integer substate from the next matrix.
	This operation is unsafe since it reads a value from the next matrix.
*/
DllExport
CALint calGetNextX2Di(struct CALModel2D* ca2D,		//!< Pointer to the cellular automaton structure.
					  struct CALSubstate2Di* Q,		//!< Pointer to a 2D real (floating point) substate.
					  int i,						//!< Row coordinate of the cell.
					  int j,						//!< Column coordinate of the cell.
					  int n							//!< Index of the n-th neighbor
					  );

/*! \brief Returns the cell (i, j) n-th neighbor value of a real (floating point) substate from the next matrix.
	This operation is unsafe since it read a value from the next matrix.
*/
DllExport
CALreal calGetNextX2Dr(struct CALModel2D* ca2D,		//!< Pointer to the cellular automaton structure.
					   struct CALSubstate2Dr* Q,	//!< Pointer to a 2D real (floating point) substate.
					   int i,						//!< Row coordinate of the cell.
					   int j,						//!< Column coordinate of the cell.
					   int n						//!< Index of the n-th neighbor
					   );



/*! \brief Sets the value of the n-th neighbor of the cell (i, j) of a byte substate.
	This operation is unsafe since it writes a value in a neighbor of the next matrix.
*/
DllExport
void calSetX2Db(struct CALModel2D* ca2D,	//!< Pointer to the cellular automaton structure.
				struct CALSubstate2Db* Q,	//!< Pointer to a 2D byte substate.
				int i,						//!< Row coordinate of the central cell.
				int j,						//!< Column coordinate of the central cell.
				int n,						//!< Index of the n-th neighbor to be initialized.
				CALbyte value				//!< initializing value.
				);

/*! \brief Sets the value of the n-th neighbor of the cell (i, j) of an integer substate.
	This operation is unsafe since it writes a value in a neighbor of the next matrix.
*/
DllExport
void calSetX2Di(struct CALModel2D* ca2D,	//!< Pointer to the cellular automaton structure.
				struct CALSubstate2Di* Q,	//!< Pointer to a 2D int substate.
				int i,						//!< Row coordinate of the central cell.
				int j,						//!< Column coordinate of the central cell.
				int n,						//!< Index of the n-th neighbor to be initialized.
				CALint value				//!< initializing value.
				);

/*! \brief Sets the value of the n-th neighbor of the cell (i, j) of a real (floating point) substate.
	This operation is unsafe since it writes a value in a neighbor of the next matrix.
*/
DllExport
void calSetX2Dr(struct CALModel2D* ca2D,	//!< Pointer to the cellular automaton structure.
				struct CALSubstate2Dr* Q,	//!< Pointer to a 2D real (floating point) substate.
				int i,						//!< Row coordinate of the central cell.
				int j,						//!< Column coordinate of the central cell.
				int n,						//!< Index of the n-th neighbor to be initialized.
				CALreal value				//!< initializing value.
				);



/*! \brief Sets the value of the n-th neighbor of the cell (i, j)x of a byte substate of the CURRENT matri.
	This operation is unsafe since it writes a value directly to the current matrix.
*/
DllExport
void calSetCurrentX2Db(struct CALModel2D* ca2D,	//!< Pointer to the cellular automaton structure.
					  struct CALSubstate2Db* Q,	//!< Pointer to a 2D byte substate.
					  int i,					//!< Row coordinate of the central cell.
					  int j,					//!< Column coordinate of the central cell.
					  int n,					//!< Index of the n-th neighbor to be initialized.
					  CALbyte value				//!< initializing value.
					  );

/*! \brief Set the value of the n-th neighbor of the  cell (i, j) of an int substate of the CURRENT matrix.
	This operation is unsafe since it writes a value directly to the current matrix.
*/
DllExport
void calSetCurrentX2Di(struct CALModel2D* ca2D,	//!< Pointer to the cellular automaton structure.
					  struct CALSubstate2Di* Q,	//!< Pointer to a 2D int substate.
					  int i,					//!< Row coordinate of the central cell.
					  int j,					//!< Column coordinate of the central cell.
					  int n,					//!< Index of the n-th neighbor to be initialized.
					  CALint value				//!< initializing value.
					  );

/*! \brief Set the value of the n-th neighbor of the  cell (i, j) of a real (floating point) substate of the CURRENT matrix.
	This operation is unsafe since it writes a value directly to the current matrix.
*/
DllExport
void calSetCurrentX2Dr(struct CALModel2D* ca2D,	//!< Pointer to the cellular automaton structure.
					  struct CALSubstate2Dr* Q,	//!< Pointer to a 2D int substate.
					  int i,					//!< Row coordinate of the central cell.
					  int j,					//!< Column coordinate of the central cell.
					  int n,					//!< Index of the n-th neighbor to be initialized.
					  CALreal value				//!< initializing value.
					  );

DllExport
void calAddNext2Dr(struct CALModel2D *, struct CALSubstate2Dr *, int , int , CALreal);

DllExport
void calAddNextX2Dr(struct CALModel2D *, struct CALSubstate2Dr *, int , int , int , CALreal);


#endif
