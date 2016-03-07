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

/*! \file calcl2DActive.h
 *	\brief calcl2DActive contains functions to add, remove and initialize active cell.
 *
 */
#ifndef calcl2DActive_h
#define calcl2DActive_h

#include "OpenCAL-CL/calcl2D.h"




/*! \brief Sets the cell (i,j) of the linearized matrix flags to CAL_TRUE.
*/
void calclAddActiveCell2D(__CALCL_MODEL_2D,					//!< Defines model parameters.
		int i, 												//!< Row coordinate of the cell to be added.
		int j												//!< Column coordinate of the cell to be added.
		);

/*! \brief Sets the n-th neighbor of the cell (i,j) of the linearized matrix flags to
	CAL_TRUE.
*/
void calclAddActiveCellX2D(__CALCL_MODEL_2D, 				//!< Defines model parameters.
		int i, 												//!< Row coordinate of the central cell.
		int j, 												//!< Column coordinate of the central cell.
		int n												//!< Index of the n-th neighbor to be added.
		);

/*! \brief \brief Sets the cell (i,j) of the linearized matrix flags to CAL_FALSE.
*/
void calclRemoveActiveCell2D(__CALCL_MODEL_2D,				//!< Defines model parameters.
		int i, 												//!< Row coordinate of the cell to be removed.
		int j												//!< Column coordinate of the cell to be removed.
		);

/*! \brief Initializes the n-th byte active cell to a constant value.
*/
void calclInitSubstateActiveCell2Db(__CALCL_MODEL_2D, 		//!< Defines model parameters.
		int substateNum,										//!< Indicates the number of the substate.
		int n,												//!< Index of the n-th neighbor to be initialized.
		CALbyte value 										//!< Value to which the cell of the substate is set.
		);

/*! \brief Initializes the n-th int active cell to a constant value.
*/
void calclInitSubstateActiveCell2Di(__CALCL_MODEL_2D, 		//!< Defines model parameters.
		int substateNum,										//!< Indicates the number of the substate.
		int n,												//!< Index of the n-th neighbor to be initialized.
		CALint value 										//!< Value to which the cell of the substate is set.
		);

/*! \brief Initializes the n-th real (floating point) active cell to a constant value.
*/
void calclInitSubstateActiveCell2Dr(__CALCL_MODEL_2D, 		//!< Defines model parameters.
		int substateNum,										//!< Indicates the number of the substate.
		int n,												//!< Index of the n-th neighbor to be initialized.
		CALreal value 										//!< Value to which the cell of the substate is set.
		);
#endif
