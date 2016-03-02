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

/*! \file cal3DActive.h
 *	\brief cal3DActive contains functions to add, remove and initialize active cell.
 *
 */
#ifndef cal3DActive_h
#define cal3DActive_h

#include "cal3D.h"

/*! \brief Sets the cell (i,j) of the linearized matrix flags to CAL_TRUE.
*/
void calAddActiveCell3D(__CALCL_MODEL_3D,				//!< Defines model parameters.
		int i,											//!< Row coordinate of the cell to be added.
		int j,											//!< Column coordinate of the cell to be added.
		int k											//!< Slice coordinate of the cell to be added.
		);

/*! \brief Sets the n-th neighbor of the cell (i,j) of the linearized matrix flags to
	CAL_TRUE.
*/
void calAddActiveCellX3D(__CALCL_MODEL_3D,			//!< Defines model parameters.
		int i,											//!< Row coordinate of the central cell.
		int j,											//!< Column coordinate of the central cell.
		int k,											//!< Slice coordinate of the central cell.
		int n											//!< Index of the n-th neighbor to be added.
		);

/*! \brief \brief Sets the cell (i,j) of the linearized matrix flags to CAL_FALSE.
*/
void calRemoveActiveCell3D(__CALCL_MODEL_3D,			//!< Defines model parameters.
		int i,											//!< Row coordinate of the cell to be removed.
		int j,											//!< Column coordinate of the cell to be removed.
		int k											//!< Slice coordinate of the cell to be removed.
		);

/*! \brief Initializes the n-th byte active cell to a constant value.
*/
void calInitSubstateActiveCell3Db(__CALCL_MODEL_3D,	//!< Defines model parameters.
		int substateNum,								//!< Indicates the number of the substate.
		int n,											//!< Index of the n-th neighbor to be initialized.
		CALbyte value									//!< Value to which the cell of the substate is set.
		);
/*! \brief Initializes the n-th int active cell to a constant value.
*/
void calInitSubstateActiveCell3Di(__CALCL_MODEL_3D,	//!< Defines model parameters.
		int substateNum,									//!< Indicates the number of the substate.
		int n,											//!< Index of the n-th neighbor to be initialized.
		CALint value									//!< Value to which the cell of the substate is set.
		);
/*! \brief Initializes the n-th real (floating point) active cell to a constant value.
*/
void calInitSubstateActiveCell3Dr(__CALCL_MODEL_3D,	//!< Defines model parameters.
		int substateNum,									//!< Indicates the number of the substate.
		int n,											//!< Index of the n-th neighbor to be initialized.
		CALreal value									//!< Value to which the cell of the substate is set.
		);

#endif



