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

/*! \file calcl3D.h
 *	\brief calcl3D contains function to perform operations on substates
 */


#ifndef calcl3D_h
#define calcl3D_h

#include "calcl3DBuffer.h"
#include "calclKernel3D.h"

/*! \brief Enumeration of 3D neighbourhood.

	Enumeration that identifies the cellular automaton's 3D neighbourhood.
*/
enum CALNeighborhood3D {
	CAL_CUSTOM_NEIGHBORHOOD_3D,			//!< Enumerator used for the definition of a custom 3D neighbourhood; this is built by calling the function calAddNeighbor3D.
	CAL_VON_NEUMANN_NEIGHBORHOOD_3D,	//!< Enumerator used for specifying the 3D von Neumann neighbourhood; no calls to calAddNeighbor3D are needed.
	CAL_MOORE_NEIGHBORHOOD_3D			//!< Enumerator used for specifying the 3D Moore neighbourhood; no calls to calAddNeighbor3D are needed.
};

/*! \brief Initializes a byte substate to a constant value.
 *
 * Initializes a byte substate to a constant value. It initializes the cell
 * i j k to a constant value in the current and next matrices.
 *
 * */
void calclInitSubstate3Db(__CALCL_MODEL_3D,		//!< Defines model parameters
		CALint substateNum,						//!< Indicates the number of the substate
		int i, 									//!< Row coordinate of the cell to be initialized
		int j,									//!< Column coordinate of the cell to be initialized
		int k,									//!< Slice coordinate of the cell to be initialized
		CALbyte value							//!< Value to which the cell is set
		);

/*! \brief Initializes a integer substate to a constant value.
 *
 * Initializes a integer substate to a constant value. It initializes the cell
 * i j k to a constant value in the current and next matrices.
 *
 * */
void calclInitSubstate3Di(__CALCL_MODEL_3D,		//!< Defines model parameters
		CALint substateNum,						//!< Indicates the number of the substate
		int i, 									//!< Row coordinate of the cell to be initialized
		int j,									//!< Column coordinate of the cell to be initialized
		int k,									//!< Slice coordinate of the cell to be initialized
		CALint value							//!< Value to which the cell is set
		);

/*! \brief Initializes a real (floating point) substate to a constant value.
 *
 * Initializes a real (floating point) substate to a constant value. It initializes the cell
 * i j k to a constant value in the current and next matrices.
 *
 * */
void calclInitSubstate3Dr(__CALCL_MODEL_3D,		//!< Defines model parameters
		CALint substateNum,						//!< Indicates the number of the substate
		int i, 									//!< Row coordinate of the cell to be initialized
		int j,									//!< Column coordinate of the cell to be initialized
		int k,									//!< Slice coordinate of the cell to be initialized
		CALreal value							//!< Value to which the cell is set
		);

/*! \brief Gets a pointer to the current byte matrix of the substate corresponding to substateNum */
__global CALbyte * calclGetCurrentSubstate3Db(__CALCL_MODEL_3D,		//!< Defines model parameters
		CALint substateNum											//!< Indicates the number of the substate
		);

/*! \brief Gets a pointer to the current integer matrix of the substate corresponding to substateNum */
__global CALreal * calclGetCurrentSubstate3Dr(__CALCL_MODEL_3D,		//!< Defines model parameters
		CALint substateNum											//!< Indicates the number of the substate
		);

/*! \brief Gets a pointer to the current real (floating-point) matrix of the substate corresponding to substateNum */
__global CALint * calclGetCurrentSubstate3Di(__CALCL_MODEL_3D,		//!< Defines model parameters
		CALint substateNum											//!< Indicates the number of the substate
		);

/*! \brief Gets a pointer to the next byte matrix of the substate corresponding to substateNum */
__global CALbyte * calclGetNextSubstate3Db(__CALCL_MODEL_3D,		//!< Defines model parameters
		CALint substateNum											//!< Indicates the number of the substate
		);

/*! \brief Gets a pointer to the next integer matrix of the substate corresponding to substateNum */
__global CALreal * calclGetNextSubstate3Dr(__CALCL_MODEL_3D,		//!< Defines model parameters
		CALint substateNum											//!< Indicates the number of the substate
		);

/*! \brief Gets a pointer to the next real (floating-point) matrix of the substate corresponding to substateNum */
__global CALint * calclGetNextSubstate3Di(__CALCL_MODEL_3D,			//!< Defines model parameters
		CALint substateNum											//!< Indicates the number of the substate
		);

/*! \brief Returns the cell (i, j, k) value of a byte substate.*/
CALbyte calclGet3Db(__CALCL_MODEL_3D,	//!< Defines model parameters
		CALint substateNum,				//!< Indicates the number of the substate
		int i, 							//!< Row coordinate of the cell
		int j,							//!< Column coordinate of the cell
		int k							//!< Slice coordinate of the cell
		);

/*! \brief Returns the cell (i, j, k) value of a integer substate.*/
CALint calclGet3Di(__CALCL_MODEL_3D,	//!< Defines model parameters
		CALint substateNum,				//!< Indicates the number of the substate
		int i, 							//!< Row coordinate of the cell
		int j,							//!< Column coordinate of the cell
		int k							//!< Slice coordinate of the cell
		);

/*! \brief Returns the cell (i, j, k) value of a real (floating point) substate.*/
CALreal calclGet3Dr(__CALCL_MODEL_3D,	//!< Defines model parameters
		CALint substateNum,				//!< Indicates the number of the substate
		int i, 							//!< Row coordinate of the cell
		int j,							//!< Column coordinate of the cell
		int k							//!< Slice coordinate of the cell
		);

/*! \brief Returns the n-th neighbor of the cell (i, j, k) value of a byte substate.*/
CALbyte calclGetX3Db(__CALCL_MODEL_3D,	//!< Defines model parameters
		CALint substateNum,				//!< Indicates the number of the substate
		int i, 							//!< Row coordinate of the cell
		int j,							//!< Column coordinate of the cell
		int k,							//!< Slice coordinate of the cell
		int n							//!< Index of the n-th neighbor.
		);

/*! \brief Returns the n-th neighbor of the cell (i, j, k) value of a integer substate.*/
CALint calclGetX3Di(__CALCL_MODEL_3D,	//!< Defines model parameters
		CALint substateNum,				//!< Indicates the number of the substate
		int i, 							//!< Row coordinate of the cell
		int j,							//!< Column coordinate of the cell
		int k,							//!< Slice coordinate of the cell
		int n							//!< Index of the n-th neighbor.
		);

/*! \brief Returns the n-th neighbor of the cell (i, j, k) value of a real (floating point) substate.*/
CALreal calclGetX3Dr(__CALCL_MODEL_3D,	//!< Defines model parameters
		CALint substateNum,				//!< Indicates the number of the substate
		int i, 							//!< Row coordinate of the cell
		int j,							//!< Column coordinate of the cell
		int k,							//!< Slice coordinate of the cell
		int n							//!< Index of the n-th neighbor.
		);

/*! \brief Sets the cell (i, j, k) value of a byte substate.  */
void calclSet3Db(__CALCL_MODEL_3D,		//!< Defines model parameters
		CALint substateNum,				//!< Indicates the number of the substate
		int i, 							//!< Row coordinate of the cell
		int j,							//!< Column coordinate of the cell
		int k,							//!< Slice coordinate of the cell
		CALbyte value					//!< Value to which the cell is set
		);

/*! \brief Sets the cell (i, j, k) value of a integer substate.  */
void calclSet3Di(__CALCL_MODEL_3D,		//!< Defines model parameters
		CALint substateNum,				//!< Indicates the number of the substate
		int i, 							//!< Row coordinate of the cell
		int j,							//!< Column coordinate of the cell
		int k,							//!< Slice coordinate of the cell
		CALint value					//!< Value to which the cell is set
		);

/*! \brief Sets the cell (i, j, k) value of a real (floating point) substate.  */
void calclSet3Dr(__CALCL_MODEL_3D,		//!< Defines model parameters
		CALint substateNum,				//!< Indicates the number of the substate
		int i, 							//!< Row coordinate of the cell
		int j,							//!< Column coordinate of the cell
		int k,							//!< Slice coordinate of the cell
		CALreal value					//!< Value to which the cell is set
		);


#endif
