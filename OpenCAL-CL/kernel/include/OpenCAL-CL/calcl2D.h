/*
 * Copyright (c) 2016 OpenCALTeam (https://github.com/OpenCALTeam),
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

	/*! \file calcl2D.h
	 *	\brief calcl2D contains function to perform operations on substates
	 */

	#ifndef calcl2D_h
	#define calcl2D_h

	#include "OpenCAL-CL/calcl2DBuffer.h"
	#include "OpenCAL-CL/calclKernel2D.h"

	/*! \brief Enumeration of 2D neighbourhood.

		Enumeration that identifies the cellular automaton's 2D neighbourhood.
	*/
	enum CALNeighborhood2D {
		CAL_CUSTOM_NEIGHBORHOOD_2D,			//!< Enumerator used for the definition of a custom 2D neighborhood; this is built by calling the function calAddNeighbor2D.
		CAL_VON_NEUMANN_NEIGHBORHOOD_2D,	//!< Enumerator used for specifying the 2D von Neumann neighborhood; no calls to calAddNeighbor2D are needed.
		CAL_MOORE_NEIGHBORHOOD_2D,			//!< Enumerator used for specifying the 2D Moore neighborhood; no calls to calAddNeighbor2D are needed.
		CAL_HEXAGONAL_NEIGHBORHOOD_2D,		//!< Enumerator used for specifying the 2D Moore Hexagonal neighborhood; no calls to calAddNeighbor2D are needed.
		CAL_HEXAGONAL_NEIGHBORHOOD_ALT_2D	//!< Enumerator used for specifying the alternative 90� rotated 2D Moore Hexagonal neighborhood; no calls to calAddNeighbor2D are needed.
	};

	#define CAL_HEXAGONAL_SHIFT 7			//<! Shif used for accessing to the correct neighbor in case hexagonal heighbourhood and odd column cell

	/*! \brief Initializes a byte substate to a constant value.
	 *
	 * Initializes a byte substate to a constant value. It initializes the cell
	 * i j to a constant value in the current and next matrices.
	 *
	 * */
	void calclInitSubstate2Db(__CALCL_MODEL_2D, 	//!< Defines model parameters
			CALint substateNum,						//!< Indicates the number of the substate
			int i, 									//!< Row coordinate of the cell to be initialized
			int j,									//!< Column coordinate of the cell to be initialized
			CALbyte value							//!< Value to which the cell is set
			);

	/*! \brief Initializes a integer substate to a constant value.
	 *
	 * Initializes an integer substate to a constant value. It initializes the cell
	 * i j to a constant value in the current and next matrices.
	 *
	 * */
	void calclInitSubstate2Di(__CALCL_MODEL_2D,		//!< Defines model parameters
			CALint substateNum,						//!< Indicates the number of the substate
			int i, 									//!< Row coordinate of the cell to be initialized
			int j,									//!< Column coordinate of the cell to be initialized
			CALint value							//!< Value to which the cell is set
			);

	/*! \brief Initializes a real (floating point) substate to a constant value.
	 *
	 * Initializes a real (floating point) substate to a constant value. It initializes the cell
	 * i j to a constant value in the current and next matrices.
	 *
	 * */
	void calclInitSubstate2Dr(__CALCL_MODEL_2D,		//!< Defines model parameters
			CALint substateNum,						//!< Indicates the number of the substate
			int i, 									//!< Row coordinate of the cell to be initialized
			int j,									//!< Column coordinate of the cell to be initialized
			CALreal value							//!< Value to which the cell is set
			);

	/*! \brief Gets a pointer to the current byte matrix of the substate corresponding to substateNum */
	__global CALbyte * calclGetCurrentSubstate2Db(__CALCL_MODEL_2D,	//!< Defines model parameters
			CALint substateNum										//!< Indicates the number of the substate
			);

	/*! \brief Gets a pointer to the current integer matrix of the substate corresponding to substateNum */
	__global CALint * calclGetCurrentSubstate2Di(__CALCL_MODEL_2D,	//!< Defines model parameters
			CALint substateNum										//!< Indicates the number of the substate
			);

	/*! \brief Gets a pointer to the current real (floating-point) matrix of the substate corresponding to substateNum */
	__global CALreal * calclGetCurrentSubstate2Dr(__CALCL_MODEL_2D,	//!< Defines model parameters
			CALint substateNum										//!< Indicates the number of the substate
			);

	/*! \brief Gets a pointer to the current byte matrix of the substate corresponding to substateNum */
	__global CALbyte * calclGetNextSubstate2Db(__CALCL_MODEL_2D,	//!< Defines model parameters
			CALint substateNum										//!< Indicates the number of the substate
			);

	/*! \brief Gets a pointer to the current integer matrix of the substate corresponding to substateNum */
	__global CALint * calclGetNextSubstate2Di(__CALCL_MODEL_2D,		//!< Defines model parameters
			CALint substateNum										//!< Indicates the number of the substate
			);

	/*! \brief Gets a pointer to the current real (floating-point) matrix of the substate corresponding to substateNum */
	__global CALreal * calclGetNextSubstate2Dr(__CALCL_MODEL_2D,	//!< Defines model parameters
			CALint substateNum										//!< Indicates the number of the substate
			);

	/*! \brief Returns the cell (i, j) value of a byte substate.*/
	CALbyte calclGet2Db(__CALCL_MODEL_2D,	//!< Defines model parameters
			CALint substateNum,				//!< Indicates the number of the substate
			int i, 							//!< Row coordinate of the cell
			int j							//!< Column coordinate of the cell
			);

	/*! \brief Returns the cell (i, j) value of an integer substate.*/
	CALint calclGet2Di(__CALCL_MODEL_2D,	//!< Defines model parameters
			CALint substateNum,				//!< Indicates the number of the substate
			int i, 							//!< Row coordinate of the cell
			int j							//!< Column coordinate of the cell
			);

	/*! \brief Returns the cell (i, j) value of a real (floating point) substate.*/
	CALreal calclGet2Dr(__CALCL_MODEL_2D,	//!< Defines model parameters
			CALint substateNum,				//!< Indicates the number of the substate
			int i, 							//!< Row coordinate of the cell
			int j							//!< Column coordinate of the cell
			);

	/*! \brief Returns the n-th neighbor of the cell (i, j) value of a byte substate.*/
	CALbyte calclGetX2Db(__CALCL_MODEL_2D,	//!< Defines model parameters
			CALint substateNum,				//!< Indicates the number of the substate
			int i,							//!< Row coordinate of the central cell.
			int j,							//!< Column coordinate of the central cell.
			int n							//!< Index of the n-th neighbor.
			);

	/*! \brief Returns the n-th neighbor of the cell (i, j) value of a integer substate.*/
	CALint calclGetX2Di(__CALCL_MODEL_2D,	//!< Defines model parameters
			CALint substateNum,				//!< Indicates the number of the substate
			int i,							//!< Row coordinate of the central cell.
			int j,							//!< Column coordinate of the central cell.
			int n							//!< Index of the n-th neighbor.
			);

	/*! \brief Returns the n-th neighbor of the cell (i, j) value of a real (floating point) substate.*/
	CALreal calclGetX2Dr(__CALCL_MODEL_2D,	//!< Defines model parameters
			CALint substateNum,				//!< Indicates the number of the substate
			int i,							//!< Row coordinate of the central cell.
			int j,							//!< Column coordinate of the central cell.
			int n							//!< Index of the n-th neighbor.
			);

	/*! \brief Sets the cell (i, j) value of a real (floating point) substate.  */
	void calclSet2Dr(__CALCL_MODEL_2D,		//!< Defines model parameters
			CALint substateNum,				//!< Indicates the number of the substate
			int i,							//!< Row coordinate of the cell.
			int j,							//!< Column coordinate of the cell.
			CALreal value					//!< Value to which the cell is set
			);

	/*! \brief Sets the cell (i, j) value of a integer substate.  */
	void calclSet2Di(__CALCL_MODEL_2D,		//!< Defines model parameters
			CALint substateNum,				//!< Indicates the number of the substate
			int i,							//!< Row coordinate of the cell.
			int j,							//!< Column coordinate of the cell.
			CALint value					//!< Value to which the cell is set
			);

	/*! \brief Sets the cell (i, j) value of a byte substate.  */
	void calclSet2Db(__CALCL_MODEL_2D,		//!< Defines model parameters
			CALint substateNum,				//!< Indicates the number of the substate
			int i,							//!< Row coordinate of the cell.
			int j,							//!< Column coordinate of the cell.
			CALbyte value					//!< Value to which the cell is set
			);

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
