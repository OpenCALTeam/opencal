/*! \file cal2DActive.h
 *	\brief cal2DActive contains functions to add, remove and initialize active cell.
 *
 */
#ifndef cal2DActive_h
#define cal2DActive_h

#include "../../../OpenCAL-CL/kernel/include/cal2D.h"




/*! \brief Sets the cell (i,j) of the linearized matrix flags to CAL_TRUE.
*/
void calAddActiveCell2D(MODEL_DEFINITION2D,					//!< Defines model parameters.
		int i, 												//!< Row coordinate of the cell to be added.
		int j												//!< Column coordinate of the cell to be added.
		);

/*! \brief Sets the n-th neighbor of the cell (i,j) of the linearized matrix flags to
	CAL_TRUE.
*/
void calAddActiveCellX2D(MODEL_DEFINITION2D, 				//!< Defines model parameters.
		int i, 												//!< Row coordinate of the central cell.
		int j, 												//!< Column coordinate of the central cell.
		int n												//!< Index of the n-th neighbor to be added.
		);

/*! \brief \brief Sets the cell (i,j) of the linearized matrix flags to CAL_FALSE.
*/
void calRemoveActiveCell2D(MODEL_DEFINITION2D,				//!< Defines model parameters.
		int i, 												//!< Row coordinate of the cell to be removed.
		int j												//!< Column coordinate of the cell to be removed.
		);

/*! \brief Initializes the n-th byte active cell to a constant value.
*/
void calInitSubstateActiveCell2Db(MODEL_DEFINITION2D, 		//!< Defines model parameters.
		CALbyte value, 										//!< Value to which the cell of the substate is set.
		int n,												//!< Index of the n-th neighbor to be initialized.
		int substateNum										//!< Indicates the number of the substate.
		);

/*! \brief Initializes the n-th int active cell to a constant value.
*/
void calInitSubstateActiveCell2Di(MODEL_DEFINITION2D, 		//!< Defines model parameters.
		CALint value, 										//!< Value to which the cell of the substate is set.
		int n,												//!< Index of the n-th neighbor to be initialized.
		int substateNum										//!< Indicates the number of the substate.
		);

/*! \brief Initializes the n-th real (floating point) active cell to a constant value.
*/
void calInitSubstateActiveCell2Dr(MODEL_DEFINITION2D, 		//!< Defines model parameters.
		CALreal value, 										//!< Value to which the cell of the substate is set.
		int n,												//!< Index of the n-th neighbor to be initialized.
		int substateNum										//!< Indicates the number of the substate.
		);
#endif
