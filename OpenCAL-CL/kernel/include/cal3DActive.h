/*! \file cal3DActive.h
 *	\brief cal3DActive contains functions to add, remove and initialize active cell.
 *
 */
#ifndef cal3DActive_h
#define cal3DActive_h

#include "../../../OpenCAL-CL/kernel/include/cal3D.h"

/*! \brief Sets the cell (i,j) of the linearized matrix flags to CAL_TRUE.
*/
void calAddActiveCell3D(MODEL_DEFINITION3D,				//!< Defines model parameters.
		int i,											//!< Row coordinate of the cell to be added.
		int j,											//!< Column coordinate of the cell to be added.
		int k											//!< Slice coordinate of the cell to be added.
		);

/*! \brief Sets the n-th neighbor of the cell (i,j) of the linearized matrix flags to
	CAL_TRUE.
*/
void calAddActiveCellX3D(MODEL_DEFINITION3D,			//!< Defines model parameters.
		int i,											//!< Row coordinate of the central cell.
		int j,											//!< Column coordinate of the central cell.
		int k,											//!< Slice coordinate of the central cell.
		int n											//!< Index of the n-th neighbor to be added.
		);

/*! \brief \brief Sets the cell (i,j) of the linearized matrix flags to CAL_FALSE.
*/
void calRemoveActiveCell3D(MODEL_DEFINITION3D,			//!< Defines model parameters.
		int i,											//!< Row coordinate of the cell to be removed.
		int j,											//!< Column coordinate of the cell to be removed.
		int k											//!< Slice coordinate of the cell to be removed.
		);

/*! \brief Initializes the n-th byte active cell to a constant value.
*/
void calInitSubstateActiveCell3Db(MODEL_DEFINITION3D,	//!< Defines model parameters.
		CALbyte value,									//!< Value to which the cell of the substate is set.
		int n,											//!< Index of the n-th neighbor to be initialized.
		int substateNum									//!< Indicates the number of the substate.
		);
/*! \brief Initializes the n-th int active cell to a constant value.
*/
void calInitSubstateActiveCell3Di(MODEL_DEFINITION3D,	//!< Defines model parameters.
		CALint value,									//!< Value to which the cell of the substate is set.
		int n,											//!< Index of the n-th neighbor to be initialized.
		int substateNum									//!< Indicates the number of the substate.
		);
/*! \brief Initializes the n-th real (floating point) active cell to a constant value.
*/
void calInitSubstateActiveCell3Dr(MODEL_DEFINITION3D,	//!< Defines model parameters.
		CALreal value,									//!< Value to which the cell of the substate is set.
		int n,											//!< Index of the n-th neighbor to be initialized.
		int substateNum									//!< Indicates the number of the substate.
		);

#endif



