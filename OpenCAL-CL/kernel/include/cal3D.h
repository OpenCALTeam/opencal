/*! \file cal3D.h
 *	\brief cal3D contains function to perform operations on substates
 */


#ifndef cal3D_h
#define cal3D_h

#include "cal3DBuffer.h"
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
void calInitSubstate3Db(MODEL_DEFINITION3D,		//!< Defines model parameters
		CALbyte value,							//!< Value to which the cell is set
		int i, 									//!< Row coordinate of the cell to be initialized
		int j,									//!< Column coordinate of the cell to be initialized
		int k,									//!< Slice coordinate of the cell to be initialized
		CALint substateNum						//!< Indicates the number of the substate
		);

/*! \brief Initializes a integer substate to a constant value.
 *
 * Initializes a integer substate to a constant value. It initializes the cell
 * i j k to a constant value in the current and next matrices.
 *
 * */
void calInitSubstate3Di(MODEL_DEFINITION3D,		//!< Defines model parameters
		CALint value,							//!< Value to which the cell is set
		int i, 									//!< Row coordinate of the cell to be initialized
		int j,									//!< Column coordinate of the cell to be initialized
		int k,									//!< Slice coordinate of the cell to be initialized
		CALint substateNum						//!< Indicates the number of the substate
		);

/*! \brief Initializes a real (floating point) substate to a constant value.
 *
 * Initializes a real (floating point) substate to a constant value. It initializes the cell
 * i j k to a constant value in the current and next matrices.
 *
 * */
void calInitSubstate3Dr(MODEL_DEFINITION3D,		//!< Defines model parameters
		CALreal value,							//!< Value to which the cell is set
		int i, 									//!< Row coordinate of the cell to be initialized
		int j,									//!< Column coordinate of the cell to be initialized
		int k,									//!< Slice coordinate of the cell to be initialized
		CALint substateNum						//!< Indicates the number of the substate
		);

/*! \brief Gets a pointer to the current byte matrix of the substate corresponding to substateNum */
__global CALbyte * calGetCurrentSubstate3Db(MODEL_DEFINITION3D,		//!< Defines model parameters
		CALint substateNum											//!< Indicates the number of the substate
		);

/*! \brief Gets a pointer to the current integer matrix of the substate corresponding to substateNum */
__global CALreal * calGetCurrentSubstate3Dr(MODEL_DEFINITION3D,		//!< Defines model parameters
		CALint substateNum											//!< Indicates the number of the substate
		);

/*! \brief Gets a pointer to the current real (floating-point) matrix of the substate corresponding to substateNum */
__global CALint * calGetCurrentSubstate3Di(MODEL_DEFINITION3D,		//!< Defines model parameters
		CALint substateNum											//!< Indicates the number of the substate
		);

/*! \brief Gets a pointer to the next byte matrix of the substate corresponding to substateNum */
__global CALbyte * calGetNextSubstate3Db(MODEL_DEFINITION3D,		//!< Defines model parameters
		CALint substateNum											//!< Indicates the number of the substate
		);

/*! \brief Gets a pointer to the next integer matrix of the substate corresponding to substateNum */
__global CALreal * calGetNextSubstate3Dr(MODEL_DEFINITION3D,		//!< Defines model parameters
		CALint substateNum											//!< Indicates the number of the substate
		);

/*! \brief Gets a pointer to the next real (floating-point) matrix of the substate corresponding to substateNum */
__global CALint * calGetNextSubstate3Di(MODEL_DEFINITION3D,			//!< Defines model parameters
		CALint substateNum											//!< Indicates the number of the substate
		);

/*! \brief Returns the cell (i, j, k) value of a byte substate.*/
CALbyte calGet3Db(MODEL_DEFINITION3D,	//!< Defines model parameters
		int i, 							//!< Row coordinate of the cell
		int j,							//!< Column coordinate of the cell
		int k,							//!< Slice coordinate of the cell
		CALint substateNum				//!< Indicates the number of the substate
		);

/*! \brief Returns the cell (i, j, k) value of a integer substate.*/
CALint calGet3Di(MODEL_DEFINITION3D,	//!< Defines model parameters
		int i, 							//!< Row coordinate of the cell
		int j,							//!< Column coordinate of the cell
		int k,							//!< Slice coordinate of the cell
		CALint substateNum				//!< Indicates the number of the substate
		);

/*! \brief Returns the cell (i, j, k) value of a real (floating point) substate.*/
CALreal calGet3Dr(MODEL_DEFINITION3D,	//!< Defines model parameters
		int i, 							//!< Row coordinate of the cell
		int j,							//!< Column coordinate of the cell
		int k,							//!< Slice coordinate of the cell
		CALint substateNum				//!< Indicates the number of the substate
		);

/*! \brief Returns the n-th neighbor of the cell (i, j, k) value of a byte substate.*/
CALbyte calGetX3Db(MODEL_DEFINITION3D,	//!< Defines model parameters
		int i, 							//!< Row coordinate of the cell
		int j,							//!< Column coordinate of the cell
		int k,							//!< Slice coordinate of the cell
		int n,							//!< Index of the n-th neighbor.
		CALint substateNum				//!< Indicates the number of the substate
		);

/*! \brief Returns the n-th neighbor of the cell (i, j, k) value of a integer substate.*/
CALint calGetX3Di(MODEL_DEFINITION3D,	//!< Defines model parameters
		int i, 							//!< Row coordinate of the cell
		int j,							//!< Column coordinate of the cell
		int k,							//!< Slice coordinate of the cell
		int n,							//!< Index of the n-th neighbor.
		CALint substateNum				//!< Indicates the number of the substate
		);

/*! \brief Returns the n-th neighbor of the cell (i, j, k) value of a real (floating point) substate.*/
CALreal calGetX3Dr(MODEL_DEFINITION3D,	//!< Defines model parameters
		int i, 							//!< Row coordinate of the cell
		int j,							//!< Column coordinate of the cell
		int k,							//!< Slice coordinate of the cell
		int n,							//!< Index of the n-th neighbor.
		CALint substateNum				//!< Indicates the number of the substate
		);

/*! \brief Sets the cell (i, j, k) value of a byte substate.  */
void calSet3Db(MODEL_DEFINITION3D,		//!< Defines model parameters
		CALbyte value,					//!< Value to which the cell is set
		int i, 							//!< Row coordinate of the cell
		int j,							//!< Column coordinate of the cell
		int k,							//!< Slice coordinate of the cell
		CALint substateNum				//!< Indicates the number of the substate
		);

/*! \brief Sets the cell (i, j, k) value of a integer substate.  */
void calSet3Di(MODEL_DEFINITION3D,		//!< Defines model parameters
		CALint value,					//!< Value to which the cell is set
		int i, 							//!< Row coordinate of the cell
		int j,							//!< Column coordinate of the cell
		int k,							//!< Slice coordinate of the cell
		CALint substateNum				//!< Indicates the number of the substate
		);

/*! \brief Sets the cell (i, j, k) value of a real (floating point) substate.  */
void calSet3Dr(MODEL_DEFINITION3D,		//!< Defines model parameters
		CALreal value,					//!< Value to which the cell is set
		int i, 							//!< Row coordinate of the cell
		int j,							//!< Column coordinate of the cell
		int k,							//!< Slice coordinate of the cell
		CALint substateNum				//!< Indicates the number of the substate
		);


#endif




