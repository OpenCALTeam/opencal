/*! \file cal2D.h
 *	\brief cal2D contains function to perform operations on substates
 */

#ifndef cal2D_h
#define cal2D_h

#include <cal2DBuffer.h>
#include <calclKernel2D.h>

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
void calInitSubstate2Db(MODEL_DEFINITION2D, 	//!< Defines model parameters
		CALbyte value,							//!< Value to which the cell is set
		int i, 									//!< Row coordinate of the cell to be initialized
		int j,									//!< Column coordinate of the cell to be initialized
		CALint substateNum						//!< Indicates the number of the substate
		);

/*! \brief Initializes a integer substate to a constant value.
 *
 * Initializes an integer substate to a constant value. It initializes the cell
 * i j to a constant value in the current and next matrices.
 *
 * */
void calInitSubstate2Di(MODEL_DEFINITION2D,		//!< Defines model parameters
		CALint value,							//!< Value to which the cell is set
		int i, 									//!< Row coordinate of the cell to be initialized
		int j,									//!< Column coordinate of the cell to be initialized
		CALint substateNum						//!< Indicates the number of the substate
		);

/*! \brief Initializes a real (floating point) substate to a constant value.
 *
 * Initializes a real (floating point) substate to a constant value. It initializes the cell
 * i j to a constant value in the current and next matrices.
 *
 * */
void calInitSubstate2Dr(MODEL_DEFINITION2D,		//!< Defines model parameters
		CALreal value,							//!< Value to which the cell is set
		int i, 									//!< Row coordinate of the cell to be initialized
		int j,									//!< Column coordinate of the cell to be initialized
		CALint substateNum						//!< Indicates the number of the substate
		);

/*! \brief Gets a pointer to the current byte matrix of the substate corresponding to substateNum */
__global CALbyte * calGetCurrentSubstate2Db(MODEL_DEFINITION2D,	//!< Defines model parameters
		CALint substateNum										//!< Indicates the number of the substate
		);

/*! \brief Gets a pointer to the current integer matrix of the substate corresponding to substateNum */
__global CALint * calGetCurrentSubstate2Di(MODEL_DEFINITION2D,	//!< Defines model parameters
		CALint substateNum										//!< Indicates the number of the substate
		);

/*! \brief Gets a pointer to the current real (floating-point) matrix of the substate corresponding to substateNum */
__global CALreal * calGetCurrentSubstate2Dr(MODEL_DEFINITION2D,	//!< Defines model parameters
		CALint substateNum										//!< Indicates the number of the substate
		);

/*! \brief Gets a pointer to the current byte matrix of the substate corresponding to substateNum */
__global CALbyte * calGetNextSubstate2Db(MODEL_DEFINITION2D,	//!< Defines model parameters
		CALint substateNum										//!< Indicates the number of the substate
		);

/*! \brief Gets a pointer to the current integer matrix of the substate corresponding to substateNum */
__global CALint * calGetNextSubstate2Di(MODEL_DEFINITION2D,		//!< Defines model parameters
		CALint substateNum										//!< Indicates the number of the substate
		);

/*! \brief Gets a pointer to the current real (floating-point) matrix of the substate corresponding to substateNum */
__global CALreal * calGetNextSubstate2Dr(MODEL_DEFINITION2D,	//!< Defines model parameters
		CALint substateNum										//!< Indicates the number of the substate
		);

/*! \brief Returns the cell (i, j) value of a byte substate.*/
CALbyte calGet2Db(MODEL_DEFINITION2D,	//!< Defines model parameters
		int i, 							//!< Row coordinate of the cell
		int j,							//!< Column coordinate of the cell
		CALint substateNum				//!< Indicates the number of the substate
		);

/*! \brief Returns the cell (i, j) value of an integer substate.*/
CALint calGet2Di(MODEL_DEFINITION2D,	//!< Defines model parameters
		int i, 							//!< Row coordinate of the cell
		int j,							//!< Column coordinate of the cell
		CALint substateNum				//!< Indicates the number of the substate
		);

/*! \brief Returns the cell (i, j) value of a real (floating point) substate.*/
CALreal calGet2Dr(MODEL_DEFINITION2D,	//!< Defines model parameters
		int i, 							//!< Row coordinate of the cell
		int j,							//!< Column coordinate of the cell
		CALint substateNum				//!< Indicates the number of the substate
		);

/*! \brief Returns the n-th neighbor of the cell (i, j) value of a byte substate.*/
CALbyte calGetX2Db(MODEL_DEFINITION2D,	//!< Defines model parameters
		int i,							//!< Row coordinate of the central cell.
		int j,							//!< Column coordinate of the central cell.
		int n,							//!< Index of the n-th neighbor.
		CALint substateNum				//!< Indicates the number of the substate
		);

/*! \brief Returns the n-th neighbor of the cell (i, j) value of a integer substate.*/
CALint calGetX2Di(MODEL_DEFINITION2D,	//!< Defines model parameters
		int i,							//!< Row coordinate of the central cell.
		int j,							//!< Column coordinate of the central cell.
		int n,							//!< Index of the n-th neighbor.
		CALint substateNum				//!< Indicates the number of the substate
		);

/*! \brief Returns the n-th neighbor of the cell (i, j) value of a real (floating point) substate.*/
CALreal calGetX2Dr(MODEL_DEFINITION2D,	//!< Defines model parameters
		int i,							//!< Row coordinate of the central cell.
		int j,							//!< Column coordinate of the central cell.
		int n,							//!< Index of the n-th neighbor.
		CALint substateNum				//!< Indicates the number of the substate
		);

/*! \brief Sets the cell (i, j) value of a real (floating point) substate.  */
void calSet2Dr(MODEL_DEFINITION2D,		//!< Defines model parameters
		CALreal value,					//!< Value to which the cell is set
		int i,							//!< Row coordinate of the cell.
		int j,							//!< Column coordinate of the cell.
		CALint substateNum				//!< Indicates the number of the substate
		);

/*! \brief Sets the cell (i, j) value of a integer substate.  */
void calSet2Di(MODEL_DEFINITION2D,		//!< Defines model parameters
		CALint value,					//!< Value to which the cell is set
		int i,							//!< Row coordinate of the cell.
		int j,							//!< Column coordinate of the cell.
		CALint substateNum				//!< Indicates the number of the substate
		);

/*! \brief Sets the cell (i, j) value of a byte substate.  */
void calSet2Db(MODEL_DEFINITION2D,		//!< Defines model parameters
		CALbyte value,					//!< Value to which the cell is set
		int i,							//!< Row coordinate of the cell.
		int j,							//!< Column coordinate of the cell.
		CALint substateNum				//!< Indicates the number of the substate
		);

#endif
