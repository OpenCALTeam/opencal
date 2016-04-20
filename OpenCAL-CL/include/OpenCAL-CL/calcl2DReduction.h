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

#ifndef calcl2DReduction_h
#define calcl2DReduction_h

#include <OpenCAL-CL/calcl2D.h>

/*! \brief Compute min reduction for CALbyte substates   */
void calclAddReductionMin2Db(struct CALCLModel2D * calclmodel2D,					//!< Pointer to a struct CALCLModel2D
		int numSubstates													//!< Number of the substate
		);

/*! \brief Compute min reduction for CALint substates   */
void calclAddReductionMin2Di(struct CALCLModel2D * calclmodel2D,					//!< Pointer to a struct CALCLModel2D
		int numSubstates													//!< Number of the substate
		);

/*! \brief Compute min reduction for CALreal substates   */
void calclAddReductionMin2Dr(struct CALCLModel2D * calclmodel2D,					//!< Pointer to a struct CALCLModel2D
		int numSubstates													//!< Number of the substate
		);

/*! \brief Compute max reduction for CALbyte substates   */
void calclAddReductionMax2Db(struct CALCLModel2D * calclmodel2D,					//!< Pointer to a struct CALCLModel2D
		int numSubstates													//!< Number of the substate
		);
/*! \brief Compute max reduction for CALint substates   */
void calclAddReductionMax2Di(struct CALCLModel2D * calclmodel2D,					//!< Pointer to a struct CALCLModel2D
		int numSubstates													//!< Number of the substate
		);
/*! \brief Compute max reduction for CALreal substates   */
void calclAddReductionMax2Dr(struct CALCLModel2D * calclmodel2D,					//!< Pointer to a struct CALCLModel2D
		int numSubstates													//!< Number of the substate
		);

/*! \brief Compute sum reduction for CALbyte substates   */
void calclAddReductionSum2Db(struct CALCLModel2D * calclmodel2D,					//!< Pointer to a struct CALCLModel2D
		int numSubstates													//!< Number of the substate
		);
/*! \brief Compute sum reduction for CALint substates   */
void calclAddReductionSum2Di(struct CALCLModel2D * calclmodel2D,					//!< Pointer to a struct CALCLModel2D
		int numSubstates													//!< Number of the substate
		);
/*! \brief Compute sum reduction for CALreal substates   */
void calclAddReductionSum2Dr(struct CALCLModel2D * calclmodel2D,					//!< Pointer to a struct CALCLModel2D
		int numSubstates													//!< Number of the substate
		);

/*! \brief Compute prod reduction for CALbyte substates   */
void calclAddReductionProd2Db(struct CALCLModel2D * calclmodel2D,					//!< Pointer to a struct CALCLModel2D
		int numSubstates													//!< Number of the substate
		);
/*! \brief Compute prod reduction for CALint substates   */
void calclAddReductionProd2Di(struct CALCLModel2D * calclmodel2D,					//!< Pointer to a struct CALCLModel2D
		int numSubstates													//!< Number of the substate
		);
/*! \brief Compute prod reduction for CALreal substates   */
void calclAddReductionProd2Dr(struct CALCLModel2D * calclmodel2D,					//!< Pointer to a struct CALCLModel2D
		int numSubstates													//!< Number of the substate
		);

/*! \brief Compute logical and reduction for CALbyte substates   */
void calclAddReductionLogicalAnd2Db(struct CALCLModel2D * calclmodel2D,					//!< Pointer to a struct CALCLModel2D
		int numSubstates													//!< Number of the substate
		);
/*! \brief Compute logical and reduction for CALint substates   */
void calclAddReductionLogicalAnd2Di(struct CALCLModel2D * calclmodel2D,					//!< Pointer to a struct CALCLModel2D
		int numSubstates													//!< Number of the substate
		);
/*! \brief Compute logical and reduction for CALreal substates   */
void calclAddReductionLogicalAnd2Dr(struct CALCLModel2D * calclmodel2D,					//!< Pointer to a struct CALCLModel2D
		int numSubstates													//!< Number of the substate
		);

/*! \brief Compute logical or reduction for CALbyte substates   */
void calclAddReductionLogicalOr2Db(struct CALCLModel2D * calclmodel2D,					//!< Pointer to a struct CALCLModel2D
		int numSubstates													//!< Number of the substate
		);
/*! \brief Compute logical or reduction for CALint substates   */
void calclAddReductionLogicalOr2Di(struct CALCLModel2D * calclmodel2D,					//!< Pointer to a struct CALCLModel2D
		int numSubstates													//!< Number of the substate
		);
/*! \brief Compute logical or reduction for CALreal substates   */
void calclAddReductionLogicalOr2Dr(struct CALCLModel2D * calclmodel2D,					//!< Pointer to a struct CALCLModel2D
		int numSubstates													//!< Number of the substate
		);

/*! \brief Compute logical xor reduction for CALbyte substates   */
void calclAddReductionLogicalXOr2Db(struct CALCLModel2D * calclmodel2D,					//!< Pointer to a struct CALCLModel2D
		int numSubstates													//!< Number of the substate
		);
/*! \brief Compute logical xor reduction for CALint substates   */
void calclAddReductionLogicalXOr2Di(struct CALCLModel2D * calclmodel2D,					//!< Pointer to a struct CALCLModel2D
		int numSubstates													//!< Number of the substate
		);
/*! \brief Compute logical xor reduction for CALreal substates   */
void calclAddReductionLogicalXOr2Dr(struct CALCLModel2D * calclmodel2D,					//!< Pointer to a struct CALCLModel2D
		int numSubstates													//!< Number of the substate
		);

/*! \brief Compute binary and reduction for CALbyte substates   */
void calclAddReductionBinaryAnd2Db(struct CALCLModel2D * calclmodel2D,					//!< Pointer to a struct CALCLModel2D
		int numSubstates													//!< Number of the substate
		);
/*! \brief Compute binary and reduction for CALint substates   */
void calclAddReductionBinaryAnd2Di(struct CALCLModel2D * calclmodel2D,					//!< Pointer to a struct CALCLModel2D
		int numSubstates													//!< Number of the substate
		);
/*! \brief Compute binary and reduction for CALreal substates   */
void calclAddReductionBinaryAnd2Dr(struct CALCLModel2D * calclmodel2D,					//!< Pointer to a struct CALCLModel2D
		int numSubstates													//!< Number of the substate
		);

/*! \brief Compute binary or reduction for CALbyte substates   */
void calclAddReductionBinaryOr2Db(struct CALCLModel2D * calclmodel2D,					//!< Pointer to a struct CALCLModel2D
		int numSubstates													//!< Number of the substate
		);
/*! \brief Compute binary or reduction for CALint substates   */
void calclAddReductionBinaryOr2Di(struct CALCLModel2D * calclmodel2D,					//!< Pointer to a struct CALCLModel2D
		int numSubstates													//!< Number of the substate
		);
/*! \brief Compute binary or reduction for CALreal substates   */
void calclAddReductionBinaryOr2Dr(struct CALCLModel2D * calclmodel2D,					//!< Pointer to a struct CALCLModel2D
		int numSubstates													//!< Number of the substate
		);

/*! \brief Compute binary xor reduction for CALbyte substates   */
void calclAddReductionBinaryXor2Db(struct CALCLModel2D * calclmodel2D,					//!< Pointer to a struct CALCLModel2D
		int numSubstates													//!< Number of the substate
		);
/*! \brief Compute binary xor reduction for CALint substates   */
void calclAddReductionBinaryXor2Di(struct CALCLModel2D * calclmodel2D,					//!< Pointer to a struct CALCLModel2D
		int numSubstates													//!< Number of the substate
		);
/*! \brief Compute binary xor reduction for CALreal substates   */
void calclAddReductionBinaryXor2Dr(struct CALCLModel2D * calclmodel2D,					//!< Pointer to a struct CALCLModel2D
		int numSubstates													//!< Number of the substate
		);



#endif
