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


#ifndef calcl2DReduction_h
#define calcl2DReduction_h

#include <OpenCAL-CL/calcl2D.h>
#include <OpenCAL-CL/dllexport.h>

/*! \brief Compute min reduction for CALbyte substates   */
DllExport
void calclAddReductionMin2Db(struct CALCLModel2D * calclmodel2D,					//!< Pointer to a struct CALCLModel2D
		int numSubstates													//!< Number of the substate
		);

/*! \brief Compute min reduction for CALint substates   */
DllExport
void calclAddReductionMin2Di(struct CALCLModel2D * calclmodel2D,					//!< Pointer to a struct CALCLModel2D
		int numSubstates													//!< Number of the substate
		);

/*! \brief Compute min reduction for CALreal substates   */
DllExport
void calclAddReductionMin2Dr(struct CALCLModel2D * calclmodel2D,					//!< Pointer to a struct CALCLModel2D
		int numSubstates													//!< Number of the substate
		);

/*! \brief Compute max reduction for CALbyte substates   */
DllExport
void calclAddReductionMax2Db(struct CALCLModel2D * calclmodel2D,					//!< Pointer to a struct CALCLModel2D
		int numSubstates													//!< Number of the substate
		);
/*! \brief Compute max reduction for CALint substates   */
DllExport
void calclAddReductionMax2Di(struct CALCLModel2D * calclmodel2D,					//!< Pointer to a struct CALCLModel2D
		int numSubstates													//!< Number of the substate
		);
/*! \brief Compute max reduction for CALreal substates   */
DllExport
void calclAddReductionMax2Dr(struct CALCLModel2D * calclmodel2D,					//!< Pointer to a struct CALCLModel2D
		int numSubstates													//!< Number of the substate
		);

/*! \brief Compute sum reduction for CALbyte substates   */
DllExport
void calclAddReductionSum2Db(struct CALCLModel2D * calclmodel2D,					//!< Pointer to a struct CALCLModel2D
		int numSubstates													//!< Number of the substate
		);
/*! \brief Compute sum reduction for CALint substates   */
DllExport
void calclAddReductionSum2Di(struct CALCLModel2D * calclmodel2D,					//!< Pointer to a struct CALCLModel2D
		int numSubstates													//!< Number of the substate
		);
/*! \brief Compute sum reduction for CALreal substates   */
DllExport
void calclAddReductionSum2Dr(struct CALCLModel2D * calclmodel2D,					//!< Pointer to a struct CALCLModel2D
		int numSubstates													//!< Number of the substate
		);

/*! \brief Compute prod reduction for CALbyte substates   */
DllExport
void calclAddReductionProd2Db(struct CALCLModel2D * calclmodel2D,					//!< Pointer to a struct CALCLModel2D
		int numSubstates													//!< Number of the substate
		);
/*! \brief Compute prod reduction for CALint substates   */
DllExport
void calclAddReductionProd2Di(struct CALCLModel2D * calclmodel2D,					//!< Pointer to a struct CALCLModel2D
		int numSubstates													//!< Number of the substate
		);
/*! \brief Compute prod reduction for CALreal substates   */
DllExport
void calclAddReductionProd2Dr(struct CALCLModel2D * calclmodel2D,					//!< Pointer to a struct CALCLModel2D
		int numSubstates													//!< Number of the substate
		);

/*! \brief Compute logical and reduction for CALbyte substates   */
DllExport
void calclAddReductionLogicalAnd2Db(struct CALCLModel2D * calclmodel2D,					//!< Pointer to a struct CALCLModel2D
		int numSubstates													//!< Number of the substate
		);
/*! \brief Compute logical and reduction for CALint substates   */
DllExport
void calclAddReductionLogicalAnd2Di(struct CALCLModel2D * calclmodel2D,					//!< Pointer to a struct CALCLModel2D
		int numSubstates													//!< Number of the substate
		);
/*! \brief Compute logical and reduction for CALreal substates   */
DllExport
void calclAddReductionLogicalAnd2Dr(struct CALCLModel2D * calclmodel2D,					//!< Pointer to a struct CALCLModel2D
		int numSubstates													//!< Number of the substate
		);

/*! \brief Compute logical or reduction for CALbyte substates   */
DllExport
void calclAddReductionLogicalOr2Db(struct CALCLModel2D * calclmodel2D,					//!< Pointer to a struct CALCLModel2D
		int numSubstates													//!< Number of the substate
		);
/*! \brief Compute logical or reduction for CALint substates   */
DllExport
void calclAddReductionLogicalOr2Di(struct CALCLModel2D * calclmodel2D,					//!< Pointer to a struct CALCLModel2D
		int numSubstates													//!< Number of the substate
		);
/*! \brief Compute logical or reduction for CALreal substates   */
DllExport
void calclAddReductionLogicalOr2Dr(struct CALCLModel2D * calclmodel2D,					//!< Pointer to a struct CALCLModel2D
		int numSubstates													//!< Number of the substate
		);

/*! \brief Compute logical xor reduction for CALbyte substates   */
DllExport
DllExport
void calclAddReductionLogicalXOr2Db(struct CALCLModel2D * calclmodel2D,					//!< Pointer to a struct CALCLModel2D
		int numSubstates													//!< Number of the substate
		);
/*! \brief Compute logical xor reduction for CALint substates   */
DllExport
void calclAddReductionLogicalXOr2Di(struct CALCLModel2D * calclmodel2D,					//!< Pointer to a struct CALCLModel2D
		int numSubstates													//!< Number of the substate
		);
/*! \brief Compute logical xor reduction for CALreal substates   */
DllExport
void calclAddReductionLogicalXOr2Dr(struct CALCLModel2D * calclmodel2D,					//!< Pointer to a struct CALCLModel2D
		int numSubstates													//!< Number of the substate
		);

/*! \brief Compute binary and reduction for CALbyte substates   */
DllExport
void calclAddReductionBinaryAnd2Db(struct CALCLModel2D * calclmodel2D,					//!< Pointer to a struct CALCLModel2D
		int numSubstates													//!< Number of the substate
		);
/*! \brief Compute binary and reduction for CALint substates   */
DllExport
void calclAddReductionBinaryAnd2Di(struct CALCLModel2D * calclmodel2D,					//!< Pointer to a struct CALCLModel2D
		int numSubstates													//!< Number of the substate
		);
/*! \brief Compute binary and reduction for CALreal substates   */
DllExport
void calclAddReductionBinaryAnd2Dr(struct CALCLModel2D * calclmodel2D,					//!< Pointer to a struct CALCLModel2D
		int numSubstates													//!< Number of the substate
		);

/*! \brief Compute binary or reduction for CALbyte substates   */
DllExport
void calclAddReductionBinaryOr2Db(struct CALCLModel2D * calclmodel2D,					//!< Pointer to a struct CALCLModel2D
		int numSubstates													//!< Number of the substate
		);
/*! \brief Compute binary or reduction for CALint substates   */
DllExport
void calclAddReductionBinaryOr2Di(struct CALCLModel2D * calclmodel2D,					//!< Pointer to a struct CALCLModel2D
		int numSubstates													//!< Number of the substate
		);
/*! \brief Compute binary or reduction for CALreal substates   */
DllExport
void calclAddReductionBinaryOr2Dr(struct CALCLModel2D * calclmodel2D,					//!< Pointer to a struct CALCLModel2D
		int numSubstates													//!< Number of the substate
		);

/*! \brief Compute binary xor reduction for CALbyte substates   */
DllExport
void calclAddReductionBinaryXor2Db(struct CALCLModel2D * calclmodel2D,					//!< Pointer to a struct CALCLModel2D
		int numSubstates													//!< Number of the substate
		);
/*! \brief Compute binary xor reduction for CALint substates   */
DllExport
void calclAddReductionBinaryXor2Di(struct CALCLModel2D * calclmodel2D,					//!< Pointer to a struct CALCLModel2D
		int numSubstates													//!< Number of the substate
		);
/*! \brief Compute binary xor reduction for CALreal substates   */
DllExport
void calclAddReductionBinaryXor2Dr(struct CALCLModel2D * calclmodel2D,					//!< Pointer to a struct CALCLModel2D
		int numSubstates													//!< Number of the substate
		);



#endif
