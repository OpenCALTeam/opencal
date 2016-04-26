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

#ifndef calcl3DReduction_h
#define calcl3DReduction_h

#include <OpenCAL-CL/calcl3D.h>
#include <OpenCAL-CL/dllexport.h>

/*! \brief Compute min reduction for CALbyte substates   */
DllExport
void calclAddReductionMin3Db(struct CALCLModel3D * calclmodel3D,					//!< Pointer to a struct CALCLModel3D
		int numSubstates													//!< Number of the substate
		);

/*! \brief Compute min reduction for CALint substates   */
DllExport
void calclAddReductionMin3Di(struct CALCLModel3D * calclmodel3D,					//!< Pointer to a struct CALCLModel3D
		int numSubstates													//!< Number of the substate
		);

/*! \brief Compute min reduction for CALreal substates   */
DllExport
void calclAddReductionMin3Dr(struct CALCLModel3D * calclmodel3D,					//!< Pointer to a struct CALCLModel3D
		int numSubstates													//!< Number of the substate
		);

/*! \brief Compute max reduction for CALbyte substates   */
DllExport
void calclAddReductionMax3Db(struct CALCLModel3D * calclmodel3D,					//!< Pointer to a struct CALCLModel3D
		int numSubstates													//!< Number of the substate
		);
/*! \brief Compute max reduction for CALint substates   */
DllExport
void calclAddReductionMax3Di(struct CALCLModel3D * calclmodel3D,					//!< Pointer to a struct CALCLModel3D
		int numSubstates													//!< Number of the substate
		);
/*! \brief Compute max reduction for CALreal substates   */
DllExport
void calclAddReductionMax3Dr(struct CALCLModel3D * calclmodel3D,					//!< Pointer to a struct CALCLModel3D
		int numSubstates													//!< Number of the substate
		);

/*! \brief Compute sum reduction for CALbyte substates   */
DllExport
void calclAddReductionSum3Db(struct CALCLModel3D * calclmodel3D,					//!< Pointer to a struct CALCLModel3D
		int numSubstates													//!< Number of the substate
		);
/*! \brief Compute sum reduction for CALint substates   */
DllExport
void calclAddReductionSum3Di(struct CALCLModel3D * calclmodel3D,					//!< Pointer to a struct CALCLModel3D
		int numSubstates													//!< Number of the substate
		);
/*! \brief Compute sum reduction for CALreal substates   */
DllExport
void calclAddReductionSum3Dr(struct CALCLModel3D * calclmodel3D,					//!< Pointer to a struct CALCLModel3D
		int numSubstates													//!< Number of the substate
		);

/*! \brief Compute prod reduction for CALbyte substates   */
DllExport
void calclAddReductionProd3Db(struct CALCLModel3D * calclmodel3D,					//!< Pointer to a struct CALCLModel3D
		int numSubstates													//!< Number of the substate
		);
/*! \brief Compute prod reduction for CALint substates   */
DllExport
void calclAddReductionProd3Di(struct CALCLModel3D * calclmodel3D,					//!< Pointer to a struct CALCLModel3D
		int numSubstates													//!< Number of the substate
		);
/*! \brief Compute prod reduction for CALreal substates   */
DllExport
void calclAddReductionProd3Dr(struct CALCLModel3D * calclmodel3D,					//!< Pointer to a struct CALCLModel3D
		int numSubstates													//!< Number of the substate
		);

/*! \brief Compute logical and reduction for CALbyte substates   */
DllExport
void calclAddReductionLogicalAnd3Db(struct CALCLModel3D * calclmodel3D,					//!< Pointer to a struct CALCLModel3D
		int numSubstates													//!< Number of the substate
		);
/*! \brief Compute logical and reduction for CALint substates   */
DllExport
void calclAddReductionLogicalAnd3Di(struct CALCLModel3D * calclmodel3D,					//!< Pointer to a struct CALCLModel3D
		int numSubstates													//!< Number of the substate
		);
/*! \brief Compute logical and reduction for CALreal substates   */
DllExport
void calclAddReductionLogicalAnd3Dr(struct CALCLModel3D * calclmodel3D,					//!< Pointer to a struct CALCLModel3D
		int numSubstates													//!< Number of the substate
		);

/*! \brief Compute logical or reduction for CALbyte substates   */
DllExport
void calclAddReductionLogicalOr3Db(struct CALCLModel3D * calclmodel3D,					//!< Pointer to a struct CALCLModel3D
		int numSubstates													//!< Number of the substate
		);
/*! \brief Compute logical or reduction for CALint substates   */
DllExport
void calclAddReductionLogicalOr3Di(struct CALCLModel3D * calclmodel3D,					//!< Pointer to a struct CALCLModel3D
		int numSubstates													//!< Number of the substate
		);
/*! \brief Compute logical or reduction for CALreal substates   */
DllExport
void calclAddReductionLogicalOr3Dr(struct CALCLModel3D * calclmodel3D,					//!< Pointer to a struct CALCLModel3D
		int numSubstates													//!< Number of the substate
		);

/*! \brief Compute logical xor reduction for CALbyte substates   */
DllExport
void calclAddReductionLogicalXOr3Db(struct CALCLModel3D * calclmodel3D,					//!< Pointer to a struct CALCLModel3D
		int numSubstates													//!< Number of the substate
		);
/*! \brief Compute logical xor reduction for CALint substates   */
DllExport
void calclAddReductionLogicalXOr3Di(struct CALCLModel3D * calclmodel3D,					//!< Pointer to a struct CALCLModel3D
		int numSubstates													//!< Number of the substate
		);
/*! \brief Compute logical xor reduction for CALreal substates   */
DllExport
void calclAddReductionLogicalXOr3Dr(struct CALCLModel3D * calclmodel3D,					//!< Pointer to a struct CALCLModel3D
		int numSubstates													//!< Number of the substate
		);

/*! \brief Compute binary and reduction for CALbyte substates   */
DllExport
void calclAddReductionBinaryAnd3Db(struct CALCLModel3D * calclmodel3D,					//!< Pointer to a struct CALCLModel3D
		int numSubstates													//!< Number of the substate
		);
/*! \brief Compute binary and reduction for CALint substates   */
DllExport
void calclAddReductionBinaryAnd3Di(struct CALCLModel3D * calclmodel3D,					//!< Pointer to a struct CALCLModel3D
		int numSubstates													//!< Number of the substate
		);
/*! \brief Compute binary and reduction for CALreal substates   */
DllExport
void calclAddReductionBinaryAnd3Dr(struct CALCLModel3D * calclmodel3D,					//!< Pointer to a struct CALCLModel3D
		int numSubstates													//!< Number of the substate
		);

/*! \brief Compute binary or reduction for CALbyte substates   */
DllExport
void calclAddReductionBinaryOr3Db(struct CALCLModel3D * calclmodel3D,					//!< Pointer to a struct CALCLModel3D
		int numSubstates													//!< Number of the substate
		);
/*! \brief Compute binary or reduction for CALint substates   */
DllExport
void calclAddReductionBinaryOr3Di(struct CALCLModel3D * calclmodel3D,					//!< Pointer to a struct CALCLModel3D
		int numSubstates													//!< Number of the substate
		);
/*! \brief Compute binary or reduction for CALreal substates   */
DllExport
void calclAddReductionBinaryOr3Dr(struct CALCLModel3D * calclmodel3D,					//!< Pointer to a struct CALCLModel3D
		int numSubstates													//!< Number of the substate
		);

/*! \brief Compute binary xor reduction for CALbyte substates   */
DllExport
void calclAddReductionBinaryXor3Db(struct CALCLModel3D * calclmodel3D,					//!< Pointer to a struct CALCLModel3D
		int numSubstates													//!< Number of the substate
		);
/*! \brief Compute binary xor reduction for CALint substates   */
DllExport
void calclAddReductionBinaryXor3Di(struct CALCLModel3D * calclmodel3D,					//!< Pointer to a struct CALCLModel3D
		int numSubstates													//!< Number of the substate
		);
/*! \brief Compute binary xor reduction for CALreal substates   */
DllExport
void calclAddReductionBinaryXor3Dr(struct CALCLModel3D * calclmodel3D,					//!< Pointer to a struct CALCLModel3D
		int numSubstates													//!< Number of the substate
		);


#endif
