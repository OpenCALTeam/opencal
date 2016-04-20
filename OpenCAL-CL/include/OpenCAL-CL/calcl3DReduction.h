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

void calclAddReductionMin3Db(struct CALCLModel3D * calclmodel3D,					//!< Pointer to a struct CALCLModel3D
		int numSubstates													//!< Number of the substate
		);
void calclAddReductionMin3Di(struct CALCLModel3D * calclmodel3D,					//!< Pointer to a struct CALCLModel3D
		int numSubstates													//!< Number of the substate
		);
void calclAddReductionMin3Dr(struct CALCLModel3D * calclmodel3D,					//!< Pointer to a struct CALCLModel3D
		int numSubstates													//!< Number of the substate
		);

void calclAddReductionMax3Db(struct CALCLModel3D * calclmodel3D, int numSubstate);
void calclAddReductionMax3Di(struct CALCLModel3D * calclmodel3D, int numSubstate);
void calclAddReductionMax3Dr(struct CALCLModel3D * calclmodel3D, int numSubstate);

void calclAddReductionSum3Db(struct CALCLModel3D * calclmodel3D, int numSubstate);
void calclAddReductionSum3Di(struct CALCLModel3D * calclmodel3D, int numSubstate);
void calclAddReductionSum3Dr(struct CALCLModel3D * calclmodel3D, int numSubstate);

void calclAddReductionProd3Db(struct CALCLModel3D * calclmodel3D, int numSubstate);
void calclAddReductionProd3Di(struct CALCLModel3D * calclmodel3D, int numSubstate);
void calclAddReductionProd3Dr(struct CALCLModel3D * calclmodel3D, int numSubstate);

void calclAddReductionLogicalAnd3Db(struct CALCLModel3D * calclmodel3D, int numSubstate);
void calclAddReductionLogicalAnd3Di(struct CALCLModel3D * calclmodel3D, int numSubstate);
void calclAddReductionLogicalAnd3Dr(struct CALCLModel3D * calclmodel3D, int numSubstate);

void calclAddReductionLogicalOr3Db(struct CALCLModel3D * calclmodel3D, int numSubstate);
void calclAddReductionLogicalOr3Di(struct CALCLModel3D * calclmodel3D, int numSubstate);
void calclAddReductionLogicalOr3Dr(struct CALCLModel3D * calclmodel3D, int numSubstate);

void calclAddReductionLogicalXOr3Db(struct CALCLModel3D * calclmodel3D, int numSubstate);
void calclAddReductionLogicalXOr3Di(struct CALCLModel3D * calclmodel3D, int numSubstate);
void calclAddReductionLogicalXOr3Dr(struct CALCLModel3D * calclmodel3D, int numSubstate);

void calclAddReductionBinaryAnd3Db(struct CALCLModel3D * calclmodel3D, int numSubstate);
void calclAddReductionBinaryAnd3Di(struct CALCLModel3D * calclmodel3D, int numSubstate);
void calclAddReductionBinaryAnd3Dr(struct CALCLModel3D * calclmodel3D, int numSubstate);

void calclAddReductionBinaryOr3Db(struct CALCLModel3D * calclmodel3D, int numSubstate);
void calclAddReductionBinaryOr3Di(struct CALCLModel3D * calclmodel3D, int numSubstate);
void calclAddReductionBinaryOr3Dr(struct CALCLModel3D * calclmodel3D, int numSubstate);

void calclAddReductionBinaryXor3Db(struct CALCLModel3D * calclmodel3D, int numSubstate);
void calclAddReductionBinaryXor3Di(struct CALCLModel3D * calclmodel3D, int numSubstate);
void calclAddReductionBinaryXor3Dr(struct CALCLModel3D * calclmodel3D, int numSubstate);


#endif
