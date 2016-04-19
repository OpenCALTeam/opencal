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

void calclAddReductionMin2Db(struct CALCLModel2D * calclmodel2D,					//!< Pointer to a struct CALCLModel2D
		int numSubstates													//!< Number of the substate
		);
void calclAddReductionMin2Di(struct CALCLModel2D * calclmodel2D,					//!< Pointer to a struct CALCLModel2D
		int numSubstates													//!< Number of the substate
		);
void calclAddReductionMin2Dr(struct CALCLModel2D * calclmodel2D,					//!< Pointer to a struct CALCLModel2D
		int numSubstates													//!< Number of the substate
		);

void calclAddReductionMax2Db(struct CALCLModel2D * calclmodel2D, int numSubstate);
void calclAddReductionMax2Di(struct CALCLModel2D * calclmodel2D, int numSubstate);
void calclAddReductionMax2Dr(struct CALCLModel2D * calclmodel2D, int numSubstate);

void calclAddReductionSum2Db(struct CALCLModel2D * calclmodel2D, int numSubstate);
void calclAddReductionSum2Di(struct CALCLModel2D * calclmodel2D, int numSubstate);
void calclAddReductionSum2Dr(struct CALCLModel2D * calclmodel2D, int numSubstate);

void calclAddReductionProd2Db(struct CALCLModel2D * calclmodel2D, int numSubstate);
void calclAddReductionProd2Di(struct CALCLModel2D * calclmodel2D, int numSubstate);
void calclAddReductionProd2Dr(struct CALCLModel2D * calclmodel2D, int numSubstate);

void calclAddReductionLogicalAnd2Db(struct CALCLModel2D * calclmodel2D, int numSubstate);
void calclAddReductionLogicalAnd2Di(struct CALCLModel2D * calclmodel2D, int numSubstate);
void calclAddReductionLogicalAnd2Dr(struct CALCLModel2D * calclmodel2D, int numSubstate);

void calclAddReductionLogicalOr2Db(struct CALCLModel2D * calclmodel2D, int numSubstate);
void calclAddReductionLogicalOr2Di(struct CALCLModel2D * calclmodel2D, int numSubstate);
void calclAddReductionLogicalOr2Dr(struct CALCLModel2D * calclmodel2D, int numSubstate);

void calclAddReductionLogicalXOr2Db(struct CALCLModel2D * calclmodel2D, int numSubstate);
void calclAddReductionLogicalXOr2Di(struct CALCLModel2D * calclmodel2D, int numSubstate);
void calclAddReductionLogicalXOr2Dr(struct CALCLModel2D * calclmodel2D, int numSubstate);

void calclAddReductionBinaryAnd2Db(struct CALCLModel2D * calclmodel2D, int numSubstate);
void calclAddReductionBinaryAnd2Di(struct CALCLModel2D * calclmodel2D, int numSubstate);
void calclAddReductionBinaryAnd2Dr(struct CALCLModel2D * calclmodel2D, int numSubstate);

void calclAddReductionBinaryOr2Db(struct CALCLModel2D * calclmodel2D, int numSubstate);
void calclAddReductionBinaryOr2Di(struct CALCLModel2D * calclmodel2D, int numSubstate);
void calclAddReductionBinaryOr2Dr(struct CALCLModel2D * calclmodel2D, int numSubstate);

void calclAddReductionBinaryXor2Db(struct CALCLModel2D * calclmodel2D, int numSubstate);
void calclAddReductionBinaryXor2Di(struct CALCLModel2D * calclmodel2D, int numSubstate);
void calclAddReductionBinaryXor2Dr(struct CALCLModel2D * calclmodel2D, int numSubstate);


#endif
