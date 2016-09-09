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

#ifndef calcl3DReduction_c
#define calcl3DReduction_c

#include <OpenCAL-CL/calcl3DReduction.h>

void calclAddReductionMin3Db(struct CALCLModel3D * calclmodel3D, int numSubstate) {
    calclmodel3D->reductionFlagsMinb[numSubstate] = CAL_TRUE;
}
void calclAddReductionMin3Di(struct CALCLModel3D * calclmodel3D, int numSubstate) {
    calclmodel3D->reductionFlagsMini[numSubstate] = CAL_TRUE;
}
void calclAddReductionMin3Dr(struct CALCLModel3D * calclmodel3D, int numSubstate) {
    calclmodel3D->reductionFlagsMinr[numSubstate] = CAL_TRUE;
}

void calclAddReductionMax3Db(struct CALCLModel3D * calclmodel3D, int numSubstate) {
    calclmodel3D->reductionFlagsMaxb[numSubstate] = CAL_TRUE;
}
void calclAddReductionMax3Di(struct CALCLModel3D * calclmodel3D, int numSubstate) {
    calclmodel3D->reductionFlagsMaxi[numSubstate] = CAL_TRUE;
}
void calclAddReductionMax3Dr(struct CALCLModel3D * calclmodel3D, int numSubstate) {
    calclmodel3D->reductionFlagsMaxr[numSubstate] = CAL_TRUE;
}

void calclAddReductionSum3Db(struct CALCLModel3D * calclmodel3D, int numSubstate) {
    calclmodel3D->reductionFlagsSumb[numSubstate] = CAL_TRUE;
}
void calclAddReductionSum3Di(struct CALCLModel3D * calclmodel3D, int numSubstate) {
    calclmodel3D->reductionFlagsSumi[numSubstate] = CAL_TRUE;
}
void calclAddReductionSum3Dr(struct CALCLModel3D * calclmodel3D, int numSubstate) {
    calclmodel3D->reductionFlagsSumr[numSubstate] = CAL_TRUE;
}

void calclAddReductionProd3Db(struct CALCLModel3D * calclmodel3D, int numSubstate) {
    calclmodel3D->reductionFlagsProdb[numSubstate] = CAL_TRUE;
}
void calclAddReductionProd3Di(struct CALCLModel3D * calclmodel3D, int numSubstate) {
    calclmodel3D->reductionFlagsProdi[numSubstate] = CAL_TRUE;
}
void calclAddReductionProd3Dr(struct CALCLModel3D * calclmodel3D, int numSubstate) {
    calclmodel3D->reductionFlagsProdr[numSubstate] = CAL_TRUE;
}

void calclAddReductionLogicalAnd3Db(struct CALCLModel3D * calclmodel3D, int numSubstate) {
    calclmodel3D->reductionFlagsLogicalAndb[numSubstate] = CAL_TRUE;
}
void calclAddReductionLogicalAnd3Di(struct CALCLModel3D * calclmodel3D, int numSubstate) {
    calclmodel3D->reductionFlagsLogicalAndi[numSubstate] = CAL_TRUE;
}
void calclAddReductionLogicalAnd3Dr(struct CALCLModel3D * calclmodel3D, int numSubstate) {
    calclmodel3D->reductionFlagsLogicalAndr[numSubstate] = CAL_TRUE;
}

void calclAddReductionLogicalOr3Db(struct CALCLModel3D * calclmodel3D, int numSubstate) {
    calclmodel3D->reductionFlagsLogicalOrb[numSubstate] = CAL_TRUE;
}
void calclAddReductionLogicalOr3Di(struct CALCLModel3D * calclmodel3D, int numSubstate) {
    calclmodel3D->reductionFlagsLogicalOri[numSubstate] = CAL_TRUE;
}
void calclAddReductionLogicalOr3Dr(struct CALCLModel3D * calclmodel3D, int numSubstate) {
    calclmodel3D->reductionFlagsLogicalOrr[numSubstate] = CAL_TRUE;
}

void calclAddReductionLogicalXOr3Db(struct CALCLModel3D * calclmodel3D, int numSubstate) {
    calclmodel3D->reductionFlagsLogicalXOrb[numSubstate] = CAL_TRUE;
}
void calclAddReductionLogicalXOr3Di(struct CALCLModel3D * calclmodel3D, int numSubstate) {
    calclmodel3D->reductionFlagsLogicalXOri[numSubstate] = CAL_TRUE;
}
void calclAddReductionLogicalXOr3Dr(struct CALCLModel3D * calclmodel3D, int numSubstate) {
    calclmodel3D->reductionFlagsLogicalXOrr[numSubstate] = CAL_TRUE;
}

void calclAddReductionBinaryAnd3Db(struct CALCLModel3D * calclmodel3D, int numSubstate) {
    calclmodel3D->reductionFlagsBinaryAndb[numSubstate] = CAL_TRUE;
}
void calclAddReductionBinaryAnd3Di(struct CALCLModel3D * calclmodel3D, int numSubstate) {
    calclmodel3D->reductionFlagsBinaryAndi[numSubstate] = CAL_TRUE;
}
void calclAddReductionBinaryAnd3Dr(struct CALCLModel3D * calclmodel3D, int numSubstate) {
    calclmodel3D->reductionFlagsBinaryAndr[numSubstate] = CAL_TRUE;
}

void calclAddReductionBinaryOr3Db(struct CALCLModel3D * calclmodel3D, int numSubstate) {
    calclmodel3D->reductionFlagsBinaryOrb[numSubstate] = CAL_TRUE;
}
void calclAddReductionBinaryOr3Di(struct CALCLModel3D * calclmodel3D, int numSubstate) {
    calclmodel3D->reductionFlagsBinaryOri[numSubstate] = CAL_TRUE;
}
void calclAddReductionBinaryOr3Dr(struct CALCLModel3D * calclmodel3D, int numSubstate) {
    calclmodel3D->reductionFlagsBinaryOrr[numSubstate] = CAL_TRUE;
}

void calclAddReductionBinaryXOr3Db(struct CALCLModel3D * calclmodel3D, int numSubstate) {
    calclmodel3D->reductionFlagsBinaryXOrb[numSubstate] = CAL_TRUE;
}
void calclAddReductionBinaryXOr3Di(struct CALCLModel3D * calclmodel3D, int numSubstate) {
    calclmodel3D->reductionFlagsBinaryXOri[numSubstate] = CAL_TRUE;
}
void calclAddReductionBinaryXOr3Dr(struct CALCLModel3D * calclmodel3D, int numSubstate) {
    calclmodel3D->reductionFlagsBinaryXOrr[numSubstate] = CAL_TRUE;
}




#endif
