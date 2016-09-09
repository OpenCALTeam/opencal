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


#ifndef calcl2DReduction_c
#define calcl2DReduction_c

#include <OpenCAL-CL/calcl2DReduction.h>

void calclAddReductionMin2Db(struct CALCLModel2D * calclmodel2D, int numSubstate) {
    calclmodel2D->reductionFlagsMinb[numSubstate] = CAL_TRUE;
}

void calclAddReductionMin2Di(struct CALCLModel2D * calclmodel2D, int numSubstate) {
    calclmodel2D->reductionFlagsMini[numSubstate] = CAL_TRUE;
}

void calclAddReductionMin2Dr(struct CALCLModel2D * calclmodel2D, int numSubstate) {
    calclmodel2D->reductionFlagsMinr[numSubstate] = CAL_TRUE;
}


void calclAddReductionMax2Db(struct CALCLModel2D * calclmodel2D, int numSubstate) {
    calclmodel2D->reductionFlagsMaxb[numSubstate] = CAL_TRUE;
}

void calclAddReductionMax2Di(struct CALCLModel2D * calclmodel2D, int numSubstate) {
    calclmodel2D->reductionFlagsMaxi[numSubstate] = CAL_TRUE;
}

void calclAddReductionMax2Dr(struct CALCLModel2D * calclmodel2D, int numSubstate) {
    calclmodel2D->reductionFlagsMaxr[numSubstate] = CAL_TRUE;
}


void calclAddReductionSum2Db(struct CALCLModel2D * calclmodel2D, int numSubstate) {
    calclmodel2D->reductionFlagsSumb[numSubstate] = CAL_TRUE;
}

void calclAddReductionSum2Di(struct CALCLModel2D * calclmodel2D, int numSubstate) {
    calclmodel2D->reductionFlagsSumi[numSubstate] = CAL_TRUE;
}

void calclAddReductionSum2Dr(struct CALCLModel2D * calclmodel2D, int numSubstate) {
    calclmodel2D->reductionFlagsSumr[numSubstate] = CAL_TRUE;
}


void calclAddReductionProd2Db(struct CALCLModel2D * calclmodel2D, int numSubstate) {
    calclmodel2D->reductionFlagsProdb[numSubstate] = CAL_TRUE;
}

void calclAddReductionProd2Di(struct CALCLModel2D * calclmodel2D, int numSubstate) {
    calclmodel2D->reductionFlagsProdi[numSubstate] = CAL_TRUE;
}

void calclAddReductionProd2Dr(struct CALCLModel2D * calclmodel2D, int numSubstate) {
    calclmodel2D->reductionFlagsProdr[numSubstate] = CAL_TRUE;
}

void calclAddReductionLogicalAnd2Db(struct CALCLModel2D * calclmodel2D, int numSubstate) {
    calclmodel2D->reductionFlagsLogicalAndb[numSubstate] = CAL_TRUE;
}

void calclAddReductionLogicalAnd2Di(struct CALCLModel2D * calclmodel2D, int numSubstate) {
    calclmodel2D->reductionFlagsLogicalAndi[numSubstate] = CAL_TRUE;
}

void calclAddReductionLogicalAnd2Dr(struct CALCLModel2D * calclmodel2D, int numSubstate) {
    calclmodel2D->reductionFlagsLogicalAndr[numSubstate] = CAL_TRUE;
}


void calclAddReductionLogicalOr2Db(struct CALCLModel2D * calclmodel2D, int numSubstate) {
    calclmodel2D->reductionFlagsLogicalOrb[numSubstate] = CAL_TRUE;
}

void calclAddReductionLogicalOr2Di(struct CALCLModel2D * calclmodel2D, int numSubstate) {
    calclmodel2D->reductionFlagsLogicalOri[numSubstate] = CAL_TRUE;
}

void calclAddReductionLogicalOr2Dr(struct CALCLModel2D * calclmodel2D, int numSubstate) {
    calclmodel2D->reductionFlagsLogicalOrr[numSubstate] = CAL_TRUE;
}

void calclAddReductionLogicalXOr2Db(struct CALCLModel2D * calclmodel2D, int numSubstate) {
    calclmodel2D->reductionFlagsLogicalXOrb[numSubstate] = CAL_TRUE;
}

void calclAddReductionLogicalXOr2Di(struct CALCLModel2D * calclmodel2D, int numSubstate) {
    calclmodel2D->reductionFlagsLogicalXOri[numSubstate] = CAL_TRUE;
}

void calclAddReductionLogicalXOr2Dr(struct CALCLModel2D * calclmodel2D, int numSubstate) {
    calclmodel2D->reductionFlagsLogicalXOrr[numSubstate] = CAL_TRUE;
}

void calclAddReductionBinaryAnd2Db(struct CALCLModel2D * calclmodel2D, int numSubstate) {
    calclmodel2D->reductionFlagsBinaryAndb[numSubstate] = CAL_TRUE;
}

void calclAddReductionBinaryAnd2Di(struct CALCLModel2D * calclmodel2D, int numSubstate) {
    calclmodel2D->reductionFlagsBinaryAndi[numSubstate] = CAL_TRUE;
}

void calclAddReductionBinaryAnd2Dr(struct CALCLModel2D * calclmodel2D, int numSubstate) {
    calclmodel2D->reductionFlagsBinaryAndr[numSubstate] = CAL_TRUE;
}


void calclAddReductionBinaryOr2Di(struct CALCLModel2D * calclmodel2D, int numSubstate) {
    calclmodel2D->reductionFlagsBinaryOri[numSubstate] = CAL_TRUE;
}

void calclAddReductionBinaryOr2Db(struct CALCLModel2D * calclmodel2D, int numSubstate) {
    calclmodel2D->reductionFlagsBinaryOrb[numSubstate] = CAL_TRUE;
}

void calclAddReductionBinaryOr2Dr(struct CALCLModel2D * calclmodel2D, int numSubstate) {
    calclmodel2D->reductionFlagsBinaryOrr[numSubstate] = CAL_TRUE;
}

void calclAddReductionBinaryXor2Di(struct CALCLModel2D * calclmodel2D, int numSubstate) {
    calclmodel2D->reductionFlagsBinaryXOri[numSubstate] = CAL_TRUE;
}

void calclAddReductionBinaryXor2Db(struct CALCLModel2D * calclmodel2D, int numSubstate) {
    calclmodel2D->reductionFlagsBinaryXOrb[numSubstate] = CAL_TRUE;
}

void calclAddReductionBinaryXor2Dr(struct CALCLModel2D * calclmodel2D, int numSubstate) {
    calclmodel2D->reductionFlagsBinaryXOrr[numSubstate] = CAL_TRUE;
}

#endif
