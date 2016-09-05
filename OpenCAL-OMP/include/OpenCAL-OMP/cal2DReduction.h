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

#ifndef cal2DReduction_h
#define cal2DReduction_h

#include <OpenCAL-OMP/cal2D.h>

/*! \brief 

	Set of functions that compute the maximum value of a substate.
*/
DllExport
CALbyte calReductionComputeMax2Db(struct CALModel2D* model, struct CALSubstate2Db* substate);
DllExport
CALint calReductionComputeMax2Di(struct CALModel2D* model, struct CALSubstate2Di* substate);
DllExport
CALreal calReductionComputeMax2Dr(struct CALModel2D* model, struct CALSubstate2Dr* substate);

/*! \brief 

	Set of functions that compute the minimum value of a substate.
*/
DllExport
CALbyte calReductionComputeMin2Db(struct CALModel2D* model, struct CALSubstate2Db* substate);
DllExport
CALint calReductionComputeMin2Di(struct CALModel2D* model, struct CALSubstate2Di* substate);
DllExport
CALreal calReductionComputeMin2Dr(struct CALModel2D* model, struct CALSubstate2Dr* substate);

/*! \brief 

	Set of functions that compute the sum of values of a substate.
*/
DllExport
CALbyte calReductionComputeSum2Db(struct CALModel2D* model, struct CALSubstate2Db* substate);
DllExport
CALint calReductionComputeSum2Di(struct CALModel2D* model, struct CALSubstate2Di* substate);
DllExport
CALreal calReductionComputeSum2Dr(struct CALModel2D* model, struct CALSubstate2Dr* substate);

/*! \brief 

	Set of functions that compute the product of values of a substate.
*/
DllExport
CALbyte calReductionComputeProd2Db(struct CALModel2D* model, struct CALSubstate2Db* substate);
DllExport
CALint calReductionComputeProd2Di(struct CALModel2D* model, struct CALSubstate2Di* substate);
DllExport
CALreal calReductionComputeProd2Dr(struct CALModel2D* model, struct CALSubstate2Dr* substate);

/*! \brief 

	Set of functions that compute the "Logic And" of values of a substate.
*/
DllExport
CALbyte calReductionComputeLogicalAnd2Db(struct CALModel2D* model, struct CALSubstate2Db* substate);
DllExport
CALint calReductionComputeLogicalAnd2Di(struct CALModel2D* model, struct CALSubstate2Di* substate);
DllExport
CALreal calReductionComputeLogicalAnd2Dr(struct CALModel2D* model, struct CALSubstate2Dr* substate);

/*! \brief 

	Set of functions that compute the "Binary And" of values of a substate.
*/
DllExport
CALbyte calReductionComputeBinaryAnd2Db(struct CALModel2D* model, struct CALSubstate2Db* substate);
DllExport
CALint calReductionComputeBinaryAnd2Di(struct CALModel2D* model, struct CALSubstate2Di* substate);
DllExport
CALreal calReductionComputeBinaryAnd2Dr(struct CALModel2D* model, struct CALSubstate2Dr* substate);

/*! \brief 

	Set of functions that compute the "Logic Or" of values of a substate.
*/
DllExport
CALbyte calReductionComputeLogicalOr2Db(struct CALModel2D* model, struct CALSubstate2Db* substate);
DllExport
CALint calReductionComputeLogicalOr2Di(struct CALModel2D* model, struct CALSubstate2Di* substate);
DllExport
CALreal calReductionComputeLogicalOr2Dr(struct CALModel2D* model, struct CALSubstate2Dr* substate);

/*! \brief 

	Set of functions that compute the "Binary Or" of values of a substate.
*/
DllExport
CALbyte calReductionComputeBinaryOr2Db(struct CALModel2D* model, struct CALSubstate2Db* substate);
DllExport
CALint calReductionComputeBinaryOr2Di(struct CALModel2D* model, struct CALSubstate2Di* substate);
DllExport
CALreal calReductionComputeBinaryOr2Dr(struct CALModel2D* model, struct CALSubstate2Dr* substate);

/*! \brief 

	Set of functions that compute the "Logic Xor" of values of a substate.
*/
DllExport
CALbyte calReductionComputeLogicalXor2Db(struct CALModel2D* model, struct CALSubstate2Db* substate);
DllExport
CALint calReductionComputeLogicalXor2Di(struct CALModel2D* model, struct CALSubstate2Di* substate);
DllExport
CALreal calReductionComputeLogicalXor2Dr(struct CALModel2D* model, struct CALSubstate2Dr* substate);

/*! \brief 

	Set of functions that compute the "Binary Xor" of values of a substate.
*/
DllExport
CALbyte calReductionComputeBinaryXor2Db(struct CALModel2D* model, struct CALSubstate2Db* substate);
DllExport
CALint calReductionComputeBinaryXor2Di(struct CALModel2D* model, struct CALSubstate2Di* substate);
DllExport
CALreal calReductionComputeBinaryXor2Dr(struct CALModel2D* model, struct CALSubstate2Dr* substate);

/*! \brief 

	Private set of functions that execute the operation specified on a given substate.
	The user must not call directly this function, but instead call the functions specifed above.
*/
DllExport
CALbyte calReductionOperation2Db(struct CALModel2D* model, struct CALSubstate2Db* substate, enum REDUCTION_OPERATION operation);
DllExport
CALint calReductionOperation2Di(struct CALModel2D* model, struct CALSubstate2Di* substate, enum REDUCTION_OPERATION operation);
DllExport
CALreal calReductionOperation2Dr(struct CALModel2D* model, struct CALSubstate2Dr* substate, enum REDUCTION_OPERATION operation);

/*! \brief 

	Utility functions used instead of "calGet2D(i, j)" for retriving the cell(i, j).
	It is sufficiently one index instead two.
*/

DllExport
CALbyte getValue2DbAtIndex(struct CALSubstate2Db* substate, CALint index);
DllExport
CALint getValue2DiAtIndex(struct CALSubstate2Di* substate, CALint index);
DllExport
CALreal getValue2DrAtIndex(struct CALSubstate2Dr* substate, CALint index);

#endif
