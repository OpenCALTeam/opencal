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

#ifndef cal2DReduction_h
#define cal2DReduction_h

#include <OpenCAL/cal2D.h>

/*! \brief 

	Set of functions that compute the maximum value of a substate.
*/
CALbyte calReductionComputeMax2Db(struct CALModel2D* model, struct CALSubstate2Db* substate);
CALint calReductionComputeMax2Di(struct CALModel2D* model, struct CALSubstate2Di* substate);
CALreal calReductionComputeMax2Dr(struct CALModel2D* model, struct CALSubstate2Dr* substate);

/*! \brief 

	Set of functions that compute the minimum value of a substate.
*/
CALbyte calReductionComputeMin2Db(struct CALModel2D* model, struct CALSubstate2Db* substate);
CALint calReductionComputeMin2Di(struct CALModel2D* model, struct CALSubstate2Di* substate);
CALreal calReductionComputeMin2Dr(struct CALModel2D* model, struct CALSubstate2Dr* substate);

/*! \brief 

	Set of functions that compute the sum of values of a substate.
*/
CALbyte calReductionComputeSum2Db(struct CALModel2D* model, struct CALSubstate2Db* substate);
CALint calReductionComputeSum2Di(struct CALModel2D* model, struct CALSubstate2Di* substate);
CALreal calReductionComputeSum2Dr(struct CALModel2D* model, struct CALSubstate2Dr* substate);

/*! \brief 

	Set of functions that compute the product of values of a substate.
*/
CALbyte calReductionComputeProd2Db(struct CALModel2D* model, struct CALSubstate2Db* substate);
CALint calReductionComputeProd2Di(struct CALModel2D* model, struct CALSubstate2Di* substate);
CALreal calReductionComputeProd2Dr(struct CALModel2D* model, struct CALSubstate2Dr* substate);

/*! \brief 

	Set of functions that compute the "Logic And" of values of a substate.
*/
CALbyte calReductionComputeLogicalAnd2Db(struct CALModel2D* model, struct CALSubstate2Db* substate);
CALint calReductionComputeLogicalAnd2Di(struct CALModel2D* model, struct CALSubstate2Di* substate);
CALreal calReductionComputeLogicalAnd2Dr(struct CALModel2D* model, struct CALSubstate2Dr* substate);

/*! \brief 

	Set of functions that compute the "Binary And" of values of a substate.
*/
CALbyte calReductionComputeBinaryAnd2Db(struct CALModel2D* model, struct CALSubstate2Db* substate);
CALint calReductionComputeBinaryAnd2Di(struct CALModel2D* model, struct CALSubstate2Di* substate);
CALreal calReductionComputeBinaryAnd2Dr(struct CALModel2D* model, struct CALSubstate2Dr* substate);

/*! \brief 

	Set of functions that compute the "Logic Or" of values of a substate.
*/
CALbyte calReductionComputeLogicalOr2Db(struct CALModel2D* model, struct CALSubstate2Db* substate);
CALint calReductionComputeLogicalOr2Di(struct CALModel2D* model, struct CALSubstate2Di* substate);
CALreal calReductionComputeLogicalOr2Dr(struct CALModel2D* model, struct CALSubstate2Dr* substate);

/*! \brief 

	Set of functions that compute the "Binary Or" of values of a substate.
*/
CALbyte calReductionComputeBinaryOr2Db(struct CALModel2D* model, struct CALSubstate2Db* substate);
CALint calReductionComputeBinaryOr2Di(struct CALModel2D* model, struct CALSubstate2Di* substate);
CALreal calReductionComputeBinaryOr2Dr(struct CALModel2D* model, struct CALSubstate2Dr* substate);

/*! \brief 

	Set of functions that compute the "Logic Xor" of values of a substate.
*/
CALbyte calReductionComputeLogicalXor2Db(struct CALModel2D* model, struct CALSubstate2Db* substate);
CALint calReductionComputeLogicalXor2Di(struct CALModel2D* model, struct CALSubstate2Di* substate);
CALreal calReductionComputeLogicalXor2Dr(struct CALModel2D* model, struct CALSubstate2Dr* substate);

/*! \brief 

	Set of functions that compute the "Binary Xor" of values of a substate.
*/
CALbyte calReductionComputeBinaryXor2Db(struct CALModel2D* model, struct CALSubstate2Db* substate);
CALint calReductionComputeBinaryXor2Di(struct CALModel2D* model, struct CALSubstate2Di* substate);
CALreal calReductionComputeBinaryXor2Dr(struct CALModel2D* model, struct CALSubstate2Dr* substate);

/*! \brief 

	Private set of functions that execute the operation specified on a given substate.
	The user must not call directly this function, but instead call the functions specifed above.
*/
CALbyte calReductionOperation2Db(struct CALModel2D* model, struct CALSubstate2Db* substate, enum REDUCTION_OPERATION operation);
CALint calReductionOperation2Di(struct CALModel2D* model, struct CALSubstate2Di* substate, enum REDUCTION_OPERATION operation);
CALreal calReductionOperation2Dr(struct CALModel2D* model, struct CALSubstate2Dr* substate, enum REDUCTION_OPERATION operation);

/*! \brief 

	Utility functions used instead of "calGet2D(i, j)" for retriving the cell(i, j).
	It is sufficiently one index instead two.
*/

CALbyte getValue2DbAtIndex(struct CALSubstate2Db* substate, CALint index);
CALint getValue2DiAtIndex(struct CALSubstate2Di* substate, CALint index);
CALreal getValue2DrAtIndex(struct CALSubstate2Dr* substate, CALint index);

#endif
