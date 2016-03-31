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

#ifndef cal3DReduction_h
#define cal3DReduction_h

#include <OpenCAL/cal3D.h>

/*! \brief 

	Set of functions that compute the maximum value of a substate.
*/
CALbyte calReductionComputeMax3Db(struct CALModel3D* model, struct CALSubstate3Db* substate);
CALint calReductionComputeMax3Di(struct CALModel3D* model, struct CALSubstate3Di* substate);
CALreal calReductionComputeMax3Dr(struct CALModel3D* model, struct CALSubstate3Dr* substate);

/*! \brief 

	Set of functions that compute the minimum value of a substate.
*/
CALbyte calReductionComputeMin3Db(struct CALModel3D* model, struct CALSubstate3Db* substate);
CALint calReductionComputeMin3Di(struct CALModel3D* model, struct CALSubstate3Di* substate);
CALreal calReductionComputeMin3Dr(struct CALModel3D* model, struct CALSubstate3Dr* substate);

/*! \brief 

	Set of functions that compute the sum of values of a substate.
*/
CALbyte calReductionComputeSum3Db(struct CALModel3D* model, struct CALSubstate3Db* substate);
CALint calReductionComputeSum3Di(struct CALModel3D* model, struct CALSubstate3Di* substate);
CALreal calReductionComputeSum3Dr(struct CALModel3D* model, struct CALSubstate3Dr* substate);

/*! \brief 

	Set of functions that compute the product of values of a substate.
*/
CALbyte calReductionComputeProd3Db(struct CALModel3D* model, struct CALSubstate3Db* substate);
CALint calReductionComputeProd3Di(struct CALModel3D* model, struct CALSubstate3Di* substate);
CALreal calReductionComputeProd3Dr(struct CALModel3D* model, struct CALSubstate3Dr* substate);

/*! \brief 

	Set of functions that compute the "Logic And" of values of a substate.
*/
CALbyte calReductionComputeLogicalAnd3Db(struct CALModel3D* model, struct CALSubstate3Db* substate);
CALint calReductionComputeLogicalAnd3Di(struct CALModel3D* model, struct CALSubstate3Di* substate);
CALreal calReductionComputeLogicalAnd3Dr(struct CALModel3D* model, struct CALSubstate3Dr* substate);

/*! \brief 

	Set of functions that compute the "Binary And" of values of a substate.
*/
CALbyte calReductionComputeBinaryAnd3Db(struct CALModel3D* model, struct CALSubstate3Db* substate);
CALint calReductionComputeBinaryAnd3Di(struct CALModel3D* model, struct CALSubstate3Di* substate);
CALreal calReductionComputeBinaryAnd3Dr(struct CALModel3D* model, struct CALSubstate3Dr* substate);

/*! \brief 

	Set of functions that compute the "Logic Or" of values of a substate.
*/
CALbyte calReductionComputeLogicalOr3Db(struct CALModel3D* model, struct CALSubstate3Db* substate);
CALint calReductionComputeLogicalOr3Di(struct CALModel3D* model, struct CALSubstate3Di* substate);
CALreal calReductionComputeLogicalOr3Dr(struct CALModel3D* model, struct CALSubstate3Dr* substate);

/*! \brief 

	Set of functions that compute the "Binary Or" of values of a substate.
*/
CALbyte calReductionComputeBinaryOr3Db(struct CALModel3D* model, struct CALSubstate3Db* substate);
CALint calReductionComputeBinaryOr3Di(struct CALModel3D* model, struct CALSubstate3Di* substate);
CALreal calReductionComputeBinaryOr3Dr(struct CALModel3D* model, struct CALSubstate3Dr* substate);

/*! \brief 

	Set of functions that compute the "Logic Xor" of values of a substate.
*/
CALbyte calReductionComputeLogicalXor3Db(struct CALModel3D* model, struct CALSubstate3Db* substate);
CALint calReductionComputeLogicalXor3Di(struct CALModel3D* model, struct CALSubstate3Di* substate);
CALreal calReductionComputeLogicalXor3Dr(struct CALModel3D* model, struct CALSubstate3Dr* substate);

/*! \brief 

	Set of functions that compute the "Binary Xor" of values of a substate.
*/
CALbyte calReductionComputeBinaryXor3Db(struct CALModel3D* model, struct CALSubstate3Db* substate);
CALint calReductionComputeBinaryXor3Di(struct CALModel3D* model, struct CALSubstate3Di* substate);
CALreal calReductionComputeBinaryXor3Dr(struct CALModel3D* model, struct CALSubstate3Dr* substate);

/*! \brief 

	Private set of functions that execute the operation specified on a given substate.
	The user must not call directly this function, but instead call the functions specifed above.
*/
CALbyte calReductionOperation3Db(struct CALModel3D* model, struct CALSubstate3Db* substate, enum REDUCTION_OPERATION operation);
CALint calReductionOperation3Di(struct CALModel3D* model, struct CALSubstate3Di* substate, enum REDUCTION_OPERATION operation);
CALreal calReductionOperation3Dr(struct CALModel3D* model, struct CALSubstate3Dr* substate, enum REDUCTION_OPERATION operation);

/*! \brief 

	Utility functions used instead of "calGet3D(i, j, k)" for retriving the cell(i, j, k).
	It is sufficiently one index instead two.
*/

CALbyte getValue3DbAtIndex(struct CALSubstate3Db* substate, CALint index);
CALint getValue3DiAtIndex(struct CALSubstate3Di* substate, CALint index);
CALreal getValue3DrAtIndex(struct CALSubstate3Dr* substate, CALint index);

#endif
