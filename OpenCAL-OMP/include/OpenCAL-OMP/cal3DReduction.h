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

#include <OpenCAL-OMP/cal3D.h>

/*! \brief 

	Set of functions that compute the maximum value of a substate.
*/
CALbyte calSteeringComputeMax3Db(struct CALModel3D* model, struct CALSubstate3Db* substate);
CALint calSteeringComputeMax3Di(struct CALModel3D* model, struct CALSubstate3Di* substate);
CALreal calSteeringComputeMax3Dr(struct CALModel3D* model, struct CALSubstate3Dr* substate);

/*! \brief 

	Set of functions that compute the minimum value of a substate.
*/
CALbyte calSteeringComputeMin3Db(struct CALModel3D* model, struct CALSubstate3Db* substate);
CALint calSteeringComputeMin3Di(struct CALModel3D* model, struct CALSubstate3Di* substate);
CALreal calSteeringComputeMin3Dr(struct CALModel3D* model, struct CALSubstate3Dr* substate);

/*! \brief 

	Set of functions that compute the sum of values of a substate.
*/
CALbyte calSteeringComputeSum3Db(struct CALModel3D* model, struct CALSubstate3Db* substate);
CALint calSteeringComputeSum3Di(struct CALModel3D* model, struct CALSubstate3Di* substate);
CALreal calSteeringComputeSum3Dr(struct CALModel3D* model, struct CALSubstate3Dr* substate);

/*! \brief 

	Set of functions that compute the product of values of a substate.
*/
CALbyte calSteeringComputeProd3Db(struct CALModel3D* model, struct CALSubstate3Db* substate);
CALint calSteeringComputeProd3Di(struct CALModel3D* model, struct CALSubstate3Di* substate);
CALreal calSteeringComputeProd3Dr(struct CALModel3D* model, struct CALSubstate3Dr* substate);

/*! \brief 

	Set of functions that compute the "Logic And" of values of a substate.
*/
CALbyte calSteeringComputeLogicalAnd3Db(struct CALModel3D* model, struct CALSubstate3Db* substate);
CALint calSteeringComputeLogicalAnd3Di(struct CALModel3D* model, struct CALSubstate3Di* substate);
CALreal calSteeringComputeLogicalAnd3Dr(struct CALModel3D* model, struct CALSubstate3Dr* substate);

/*! \brief 

	Set of functions that compute the "Binary And" of values of a substate.
*/
CALbyte calSteeringComputeBinaryAnd3Db(struct CALModel3D* model, struct CALSubstate3Db* substate);
CALint calSteeringComputeBinaryAnd3Di(struct CALModel3D* model, struct CALSubstate3Di* substate);
CALreal calSteeringComputeBinaryAnd3Dr(struct CALModel3D* model, struct CALSubstate3Dr* substate);

/*! \brief 

	Set of functions that compute the "Logic Or" of values of a substate.
*/
CALbyte calSteeringComputeLogicalOr3Db(struct CALModel3D* model, struct CALSubstate3Db* substate);
CALint calSteeringComputeLogicalOr3Di(struct CALModel3D* model, struct CALSubstate3Di* substate);
CALreal calSteeringComputeLogicalOr3Dr(struct CALModel3D* model, struct CALSubstate3Dr* substate);

/*! \brief 

	Set of functions that compute the "Binary Or" of values of a substate.
*/
CALbyte calSteeringComputeBinaryOr3Db(struct CALModel3D* model, struct CALSubstate3Db* substate);
CALint calSteeringComputeBinaryOr3Di(struct CALModel3D* model, struct CALSubstate3Di* substate);
CALreal calSteeringComputeBinaryOr3Dr(struct CALModel3D* model, struct CALSubstate3Dr* substate);

/*! \brief 

	Set of functions that compute the "Logic Xor" of values of a substate.
*/
CALbyte calSteeringComputeLogicalXor3Db(struct CALModel3D* model, struct CALSubstate3Db* substate);
CALint calSteeringComputeLogicalXor3Di(struct CALModel3D* model, struct CALSubstate3Di* substate);
CALreal calSteeringComputeLogicalXor3Dr(struct CALModel3D* model, struct CALSubstate3Dr* substate);

/*! \brief 

	Set of functions that compute the "Binary Xor" of values of a substate.
*/
CALbyte calSteeringComputeBinaryXor3Db(struct CALModel3D* model, struct CALSubstate3Db* substate);
CALint calSteeringComputeBinaryXor3Di(struct CALModel3D* model, struct CALSubstate3Di* substate);
CALreal calSteeringComputeBinaryXor3Dr(struct CALModel3D* model, struct CALSubstate3Dr* substate);

/*! \brief 

	Private set of functions that execute the operation specified on a given substate.
	The user must not call directly this function, but instead call the functions specifed above.
*/
CALbyte calSteeringOperation3Db(struct CALModel3D* model, struct CALSubstate3Db* substate, enum REDUCTION_OPERATION operation);
CALint calSteeringOperation3Di(struct CALModel3D* model, struct CALSubstate3Di* substate, enum REDUCTION_OPERATION operation);
CALreal calSteeringOperation3Dr(struct CALModel3D* model, struct CALSubstate3Dr* substate, enum REDUCTION_OPERATION operation);

/*! \brief 

	Utility functions used instead of "calGet3D(i, j, k)" for retriving the cell(i, j, k).
	It is sufficiently one index instead two.
*/

CALbyte getValue3DbAtIndex(struct CALSubstate3Db* substate, CALint index);
CALint getValue3DiAtIndex(struct CALSubstate3Di* substate, CALint index);
CALreal getValue3DrAtIndex(struct CALSubstate3Dr* substate, CALint index);

#endif
