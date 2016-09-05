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

#ifndef cal3DReduction_h
#define cal3DReduction_h

#include <OpenCAL-OMP/cal3D.h>

/*! \brief 

	Set of functions that compute the maximum value of a substate.
*/
DllExport
CALbyte calReductionComputeMax3Db(struct CALModel3D* model, struct CALSubstate3Db* substate);
DllExport
CALint calReductionComputeMax3Di(struct CALModel3D* model, struct CALSubstate3Di* substate);
DllExport
CALreal calReductionComputeMax3Dr(struct CALModel3D* model, struct CALSubstate3Dr* substate);

/*! \brief 

	Set of functions that compute the minimum value of a substate.
*/
DllExport
CALbyte calReductionComputeMin3Db(struct CALModel3D* model, struct CALSubstate3Db* substate);
DllExport
CALint calReductionComputeMin3Di(struct CALModel3D* model, struct CALSubstate3Di* substate);
DllExport
CALreal calReductionComputeMin3Dr(struct CALModel3D* model, struct CALSubstate3Dr* substate);

/*! \brief 

	Set of functions that compute the sum of values of a substate.
*/
DllExport
CALbyte calReductionComputeSum3Db(struct CALModel3D* model, struct CALSubstate3Db* substate);
DllExport
CALint calReductionComputeSum3Di(struct CALModel3D* model, struct CALSubstate3Di* substate);
DllExport
CALreal calReductionComputeSum3Dr(struct CALModel3D* model, struct CALSubstate3Dr* substate);

/*! \brief 

	Set of functions that compute the product of values of a substate.
*/
DllExport
CALbyte calReductionComputeProd3Db(struct CALModel3D* model, struct CALSubstate3Db* substate);
DllExport
CALint calReductionComputeProd3Di(struct CALModel3D* model, struct CALSubstate3Di* substate);
DllExport
CALreal calReductionComputeProd3Dr(struct CALModel3D* model, struct CALSubstate3Dr* substate);

/*! \brief 

	Set of functions that compute the "Logic And" of values of a substate.
*/
DllExport
CALbyte calReductionComputeLogicalAnd3Db(struct CALModel3D* model, struct CALSubstate3Db* substate);
DllExport
CALint calReductionComputeLogicalAnd3Di(struct CALModel3D* model, struct CALSubstate3Di* substate);
DllExport
CALreal calReductionComputeLogicalAnd3Dr(struct CALModel3D* model, struct CALSubstate3Dr* substate);

/*! \brief 

	Set of functions that compute the "Binary And" of values of a substate.
*/
DllExport
CALbyte calReductionComputeBinaryAnd3Db(struct CALModel3D* model, struct CALSubstate3Db* substate);
DllExport
CALint calReductionComputeBinaryAnd3Di(struct CALModel3D* model, struct CALSubstate3Di* substate);
DllExport
CALreal calReductionComputeBinaryAnd3Dr(struct CALModel3D* model, struct CALSubstate3Dr* substate);

/*! \brief 

	Set of functions that compute the "Logic Or" of values of a substate.
*/
DllExport
CALbyte calReductionComputeLogicalOr3Db(struct CALModel3D* model, struct CALSubstate3Db* substate);
DllExport
CALint calReductionComputeLogicalOr3Di(struct CALModel3D* model, struct CALSubstate3Di* substate);
DllExport
CALreal calReductionComputeLogicalOr3Dr(struct CALModel3D* model, struct CALSubstate3Dr* substate);

/*! \brief 

	Set of functions that compute the "Binary Or" of values of a substate.
*/
DllExport
CALbyte calReductionComputeBinaryOr3Db(struct CALModel3D* model, struct CALSubstate3Db* substate);
DllExport
CALint calReductionComputeBinaryOr3Di(struct CALModel3D* model, struct CALSubstate3Di* substate);
DllExport
CALreal calReductionComputeBinaryOr3Dr(struct CALModel3D* model, struct CALSubstate3Dr* substate);

/*! \brief 

	Set of functions that compute the "Logic Xor" of values of a substate.
*/
DllExport
CALbyte calReductionComputeLogicalXor3Db(struct CALModel3D* model, struct CALSubstate3Db* substate);
DllExport
CALint calReductionComputeLogicalXor3Di(struct CALModel3D* model, struct CALSubstate3Di* substate);
DllExport
CALreal calReductionComputeLogicalXor3Dr(struct CALModel3D* model, struct CALSubstate3Dr* substate);

/*! \brief 

	Set of functions that compute the "Binary Xor" of values of a substate.
*/
DllExport
CALbyte calReductionComputeBinaryXor3Db(struct CALModel3D* model, struct CALSubstate3Db* substate);
DllExport
CALint calReductionComputeBinaryXor3Di(struct CALModel3D* model, struct CALSubstate3Di* substate);
DllExport
CALreal calReductionComputeBinaryXor3Dr(struct CALModel3D* model, struct CALSubstate3Dr* substate);

/*! \brief 

	Private set of functions that execute the operation specified on a given substate.
	The user must not call directly this function, but instead call the functions specifed above.
*/
DllExport
CALbyte calReductionOperation3Db(struct CALModel3D* model, struct CALSubstate3Db* substate, enum REDUCTION_OPERATION operation);
DllExport
CALint calReductionOperation3Di(struct CALModel3D* model, struct CALSubstate3Di* substate, enum REDUCTION_OPERATION operation);
DllExport
CALreal calReductionOperation3Dr(struct CALModel3D* model, struct CALSubstate3Dr* substate, enum REDUCTION_OPERATION operation);

/*! \brief 

	Utility functions used instead of "calGet3D(i, j, k)" for retriving the cell(i, j, k).
	It is sufficiently one index instead two.
*/

DllExport
CALbyte getValue3DbAtIndex(struct CALSubstate3Db* substate, CALint index);
DllExport
CALint getValue3DiAtIndex(struct CALSubstate3Di* substate, CALint index);
DllExport
CALreal getValue3DrAtIndex(struct CALSubstate3Dr* substate, CALint index);

#endif
