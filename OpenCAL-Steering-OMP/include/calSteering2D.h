#ifndef calSteering2D_h
#define calSteering2D_h

#include <cal2D.h>
#include <calSteeringCommon.h>

/*! \brief 

	Set of functions that compute the maximum value of a substate.
*/
CALbyte calSteeringComputeMax2Db(struct CALModel2D* model, struct CALSubstate2Db* substate);
CALint calSteeringComputeMax2Di(struct CALModel2D* model, struct CALSubstate2Di* substate);
CALreal calSteeringComputeMax2Dr(struct CALModel2D* model, struct CALSubstate2Dr* substate);

/*! \brief 

	Set of functions that compute the minimum value of a substate.
*/
CALbyte calSteeringComputeMin2Db(struct CALModel2D* model, struct CALSubstate2Db* substate);
CALint calSteeringComputeMin2Di(struct CALModel2D* model, struct CALSubstate2Di* substate);
CALreal calSteeringComputeMin2Dr(struct CALModel2D* model, struct CALSubstate2Dr* substate);

/*! \brief 

	Set of functions that compute the sum of values of a substate.
*/
CALbyte calSteeringComputeSum2Db(struct CALModel2D* model, struct CALSubstate2Db* substate);
CALint calSteeringComputeSum2Di(struct CALModel2D* model, struct CALSubstate2Di* substate);
CALreal calSteeringComputeSum2Dr(struct CALModel2D* model, struct CALSubstate2Dr* substate);

/*! \brief 

	Set of functions that compute the product of values of a substate.
*/
CALbyte calSteeringComputeProd2Db(struct CALModel2D* model, struct CALSubstate2Db* substate);
CALint calSteeringComputeProd2Di(struct CALModel2D* model, struct CALSubstate2Di* substate);
CALreal calSteeringComputeProd2Dr(struct CALModel2D* model, struct CALSubstate2Dr* substate);

/*! \brief 

	Set of functions that compute the "Logic And" of values of a substate.
*/
CALbyte calSteeringComputeLogicalAnd2Db(struct CALModel2D* model, struct CALSubstate2Db* substate);
CALint calSteeringComputeLogicalAnd2Di(struct CALModel2D* model, struct CALSubstate2Di* substate);
CALreal calSteeringComputeLogicalAnd2Dr(struct CALModel2D* model, struct CALSubstate2Dr* substate);

/*! \brief 

	Set of functions that compute the "Binary And" of values of a substate.
*/
CALbyte calSteeringComputeBinaryAnd2Db(struct CALModel2D* model, struct CALSubstate2Db* substate);
CALint calSteeringComputeBinaryAnd2Di(struct CALModel2D* model, struct CALSubstate2Di* substate);
CALreal calSteeringComputeBinaryAnd2Dr(struct CALModel2D* model, struct CALSubstate2Dr* substate);

/*! \brief 

	Set of functions that compute the "Logic Or" of values of a substate.
*/
CALbyte calSteeringComputeLogicalOr2Db(struct CALModel2D* model, struct CALSubstate2Db* substate);
CALint calSteeringComputeLogicalOr2Di(struct CALModel2D* model, struct CALSubstate2Di* substate);
CALreal calSteeringComputeLogicalOr2Dr(struct CALModel2D* model, struct CALSubstate2Dr* substate);

/*! \brief 

	Set of functions that compute the "Binary Or" of values of a substate.
*/
CALbyte calSteeringComputeBinaryOr2Db(struct CALModel2D* model, struct CALSubstate2Db* substate);
CALint calSteeringComputeBinaryOr2Di(struct CALModel2D* model, struct CALSubstate2Di* substate);
CALreal calSteeringComputeBinaryOr2Dr(struct CALModel2D* model, struct CALSubstate2Dr* substate);

/*! \brief 

	Set of functions that compute the "Logic Xor" of values of a substate.
*/
CALbyte calSteeringComputeLogicalXor2Db(struct CALModel2D* model, struct CALSubstate2Db* substate);
CALint calSteeringComputeLogicalXor2Di(struct CALModel2D* model, struct CALSubstate2Di* substate);
CALreal calSteeringComputeLogicalXor2Dr(struct CALModel2D* model, struct CALSubstate2Dr* substate);

/*! \brief 

	Set of functions that compute the "Binary Xor" of values of a substate.
*/
CALbyte calSteeringComputeBinaryXor2Db(struct CALModel2D* model, struct CALSubstate2Db* substate);
CALint calSteeringComputeBinaryXor2Di(struct CALModel2D* model, struct CALSubstate2Di* substate);
CALreal calSteeringComputeBinaryXor2Dr(struct CALModel2D* model, struct CALSubstate2Dr* substate);

/*! \brief 

	Private set of functions that execute the operation specified on a given substate.
	The user must not call directly this function, but instead call the functions specifed above.
*/
CALbyte calSteeringOperation2Db(struct CALModel2D* model, struct CALSubstate2Db* substate, enum STEERING_OPERATION operation);
CALint calSteeringOperation2Di(struct CALModel2D* model, struct CALSubstate2Di* substate, enum STEERING_OPERATION operation);
CALreal calSteeringOperation2Dr(struct CALModel2D* model, struct CALSubstate2Dr* substate, enum STEERING_OPERATION operation);

/*! \brief 

	Utility functions used instead of "calGet2D(i, j)" for retriving the cell(i, j).
	It is sufficiently one index instead two.
*/

CALbyte getValue2DbAtIndex(struct CALSubstate2Db* substate, CALint index);
CALint getValue2DiAtIndex(struct CALSubstate2Di* substate, CALint index);
CALreal getValue2DrAtIndex(struct CALSubstate2Dr* substate, CALint index);

#endif
