#ifndef opencal_explicit
#define opencal_explicit
#include <OpenCAL-CPU/calModel.h>

/*! \brief Apply a local process to all the cellular space.
*/
void calApplyLocalProcess(struct CALModel* calModel,	//!< Pointer to the cellular automaton structure.
                          CALLocalProcess local_process //!< Pointer to a local function.
                          );

/*! \brief Updates all the substates registered in CALModel::pQb_array,
    CALModel::pQi_array and CALModel::pQr_array.
    It is called by the global transition function.
*/
void calUpdate(struct CALModel* calModel	//!< Pointer to the cellular automaton structure.
                 );

/*! \brief Copies the next matrix of a byte substate to the current one: current = next.
    If the active cells optimization is considered, it only updates the active cells.
*/
void calUpdateSubstate_b(struct CALModel* calModel,	//!< Pointer to the cellular automaton structure.
                         struct CALSubstate_b* Q	//!< Pointer to a byte substate.
                         );
/*! \brief Copies the next matrix of a integer substate to the current one: current = next.
    If the active cells optimization is considered, it only updates the active cells.
*/
void calUpdateSubstate_i(struct CALModel* calModel,	//!< Pointer to the cellular automaton structure.
                         struct CALSubstate_i* Q	//!< Pointer to a int substate.
                         );

/*! \brief Copies the next matrix of a real (floating point) substate to the current one: current = next.
    If the active cells optimization is considered, it only updates the active cells.
*/
void calUpdateSubstate_r(struct CALModel* calModel,	//!< Pointer to the cellular automaton structure.
                         struct CALSubstate_r* Q	//!< Pointer to a real (floating point) substate.
                         );

void calAddGlobalTransitionFunction(struct CALModel* calModel, void(*globalTransition)(struct CALModel*));


#endif
