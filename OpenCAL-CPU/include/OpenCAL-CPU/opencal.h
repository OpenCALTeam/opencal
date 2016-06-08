#ifndef opencal_implicit
#define opencal_implicit
#include <OpenCAL-CPU/calModel.h>

/*! \brief Adds a local function to the CALModel::modelFunctions array.
*/
void calAddLocalProcess(struct CALModel* calModel,	//!< Pointer to the cellular automaton structure.
                        CALLocalProcess elementary_process //!< Pointer to a transition function's elementary process.
                        );

/*! \brief Adds a global function to the CALModel::modelFunctions array.
*/
void calAddGlobalProcess(struct CALModel* calModel,	//!< Pointer to the cellular automaton structure.
                         CALGlobalProcess elementary_process //!< Pointer to a global function.
                         );

void calAddInitFunc(struct CALModel* calModel, void(*init)(struct CALModel* calModel));

void calAddStopCondition(struct CALModel* calModel, CALbyte(*stopCondition)(struct CALModel* calModel));

void calAddFinalizeFunc(struct CALModel* calModel, void(*finalize)(struct CALModel* calModel) );


#endif
