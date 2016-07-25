#ifndef cal_run
#define cal_run

#include <OpenCAL-CPU/calModel.h>

struct CALRun {
        int step;			//!< Current simulation step.
        int initial_step;	//!< Initial simulation step.
        int final_step;		//!< Final simulation step; if 0 the simulation becomes a loop.

        void (**init)(struct CALModel*);				//!< Simulation's initialization callback function.
        int num_of_init_func;
        void (*globalTransition)(struct CALModel*);	//!< CA's globalTransition callback function. If defined, it is executed instead of cal2D.c::calGlobalTransitionFunction2D.
        CALbyte (*stopCondition)(struct CALModel*);	//!< Simulation's stopCondition callback function.
        void (**finalize)(struct CALModel*);			//!< Simulation's finalize callback function.
        int num_of_fin_func;
        CALbyte (*active_cells_def)(struct CALModel* calModel, CALIndices cell, int number_of_dimensions);
        CALbyte (*rem_active_cells)(struct CALModel* calModel, CALIndices cell, int number_of_dimensions);

#if CAL_PARALLEL == 1
        CAL_LOCKS_DEFINE(locks);
#endif
};


struct CALRun* makeCALRun(int initial_step, int final_step);

//#include <OpenCAL-CPU/calActiveCells.h>
void calAddActiveCellsDefinition(struct CALModel* calModel, CALbyte (*active_cells_def)(struct CALModel* calModel, CALIndices cell, int number_of_dimensions));
void calAddInactiveCellsDefinition(struct CALModel* calModel, CALbyte (*active_cells_def)(struct CALModel* calModel, CALIndices cell, int number_of_dimensions));

void calRunApplyLocalProcess(struct CALModel* calModel, CALLocalProcess local_process);

///*! \brief The cellular automaton global transition function.
//    It applies the transition function to each cell of the cellular space.
//    After each local process, a global substates update is performed.
//*/
//void calGlobalTransitionFunction(struct CALModel* calModel	//!< Pointer to the cellular automaton structure.
//                                 );



CALint calRunSimulation(struct CALModel* calModel);
CALbyte calRunCAStep(struct CALModel* calModel);

void calForceInit(struct CALModel* calModel);

#define calJumpToNextStep(calModel)(calModel->calRun->step++)
#define calGetCurrentStep(calModel)(calModel->calRun->step)








#endif
