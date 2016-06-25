﻿#ifndef cal_run
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
};


struct CALRun* makeCALRun(enum CALExecutionType executionType);

void calRunApplyLocalProcess( struct CALModel* calModel, CALLocalProcess local_process );

void calRunUpdate (struct CALModel* calModel);




///*! \brief The cellular automaton global transition function.
//    It applies the transition function to each cell of the cellular space.
//    After each local process, a global substates update is performed.
//*/
//void calGlobalTransitionFunction(struct CALModel* calModel	//!< Pointer to the cellular automaton structure.
//                                 );



CALint calRunSimulation(struct CALModel* calModel);
CALbyte calRunCAStep(struct CALModel* calModel);








#endif