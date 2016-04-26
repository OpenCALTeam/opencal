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

#ifndef calRun_h
#define calRun_h

#include <OpenCAL++/calModelFunctor.h>
#include <OpenCAL++/calModel.h>


typedef CALModelFunctor<CALModel,void> InitFunction;
typedef CALModelFunctor<CALModel,void> SteeringFunction;
typedef CALModelFunctor<CALModel,calCommon :: CALbyte> StopConditionFunction;
typedef CALModelFunctor<CALModel,void> GlobalTransitionFunction;
typedef CALModelFunctor<CALModel,void> FinalizeFunction;

/*! \brief Class that defines the cellular automaton's simulation run specifications.
*/
class CALRun
{
private:
   CALModel* calModel;	//!< Pointer to the cellular automaton structure.

    int step;			//!< Current simulation step.
    int initial_step;	//!< Initial simulation step.
    int final_step;		//!< Final simulation step; if 0 the simulation becomes a loop.

    enum calCommon :: CALUpdateMode UPDATE_MODE;	//!< Callbacks substates' update mode; it can be CAL_UPDATE_EXPLICIT or CAL_UPDATE_IMPLICIT.

    InitFunction* init;								//!< Simulation's initialization callback functor.
    GlobalTransitionFunction* globalTransition;		//!< CA's globalTransition callback function. If defined, it is executed instead of CALModel::globalTransitionFunction.
    SteeringFunction* steering;						//!< Simulation's steering callback function.
    StopConditionFunction* stopCondition;			//!< Simulation's stopCondition callback function.
    FinalizeFunction* finalize;						//!< Simulation's finalize callback function.



public:

    /*! \brief CALRun's constructor, it defines the cellular automaton simulation structure.
    */

    CALRun (CALModel* calModel,			//!< Pointer to the cellular automaton structure.
            int initial_step,					//!< Initial simulation step; default value is 0.
            int final_step,					//!< Finale step; if it is 0, a loop is obtained. In order to set final_step to 0, the constant CAL_RUN_LOOP can be used.
            enum calCommon :: CALUpdateMode UPDATE_MODE		//!< Update mode: explicit on or explicit off (implicit).
            );

    /*! \brief CALRun's destructor.
    */
    ~CALRun ();

    /*! \brief Adds a simulation initialization function to CALRun.
    */
    void addInitFunc(InitFunction* init 		//!< Simulation's initialization callback function.
                             );

    /*! \brief Adds a CA's globalTransition callback function.
        If defined, it is executed instead of CALModel::globalTransitionFunction.
    */
    void addGlobalTransitionFunc(GlobalTransitionFunction* globalTransition	//!< CA's globalTransition callback function. If defined, it is executed instead of CALModel::globalTransitionFunction.
                                         );

    /*! \brief Adds a simulation steering function to CALRun.
    */
    void addSteeringFunc(SteeringFunction*	steering //!< Simulation's steering callback function.
                                 );

    /*! \brief Adds a stop condition function to CALRun.
    */
    void addStopConditionFunc(StopConditionFunction* stopCondition	//!< Simulation's stopCondition callback function.
                                      );

    /*! \brief Adds a finalization function to CALRun.
    */
    void addFinalizeFunc(FinalizeFunction* finalize	//!< Simulation's finalize callback function.
                                 );



    /*! \brief It executes the simulation initialization function.
    */
    void runInitSimulation();

    /*! \brief A single step of the cellular automaton. It executes the transition function, the steering and check for the stop condition.
    */
    calCommon :: CALbyte runCAStep();

    /*! \brief It executes the simulation finalization function.
    */
    void runFinalizeSimulation();

    /*! \brief Main simulation cicle. It can become a loop is CALRun::final_step == 0.
    */
    void run();


};


#endif
