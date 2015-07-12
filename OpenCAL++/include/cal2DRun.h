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

#ifndef cal2DRun_h
#define cal2DRun_h

#include <cal2D.h>

typedef CalModelFunctor<CALModel2D,void> InitFunction2D;
typedef CalModelFunctor<CALModel2D,void> SteeringFunction2D;
typedef CalModelFunctor<CALModel2D,CALbyte> StopConditionFunction2D;
typedef CalModelFunctor<CALModel2D,void> GlobalTransitionFunction2D;
typedef CalModelFunctor<CALModel2D,void> FinalizeFunction2D;


/*! \brief Structure that defines the cellular automaton's simulation run specifications.
*/
struct CALRun2D
{
	struct CALModel2D* ca2D;	//!< Pointer to the cellular automaton structure.
	
	int step;			//!< Current simulation step.
	int initial_step;	//!< Initial simulation step.
	int final_step;		//!< Final simulation step; if 0 the simulation becomes a loop.
	
	enum CALUpdateMode UPDATE_MODE;	//!< Callbacks substates' update mode; it can be CAL_UPDATE_EXPLICIT or CAL_UPDATE_IMPLICIT.

	InitFunction2D* init;								//!< Simulation's initialization callback functor.
	GlobalTransitionFunction2D* globalTransition;		//!< CA's globalTransition callback function. If defined, it is executed instead of cal2D.c::calGlobalTransitionFunction2D.
	SteeringFunction2D* steering;						//!< Simulation's steering callback function.
	StopConditionFunction2D* stopCondition;			//!< Simulation's stopCondition callback function.
	FinalizeFunction2D* finalize;						//!< Simulation's finalize callback function.
};



/*! \brief Creates an object of type calRunDef2D, sets its records and returns it as a pointer; it defines the cellular automaton simulation structure.
*/
struct CALRun2D* calRunDef2D(struct CALModel2D* ca2D,			//!< Pointer to the cellular automaton structure.
							 int initial_step,					//!< Initial simulation step; default value is 0.
							 int final_step,					//!< Finale step; if it is 0, a loop is obtained. In order to set final_step to 0, the constant CAL_RUN_LOOP can be used.
							 enum CALUpdateMode UPDATE_MODE		//!< Update mode: explicit on or explicit off (implicit).
							 );	



/*! \brief Adds a simulation initialization function to CALRun2D.
*/
void calRunAddInitFunc2D(struct CALRun2D* simulation,			//!< Pointer to the run structure.
						 InitFunction2D*		//!< Simulation's initialization callback function.
						 );

/*! \brief Adds a CA's globalTransition callback function.
	If defined, it is executed instead of cal2D.c::calGlobalTransitionFunction2D.
*/
void calRunAddGlobalTransitionFunc2D(struct CALRun2D* simulation,					//!< Pointer to the run structure.
									 GlobalTransitionFunction2D*	//!< CA's globalTransition callback function. If defined, it is executed instead of cal2D.c::calGlobalTransitionFunction2D.
									 );

/*! \brief Adds a simulation steering function to CALRun2D.
*/
void calRunAddSteeringFunc2D(struct CALRun2D* simulation,			//!< Pointer to the run structure.
							 SteeringFunction2D*	//!< Simulation's steering callback function.
							 );

/*! \brief Adds a stop condition function to CALRun2D.
*/
void calRunAddStopConditionFunc2D(struct CALRun2D* simulation,					//!< Pointer to the run structure.
								  StopConditionFunction2D*	//!< Simulation's stopCondition callback function.
								  );

/*! \brief Adds a finalization function to CALRun2D.
*/
void calRunAddFinalizeFunc2D(struct CALRun2D* simulation,			//!< Pointer to the run structure.
							 FinalizeFunction2D*	//!< Simulation's finalize callback function.
							 );



/*! \brief It executes the simulation initialization function.
*/
void calRunInitSimulation2D(struct CALRun2D* simulation	//!< Pointer to the run structure.
							);


/*! \brief A single step of the cellular automaton. It executes the transition function, the steering and check for the stop condition.
*/
CALbyte calRunCAStep2D(struct CALRun2D* simulation  //!< Pointer to the run structure.
					   );

/*! \brief It executes the simulation finalization function.
*/
void calRunFinalizeSimulation2D(struct CALRun2D* simulation	//!< Pointer to the run structure.
								);

/*! \brief Main simulation cicle. It can become a loop is CALRun2D::final_step == 0.
*/
void calRun2D(struct CALRun2D* simulation	//!< Pointer to the run structure.
			  );



/*! \brief Finalization function. It releases the allocated memory.
*/
void calRunFinalize2D(struct CALRun2D* cal2DRun		//!< Pointer to the run structure.
					  );


#endif
