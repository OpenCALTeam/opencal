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

#ifndef cal3DRun_h
#define cal3DRun_h

#include <OpenCAL/cal3D.h>



/*! \brief Structure that defines the cellular automaton's simulation run specifications.
*/
struct CALRun3D
{
	struct CALModel3D* ca3D;	//!< Pointer to the cellular automaton structure.
	
	int step;			//!< Current simulation step.
	int initial_step;	//!< Initial simulation step.
	int final_step;		//!< Final simulation step; if 0 the simulation becomes a loop.
	
	enum CALUpdateMode UPDATE_MODE;	//!< Callbacks substates' update mode; it can be CAL_UPDATE_EXPLICIT or CAL_UPDATE_IMPLICIT.

	void (*init)(struct CALModel3D*);				//!< Simulation's initialization callback function.
	void (*globalTransition)(struct CALModel3D*);	//!< CA's globalTransition callback function. If defined, it is executed instead of cal3D.c::calGlobalTransitionFunction3D.
	void (*steering)(struct CALModel3D*);			//!< Simulation's steering callback function.
	CALbyte (*stopCondition)(struct CALModel3D*);	//!< Simulation's stopCondition callback function.
	void (*finalize)(struct CALModel3D*);			//!< Simulation's finalize callback function.
};



/*! \brief Creates an object of type calRunDef3D, sets its records and returns it as a pointer; it defines the cellular automaton simulation structure.
*/
struct CALRun3D* calRunDef3D(struct CALModel3D* ca3D,			//!< Pointer to the cellular automaton structure.
							 int initial_step,					//!< Initial simulation step; default value is 0.
							 int final_step,					//!< Finale step; if it is 0, a loop is obtained. In order to set final_step to 0, the constant CAL_RUN_LOOP can be used.
							 enum CALUpdateMode UPDATE_MODE		//!< Update mode: explicit on or explicit off (implicit).
							 );	



/*! \brief Adds a simulation initialization function to CALRun3D.
*/
void calRunAddInitFunc3D(struct CALRun3D* simulation,			//!< Pointer to the run structure.
						 void (*init)(struct CALModel3D*)		//!< Simulation's initialization callback function.
						 );

/*! \brief Adds a CA's globalTransition callback function.
	If defined, it is executed instead of cal3D.c::calGlobalTransitionFunction3D.
*/
void calRunAddGlobalTransitionFunc3D(struct CALRun3D* simulation,					//!< Pointer to the run structure.
									 void (*globalTransition)(struct CALModel3D*)	//!< CA's globalTransition callback function. If defined, it is executed instead of cal3D.c::calGlobalTransitionFunction3D.
									 );

/*! \brief Adds a simulation steering function to CALRun3D.
*/
void calRunAddSteeringFunc3D(struct CALRun3D* simulation,			//!< Pointer to the run structure.
							 void (*steering)(struct CALModel3D*)	//!< Simulation's steering callback function.
							 );

/*! \brief Adds a stop condition function to CALRun3D.
*/
void calRunAddStopConditionFunc3D(struct CALRun3D* simulation,					//!< Pointer to the run structure.
								  CALbyte (*stopCondition)(struct CALModel3D*)	//!< Simulation's stopCondition callback function.
								  );

/*! \brief Adds a finalization function to CALRun3D.
*/
void calRunAddFinalizeFunc3D(struct CALRun3D* simulation,			//!< Pointer to the run structure.
							 void (*finalize)(struct CALModel3D*)	//!< Simulation's finalize callback function.
							 );



/*! \brief It executes the simulation initialization function.
*/
void calRunInitSimulation3D(struct CALRun3D* simulation	//!< Pointer to the run structure.
	);

/*! \brief A single step of the cellular automaton. It executes the transition function, the steering and check for the stop condition.
*/
CALbyte calRunCAStep3D(struct CALRun3D* simulation  //!< Pointer to the run structure.
					   );

/*! \brief It executes the simulation finalization function.
*/
void calRunFinalizeSimulation3D(struct CALRun3D* simulation	//!< Pointer to the run structure.
								);

/*! \brief Main simulation cicle. It can become a loop is CALRun3D::final_step == 0.
*/
void calRun3D(struct CALRun3D* simulation			//!< Pointer to the run structure.
			  );



/*! \brief Finalization function. It releases the allocated memory.
*/
void calRunFinalize3D(struct CALRun3D* cal3DRun		//!< Pointer to the run structure.
					  );


#endif
