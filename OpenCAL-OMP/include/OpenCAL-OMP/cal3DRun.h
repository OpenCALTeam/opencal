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

#ifndef cal3DRun_h
#define cal3DRun_h

#include <OpenCAL-OMP/cal3D.h>



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
DllExport
struct CALRun3D* calRunDef3D(struct CALModel3D* ca3D,			//!< Pointer to the cellular automaton structure.
							 int initial_step,					//!< Initial simulation step; default value is 0.
							 int final_step,					//!< Finale step; if it is 0, a loop is obtained. In order to set final_step to 0, the constant CAL_RUN_LOOP can be used.
							 enum CALUpdateMode UPDATE_MODE		//!< Update mode: explicit on or explicit off (implicit).
							 );	



/*! \brief Adds a simulation initialization function to CALRun3D.
*/
DllExport
void calRunAddInitFunc3D(struct CALRun3D* simulation,			//!< Pointer to the run structure.
						 void (*init)(struct CALModel3D*)		//!< Simulation's initialization callback function.
						 );

/*! \brief Adds a CA's globalTransition callback function.
	If defined, it is executed instead of cal3D.c::calGlobalTransitionFunction3D.
*/
DllExport
void calRunAddGlobalTransitionFunc3D(struct CALRun3D* simulation,					//!< Pointer to the run structure.
									 void (*globalTransition)(struct CALModel3D*)	//!< CA's globalTransition callback function. If defined, it is executed instead of cal3D.c::calGlobalTransitionFunction3D.
									 );

/*! \brief Adds a simulation steering function to CALRun3D.
*/
DllExport
void calRunAddSteeringFunc3D(struct CALRun3D* simulation,			//!< Pointer to the run structure.
							 void (*steering)(struct CALModel3D*)	//!< Simulation's steering callback function.
							 );

/*! \brief Adds a stop condition function to CALRun3D.
*/
DllExport
void calRunAddStopConditionFunc3D(struct CALRun3D* simulation,					//!< Pointer to the run structure.
								  CALbyte (*stopCondition)(struct CALModel3D*)	//!< Simulation's stopCondition callback function.
								  );

/*! \brief Adds a finalization function to CALRun3D.
*/
DllExport
void calRunAddFinalizeFunc3D(struct CALRun3D* simulation,			//!< Pointer to the run structure.
							 void (*finalize)(struct CALModel3D*)	//!< Simulation's finalize callback function.
							 );

DllExport
void calRunInitSimulation3D(struct CALRun3D* simulation	//!< Pointer to the run structure.
	);


/*! \brief A single step of the cellular automaton. It execute the transition function, the steering and check for the stop condition.
*/
DllExport
CALbyte calRunCAStep3D(struct CALRun3D* simulation  //!< Pointer to the run structure.
					   );

DllExport
void calRunFinalizeSimulation3D(struct CALRun3D* simulation	//!< Pointer to the run structure.
								);


/*! \brief Main simulation cicle. It can become a loop is CALRun3D::final_step == 0.
*/
DllExport
void calRun3D(struct CALRun3D* simulation			//!< Pointer to the run structure.
			  );



/*! \brief Finalization function. It releases the allocated memory.
*/
DllExport
void calRunFinalize3D(struct CALRun3D* cal3DRun		//!< Pointer to the run structure.
					  );


#endif
