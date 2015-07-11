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

#include <cal2DRun.h>
#include <stdlib.h>
#include <stdio.h>

struct CALRun2D* calRunDef2D(struct CALModel2D* ca2D,
	int initial_step,
	int final_step,
enum CALUpdateMode UPDATE_MODE)
{
	struct CALRun2D* simulation = (struct CALRun2D*)malloc(sizeof(struct CALRun2D));
	if (!simulation)
		return NULL;

	simulation->ca2D = ca2D;

	simulation->step = 0;
	simulation->initial_step = initial_step;
	simulation->final_step = final_step;

	simulation->UPDATE_MODE = UPDATE_MODE;

	simulation->init = NULL;
	simulation->globalTransition = NULL;
	simulation->steering = NULL;
	simulation->stopCondition = NULL;
	simulation->finalize = NULL;

	return simulation;
}



void calRunAddInitFunc2D(struct CALRun2D* simulation, void(*init)(struct CALModel2D*))
{
	simulation->init = init;
}



void calRunAddGlobalTransitionFunc2D(struct CALRun2D* simulation, void(*globalTransition)(struct CALModel2D*))
{
	simulation->globalTransition = globalTransition;
}



void calRunAddSteeringFunc2D(struct CALRun2D* simulation, void(*steering)(struct CALModel2D*))
{
	simulation->steering = steering;
}



void calRunAddStopConditionFunc2D(struct CALRun2D* simulation, CALbyte(*stopCondition)(struct CALModel2D*))
{
	simulation->stopCondition = stopCondition;
}



void calRunAddFinalizeFunc2D(struct CALRun2D* simulation, void(*finalize)(struct CALModel2D*))
{
	simulation->finalize = finalize;
}


void calRunInitSimulation2D(struct CALRun2D* simulation)
{
	if (simulation->init)
	{
		simulation->init(simulation->ca2D);
		if (simulation->UPDATE_MODE == CAL_UPDATE_IMPLICIT)
			calUpdate2D(simulation->ca2D);
	}
}


CALbyte calRunCAStep2D(struct CALRun2D* simulation)
{
	if (simulation->globalTransition)
	{
		simulation->globalTransition(simulation->ca2D);
		if (simulation->UPDATE_MODE == CAL_UPDATE_IMPLICIT)
			calUpdate2D(simulation->ca2D);
	}
	else
		calGlobalTransitionFunction2D(simulation->ca2D);
	//No explicit substates and active cells updates are needed in this case

	if (simulation->steering)
	{
		simulation->steering(simulation->ca2D);
		if (simulation->UPDATE_MODE == CAL_UPDATE_IMPLICIT)
			calUpdate2D(simulation->ca2D);
	}

	if (simulation->stopCondition)
		if (simulation->stopCondition(simulation->ca2D))
			return CAL_FALSE;

	return CAL_TRUE;
}


void calRunFinalizeSimulation2D(struct CALRun2D* simulation)
{
	if (simulation->finalize)
	{
		simulation->finalize(simulation->ca2D);
		if (simulation->UPDATE_MODE == CAL_UPDATE_IMPLICIT)
			calUpdate2D(simulation->ca2D);
	}
}


void calRun2D(struct CALRun2D* simulation)
{
	CALbyte again;

	calRunInitSimulation2D(simulation);

	for (simulation->step = simulation->initial_step; (simulation->step <= simulation->final_step || simulation->final_step == CAL_RUN_LOOP); simulation->step++)
	{
		again = calRunCAStep2D(simulation);
		if (!again)
			break;
	}

	calRunFinalizeSimulation2D(simulation);
}


void calRunFinalize2D(struct CALRun2D* cal2DRun)
{
	//Note that cal2DRun->ca2D MUST NOT BE DEALLOCATED as it is not allocated within cal2DRun.
	free(cal2DRun);
	cal2DRun = NULL;
}
