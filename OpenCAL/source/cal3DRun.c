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

#include <cal3DRun.h>
#include <stdlib.h>
#include <stdio.h>

struct CALRun3D* calRunDef3D(struct CALModel3D* ca3D,
							 int initial_step, 
							 int final_step,
							 enum CALUpdateMode UPDATE_MODE)
{
	struct CALRun3D* simulation = (struct CALRun3D*)malloc(sizeof(struct CALRun3D));
	if (!simulation)
		return NULL;

	simulation->ca3D = ca3D;

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



void calRunAddInitFunc3D(struct CALRun3D* simulation, void (*init)(struct CALModel3D*))
{
	simulation->init = init;
}



void calRunAddGlobalTransitionFunc3D(struct CALRun3D* simulation, void (*globalTransition)(struct CALModel3D*))
{
	simulation->globalTransition = globalTransition;
}



void calRunAddSteeringFunc3D(struct CALRun3D* simulation, void (*steering)(struct CALModel3D*))
{
	simulation->steering = steering;
}



void calRunAddStopConditionFunc3D(struct CALRun3D* simulation, CALbyte (*stopCondition)(struct CALModel3D*))
{
	simulation->stopCondition = stopCondition;
}



void calRunAddFinalizeFunc3D(struct CALRun3D* simulation, void (*finalize)(struct CALModel3D*))
{
	simulation->finalize = finalize;
}



void calRunInitSimulation3D(struct CALRun3D* simulation)
{
	if (simulation->init)
	{
		simulation->init(simulation->ca3D);
		if (simulation->UPDATE_MODE == CAL_UPDATE_IMPLICIT)
			calUpdate3D(simulation->ca3D);
	}
}



CALbyte calRunCAStep3D(struct CALRun3D* simulation)
{
    if (simulation->globalTransition)
		{
			simulation->globalTransition(simulation->ca3D);
			if (simulation->UPDATE_MODE == CAL_UPDATE_IMPLICIT)
				calUpdate3D(simulation->ca3D);
		}
		else
			calGlobalTransitionFunction3D(simulation->ca3D);
            //No explicit substates and active cells updates are needed in this case
		
		if (simulation->steering)
		{
			simulation->steering(simulation->ca3D);
			if (simulation->UPDATE_MODE == CAL_UPDATE_IMPLICIT)
				calUpdate3D(simulation->ca3D);
		}

        if (simulation->stopCondition)
			if (simulation->stopCondition(simulation->ca3D)) 
				return CAL_FALSE;

        return CAL_TRUE;
}



void calRunFinalizeSimulation3D(struct CALRun3D* simulation)
{
	if (simulation->finalize)
	{
		simulation->finalize(simulation->ca3D);
		if (simulation->UPDATE_MODE == CAL_UPDATE_IMPLICIT)
			calUpdate3D(simulation->ca3D);
	}
}



void calRun3D(struct CALRun3D* simulation)
{
    CALbyte again;

	calRunInitSimulation3D(simulation);

	for (simulation->step = simulation->initial_step; (simulation->step <= simulation->final_step || simulation->final_step == CAL_RUN_LOOP); simulation->step++)
	{
		again = calRunCAStep3D(simulation);
        if (!again)
            break;		
	}

	calRunFinalizeSimulation3D(simulation);
}



void calRunFinalize3D(struct CALRun3D* cal3DRun)
{
	//Note that cal3DRun->ca3D MUST NOT BE DEALLOCATED as it is not allocated within cal3DRun.
	free(cal3DRun);
	cal3DRun = NULL;
}
