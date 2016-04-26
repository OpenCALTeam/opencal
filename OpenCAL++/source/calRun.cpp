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

#include <OpenCAL++/calRun.h>

#include <stdlib.h>
#include <stdio.h>
#include <time.h>

CALRun :: CALRun(struct CALModel* calModel,
                 int initial_step,
                 int final_step,
                 enum calCommon :: CALUpdateMode UPDATE_MODE)
{
    this->calModel = calModel;

    this->step = 0;
    this->initial_step = initial_step;
    this->final_step = final_step;

    this->UPDATE_MODE = UPDATE_MODE;

    this->init = nullptr;
    this->globalTransition = nullptr;
    this->steering = nullptr;
    this->stopCondition = nullptr;
    this->finalize = nullptr;

}


CALRun :: ~ CALRun ()
{
    delete init;
    delete globalTransition;
    delete steering;
    delete stopCondition;
    delete finalize;
}


void CALRun :: addInitFunc(InitFunction* init)
{
    this->init = init;
}



void CALRun :: addGlobalTransitionFunc(GlobalTransitionFunction* globalTransition)
{
    this->globalTransition = globalTransition;
}



void CALRun :: addSteeringFunc(SteeringFunction* steering)
{
    this->steering = steering;
}



void CALRun :: addStopConditionFunc(StopConditionFunction* stopCondition)
{
    this->stopCondition = stopCondition;
}



void CALRun :: addFinalizeFunc(FinalizeFunction* finalize)
{
    this->finalize = finalize;
}



void CALRun :: runInitSimulation()
{
    if (this->init != nullptr)
    {
        this->init->run(this->calModel);
        if (this->UPDATE_MODE == calCommon :: CAL_UPDATE_IMPLICIT)
            this->calModel->update();
    }
}



calCommon :: CALbyte CALRun :: runCAStep(){

    if (this->globalTransition!= nullptr){
        this->globalTransition->run(this->calModel);
        if (this->UPDATE_MODE == calCommon :: CAL_UPDATE_IMPLICIT)
            this->calModel->update();
    }
    else{
        this->calModel->globalTransitionFunction();

    }
    //No explicit substates and active cells updates are needed in this case

    if (this->steering != nullptr)
    {
        this->steering->run(this->calModel);
        if (this->UPDATE_MODE == calCommon :: CAL_UPDATE_IMPLICIT)
            this->calModel->update();
    }

    if (this->stopCondition != nullptr)
        if (this->stopCondition->run(this->calModel))
            return CAL_FALSE;

    return CAL_TRUE;
}



void CALRun :: runFinalizeSimulation()
{
    if (this->finalize!= nullptr)
    {
        this->finalize->run(this->calModel);
        if (this->UPDATE_MODE == calCommon :: CAL_UPDATE_IMPLICIT)
            this->calModel->update();
    }
}



void CALRun :: run()
{
    calCommon :: CALbyte again;

    runInitSimulation();

    for (this->step = this->initial_step; (this->step <= this->final_step || this->final_step == CAL_RUN_LOOP); this->step++)
    {
        again = runCAStep();
        if (!again)
            break;
    }

    runFinalizeSimulation();
}