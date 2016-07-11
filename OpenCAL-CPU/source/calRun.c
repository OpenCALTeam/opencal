#include <OpenCAL-CPU/calRun.h>

static void calRunInitSimulation(struct CALModel* calModel)
{
    int n;

    for(n = 0; n < calModel->calRun->num_of_init_func; n++)
        calModel->calRun->init[n](calModel);
}

static void calRunFinalizeSimulation(struct CALModel* calModel)
{
    int n;

    for(n = 0; n < calModel->calRun->num_of_fin_func; n++)
        calModel->calRun->finalize[n](calModel);
}

struct CALRun*makeCALRun(int initial_step, int final_step)
{
    struct CALRun* simulation = (struct CALRun*)malloc(sizeof(struct CALRun));
    if (!simulation)
        return NULL;

    simulation->step = initial_step;
    simulation->initial_step = initial_step;
    simulation->final_step = final_step;

    simulation->init = NULL;
    simulation->globalTransition = NULL;
    simulation->stopCondition = NULL;
    simulation->finalize = NULL;

    return simulation;
}


CALbyte calRunCAStep(struct CALModel* calModel)
{
    if(calModel->calRun->globalTransition)
        calModel->calRun->globalTransition(calModel);
    else
    {
        int b;
        for (b=0; b<calModel->num_of_processes; b++)
        {
            //applying the b-th process
            if(calModel->model_processes->type == 'L')
                calRunApplyLocalProcess(calModel, calModel->model_processes[b].localProcess);
            else
                calModel->model_processes[b].globalProcess(calModel);

            //updating substates
            calUpdate(calModel);
        }
    }

    return(calModel->calRun->stopCondition(calModel));
}



void calApplyLocalProcess(struct CALModel* calModel, CALLocalProcess local_process)
{
    calRunApplyLocalProcess(calModel, local_process);
}

void calApplyGlobalProcess(struct CALModel* calModel, CALGlobalProcess global_process)
{
    global_process(calModel);
}

void calGlobalTransitionFunction(struct CALModel* calModel)
{
    calModel->calRun->globalTransition(calModel);
}

CALint calRunSimulation(struct CALModel* calModel)
{
    CALbyte again;

    calRunInitSimulation(calModel);
    struct CALRun* simulation = calModel->calRun;

    for (simulation->step = simulation->initial_step; (simulation->step <= simulation->final_step || simulation->final_step == CAL_RUN_LOOP); simulation->step++)
    {
        again = calRunCAStep(calModel);
        if (!again)
            break;
    }

    return 0;

}


void calRunApplyLocalProcess(struct CALModel* calModel, CALLocalProcess local_process)
{
    int n;
    int cellular_space_dimension = calModel->calIndexesPool->cellular_space_dimension;
    int number_of_dimensions = calModel->calIndexesPool->number_of_dimensions;
    CALIndices* pool = calModel->calIndexesPool->pool;

    for(n = 0; n < cellular_space_dimension; n++)
        local_process(calModel, pool[n], number_of_dimensions);

}
