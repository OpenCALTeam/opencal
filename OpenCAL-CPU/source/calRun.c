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
            calRunUpdate(calModel);
        }
    }
}


struct CALRun*makeCALRun(enum CALExecutionType executionType)
{

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

}

void calRunUpdate(struct CALModel* calModel)
{

}
