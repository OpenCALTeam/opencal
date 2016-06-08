#include <OpenCAL-CPU/calRun.h>

void (* calRunApplyLocalProcess)( struct CALModel* calModel, CALLocalProcess local_process );

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

void calUpdate(struct CALModel* calModel)
{

}
