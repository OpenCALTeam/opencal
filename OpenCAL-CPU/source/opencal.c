#include <OpenCAL-CPU/opencal.h>

void calAddLocalProcess(struct CALModel* calModel,	//!< Pointer to the cellular automaton structure.
                        CALLocalProcess elementary_process //!< Pointer to a transition function's elementary process.
                        )
{
    struct CALProcess* prv = calModel->model_processes;
    calModel->num_of_processes++;
    calModel->model_processes = (struct CALProcess*) malloc(sizeof(struct CALProcess) * (calModel->num_of_processes));

    int n = 0;

    if(calModel->num_of_processes > 1)
    {
        for( ; n < calModel->num_of_processes - 1; n++)

            calModel->model_processes[n] = prv[n];
        free(prv);
    }
    calModel->model_processes[n].localProcess = elementary_process;
    calModel->model_processes[n].type = 'L';

}

void calAddGlobalProcess(struct CALModel* calModel,	//!< Pointer to the cellular automaton structure.
                         CALGlobalProcess elementary_process //!< Pointer to a global function.
                         )
{
    struct CALProcess* prv = calModel->model_processes;
    calModel->model_processes = (struct CALProcess*) malloc(sizeof(struct CALProcess) * ++calModel->num_of_processes);

    int n = 0;

    for( ; n < calModel->num_of_processes - 1; n++)
        calModel->model_processes[n] = prv[n];

    calModel->model_processes[n].globalProcess = elementary_process;
    calModel->model_processes[n].type = 'G';

    free(prv);
}

void calAddInitFunc(struct CALModel* calModel, void (*init)(struct CALModel*))
{
    if(!calModel->calRun->init)
        calModel->calRun->init = (void (**)(struct CALModel*)) malloc(sizeof(void (*)(struct CALModel*)));
    void (**prv)(struct CALModel*) = calModel->calRun->init;
    calModel->calRun->init = (void (**)(struct CALModel*)) malloc(sizeof(void (*)(struct CALModel*)) * ++calModel->calRun->num_of_init_func);

    int n = 0;

    for( ; n < calModel->calRun->num_of_init_func - 1; n++)
        calModel->calRun->init[n] = prv[n];

    calModel->calRun->init[n] = init;

    free(prv);
}

void calAddStopCondition(struct CALModel* calModel, CALbyte (*stopCondition)(struct CALModel*))
{
    calModel->calRun->stopCondition = stopCondition;
}

void calAddFinalizeFunc(struct CALModel* calModel, void (*finalize)(struct CALModel*))
{
    void (**prv)(struct CALModel*) = calModel->calRun->finalize;
    calModel->calRun->finalize = (void (**)(struct CALModel*)) malloc(sizeof(void (*)(struct CALModel*)) * ++calModel->calRun->num_of_fin_func);

    int n = 0;

    for( ; n < calModel->calRun->num_of_fin_func - 1; n++)
        calModel->calRun->finalize[n] = prv[n];

    free(prv);
}
