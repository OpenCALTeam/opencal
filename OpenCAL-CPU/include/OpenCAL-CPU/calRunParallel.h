#ifndef cal_run_parallel
#define cal_run_parallel

#include <OpenCAL-CPU/calRun.h>

void calParallelApplyLocalProcess( struct CALModel* calModel, CALLocalProcess local_process );

void calParallelUpdate (struct CALModel* calModel);


CALbyte calParallelGet_b(struct CALModel* calModel, struct CALSubstate_b* Q, CALIndexes indexes);
CALint calParallelGet_i(struct CALModel* calModel, struct CALSubstate_i* Q, CALIndexes indexes);
CALreal  calParallelGet_r(struct CALModel* calModel, struct CALSubstate_r* Q, CALIndexes indexes);

CALbyte  calParallelGetX_b(struct CALModel* calModel, struct CALSubstate_b* Q, CALIndexes central_cell, int n);
CALint  calParallelGetX_i(struct CALModel* calModel, struct CALSubstate_i* Q, CALIndexes central_cell, int n);
CALreal  calParallelGetX_r(struct CALModel* calModel, struct CALSubstate_r* Q, CALIndexes central_cell, int n);

void  calParallelSet_b(struct CALModel* calModel, struct CALSubstate_b* Q, CALIndexes central_cell,CALbyte value);
void  calParallelSet_i(struct CALModel* calModel, struct CALSubstate_i* Q, CALIndexes central_cell,CALbyte value);
void  calParallelSet_r(struct CALModel* calModel, struct CALSubstate_r* Q, CALIndexes central_cell,CALbyte value);

void  calParallelSetCurrent_b(struct CALModel* calModel, struct CALSubstate_b* Q, CALIndexes central_cell, CALbyte value);
void  calParallelSetCurrent_i(struct CALModel* calModel, struct CALSubstate_i* Q, CALIndexes central_cell, CALbyte value);
void  calParallelSetCurrent_r(struct CALModel* calModel, struct CALSubstate_r* Q, CALIndexes central_cell, CALbyte value);

void  calParallelCopyBuffer_b(CALbyte* M_src, CALbyte* M_dest, int buffer_dimension);
void  calParallelCopyBuffer_i(CALint* M_src, CALint* M_dest, int buffer_dimension);
void  calParallelCopyBuffer_r(CALreal* M_src, CALreal* M_dest, int buffer_dimension);

void  calParallelAddBuffer_b(CALbyte* M_op1, CALbyte* M_op2,  CALbyte* M_dest, int buffer_dimension);
void  calParallelAddBuffer_i(CALint* M_op1, CALint* M_op2,  CALint* M_dest, int buffer_dimension);
void  calParallelAddBuffer_r(CALreal* M_op1, CALreal* M_op2,  CALreal* M_dest, int buffer_dimension);

void  calParallelSubtractBuffer_b(CALbyte* M_op1, CALbyte* M_op2,  CALbyte* M_dest, int buffer_dimension);
void  calParallelSubtractBuffer_i(CALint* M_op1, CALint* M_op2,  CALint* M_dest, int buffer_dimension);
void  calParallelSubtractBuffer_r(CALreal* M_op1, CALreal* M_op2,  CALreal* M_dest, int buffer_dimension);


#endif

