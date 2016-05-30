#ifndef cal_run
#define cal_run

#include <OpenCAL-CPU/calModel.h>

enum CALExecutionType {SERIAL = 0, PARALLEL};

struct CALRun {

        void (* calApplyLocalProcess)( struct CALModel* calModel, CALLocalFunction local_process );

        void (* calUpdate) (struct CALModel* calModel);


        CALbyte (* calGet_b)(struct CALModel* calModel, struct CALSubstate_b* Q, CALIndexes indexes);
        CALint (* calGet_i)(struct CALModel* calModel, struct CALSubstate_i* Q, CALIndexes indexes);
        CALreal (* calGet_r)(struct CALModel* calModel, struct CALSubstate_r* Q, CALIndexes indexes);

        CALbyte (* calGetX_b)(struct CALModel* calModel, struct CALSubstate_b* Q, CALIndexes central_cell, int n);
        CALint (* calGetX_i)(struct CALModel* calModel, struct CALSubstate_i* Q, CALIndexes central_cell, int n);
        CALreal (* calGetX_r)(struct CALModel* calModel, struct CALSubstate_r* Q, CALIndexes central_cell, int n);

        void (* calSet_b)(struct CALModel* calModel, struct CALSubstate_b* Q, CALIndexes central_cell,CALbyte value);
        void (* calSet_i)(struct CALModel* calModel, struct CALSubstate_i* Q, CALIndexes central_cell,CALbyte value);
        void (* calSet_r)(struct CALModel* calModel, struct CALSubstate_r* Q, CALIndexes central_cell,CALbyte value);

        void (* calSetCurrent_b)(struct CALModel* calModel, struct CALSubstate_b* Q, CALIndexes central_cell, CALbyte value);
        void (* calSetCurrent_i)(struct CALModel* calModel, struct CALSubstate_i* Q, CALIndexes central_cell, CALbyte value);
        void (* calSetCurrent_r)(struct CALModel* calModel, struct CALSubstate_r* Q, CALIndexes central_cell, CALbyte value);

        void (* calCopyBuffer_b)(CALbyte* M_src, CALbyte* M_dest, int buffer_dimension);
        void (* calCopyBuffer_i)(CALint* M_src, CALint* M_dest, int buffer_dimension);
        void (* calCopyBuffer_r)(CALreal* M_src, CALreal* M_dest, int buffer_dimension);

        void (* calAddBuffer_b)(CALbyte* M_op1, CALbyte* M_op2,  CALbyte* M_dest, int buffer_dimension);
        void (* calAddBuffer_i)(CALint* M_op1, CALint* M_op2,  CALint* M_dest, int buffer_dimension);
        void (* calAddBuffer_r)(CALreal* M_op1, CALreal* M_op2,  CALreal* M_dest, int buffer_dimension);

        void (* calSubtractBuffer_b)(CALbyte* M_op1, CALbyte* M_op2,  CALbyte* M_dest, int buffer_dimension);
        void (* calSubtractBuffer_i)(CALint* M_op1, CALint* M_op2,  CALint* M_dest, int buffer_dimension);
        void (* calSubtractBuffer_r)(CALreal* M_op1, CALreal* M_op2,  CALreal* M_dest, int buffer_dimension);

};

struct CALRun* makeCALRun(enum CALExecutionType executionType);

#endif
