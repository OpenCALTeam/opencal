﻿#ifndef cal_run_serial
#define cal_run_serial

#include <OpenCAL-CPU/calRun.h>

void calSerialApplyLocalProcess( struct CALModel* calModel, CALLocalProcess local_process );

void calSerialUpdate (struct CALModel* calModel);


CALbyte calSerialGet_b(struct CALModel* calModel, struct CALSubstate_b* Q, CALIndices indexes);
CALint calSerialGet_i(struct CALModel* calModel, struct CALSubstate_i* Q, CALIndices indexes);
CALreal  calSerialGet_r(struct CALModel* calModel, struct CALSubstate_r* Q, CALIndices indexes);

CALbyte  calSerialGetX_b(struct CALModel* calModel, struct CALSubstate_b* Q, CALIndices central_cell, int n);
CALint  calSerialGetX_i(struct CALModel* calModel, struct CALSubstate_i* Q, CALIndices central_cell, int n);
CALreal  calSerialGetX_r(struct CALModel* calModel, struct CALSubstate_r* Q, CALIndices central_cell, int n);

void  calSerialSet_b(struct CALModel* calModel, struct CALSubstate_b* Q, CALIndices central_cell,CALbyte value);
void  calSerialSet_i(struct CALModel* calModel, struct CALSubstate_i* Q, CALIndices central_cell,CALbyte value);
void  calSerialSet_r(struct CALModel* calModel, struct CALSubstate_r* Q, CALIndices central_cell,CALbyte value);

void  calSerialSetCurrent_b(struct CALModel* calModel, struct CALSubstate_b* Q, CALIndices central_cell, CALbyte value);
void  calSerialSetCurrent_i(struct CALModel* calModel, struct CALSubstate_i* Q, CALIndices central_cell, CALbyte value);
void  calSerialSetCurrent_r(struct CALModel* calModel, struct CALSubstate_r* Q, CALIndices central_cell, CALbyte value);

void  calSerialCopyBuffer_b(CALbyte* M_src, CALbyte* M_dest, int buffer_dimension);
void  calSerialCopyBuffer_i(CALint* M_src, CALint* M_dest, int buffer_dimension);
void  calSerialCopyBuffer_r(CALreal* M_src, CALreal* M_dest, int buffer_dimension);

void  calSerialAddBuffer_b(CALbyte* M_op1, CALbyte* M_op2,  CALbyte* M_dest, int buffer_dimension);
void  calSerialAddBuffer_i(CALint* M_op1, CALint* M_op2,  CALint* M_dest, int buffer_dimension);
void  calSerialAddBuffer_r(CALreal* M_op1, CALreal* M_op2,  CALreal* M_dest, int buffer_dimension);

void  calSerialSubtractBuffer_b(CALbyte* M_op1, CALbyte* M_op2,  CALbyte* M_dest, int buffer_dimension);
void  calSerialSubtractBuffer_i(CALint* M_op1, CALint* M_op2,  CALint* M_dest, int buffer_dimension);
void  calSerialSubtractBuffer_r(CALreal* M_op1, CALreal* M_op2,  CALreal* M_dest, int buffer_dimension);


#endif
