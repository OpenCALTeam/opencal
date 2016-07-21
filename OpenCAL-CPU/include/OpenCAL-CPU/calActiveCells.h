#ifndef cal_act_cells
#define cal_act_cells

#include <OpenCAL-CPU/calModel.h>


struct CALActiveCells
{
        struct CALModel* calModel;
        enum CALOptimization OPTIMIZATION;	//!< Type of optimization used. It can be CAL_NO_OPT or CAL_OPT_ACTIVE_CELLS.
};

void calAddActiveCells(struct CALActiveCells* A, CALIndices cell);
void calRemoveActiveCells(struct CALActiveCells* A, CALIndices cell);

CALbyte calApplyLocalFunctionOpt(struct CALActiveCells* A, CALLocalProcess local_process);

void calUpdateActiveCells(struct CALActiveCells* A);

CALbyte calCopyBufferActiveCells_b(CALbyte* M_src, CALbyte* M_dest,  struct CALActiveCells* A);
CALbyte calCopyBufferActiveCells_i(CALint* M_src, CALint* M_dest,  struct CALActiveCells* A);
CALbyte calCopyBufferActiveCells_r(CALreal* M_src, CALreal* M_dest,  struct CALActiveCells* A);


CALbyte calSetActiveCellsBuffer_b(CALbyte* M, CALbyte value, struct CALActiveCells* A);
CALbyte calSetActiveCellsBuffer_i(CALint* M, CALint value, struct CALActiveCells* A);
CALbyte calSetActiveCellsBuffer_r(CALreal* M, CALreal value, struct CALActiveCells* A);

void calFreeActiveCells(struct CALActiveCells* A);

#endif
