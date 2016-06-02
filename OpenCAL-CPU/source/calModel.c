#include <OpenCAL-CPU/calModel.h>

CALModel*calCADef(int numberOfCoordinates, CALIndexes coordinatesDimensions, CALSpaceBoundaryCondition CAL_TOROIDALITY, CALExecutionType executionType, CALOptimization CAL_OPTIMIZATION)
{

}

void calAddNeighbor2D(CALModel* calModel, CALIndexes neighbourIndex)
{

}

CALSubstate_b*calAddSubstate_b(CALModel* calModel, CALInitMethod initMethod, CALbyte value = 0)
{

}

CALSubstate_i*calAddSubstate_i(CALModel* calModel, CALInitMethod initMethod, CALint value)
{

}

CALSubstate_r*calAddSubstate_r(CALModel* calModel)
{

}

CALSubstate_b*calAddSingleLayerSubstate_b(CALModel* calModel, CALbyte init_value)
{

}

CALSubstate_i*calAddSingleLayerSubstate_i(CALModel* calModel, CALint init_value)
{

}

CALSubstate_r*calAddSingleLayerSubstate_r(CALModel* calModel, CALreal init_value)
{

}

void calAddLocalProcess(CALModel* calModel, CALLocalProcess elementary_process)
{

}

void calAddGlobalProcess(CALModel* calModel, CALGlobalProcess elementary_process)
{

}

void calUpdateSubstate_i(CALModel* calModel, CALSubstate_i* Q)
{

}

void calUpdateSubstate_r(CALModel* calModel, CALSubstate_r* Q)
{

}

void calApplyLocalProcess(CALModel* calModel, CALLocalProcess local_process)
{

}

void calUpdate2D(CALModel* calModel)
{

}

void calInit_b(CALModel* calModel, CALSubstate_b* Q, CALIndexes indexes, CALbyte value)
{

}

void calInit_i(CALModel* calModel, CALSubstate_i* Q, CALIndexes indexes, CALint value)
{

}

void calInit_r(CALModel* calModel, CALSubstate_r* Q, CALIndexes indexes, CALreal value)
{

}

CALbyte calGet_b(CALModel* calModel, CALSubstate_b* Q, CALIndexes indexes)
{

}

CALreal calGet_r(CALModel* calModel, CALSubstate_r* Q, CALIndexes indexes)
{

}

CALint calGet_i(CALModel* calModel, CALSubstate_i* Q, CALIndexes indexes)
{

}

CALbyte calGetX_b(CALModel* calModel, CALSubstate_b* Q, CALIndexes central_cell, int n)
{

}

void calSet_b(CALModel* calModel, CALSubstate_b* Q, CALIndexes central_cell, CALbyte value)
{

}

void calSet_i(CALModel* calModel, CALSubstate_i* Q, CALIndexes central_cell, CALint value)
{

}

void calSet_r(CALModel* calModel, CALSubstate_r* Q, CALIndexes central_cell, CALreal value)
{

}

void calSetCurrent_b(CALModel* calModel, CALSubstate_b* Q, CALIndexes central_cell, CALbyte value)
{

}

void calSetCurrent_i(CALModel* calModel, CALSubstate_i* Q, CALIndexes central_cell, CALint value)
{

}

void calSetCurrent_r(CALModel* calModel, CALSubstate_r* Q, CALIndexes central_cell, CALreal value)
{

}

void calFinalize2D(CALModel* calModel)
{

}

void calApplyGlobalProcess(CALModel* calModel, CALGlobalProcess global_process)
{

}
