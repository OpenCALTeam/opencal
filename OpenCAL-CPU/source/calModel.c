#include <OpenCAL-CPU/calModel.h>
#include <OpenCAL-CPU/calRun.h>
struct CALModel*calCADef(int numberOfCoordinates, CALIndexes coordinatesDimensions, enum CALSpaceBoundaryCondition CAL_TOROIDALITY, enum CALExecutionType executionType, enum CALOptimization CAL_OPTIMIZATION)
{

}

void calAddNeighbor(struct CALModel* calModel, CALIndexes neighbourIndex)
{

}

struct CALSubstate_b*calAddSubstate_b(struct CALModel* calModel, enum CALInitMethod initMethod, CALbyte value)
{

}

struct CALSubstate_i*calAddSubstate_i(struct CALModel* calModel, enum CALInitMethod initMethod, CALint value)
{

}

struct CALSubstate_r*calAddSubstate_r(struct CALModel* calModel, enum CALInitMethod initMethod, CALreal value)
{

}

struct CALSubstate_b*calAddSingleLayerSubstate_b(struct CALModel* calModel, CALbyte init_value)
{

}

struct CALSubstate_i*calAddSingleLayerSubstate_i(struct CALModel* calModel, CALint init_value)
{

}

struct CALSubstate_r*calAddSingleLayerSubstate_r(struct CALModel* calModel, CALreal init_value)
{

}

void calUpdateSubstate_i(struct CALModel* calModel, struct CALSubstate_i* Q)
{

}

void calUpdateSubstate_r(struct CALModel* calModel, struct CALSubstate_r* Q)
{

}

void calInit_b(struct CALModel* calModel, struct CALSubstate_b* Q, CALIndexes indexes, CALbyte value)
{

}

void calInit_i(struct CALModel* calModel, struct CALSubstate_i* Q, CALIndexes indexes, CALint value)
{

}

void calInit_r(struct CALModel* calModel, struct CALSubstate_r* Q, CALIndexes indexes, CALreal value)
{

}

//CALbyte calGet_b(struct CALModel* calModel, struct CALSubstate_b* Q, CALIndexes indexes)
//{

//}

//CALreal calGet_r(struct CALModel* calModel, struct CALSubstate_r* Q, CALIndexes indexes)
//{

//}

//CALint calGet_i(struct CALModel* calModel, struct CALSubstate_i* Q, CALIndexes indexes)
//{

//}

//CALbyte calGetX_b(struct CALModel* calModel, struct CALSubstate_b* Q, CALIndexes central_cell, int n)
//{

//}

//void calSet_b(struct CALModel* calModel, struct CALSubstate_b* Q, CALIndexes central_cell, CALbyte value)
//{

//}

//void calSet_i(struct CALModel* calModel, struct CALSubstate_i* Q, CALIndexes central_cell, CALint value)
//{

//}

//void calSet_r(struct CALModel* calModel, struct CALSubstate_r* Q, CALIndexes central_cell, CALreal value)
//{

//}

//void calSetCurrent_b(struct CALModel* calModel, struct CALSubstate_b* Q, CALIndexes central_cell, CALbyte value)
//{

//}

//void calSetCurrent_i(struct CALModel* calModel, struct CALSubstate_i* Q, CALIndexes central_cell, CALint value)
//{

//}

//void calSetCurrent_r(struct CALModel* calModel, struct CALSubstate_r* Q, CALIndexes central_cell, CALreal value)
//{

//}

void calFinalize2D(struct CALModel* calModel)
{

}
