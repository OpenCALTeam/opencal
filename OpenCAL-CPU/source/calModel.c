#include <OpenCAL-CPU/calModel.h>
#include <OpenCAL-CPU/calRun.h>
struct CALModel*calCADef(int numberOfCoordinates, CALIndexes coordinatesDimensions, enum CALNeighborhood CAL_NEIGHBORHOOD, enum CALSpaceBoundaryCondition CAL_TOROIDALITY, enum CALExecutionType executionType, enum CALOptimization CAL_OPTIMIZATION)
{

    struct CALModel *calModel = (struct CALModel *)malloc(sizeof(struct CALModel));
    if (!calModel)
        return NULL;

    calModel->numberOfCoordinates = numberOfCoordinates;

    calModel->coordinatesDimensions = coordinatesDimensions;

    calModel->calIndexesPool =  calDefIndexesPool(coordinatesDimensions,numberOfCoordinates);
//    calModel->calNeighborPool = calDefNeighborPool(calModel->calIndexesPool, CAL_TOROIDALITY, );

    //CALL calRun constructor and set optimization
    int ** cellPattern;
    switch (CAL_NEIGHBORHOOD) {
    case CAL_VON_NEUMANN_NEIGHBORHOOD:
        cellPattern = defineVonNeumannNeighborhood(1,numberOfCoordinates);
        break;
    case CAL_MOORE_NEIGHBORHOOD:
        cellPattern = defineMooreNeighborhood(1,numberOfCoordinates);
        break;
//    case CAL_HEXAGONAL_NEIGHBORHOOD:
//        break;
//    case CAL_HEXAGONAL_NEIGHBORHOOD_ALT:
//        break;
    }
    calModel->calNeighborPool = calDefNeighborPool(calModel->calIndexesPool,CAL_TOROIDALITY, cellPattern);

    calModel->OPTIMIZATION = CAL_OPTIMIZATION;

    //Manage Optimization

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

void calInit_b(struct CALModel* calModel, struct CALSubstate_b* Q, CALIndexes indexes, CALbyte value)
{

}

void calInit_i(struct CALModel* calModel, struct CALSubstate_i* Q, CALIndexes indexes, CALint value)
{

}

void calInit_r(struct CALModel* calModel, struct CALSubstate_r* Q, CALIndexes indexes, CALreal value)
{

}

CALbyte calGet_b(struct CALModel* calModel, struct CALSubstate_b* Q, CALIndexes indexes)
{
    CALbyte ret;
#if CAL_PARALLEL == 1
        CAL_SET_CELL_LOCK(i, j, calModel);
#endif

    //ret = calGetMatrixElement(Q->current, calModel->columns, i, j);

#if CAL_PARALLEL == 1
    CAL_UNSET_CELL_LOCK(i, j, ca2D);
#endif

    return ret;
}

CALreal calGet_r(struct CALModel* calModel, struct CALSubstate_r* Q, CALIndexes indexes)
{
    CALreal ret;
#if CAL_PARALLEL == 1
        CAL_SET_CELL_LOCK(i, j, calModel);
#endif

    //ret = calGetMatrixElement(Q->current, calModel->columns, i, j);

#if CAL_PARALLEL == 1
    CAL_UNSET_CELL_LOCK(i, j, ca2D);
#endif

    return ret;
}

CALint calGet_i(struct CALModel* calModel, struct CALSubstate_i* Q, CALIndexes indexes)
{
    CALint ret;
#if CAL_PARALLEL == 1
        CAL_SET_CELL_LOCK(i, j, calModel);
#endif

    //ret = calGetMatrixElement(Q->current, calModel->columns, i, j);

#if CAL_PARALLEL == 1
    CAL_UNSET_CELL_LOCK(i, j, ca2D);
#endif

    return ret;
}

CALbyte calGetX_b(struct CALModel* calModel, struct CALSubstate_b* Q, CALIndexes central_cell, int n)
{

}

void calSet_b(struct CALModel* calModel, struct CALSubstate_b* Q, CALIndexes central_cell, CALbyte value)
{

}

void calSet_i(struct CALModel* calModel, struct CALSubstate_i* Q, CALIndexes central_cell, CALint value)
{

}

void calSet_r(struct CALModel* calModel, struct CALSubstate_r* Q, CALIndexes central_cell, CALreal value)
{

}

void calSetCurrent_b(struct CALModel* calModel, struct CALSubstate_b* Q, CALIndexes central_cell, CALbyte value)
{

}

void calSetCurrent_i(struct CALModel* calModel, struct CALSubstate_i* Q, CALIndexes central_cell, CALint value)
{

}

void calSetCurrent_r(struct CALModel* calModel, struct CALSubstate_r* Q, CALIndexes central_cell, CALreal value)
{

}

void calFinalize(struct CALModel* calModel)
{

}
