#ifndef sciddicaTModel_h
#define sciddicaTModel_h

#include <OpenCAL++/calModel.h>
#include <OpenCAL++/calVonNeumannNeighborhood.h>
#include <OpenCAL++/calRun.h>
#include <OpenCAL++/calIORealConverter.h>

#include <stdlib.h>
#include <time.h>

#define P_R 0.5
#define P_EPSILON 0.001
#define STEPS 200
#define DEM_PATH "./testData/sciddicaT-data/dem.txt"

#define SOURCE_PATH "./testData/sciddicaT-data/source.txt"

#define NUMBER_OF_OUTFLOWS 4



typedef uint COORD_TYPE;

struct SciddicaTSubstates {
    opencal::CALSubstate<double, 2, COORD_TYPE, opencal::calCommon::OPT> *z;
    opencal::CALSubstate<double, 2, COORD_TYPE, opencal::calCommon::OPT> *h;
    opencal::CALSubstate<double, 2, COORD_TYPE, opencal::calCommon::OPT> *f[NUMBER_OF_OUTFLOWS];
};
struct SciddicaTParameters {
    double epsilon;
    double r;
};
class SciddicaTModel
{

private:
    // pointer to cellular automaton
    opencal::CALModel<2,opencal::CALVonNeumannNeighborhood<2>,COORD_TYPE> sciddicaT;
    // simulation object
    opencal::CALRun<opencal::CALModel<2,opencal::CALVonNeumannNeighborhood<2>,COORD_TYPE>> sciddicaT_simulation;
    // set of substates used in the simulation
    struct SciddicaTSubstates* Q;
    // paramenters
    struct SciddicaTParameters* P;
    opencal::CALRealConverter converter;

    opencal::CALVonNeumannNeighborhood<2> neighbor;

    //private method used by contructor to load substates from file
    void sciddicaTLoadConfig();
public:
    // SciddicaT's constructor
    SciddicaTModel (std::array<COORD_TYPE,2>& coords);
    // SciddicaT's distructor
    ~SciddicaTModel ();
    // saves configuration
    void sciddicaTSaveConfig();
    //runs simulation
    void sciddicaTRun();


};
#endif
