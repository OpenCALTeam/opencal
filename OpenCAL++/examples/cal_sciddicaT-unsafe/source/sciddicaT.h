#ifndef sciddicaTModel_h
#define sciddicaTModel_h

#include <OpenCAL++11/calModel.h>
#include <OpenCAL++11/calUnsafe.h>
#include <OpenCAL++11/calVonNeumannNeighborhood.h>
#include <OpenCAL++11/calRun.h>
#include <OpenCAL++11/calRealConverterIO.h>
#include <OpenCAL++11/calModelFunctor.h>
#include <stdlib.h>
#include <time.h>



#define P_R 0.5
#define P_EPSILON 0.001
#define STEPS 4000
#define DEM_PATH "./data/dem.txt"
#define SOURCE_PATH "./data/source.txt"
#define OUTPUT_PATH "./data/width_final.txt"

#define NUMBER_OF_OUTFLOWS 4
struct SciddicaTSubstates {
    CALSubstate<double> *z;
    CALSubstate<double> *h;
};
struct SciddicaTParameters {
    double epsilon;
    double r;
};

class SciddicaTModel
{
private:
//#define VERBOSE
    CALModel* sciddicaT;
    CALRun* sciddicaT_simulation;
    CALUnsafe* sciddicaT_unsafe;
    CALConverterIO* sciddicaConverterInputOutput;

    struct SciddicaTSubstates* Q;
    struct SciddicaTParameters* P;


    void sciddicaTLoadConfig();
public:

    SciddicaTModel (int* coordinates, size_t dimension);
    ~SciddicaTModel ();
    void sciddicaTSaveConfig();
    void sciddicaTRun();
    void sciddicaTExit();




};
#endif
