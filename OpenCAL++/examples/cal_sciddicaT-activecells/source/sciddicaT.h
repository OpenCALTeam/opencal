#ifndef sciddicaTModel_h
#define sciddicaTModel_h

#include <OpenCAL++/calModel.h>
#include <OpenCAL++/calUnsafe.h>
#include <OpenCAL++/calVonNeumannNeighborhood.h>
#include <OpenCAL++/calRun.h>
#include <OpenCAL++/calRealConverterIO.h>
#include <OpenCAL++/calModelFunctor.h>

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
    CALSubstate<double> *f[NUMBER_OF_OUTFLOWS];
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
