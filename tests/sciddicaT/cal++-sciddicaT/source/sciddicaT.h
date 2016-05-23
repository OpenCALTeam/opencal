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

using namespace opencal;

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


typedef opencal::CALModel<2,opencal::CALVonNeumannNeighborhood<2>,COORD_TYPE> CALMODEL;

class MyRun : public opencal::CALRun<CALMODEL >
{
private:
    struct SciddicaTSubstates* Q;
    struct SciddicaTParameters* P;
public:

    MyRun (CALMODEL_pointer model,   int _initial_step,int _final_step,enum calCommon :: CALUpdateMode _UPDATE_MODE
           )	:
        CALRun(model, _initial_step,_final_step, _UPDATE_MODE)
    {

    }

    void init(SciddicaTSubstates* _Q, SciddicaTParameters* _P)
    {
        this->Q = _Q;
        this->P = _P;

    }

    void steering()
    {
        // set flow to 0 everywhere
        calModel->initSubstate(Q->f[0],0.0);
        calModel->initSubstate(Q->f[1],0.0);
        calModel->initSubstate(Q->f[2],0.0);
        calModel->initSubstate(Q->f[3],0.0);

    }

    void init()
    {
        double z, h;
        int i;
        //initializing substates to 0
        calModel->initSubstate(Q->f[0],0.0);
        calModel->initSubstate(Q->f[1],0.0);
        calModel->initSubstate(Q->f[2],0.0);
        calModel->initSubstate(Q->f[3],0.0);

        //sciddicaT parameters setting
        P->r = P_R;
        P->epsilon = P_EPSILON;

        int size = calModel->getSize();
        //sciddicaT source initialization
        for (i = 0;  i< size; i++)
        {
            h = Q->h->getElement(i);
            if ( h > 0.0 )
            {
                z = Q->z->getElement(i);
                Q->z->setElement(i, z-h);
            }
        }
    }
};





class SciddicaTModel
{

private:
    // pointer to cellular automaton
    opencal::CALModel<2,opencal::CALVonNeumannNeighborhood<2>,COORD_TYPE> sciddicaT;
    // simulation object
    MyRun sciddicaT_simulation;
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
