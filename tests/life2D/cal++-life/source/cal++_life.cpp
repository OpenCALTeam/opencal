//
// Created by Davide Spataro on 12/04/16.
//
#include <OpenCAL++/calCommon.h>
#include <OpenCAL++/calModel.h>
#include <OpenCAL++/calRun.h>
#include <OpenCAL++/calMooreNeighborhood.h>


using namespace std;
template <typename T>
string NumberToString ( T Number )
{
    ostringstream ss;
    ss << Number;
    return ss.str();
}

//-----------------------------------------------------------------------
//   THE LIFE CELLULAR AUTOMATON
//-----------------------------------------------------------------------
#define DIMX 	(100)
#define DIMY 	(100)
#define STEPS 	(100)


//if versio == 0 -> write in serial folder
#define PREFIX_PATH(version,name,pathVarName) \
    if(version==0)\
    pathVarName="./testsout/serial/" name;\
    else if(version>0)\
    pathVarName="./testsout/other/" name;


typedef unsigned int COORD_TYPE;


using namespace opencal;

int version=1;
string path;
string step="";


void print (int currentStep, opencal::CALSubstate<int, 2, COORD_TYPE> * Q)
{
    step=NumberToString( currentStep);
    PREFIX_PATH(version,"",path);
    path+=step;
    path+=".txt";
    Q->saveSubstate(opencal::tostring_fn<int>(4),  path);
}


class Life_transition_function : public opencal::CALLocalFunction<2,
        opencal::CALMooreNeighborhood<2>
        >{
private:

    opencal::CALSubstate<int, 2> *Q;

public:

    Life_transition_function(opencal::CALSubstate<int, 2> *_Q)
    {
        Q = _Q;
    }

    void run(opencal::CALModel<2, opencal::CALMooreNeighborhood<2> > *calModel,
             std::array<COORD_TYPE, 2>& indexes)
    {
        int sum              = 0, n;
        int neighborhoodSize = calModel->getNeighborhoodSize();

        for (n = 1; n < neighborhoodSize; n++)
        {
            sum += Q->getX(indexes, n);
        }


        if ((sum == 3) || ( sum == 2 && Q->getElement(indexes) == 1 ))
        {
            Q->setElement(indexes, 1);
        }
        else Q->setElement(indexes, 0);
    }
};

//class LifeSimulationStopCondition : public opencal::CALModelFunctor <opencal::CALModel<2, opencal::CALMooreNeighborhood<2>, COORD_TYPE>, bool>
//{
//private:
//    opencal::CALRun < opencal::CALModel < 2, opencal::CALMooreNeighborhood<2>,COORD_TYPE >>* life_simulation;
//    opencal::CALSubstate<int, 2, COORD_TYPE> *Q;
//public:
//    LifeSimulationStopCondition(opencal::CALRun < opencal::CALModel < 2, opencal::CALMooreNeighborhood<2>,COORD_TYPE >>* _life_simulation,  opencal::CALSubstate<int, 2, COORD_TYPE> * _Q)
//    {
//        this->life_simulation = _life_simulation;
//        this->Q = _Q;
//    }

//    bool run(opencal::CALModel<2, opencal::CALMooreNeighborhood<2>, COORD_TYPE>* model)
//    {


//        if (life_simulation->getStep() >= STEPS)
//        {
//            return true;
//        }

//        print (life_simulation->getStep()+1, Q);

//        return false;
//    }

//};



//class Init : public opencal::CALModelFunctor <opencal::CALModel<2, opencal::CALMooreNeighborhood<2>, COORD_TYPE>, void>
//{
//private:

//    opencal::CALSubstate<int, 2> *Q;
//    opencal::CALRun < opencal::CALModel < 2, opencal::CALMooreNeighborhood<2>,COORD_TYPE >>* life_simulation;

//public:

//    Init(opencal::CALSubstate<int, 2> *_Q, opencal::CALRun < opencal::CALModel < 2, opencal::CALMooreNeighborhood<2>,COORD_TYPE >>* _life_simulation)
//    {
//        Q = _Q;
//        this->life_simulation = _life_simulation;
//    }

//    void run(opencal::CALModel<2, opencal::CALMooreNeighborhood<2>, COORD_TYPE>* model)
//    {
//        setGlider(model,1,1);
//        setGlider(model,DIMX-5,DIMY-5);
//        setToad(model,5,DIMY-5);

//        print (life_simulation->getStep(), Q);
//    }

//    void setToad(opencal::CALModel<2, opencal::CALMooreNeighborhood<2>, COORD_TYPE>* model, uint dx, uint dy){
//        //set a Toad Pulsar

//        std::array<std::array<COORD_TYPE, 2>, 6> indexes = {        {
//                                                                        { { 0+dx, 1+dy } },
//                                                                        { { 0+dx, 2+dy } },
//                                                                        { { 0+dx, 3+dy } },
//                                                                        { { 1+dx, 0+dy } },
//                                                                        { { 1+dx, 1+dy } },
//                                                                        { { 1+dx, 2+dy } }
//                                                                    } };


//        for (uint i = 0; i < 6; i++)
//        {
//            model->init(Q, indexes[i], 1);
//        }

//    }

//    void setGlider(opencal::CALModel<2, opencal::CALMooreNeighborhood<2>, COORD_TYPE>* model, uint dx, uint dy){
//        //set a glider
//        std::array<std::array<COORD_TYPE, 2>, 5> indexes = {        {
//                                                                        { { 0+dx, 2+dy } },
//                                                                        { { 1+dx, 0+dy } },
//                                                                        { { 1+dx, 2+dy } },
//                                                                        { { 2+dx, 1+dy } },
//                                                                        { { 2+dx, 2+dy } }
//                                                                    } };


//        for (uint i = 0; i < 5; i++)
//        {
//            model->init(Q, indexes[i], 1);
//        }
//    }


//};
 typedef opencal::CALModel<2, opencal::CALMooreNeighborhood<2>, COORD_TYPE> CALMODEL;

class MyRun : public opencal::CALRun<CALMODEL>
{
private:
    opencal::CALSubstate<int, 2> *Q;
public:

    MyRun (CALMODEL_pointer model,   int _initial_step,int _final_step,enum calCommon :: CALUpdateMode _UPDATE_MODE
           )	:
        CALRun(model, _initial_step,_final_step, _UPDATE_MODE)
    {

    }

    void init(opencal::CALSubstate<int, 2> *_Q)
    {
        this->Q = _Q;


    }

    bool stopCondition()
    {
        if (this->step >= STEPS)
        {
            return true;
        }

        print (this->step+1, Q);

        return false;
    }

    void init()
    {
        setGlider(1,1);
        setGlider(DIMX-5,DIMY-5);
        setToad(5,DIMY-5);

        print (this->step, Q);
    }

    void setToad( uint dx, uint dy){
        //set a Toad Pulsar

        std::array<std::array<COORD_TYPE, 2>, 6> indexes = {        {
                                                                        { { 0+dx, 1+dy } },
                                                                        { { 0+dx, 2+dy } },
                                                                        { { 0+dx, 3+dy } },
                                                                        { { 1+dx, 0+dy } },
                                                                        { { 1+dx, 1+dy } },
                                                                        { { 1+dx, 2+dy } }
                                                                    } };


        for (uint i = 0; i < 6; i++)
        {
            calModel->init(Q, indexes[i], 1);
        }

    }

    void setGlider(uint dx, uint dy){
        //set a glider
        std::array<std::array<COORD_TYPE, 2>, 5> indexes = {        {
                                                                        { { 0+dx, 2+dy } },
                                                                        { { 1+dx, 0+dy } },
                                                                        { { 1+dx, 2+dy } },
                                                                        { { 2+dx, 1+dy } },
                                                                        { { 2+dx, 2+dy } }
                                                                    } };


        for (uint i = 0; i < 5; i++)
        {
            calModel->init(Q, indexes[i], 1);
        }
    }
};




int main(int argc, char **argv) {
    std::array<COORD_TYPE, 2> coords = { DIMX, DIMY };
    opencal::CALMooreNeighborhood<2> neighbor;

    opencal::CALModel<2, opencal::CALMooreNeighborhood<2>, COORD_TYPE> calmodel(
                coords,
                &neighbor,
                opencal::calCommon::CAL_SPACE_TOROIDAL,
                opencal::calCommon::CAL_NO_OPT);

    MyRun calrun(&calmodel, 1, STEPS, opencal::calCommon::CAL_UPDATE_IMPLICIT);


    opencal::CALSubstate<int, 2, COORD_TYPE> *Q = calmodel.addSubstate<int>();

    calmodel.initSubstate(Q, 0);

    calrun.init(Q);
    calmodel.addElementaryProcess(new Life_transition_function(Q) );


    calrun.runInitSimulation();


    //PREFIX_PATH(version,"1.txt",path);
    //Q->saveSubstate(opencal::tostring_fn<int>(4),  path);


    calrun.run();
    //PREFIX_PATH(version,"2.txt",path);
    //Q->saveSubstate(opencal::tostring_fn<int>(4),  path);


    return 0;
}
