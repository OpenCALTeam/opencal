#include "sciddicaT.h"
#include <stdlib.h>



//------------------------------------------------------------------------------
//					sciddicaT transition function
//------------------------------------------------------------------------------

//first elementary process
//// The sigma_1 elementary process
class SciddicaT_flows_computation : public CALElementaryProcessFunctor
{
private:
    struct SciddicaTSubstates* Q;
    struct SciddicaTParameters* P;
    CALUnsafe* sciddicaT_unsafe;
public:
    SciddicaT_flows_computation (SciddicaTSubstates* Q, SciddicaTParameters* P,CALUnsafe* sciddicaT_unsafe)
    {
        this->Q = Q;
        this->P = P;
        this->sciddicaT_unsafe = sciddicaT_unsafe;
    }

    inline void run(CALModel * calModel, int * indexes)
    {
        bool eliminated_cells[5]={false,false,false,false,false};
        bool again;
        int cells_count;
        double average, m, f;

        double u[5];
        int n;
        double z, h;
        double toSet;

        int linearIndex = calCommon::cellLinearIndex(indexes, calModel->getCoordinates(), calModel->getDimension());

        int neighboorhoodSize = calModel->getNeighborhoodSize();

        m = Q->h->getElement(linearIndex) - P->epsilon;
        u[0] = Q->z->getElement(linearIndex) + P->epsilon;
        for (n=1; n<neighboorhoodSize; n++)
        {
            z = Q->z->getX(linearIndex, n);
            h = Q->h->getX(linearIndex, n);
            u[n] = z + h;
        }

        //computes outflows
        do{
            again = CAL_FALSE;
            average = m;
            cells_count = 0;

            for (n=0; n<neighboorhoodSize; n++)
                if (!eliminated_cells[n]){
                    average += u[n];
                    cells_count++;
                }

                if (cells_count != 0)
                    average /= cells_count;

                for (n=0; n<neighboorhoodSize; n++)
                    if( (average<=u[n]) && (!eliminated_cells[n]) ){
                        eliminated_cells[n]=CAL_TRUE;
                        again=CAL_TRUE;
                    }
        }while (again);

        for (n=1; n<neighboorhoodSize; n++)
            if (!eliminated_cells[n])
            {
                f = (average-u[n])*P->r;
                toSet = sciddicaT_unsafe->calGetNext (Q->h, indexes) -f;
                Q->h->setElement(linearIndex, toSet);
                toSet = sciddicaT_unsafe->calGetNextX(Q->h, indexes, n) + f;
                this->sciddicaT_unsafe->calSetX (Q->h, linearIndex, n, toSet);
                this->sciddicaT_unsafe->calAddActiveCellX(linearIndex, n);
           }
    }

};

// The sigma_3 elementary process
class SciddicaT_remove_inactive_cells :public CALElementaryProcessFunctor
{
private:
    struct SciddicaTSubstates* Q;
    struct SciddicaTParameters* P;
public:
    SciddicaT_remove_inactive_cells (SciddicaTSubstates* Q, SciddicaTParameters* P)
    {
        this->Q = Q;
        this->P = P;
    }

    inline void run (CALModel* calModel, int* indexes)
    {
        if (Q->h->getElement(indexes, calModel->getCoordinates(),calModel->getDimension())<= this->P->epsilon)
        {
            calModel->removeActiveCell(indexes);
        }

    }


};
//------------------------------------------------------------------------------
//					sciddicaT simulation functions
//------------------------------------------------------------------------------

class Simulation_Init : public CALModelFunctor <CALModel, void>
{
private:
    struct SciddicaTParameters* P;
    struct SciddicaTSubstates* Q;
public:
    Simulation_Init (SciddicaTParameters* P, SciddicaTSubstates* Q)
    {
        this->P = P;
        this->Q= Q;
    }

    void run(CALModel * model)
    {
       calCommon:: CALreal z, h;

       //sciddicaT parameters setting
        P->r = P_R;
        P->epsilon = P_EPSILON;

        int size = model->getSize();
        //sciddicaT source initialization
        for (int i = 0;  i< size; i++)
        {
            h = Q->h->getElement (i);

            if ( h > 0.0 )
            {
                z = Q->z->getElement (i);
                Q->z->setElementCurrent(i, z-h);
                model->addActiveCell(i);


            }
        }
    }

};


//------------------------------------------------------------------------------
//					sciddicaT CADef and runDef
//------------------------------------------------------------------------------


SciddicaTModel::SciddicaTModel (int* coordinates, size_t dimension)
{
    //cadef and rundef
    sciddicaT = new CALModel (coordinates,dimension, new CALVonNeumannNeighborhood () , calCommon::CAL_SPACE_TOROIDAL, calCommon::CAL_OPT_ACTIVE_CELLS);
    sciddicaT_simulation = new CALRun(sciddicaT, 1, STEPS, calCommon:: CAL_UPDATE_IMPLICIT);

    sciddicaT_unsafe = new CALUnsafe (sciddicaT);

    sciddicaConverterInputOutput = new CALRealConverterIO();
    //add substates

    Q = new SciddicaTSubstates ();
    P = new SciddicaTParameters ();
    Q->z = sciddicaT->addSingleLayerSubstate<double> ();
    Q->h = sciddicaT->addSubstate<calCommon::CALreal>();


    //load configuration
    sciddicaTLoadConfig();

    sciddicaT-> addElementaryProcess(new SciddicaT_flows_computation(this->Q, this->P, this->sciddicaT_unsafe));
    sciddicaT->addElementaryProcess((new SciddicaT_remove_inactive_cells (this->Q, this->P)));
    //simulation run setup
    sciddicaT_simulation->addInitFunc(new Simulation_Init(this->P, this->Q) );

}


SciddicaTModel :: ~SciddicaTModel ()
{
    delete sciddicaT;
    delete sciddicaT_simulation;
    delete sciddicaConverterInputOutput;
    delete Q->z;
    delete Q;
    delete P;
    delete sciddicaT_unsafe;
}
void SciddicaTModel :: sciddicaTRun ()
{

    sciddicaT_simulation->run();
}

//------------------------------------------------------------------------------
//					sciddicaT I/O functions
//------------------------------------------------------------------------------

void SciddicaTModel::sciddicaTLoadConfig()
{
    //load configuration
    Q->z->loadSubstate (sciddicaT->getCoordinates(), sciddicaT->getDimension(), sciddicaConverterInputOutput,(char*) DEM_PATH);
    Q->h->loadSubstate (sciddicaT->getCoordinates(), sciddicaT->getDimension(), sciddicaConverterInputOutput,(char*) SOURCE_PATH);
}

void SciddicaTModel::sciddicaTSaveConfig()
{
    Q->h->saveSubstate(sciddicaT->getCoordinates(), sciddicaT->getDimension(), sciddicaConverterInputOutput, (char*) OUTPUT_PATH);
}





