#include "sciddicaT.h"
//-------------------------------------------------
//		sciddicaT transition function
//-------------------------------------------------

//first elementary process
class SciddicaT_flows_computation : public CALElementaryProcessFunctor
{
private:
    struct SciddicaTSubstates* Q;
    struct SciddicaTParameters* P;
public:
    SciddicaT_flows_computation (SciddicaTSubstates* Q, SciddicaTParameters* P)
    {
        this->Q = Q;
        this->P = P;
    }
    void run(CALModel *calModel, int *indexes)
    {
        bool eliminated_cells[5]={false,false,false,false,false};
        bool again;
        int cells_count;
        double average, m;

        double u[5];
        int n;
        double z, h;

        int linearIndex = calCommon::cellLinearIndex(indexes, calModel->getCoordinates(), calModel->getDimension());
        int neighboorhoodSize = calModel->getNeighborhoodSize();

        if (Q->h->getElement(linearIndex) <= P->epsilon)
            return;
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
            if (eliminated_cells[n])
                Q->f[n-1]->setElement(linearIndex, 0.0);
            else
                Q->f[n-1]->setElement(linearIndex, (average-u[n])*P->r);
    }

};

//second (and last) elementary process
class SciddicaT_width_update: public CALElementaryProcessFunctor
{
private:
    struct SciddicaTSubstates* Q;
public:
    SciddicaT_width_update (SciddicaTSubstates* Q)
    {
        this->Q= Q;
    }
    void run (CALModel *calModel, int *indexes)
    {
        int linearIndex = calCommon::cellLinearIndex(indexes, calModel->getCoordinates(), calModel->getDimension());
        calCommon:: CALreal h_next;
        calCommon:: CALint n;
        h_next = Q->h->getElement(linearIndex);
        int neighboorhoodSize = calModel->getNeighborhoodSize();
        for(n=1; n<neighboorhoodSize; n++)
        {
            h_next +=  Q->f[NUMBER_OF_OUTFLOWS - n]->getX(linearIndex, n) - Q->f[n-1]->getElement(linearIndex);
        }
        Q->h->setElement(linearIndex, h_next);
    }
};
//-------------------------------------------------
//		sciddicaT simulation functions
//-------------------------------------------------

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
       calCommon:: CALint i;
        //initializing substates to 0
       model->initSubstate(Q->f[0],0.0);
       model->initSubstate(Q->f[1],0.0);
       model->initSubstate(Q->f[2],0.0);
       model->initSubstate(Q->f[3],0.0);

       //sciddicaT parameters setting
        P->r = P_R;
        P->epsilon = P_EPSILON;

        int size = model->getSize();
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
class SciddicaTSteering : public CALModelFunctor <CALModel, void>
{
private:
    struct SciddicaTSubstates* Q;
public:
    SciddicaTSteering (SciddicaTSubstates* Q)
    {
        this->Q = Q;
    }
    void run(CALModel * model)
    {
        // set flow to 0 everywhere
        model->initSubstate(Q->f[0],0.0);
        model->initSubstate(Q->f[1],0.0);
        model->initSubstate(Q->f[2],0.0);
        model->initSubstate(Q->f[3],0.0);
    }
};
//-------------------------------------------------
//		sciddicaT constructor and destructor
//-------------------------------------------------

SciddicaTModel::SciddicaTModel (int* coordinates, size_t dimension)
{
    //inizialize CALModel object and CALRun object
    sciddicaT = new CALModel (coordinates,dimension, new CALVonNeumannNeighborhood () , calCommon::CAL_SPACE_TOROIDAL, calCommon::CAL_NO_OPT);
    sciddicaT_simulation = new CALRun(sciddicaT, 1, STEPS, calCommon:: CAL_UPDATE_IMPLICIT);

    //inizializes object for I/O operation for real data type
    sciddicaT_ConverterInputOutput = new CALRealConverterIO();

    //adds substates
    Q = new SciddicaTSubstates ();
    P = new SciddicaTParameters ();
    Q->z = sciddicaT->addSubstate<calCommon::CALreal>();
    Q->h = sciddicaT->addSubstate<calCommon::CALreal>();

    Q->f[0] = sciddicaT->addSubstate<calCommon::CALreal>();;
    Q->f[1] = sciddicaT->addSubstate<calCommon::CALreal>();;
    Q->f[2] = sciddicaT->addSubstate<calCommon::CALreal>();;
    Q->f[3] = sciddicaT->addSubstate<calCommon::CALreal>();;

    //loads configuration
    sciddicaTLoadConfig();

    //adds elementary processes
    sciddicaT-> addElementaryProcess(new SciddicaT_flows_computation(this->Q, this->P));
    sciddicaT-> addElementaryProcess(new SciddicaT_width_update (this->Q));
    //simulation run setup
    sciddicaT_simulation->addInitFunc(new Simulation_Init(this->P, this->Q) );
    sciddicaT_simulation->addSteeringFunc(new SciddicaTSteering (this->Q));
}
SciddicaTModel :: ~SciddicaTModel ()
{
    delete sciddicaT;
    delete sciddicaT_simulation;
    delete sciddicaT_ConverterInputOutput;
    delete Q;
    delete P;
}
void SciddicaTModel :: sciddicaTRun ()
{
    sciddicaT_simulation->run();
}

//-------------------------------------------------
//			sciddicaT I/O functions
//-------------------------------------------------

void SciddicaTModel::sciddicaTLoadConfig()
{
    Q->z->loadSubstate (sciddicaT->getCoordinates(), sciddicaT->getDimension(),sciddicaT_ConverterInputOutput,(char*) DEM_PATH);
    Q->h->loadSubstate (sciddicaT->getCoordinates(), sciddicaT->getDimension(),sciddicaT_ConverterInputOutput,(char*) SOURCE_PATH);
}
void SciddicaTModel::sciddicaTSaveConfig()
{
    Q->h->saveSubstate(sciddicaT->getCoordinates(), sciddicaT->getDimension(), sciddicaT_ConverterInputOutput, (char*) OUTPUT_PATH);
}





