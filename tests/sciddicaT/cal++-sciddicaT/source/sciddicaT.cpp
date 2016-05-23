#include "sciddicaT.h"
//-------------------------------------------------
//		sciddicaT transition function
//-------------------------------------------------

const int version=1;
#define PREFIX_PATH(version,name,pathVarName) \
    if(version==0)\
         pathVarName="./testsout/serial/" name;\
     else if(version>0)\
         pathVarName="./testsout/other/" name;



//first elementary process
class SciddicaT_flows_computation : public opencal::CALElementaryProcessFunctor<2 , opencal::CALVonNeumannNeighborhood<2> , COORD_TYPE>
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
    void run(opencal::CALModel<2 , opencal::CALVonNeumannNeighborhood<2> , COORD_TYPE>* calModel, std::array<uint,2>& indexes)
    {
        bool eliminated_cells[5]={false,false,false,false,false};
        bool again;
        int cells_count;
        double average, m;

        double u[5];
        int n;
        double z, h;

        int linearIndex = opencal::calCommon::cellLinearIndex<2,COORD_TYPE>(indexes, calModel->getCoordinates());
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
class SciddicaT_width_update: public opencal::CALElementaryProcessFunctor<2 , opencal::CALVonNeumannNeighborhood<2> , COORD_TYPE>
{
private:
    struct SciddicaTSubstates* Q;
public:
    SciddicaT_width_update (SciddicaTSubstates* Q)
    {
        this->Q= Q;
    }
    void run (opencal::CALModel<2 , opencal::CALVonNeumannNeighborhood<2> , COORD_TYPE>* calModel, std::array<uint,2>& indexes)
    {
        int linearIndex = opencal::calCommon::cellLinearIndex<2,COORD_TYPE> (indexes, calModel->getCoordinates());
        double h_next;
        int n;
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
//		sciddicaT constructor and destructor
//-------------------------------------------------

SciddicaTModel::SciddicaTModel (std::array<COORD_TYPE,2>& coords): sciddicaT(coords,&neighbor,opencal::calCommon::CAL_SPACE_TOROIDAL, opencal::calCommon::CAL_NO_OPT),
    sciddicaT_simulation(&sciddicaT, 1, STEPS, opencal::calCommon:: CAL_UPDATE_IMPLICIT)
{
    //adds substates
    Q = new SciddicaTSubstates ();
    P = new SciddicaTParameters ();

    sciddicaT_simulation.init(Q,P);
    Q->z = sciddicaT.addSubstate<double, opencal::calCommon::OPT>();
    Q->h = sciddicaT.addSubstate<double, opencal::calCommon::OPT>();

    Q->f[0] = sciddicaT.addSubstate<double, opencal::calCommon::OPT>();
    Q->f[1] = sciddicaT.addSubstate<double, opencal::calCommon::OPT>();
    Q->f[2] = sciddicaT.addSubstate<double, opencal::calCommon::OPT>();
    Q->f[3] = sciddicaT.addSubstate<double,opencal::calCommon::OPT>();

    //loads configuration
    sciddicaTLoadConfig();

    //adds elementary processes
    sciddicaT.addElementaryProcess(new SciddicaT_flows_computation(this->Q, this->P));
    sciddicaT.addElementaryProcess(new SciddicaT_width_update (this->Q));

}
SciddicaTModel :: ~SciddicaTModel ()
{
    delete Q;
    delete P;
}
void SciddicaTModel :: sciddicaTRun ()
{
    sciddicaT_simulation.run();
}

//-------------------------------------------------
//			sciddicaT I/O functions
//-------------------------------------------------


std::string convertOutput (double in)
{
    std::string c= std::to_string(in);
    return c;
}


void SciddicaTModel::sciddicaTLoadConfig()
{
    Q->z->loadSubstate(opencal::stof,(char*) DEM_PATH);


    Q->h->loadSubstate(opencal::stof,(char*) SOURCE_PATH);

}
void SciddicaTModel::sciddicaTSaveConfig()
{
    std::string path;
    PREFIX_PATH(version,"1.txt",path);
    Q->h->saveSubstate (convertOutput, (char*)path.c_str());
}
