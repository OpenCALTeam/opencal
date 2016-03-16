// mod2 3D Cellular Automaton


#include <OpenCAL++/calModel.h>
#include <OpenCAL++/calRun.h>
#include <OpenCAL++/calMooreNeighborhood.h>
#include <OpenCAL++/calIntConverterIO.h>


#define ROWS 5
#define COLS 7
#define LAYERS 3

// declare CA, substate and simulation objects
CALModel* mod2;
CALSubstate<int>* Q;
CALRun* mod2_simulation;

// The cell's transition function

class Mod2_transition_function : public CALElementaryProcessFunctor
{
private:
    CALSubstate<int>* Q;
public:
    Mod2_transition_function (CALSubstate<int>* Q)
    {
        this->Q = Q;
    }

    void run(CALModel* calModel, int* indexes)
    {
        int sum = 0, n;
        int neighborhoodSize =calModel->getNeighborhoodSize();

        for (n=0; n<neighborhoodSize; n++)
            sum += Q->getX (indexes, calModel->getCoordinates(), calModel->getDimension(), n);
//        printf ("Sto settando %d alla cella %d %d %d \n", sum%2, indexes[0], indexes[1], indexes[2]);
        Q->setElement(indexes, calModel->getCoordinates(), calModel->getDimension(), sum%2);

    }
};


int main()
{
	// define of the mod2 CA and mod2_simulation simulation objects
    int* coordinates = new int [3] {ROWS,COLS,LAYERS};
    size_t dimension = 3;
    mod2 = new CALModel (coordinates, dimension,new CALMooreNeighborhood(), calCommon:: CAL_SPACE_TOROIDAL, calCommon:: CAL_NO_OPT);
    mod2_simulation = new CALRun (mod2, 1,1,calCommon:: CAL_UPDATE_IMPLICIT);
    CALConverterIO* calConverterInputOutput = new CALIntConverterIO();
	// add the Q substate to the mod2 CA
    Q = mod2->addSubstate<int> ();

	// add transition function's elementary process
    mod2->addElementaryProcess(new Mod2_transition_function(Q) );

	// set the whole substate to 0
    mod2->initSubstate (Q, 0);

	// set a seed at position (2, 3, 1)

    int * indexes = new int [3] {2,3,1};
    mod2->init (Q, indexes, 1 );



	// save the Q substate to file
    Q->saveSubstate(coordinates, dimension, calConverterInputOutput, (char*)"./mod2_0000.txt" );

	// simulation run
    mod2_simulation->run();

	// save the Q substate to file
    Q->saveSubstate(coordinates, dimension, calConverterInputOutput, (char*)"./mod2_LAST.txt" );

	// finalize simulation and CA objects
    delete indexes;
    delete mod2_simulation;
    delete mod2;


	return 0;
}
