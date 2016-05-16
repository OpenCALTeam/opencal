#include <OpenCAL++/calCommon.h>
#include <OpenCAL++/calModel.h>
#include <OpenCAL++/calRun.h>
#include <OpenCAL++/calMooreNeighborhood.h>
#include <OpenCAL++/calRealConverter.h>
using namespace std;

#define DIMX 	(30)
#define DIMY 	(30)
#define LAYERS 	(30)
#define STEPS 	(100)

#define PREFIX_PATH(version,name,pathVarName) \
	if(version==0)\
		 pathVarName="./testsout/serial/" name;\
	 else if(version>0)\
		 pathVarName="./testsout/other/" name;

typedef unsigned int COORD_TYPE;

class IdentityFunctor : public opencal::CALElementaryProcessFunctor<3,
		                                                                  opencal::CALMooreNeighborhood<3>
		                                                                 >{
private:

		 opencal::CALSubstate<int, 3> *I;
		 opencal::CALSubstate<double, 3> *R;
		 opencal::CALSubstate<bool, 3> *B;

		 public:

		   IdentityFunctor(auto& _I,auto& _R,auto& _B):I(_I),R(_R),B(_B){}

		   void run(opencal::CALModel<3, opencal::CALMooreNeighborhood<3> > *calModel,
		            std::array<COORD_TYPE, 3>& indexes)
		   {
			    B->setElement(indexes, B->getElement(indexes));
				R->setElement(indexes, R->getElement(indexes));
				I->setElement(indexes, I->getElement(indexes));
		   }
		 };



int main(int argc, char** argv)
{
	int version=0;
	if (sscanf (argv[1], "%i", &version)!=1 && version >=0) {
		printf ("error - not an integer");
		exit(-1);
	 }

	 std::array<COORD_TYPE, 3> coords = { DIMX, DIMY, LAYERS };
	 opencal::CALMooreNeighborhood<3> neighbor;

	 opencal::CALModel<3, opencal::CALMooreNeighborhood<3>, COORD_TYPE> calmodel(
	   coords,
	   &neighbor,
	   opencal::calCommon::CAL_SPACE_TOROIDAL,
	   opencal::calCommon::CAL_NO_OPT);

	opencal::CALRun < decltype(calmodel) > calrun(&calmodel, 1, STEPS, opencal::calCommon::CAL_UPDATE_IMPLICIT);

opencal::CALSubstate<int, 3, COORD_TYPE> *I = calmodel.addSubstate<int>();
opencal::CALSubstate<double, 3, COORD_TYPE> *R = calmodel.addSubstate<double>();
opencal::CALSubstate<bool, 3, COORD_TYPE> *B = calmodel.addSubstate<bool>();

calmodel.initSubstate(I, 12345);
calmodel.initSubstate(R, 1.98765432);
calmodel.initSubstate(B, false);

string path;

//saving initial configuration
//auto tostring_fn = []  (const auto& s)  -> std::string { return std::to_string(s);};
PREFIX_PATH(version,"1.txt",path);
I->saveSubstate(opencal::tostring_fn<int>(),  path);
PREFIX_PATH(version,"2.txt",path);
R->saveSubstate(opencal::tostring_fn<double>(6), path);
PREFIX_PATH(version,"3.txt",path);
B->saveSubstate(opencal::tostring_fn<bool>(), path);

calmodel.addElementaryProcess(new IdentityFunctor(I,R,B) );
calrun.run();

PREFIX_PATH(version,"4.txt",path);
I->saveSubstate(opencal::tostring_fn<int>(),  path);
PREFIX_PATH(version,"5.txt",path);
R->saveSubstate(opencal::tostring_fn<double>(6), path);
PREFIX_PATH(version,"6.txt",path);
B->saveSubstate(opencal::tostring_fn<bool>(), path);


	return 0;
}

//-----------------------------------------------------------------------
