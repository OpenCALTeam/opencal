#include <OpenCAL++/calCommon.h>
#include <OpenCAL++/calModel.h>
#include <OpenCAL++/calRun.h>
#include <OpenCAL++/calMooreNeighborhood.h>

using namespace std;

#define DIMX 	(100)
#define DIMY 	(100)
#define STEPS 	(1000)

#define PREFIX_PATH(version,name,pathVarName) \
	if(version==0)\
		 pathVarName="./testsout/serial/" name;\
	 else if(version>0)\
		 pathVarName="./testsout/other/" name;

typedef unsigned int COORD_TYPE;

class IdentityFunctor : public opencal::CALElementaryProcessFunctor<2,
		                                                                  opencal::CALMooreNeighborhood<2>
		                                                                 >{
private:

		 opencal::CALSubstate<int, 2> *I;
		 opencal::CALSubstate<double, 2> *R;
		 opencal::CALSubstate<bool, 2> *B;

		 public:

		   IdentityFunctor(auto& _I,auto& _R,auto& _B):I(_I),R(_R),B(_B){}

		   void run(opencal::CALModel<2, opencal::CALMooreNeighborhood<2> > *calModel,
		            std::array<COORD_TYPE, 2>& indexes)
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

	 std::array<COORD_TYPE, 2> coords = { DIMX, DIMY };
	 opencal::CALMooreNeighborhood<2> neighbor;

	 opencal::CALModel<2, opencal::CALMooreNeighborhood<2>, COORD_TYPE> calmodel(
	   coords,
	   &neighbor,
	   opencal::calCommon::CAL_SPACE_TOROIDAL,
	   opencal::calCommon::CAL_NO_OPT);

	opencal::CALRun < decltype(calmodel) > calrun(&calmodel, 1, STEPS, opencal::calCommon::CAL_UPDATE_IMPLICIT);

opencal::CALSubstate<int, 2, COORD_TYPE> *I = calmodel.addSubstate<int>();
opencal::CALSubstate<double, 2, COORD_TYPE> *R = calmodel.addSubstate<double>();
opencal::CALSubstate<bool, 2, COORD_TYPE> *B = calmodel.addSubstate<bool>();

calmodel.initSubstate(I, 12345);
calmodel.initSubstate(R, 1.98765432);
calmodel.initSubstate(B, false);

string path;
/*I->loadSubstate(opencal::stoi,  path);
R->loadSubstate(opencal::stod,  path);
R->loadSubstate(opencal::stoi,  path);*/
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
