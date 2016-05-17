#define CAL_OMP 1

extern "C"{
#include <OpenCAL-OMP/cal2DIO.h>
#include <OpenCAL-OMP/cal2D.h>
#include <OpenCAL-OMP/cal2DRun.h>
}
#include <stdlib.h>
#include <iostream>
#include <string>
#include <sstream>

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
#define DIMX 	(120)
#define DIMY 	(120)
#define STEPS 	(50000)

struct CALModel2D* life;
struct CALSubstate2Di *Q;
struct CALRun2D* life_simulation;
//if versio == 0 -> write in serial folder
#define PREFIX_PATH(version,name,pathVarName) \
	if(version==0)\
		 pathVarName="./testsout/serial/"name;\
	 else if(version>0)\
		 pathVarName="./testsout/other/"name;

void life_transition_function(struct CALModel2D* life, int i, int j)
{
	int sum = 0, n;
	for (n=1; n<life->sizeof_X; n++)
		sum += calGetX2Di(life, Q, i, j, n);

	if ((sum == 3) || (sum == 2 && calGet2Di(life, Q, i, j) == 1))
		calSet2Di(life, Q, i, j, 1);
	else
		calSet2Di(life, Q, i, j, 0);
}

void setGlider(struct CALModel2D* life,int dx, int dy){
	//set a glider
	calInit2Di(life, Q, 0+dx, 2+dy, 1);
	calInit2Di(life, Q, 1+dx, 0+dy, 1);
	calInit2Di(life, Q, 1+dx, 2+dy, 1);
	calInit2Di(life, Q, 2+dx, 1+dy, 1);
	calInit2Di(life, Q, 2+dx, 2+dy, 1);

}

void setToad(struct  CALModel2D* life,int dx, int dy){
	//set a Toad Pulsar
	calInit2Di(life, Q, 0+dx, 1+dy, 1);
	calInit2Di(life, Q, 0+dx, 2+dy, 1);
	calInit2Di(life, Q, 0+dx, 3+dy, 1);

	calInit2Di(life, Q, 1+dx, 0+dy, 1);
	calInit2Di(life, Q, 1+dx, 1+dy, 1);
	calInit2Di(life, Q, 1+dx, 2+dy, 1);


}

//size DIMX DIMY alway >= 20
void init(CALModel2D* life){
    setGlider(life,1,1);
    setGlider(life,DIMX-5,DIMY-5);
    setToad(life,5,DIMY-5);

}

CALbyte lifeSimulationStopCondition(struct CALModel2D* life)
{

	if (life_simulation->step >= STEPS)
		return CAL_TRUE;
	return CAL_FALSE;
}

int version=0;
string path;
string step="";
CALbyte simulationRun(){



//printf("STEPS IS ddfd%d\n",life_simulation->step);
	CALbyte again;

	//simulation main loop
	//exectutes the global transition function, the steering function and check for the stop 	condition.
	again = calRunCAStep2D(life_simulation);
	life_simulation->step++;
/*	step=NumberToString( life_simulation->step );
	PREFIX_PATH(version,"",path);
	path+=step;
	path+=".txt";
	calSaveSubstate2Di(life, Q, (char*)path.c_str());*/

	return again;
}

int main(int argc, char**argv)
{


	if ( sscanf (argv[1], "%i", &version)!=1 && version >=0) {
		printf ("error");
		exit(-1);
	 }

	//cadef and rundef
	life = calCADef2D(DIMX, DIMY, CAL_MOORE_NEIGHBORHOOD_2D, CAL_SPACE_TOROIDAL, CAL_NO_OPT);
	life_simulation = calRunDef2D(life, 1, CAL_RUN_LOOP, CAL_UPDATE_EXPLICIT);

	//add substates
	Q = calAddSubstate2Di(life);

	//add transition function's elementary processes.
	calAddElementaryProcess2D(life, life_transition_function);


	//set the whole substate to 0
	calInitSubstate2Di(life, Q, 0);

	calRunAddInitFunc2D(life_simulation, init);
	calRunInitSimulation2D(life_simulation);
	calRunAddStopConditionFunc2D(life_simulation, lifeSimulationStopCondition);

    PREFIX_PATH(version,"1.txt",path);
    calSaveSubstate2Di(life, Q, (char*)path.c_str());
	while(simulationRun())	{

	}
    PREFIX_PATH(version,"2.txt",path);
    calSaveSubstate2Di(life, Q, (char*)path.c_str());
	//finalization
	calRunFinalize2D(life_simulation);
	calFinalize2D(life);

	return 0;
}

//-----------------------------------------------------------------------
