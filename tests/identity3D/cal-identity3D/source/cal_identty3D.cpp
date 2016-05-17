
extern "C" {
	#include <OpenCAL/cal3DIO.h>
	#include <OpenCAL/cal3D.h>
	#include <OpenCAL/cal3DRun.h>
}
#include <stdlib.h>
#include<string>
#include<iostream>

using namespace std;
//-----------------------------------------------------------------------
//   THE LIFE CELLULAR AUTOMATON
//-----------------------------------------------------------------------
#define DIMX 	(30)
#define DIMY 	(30)
#define LAYERS 	(30)
#define STEPS 	(1000)

struct CALModel3D* life;
struct CALSubstate3Di *I;
struct CALSubstate3Dr *R;
struct CALSubstate3Db *B;

struct CALRun3D* life_simulation;
//if versio == 0 -> write in serial folder
#define PREFIX_PATH(version,name,pathVarName) \
	if(version==0)\
		 pathVarName="./testsout/serial/"name;\
	 else if(version>0)\
		 pathVarName="./testsout/other/"name;
void life_transition_function(struct CALModel3D* life, int i, int j,int k)
{
		calSet3Di(life, I, i, j,k, calGet3Di(life,I,i,j,k));
		calSet3Dr(life, R, i, j,k, calGet3Dr(life,R,i,j,k));
		calSet3Db(life, B, i, j,k, calGet3Db(life,B,i,j,k));
}

int main(int argc, char** argv)
{
	int version=0;
	if (sscanf (argv[1], "%i", &version)!=1 && version >=0) {
		printf ("error - not an integer");
		exit(-1);
	 }

	//cadef and rundef
	life = calCADef3D(DIMX, DIMY, LAYERS,CAL_MOORE_NEIGHBORHOOD_3D, CAL_SPACE_TOROIDAL, CAL_NO_OPT);
	life_simulation = calRunDef3D(life, 1, STEPS, CAL_UPDATE_EXPLICIT);

	//add substates
	I = calAddSubstate3Di(life);
	R = calAddSubstate3Dr(life);
	B = calAddSubstate3Db(life);

	//add transition function's elementary processes.
	calAddElementaryProcess3D(life, life_transition_function);


	//set the whole substate to 0
	calInitSubstate3Di(life, I, 12345);
	calInitSubstate3Dr(life, R, 1.98765432);
	calInitSubstate3Db(life, B, 0);


	//saving initial configuration
	string path;
	PREFIX_PATH(version,"1.txt",path);
	calSaveSubstate3Di(life, I, (char*)path.c_str());

	PREFIX_PATH(version,"2.txt",path);
	calSaveSubstate3Dr(life, R, (char*)path.c_str());

	PREFIX_PATH(version,"3.txt",path);
	calSaveSubstate3Db(life, B, (char*)path.c_str());


	//simulation run
	calRun3D(life_simulation);

	//saving configuration
	PREFIX_PATH(version,"4.txt",path);
	calSaveSubstate3Di(life, I, (char*)path.c_str());

	PREFIX_PATH(version,"5.txt",path);
	calSaveSubstate3Dr(life, R, (char*)path.c_str());

	PREFIX_PATH(version,"6.txt",path);
	calSaveSubstate3Db(life, B, (char*)path.c_str());

	//finalization
	calRunFinalize3D(life_simulation);
	calFinalize3D(life);

	return 0;
}

//-----------------------------------------------------------------------
