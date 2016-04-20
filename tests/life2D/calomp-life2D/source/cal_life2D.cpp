/*
 * Copyright (c) 2016 OpenCALTeam (https://github.com/OpenCALTeam),
 * University of Calabria, Italy.
 *
 * This file is part of OpenCAL (Open Computing Abstraction Layer).
 *
 * OpenCAL is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3 of
 * the License, or (at your option) any later version.
 *
 * OpenCAL is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with OpenCAL. If not, see <http://www.gnu.org/licenses/>.
 */

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

#define DIMX 	(100)
#define DIMY 	(100)
#define STEPS 	(100)

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
	setGlider(life,94,94);
	setToad(life,15,22);

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

	step=NumberToString( life_simulation->step );
	PREFIX_PATH(version,"",path);
	path+=step;
	path+=".txt";
	calSaveSubstate2Di(life, Q, (char*)path.c_str());

	CALbyte again;

	//simulation main loop
	life_simulation->step++;

	//exectutes the global transition function, the steering function and check for the stop 	condition.
	again = calRunCAStep2D(life_simulation);
	step=NumberToString( life_simulation->step );
	PREFIX_PATH(version,"",path);
	path+=step;
	path+=".txt";
	calSaveSubstate2Di(life, Q, (char*)path.c_str());

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

	while(simulationRun())	{

	}

	//finalization
	calRunFinalize2D(life_simulation);
	calFinalize2D(life);

	return 0;
}

//-----------------------------------------------------------------------
