/*
 * Copyright (c) 2016 OpenCALTeam (https://github.com/OpenCALTeam),
 * Telesio Research Group,
 * Department of Mathematics and Computer Science,
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


extern "C" {
	#include <OpenCAL/cal2DIO.h>
	#include <OpenCAL/cal2D.h>
	#include <OpenCAL/cal2DRun.h>
}
#include <stdlib.h>
#include<string>
#include<iostream>

using namespace std;
//-----------------------------------------------------------------------
//   THE LIFE CELLULAR AUTOMATON
//-----------------------------------------------------------------------
#define DIMX 	(100)
#define DIMY 	(100)
#define STEPS 	(100)

struct CALModel2D* life;
struct CALSubstate2Di *I;
struct CALSubstate2Dr *R;
struct CALSubstate2Db *B;

struct CALRun2D* life_simulation;
//if versio == 0 -> write in serial folder
#define PREFIX_PATH(version,name,pathVarName) \
	if(version==0)\
		 pathVarName="./testsout/serial/"name;\
	 else if(version>0)\
		 pathVarName="./testsout/other/"name;
void life_transition_function(struct CALModel2D* life, int i, int j)
{
		calSet2Di(life, I, i, j, calGet2Di(life,I,i,j));
		calSet2Dr(life, R, i, j, calGet2Dr(life,R,i,j));
		calSet2Db(life, B, i, j, calGet2Db(life,B,i,j));
}

int main(int argc, char** argv)
{
	int version=0;
	if (sscanf (argv[1], "%i", &version)!=1 && version >=0) {
		printf ("error - not an integer");
		exit(-1);
	 }

	//cadef and rundef
	life = calCADef2D(DIMX, DIMY, CAL_MOORE_NEIGHBORHOOD_2D, CAL_SPACE_TOROIDAL, CAL_NO_OPT);
	life_simulation = calRunDef2D(life, 1, STEPS, CAL_UPDATE_EXPLICIT);

	//add substates
	I = calAddSubstate2Di(life);
	R = calAddSubstate2Dr(life);
	B = calAddSubstate2Db(life);

	//add transition function's elementary processes.
	calAddElementaryProcess2D(life, life_transition_function);


	//set the whole substate to 0
	calInitSubstate2Di(life, I, 12345);
	calInitSubstate2Dr(life, R, 1.98765432);
	calInitSubstate2Db(life, B, 0);


	//saving initial configuration
	string path;
	PREFIX_PATH(version,"1.txt",path);
	calSaveSubstate2Di(life, I, (char*)path.c_str());

	PREFIX_PATH(version,"2.txt",path);
	calSaveSubstate2Dr(life, R, (char*)path.c_str());

	PREFIX_PATH(version,"3.txt",path);
	calSaveSubstate2Db(life, B, (char*)path.c_str());


	//simulation run
	calRun2D(life_simulation);

	//saving configuration
	PREFIX_PATH(version,"4.txt",path);
	calSaveSubstate2Di(life, I, (char*)path.c_str());

	PREFIX_PATH(version,"5.txt",path);
	calSaveSubstate2Dr(life, R, (char*)path.c_str());

	PREFIX_PATH(version,"6.txt",path);
	calSaveSubstate2Db(life, B, (char*)path.c_str());

	//finalization
	calRunFinalize2D(life_simulation);
	calFinalize2D(life);

	return 0;
}

//-----------------------------------------------------------------------
