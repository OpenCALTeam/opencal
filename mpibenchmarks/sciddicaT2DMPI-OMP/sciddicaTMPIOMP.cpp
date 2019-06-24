#include <mpi.h>
#include <stdio.h>
#include <vector>
#include <string>
#include <iostream>
#include <utility>
#include <OpenCAL-OMP/calMultiNodeOMP2D.h>
//#define ACTIVE_CELLS

#define OUTPUT_PATH "./data/width_final_"
#define NUMBER_OF_OUTFLOWS 4


//test local
// #define DEM_PATH "/home/mpiuser/git/sciddicaT-local/testData/testLocal/dem.txt"
// #define SOURCE_PATH "/home/mpiuser/git/sciddicaT-local/testData/testLocal/source.txt"


//STANDARD
#define DEM_PATH "./data/dem_standard.txt"
#define SOURCE_PATH "./data/source_standard.txt"
// #define DEM_PATH "./data/debugz.txt"
// #define SOURCE_PATH "./data/debugh.txt"
//STRESS TEST R
//#define DEM_PATH "./testData/stress_test_R/dem_stress_test_R.txt"
//#define SOURCE_PATH "./testData/stress_test_R/source_stress_test_R.txt"

//STRESS TEST RD Rick
//#define DEM_PATH "/home/mpiuser/git/sciddicaT/testData/stress_test_RD/etnad.txt"
//#define SOURCE_PATH "/home/mpiuser/git/sciddicaT/testData/stress_test_RD/sourced.txt"

//STRESS TEST RD 
//#define DEM_PATH "/home/mpiuser/git/sciddicaT/testData/stress_test_R/dem_stress_test_RD.txt"
//#define SOURCE_PATH "/home/mpiuser/git/sciddicaT/testData/stress_test_R/source_stress_test_RD.txt"

#define P_R 0.5
#define P_EPSILON 0.001


// Declare XCA model (host_CA), substates (Q), parameters (P)
struct CALModel2D* host_CA;
struct sciddicaTSubstates {
    struct CALSubstate2Dr *f[NUMBER_OF_OUTFLOWS];
    struct CALSubstate2Dr *z;
    struct CALSubstate2Dr *h;
} Q;

struct sciddicaTParameters {
    CALParameterr epsilon;
    CALParameterr r;
} P;



// The sigma_1 elementary process
void sciddicaTFlowsComputation(struct CALModel2D* sciddicaT, int i, int j)
{
	CALbyte eliminated_cells[5]={CAL_FALSE,CAL_FALSE,CAL_FALSE,CAL_FALSE,CAL_FALSE};
	CALbyte again;
	CALint cells_count;
	CALreal average;
	CALreal m;
	CALreal u[5];
	CALint n;
	CALreal z, h;

	if (calGet2Dr(sciddicaT, Q.h, i, j) <= P.epsilon)
		return;

	m = calGet2Dr(sciddicaT, Q.h, i, j) - P.epsilon;
	u[0] = calGet2Dr(sciddicaT, Q.z, i, j) + P.epsilon;
	for (n=1; n<sciddicaT->sizeof_X; n++)
	{
		z = calGetX2Dr(sciddicaT, Q.z, i, j, n);
		h = calGetX2Dr(sciddicaT, Q.h, i, j, n);
		u[n] = z + h;
	}

	//computes outflows
	do{
		again = CAL_FALSE;
		average = m;
		cells_count = 0;

		for (n=0; n<sciddicaT->sizeof_X; n++)
			if (!eliminated_cells[n]){
				average += u[n];
				cells_count++;
			}

			if (cells_count != 0)
				average /= cells_count;

			for (n=0; n<sciddicaT->sizeof_X; n++)
				if( (average<=u[n]) && (!eliminated_cells[n]) ){
					eliminated_cells[n]=CAL_TRUE;
					again=CAL_TRUE;
				}
	}while (again);

	for (n=1; n<sciddicaT->sizeof_X; n++)
		if (eliminated_cells[n])
			calSet2Dr(sciddicaT, Q.f[n-1], i, j, 0.0);
		else
			calSet2Dr(sciddicaT, Q.f[n-1], i, j, (average-u[n])*P.r);
}

// The sigma_2 elementary process
void sciddicaTWidthUpdate(struct CALModel2D* sciddicaT, int i, int j)
{
	CALreal h_next;
	CALint n;

	h_next = calGet2Dr(sciddicaT, Q.h, i, j);
	for(n=1; n<sciddicaT->sizeof_X; n++)
		h_next +=  calGetX2Dr(sciddicaT, Q.f[NUMBER_OF_OUTFLOWS - n], i, j, n) - calGet2Dr(sciddicaT, Q.f[n-1], i, j);

	calSet2Dr(sciddicaT, Q.h, i, j, h_next);
}

// SciddicaT simulation init function
void sciddicaTSimulationInit(struct CALModel2D* sciddicaT)
{
	CALreal z, h;
	CALint i, j;

	//initializing substates to 0
	calInitSubstate2Dr(sciddicaT, Q.f[0], 0);
	calInitSubstate2Dr(sciddicaT, Q.f[1], 0);
	calInitSubstate2Dr(sciddicaT, Q.f[2], 0);
	calInitSubstate2Dr(sciddicaT, Q.f[3], 0);

	//sciddicaT parameters setting
	P.r = P_R;
	P.epsilon = P_EPSILON;

	//sciddicaT source initialization
	for (i=1; i<sciddicaT->rows-1; i++)
		for (j=0; j<sciddicaT->columns; j++)
		{
			h = calGet2Dr(sciddicaT, Q.h, i, j);

			if ( h > 0.0 ) {
				z = calGet2Dr(sciddicaT, Q.z, i, j);
				calSet2Dr(sciddicaT, Q.z, i, j, z-h);
			}
		}
}

// SciddicaT steering function
void sciddicaTSteering(struct CALModel2D* sciddicaT)
{
	// set flow to 0 everywhere
	calInitSubstate2Dr(sciddicaT, Q.f[0], 0);
	calInitSubstate2Dr(sciddicaT, Q.f[1], 0);
	calInitSubstate2Dr(sciddicaT, Q.f[2], 0);
	calInitSubstate2Dr(sciddicaT, Q.f[3], 0);
}

void init( struct MultiNode * multinode, const Node& mynode){

CALModel2D* host_CA;
    int rank;
	int borderSize=1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::cout<<"sono il processo "<<rank<<" workload = " << mynode.workload+2 << " columns = " <<  mynode.columns<< "\n";
#ifdef ACTIVE_CELLS
    host_CA = calCADef2D(mynode.workload, mynode.columns, CAL_VON_NEUMANN_NEIGHBORHOOD_2D, CAL_SPACE_TOROIDAL, CAL_OPT_ACTIVE_CELLS_NAIVE);
#else
    host_CA = calCADef2DMN(mynode.workload, mynode.columns, CAL_VON_NEUMANN_NEIGHBORHOOD_2D, CAL_SPACE_TOROIDAL, CAL_NO_OPT, borderSize);
#endif
    // Add substates
    Q.f[0] = calAddSubstate2Dr(host_CA);
    Q.f[1] = calAddSubstate2Dr(host_CA);
    Q.f[2] = calAddSubstate2Dr(host_CA);
    Q.f[3] = calAddSubstate2Dr(host_CA);
    Q.z = calAddSubstate2Dr(host_CA);
    Q.h = calAddSubstate2Dr(host_CA);

    calAddElementaryProcess2D(host_CA, sciddicaTFlowsComputation);
    calAddElementaryProcess2D(host_CA, sciddicaTWidthUpdate);

    // Load configuration
	std::cout<<"sono il processo "<<rank<<" offset " << mynode.offset << "\n";
    calNodeLoadSubstate2Dr(host_CA, Q.z, DEM_PATH, mynode);//TODO offset e workload
    calNodeLoadSubstate2Dr(host_CA, Q.h, SOURCE_PATH, mynode);//TODO offset e workload

    // Initialization
    sciddicaTSimulationInit(host_CA);
    calUpdate2D(host_CA);

    struct CALRun2D * host_simulation = calRunDef2D(host_CA, 1, 1, CAL_UPDATE_IMPLICIT);
    //calRunAddInitFunc2D(host_simulation, sciddicaTSimulationInit);
    calRunAddSteeringFunc2D(host_simulation, sciddicaTSteering);
    multinode->setRunSimulation(host_simulation);

    std::string s = "h_" + std::to_string(rank) + ".txt";
    calSaveSubstate2Dr(multinode->host_CA, Q.h, (char*)s.c_str());

}

void finalize(struct MultiNode * multinode, const Node& mynode){
    // Saving results
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::string s = OUTPUT_PATH + std::to_string(rank) + ".txt";
    calNodeSaveSubstate2Dr(multinode->host_CA, Q.h, (char*)s.c_str(), mynode);
}

int main(int argc, char** argv){

    CALDistributedDomain2D domain = calDomainPartition2D(argc,argv);

    MultiNode mn(domain, init, finalize);
    mn.allocateAndInit();
    mn.run(4000);

    return 0;
}
