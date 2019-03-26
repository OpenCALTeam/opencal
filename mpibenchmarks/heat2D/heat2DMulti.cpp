#include <mpi.h>
#include <stdio.h>
#include <vector>
#include <string>
#include <iostream>
#include <utility>
#include <time.h>
#include <OpenCAL-CL/calclMultiNode.h>

#define SIZE (500)
#define ROWS (SIZE)
#define COLS (SIZE)

#define STEPS (50000)
#define MATERIAL_START_ROW (ROWS/2-ROWS/8)
#define SOURCE_SIZE (20)
#define MATERIAL_END_ROW (MATERIAL_START_ROW+SOURCE_SIZE)


#define KERNEL_SRC "./kernel_heat2D/source/"
#define KERNEL_INC "./kernel_heat2D/include/"


// Defining kernels' names(struct CALCLMultiGPU*)
#define KERNEL_LIFE_TRANSITION_FUNCTION "heat2D_transitionFunction"

// Declare XCA model (host_CA), substates (Q), parameters (P)
struct CALSubstate2Dr *Q_temperature;							//the substate Q
struct CALSubstate2Db *Q_material;


void heatModel_initMaterials (struct CALModel2D* host_CA, int i, int j){
		if(i > MATERIAL_START_ROW && i<MATERIAL_END_ROW)
        calSet2Db(host_CA, Q_material, i, j, CAL_TRUE);
    else
        calSet2Db(host_CA, Q_material, i, j, CAL_FALSE);
    
}

void init(struct CALCLMultiGPU* multigpu, const Node& mynode)
{
    struct CALModel2D* host_CA =
        calCADef2D(mynode.workload, mynode.columns, CAL_MOORE_NEIGHBORHOOD_2D, CAL_SPACE_TOROIDAL, CAL_NO_OPT);

    // Register the substate to the host CA
    Q_temperature = calAddSubstate2Dr(host_CA);
	Q_material = calAddSubstate2Db(host_CA);

    // Initialize the substate to 0 everywhere
	calInitSubstate2Dr(host_CA, Q_temperature, (CALreal)0.0f);
	calInitSubstate2Db(host_CA, Q_material, CAL_FALSE);

    calApplyElementaryProcess2D(host_CA, heatModel_initMaterials);
	calUpdate2D(host_CA);

    int borderSize = 1;

    // Define a device-side CAs
    calclMultiGPUDef2D(multigpu, host_CA, KERNEL_SRC, KERNEL_INC, borderSize, mynode.devices);

    calclAddElementaryProcessMultiGPU2D(multigpu, KERNEL_LIFE_TRANSITION_FUNCTION);
    
	// std::string temperature_str = "./heat_" + std::to_string(rank)+"_initial.txt";
	// std::string material_str = "./material_" + std::to_string(rank)+"_initial.txt";
    // calSaveSubstate2Dr(multigpu->device_models[0]->host_CA, Q_temperature, (char*)temperature_str.c_str());
	// calSaveSubstate2Db(multigpu->device_models[0]->host_CA, Q_material, (char*)material_str.c_str());
}

void finalize(struct CALCLMultiGPU* multigpu)
{
    // Saving results
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::cout << "sono il processo " << rank << " finalizzo\n";

	std::string temperature_str = "./heat_" + std::to_string(rank)+".txt";
	std::string material_str = "./material_" + std::to_string(rank)+".txt";  
  
     calSaveSubstate2Dr(multigpu->device_models[0]->host_CA, Q_temperature, (char*)temperature_str.c_str());
	calSaveSubstate2Db(multigpu->device_models[0]->host_CA, Q_material, (char*)material_str.c_str());

}

int main(int argc, char** argv)
{
	CALDistributedDomain2D domain = calDomainPartition2D(argc,argv);

	MultiNode mn(domain, init , finalize);
	mn.allocateAndInit();
	mn.run(STEPS);

	return 0;
}
