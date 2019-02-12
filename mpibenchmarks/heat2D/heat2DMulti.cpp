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

void init(struct CALCLMultiGPU* multigpu, const Cluster* c)
{

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    Node mynode = c->nodes[rank];
    auto devices = mynode.devices;

    // calclPrintPlatformsAndDevices(calcl_device_manager);
    struct CALCLDeviceManager* calcl_device_manager = calclCreateManager();

    calclSetNumDevice(multigpu, devices.size());
    for (auto& d : devices) {
	calclAddDevice(multigpu, calclGetDevice(calcl_device_manager, d.num_platform, d.num_device), d.workload);
    }


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
    calclMultiGPUDef2D(multigpu, host_CA, KERNEL_SRC, KERNEL_INC, borderSize, devices, c->is_full_exchange());

    calclAddElementaryProcessMultiGPU2D(multigpu, KERNEL_LIFE_TRANSITION_FUNCTION);
    
	std::string temperature_str = "./heat_" + std::to_string(rank)+"_initial.txt";
	std::string material_str = "./material_" + std::to_string(rank)+"_initial.txt";
    calSaveSubstate2Dr(multigpu->device_models[0]->host_CA, Q_temperature, (char*)temperature_str.c_str());
	calSaveSubstate2Db(multigpu->device_models[0]->host_CA, Q_material, (char*)material_str.c_str());
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


string parseCommandLineArgs(int argc, char** argv)
{
    using std::cerr;
    using std::cout;
    using std::endl;
    bool go = true;
    string s;
    if (argc != 2) {
	cout << "Usage ./mytest clusterfile" << endl;
	go = false;
    } else {
	s = argv[1];
    }

    if (!go) {
	cout << "exiting..." << endl;
	exit(-1);
    }
    return s;
}
int main(int argc, char** argv)
{

    try{
		//force kernel recompilation
		setenv("CUDA_CACHE_DISABLE", "1", 1);
		string clusterfile;
		clusterfile = parseCommandLineArgs(argc, argv);
		Cluster cluster;
		// Initialize the MPI environment
		MPI_Init(NULL, NULL);

		// Get the number of processes
		int world_size;
		MPI_Comm_size(MPI_COMM_WORLD, &world_size);

		// Get the rank of the process
		int world_rank;
		MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

		// Get the name of the processor
		char processor_name[MPI_MAX_PROCESSOR_NAME];
		int name_len;
		MPI_Get_processor_name(processor_name, &name_len);

		// TODO registrare funzioni di init e finalize all'interno di OpenCALMPI

		cluster.fromClusterFile(clusterfile);

		MPI_Barrier(MPI_COMM_WORLD);

		MultiNode<decltype(init), decltype(finalize)> mn(cluster, world_rank, init, finalize);

		mn.allocateAndInit();

		MPI_Barrier(MPI_COMM_WORLD);

		mn.run(STEPS);

		// Print off a hello world message
		printf("Hello world from processor %s, rank %d"
			   " out of %d processors\n",
			   processor_name,
			   world_rank,
			   world_size);

		MPI_Barrier(MPI_COMM_WORLD);

		// Finalize the MPI environment.
		MPI_Finalize();

		return 0;
    }
    catch (const std::exception& e){
		return -1;
    }
}
