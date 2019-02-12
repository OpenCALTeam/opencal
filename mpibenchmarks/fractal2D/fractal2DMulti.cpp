#include <mpi.h>
#include <stdio.h>
#include <vector>
#include <string>
#include <iostream>
#include <utility>
#include <time.h>

#include <OpenCAL-CL/calclMultiNode.h>





#define KERNEL_SRC "./kernel_fractal2D/source/"
#define KERNEL_INC "./kernel_fractal2D/include/"


// Defining kernels' names(struct CALCLMultiGPU*)
#define KERNEL_LIFE_TRANSITION_FUNCTION "fractal2D_transitionFunction"

// Declare XCA model (host_CA), substates (Q), parameters (P)
struct CALSubstate2Di *Q_fractal;		//the substate Q





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
		calclAddDevice(multigpu, 
						calclGetDevice(calcl_device_manager, d.num_platform, d.num_device),
						d.workload);
    }


    struct CALModel2D* host_CA =
        calCADef2D(mynode.workload, mynode.columns, CAL_MOORE_NEIGHBORHOOD_2D, CAL_SPACE_TOROIDAL, CAL_NO_OPT);

    // Register the substate to the host CA
    Q_fractal = calAddSubstate2Di(host_CA);



    int borderSize = 1;

    // Define a device-side CAs
    calclMultiGPUDef2D(multigpu, host_CA, KERNEL_SRC, KERNEL_INC, borderSize, mynode.devices, c->is_full_exchange());
	calclAddElementaryProcessMultiGPU2D(multigpu, KERNEL_LIFE_TRANSITION_FUNCTION);
    
	
}

void finalize(struct CALCLMultiGPU* multigpu)
{
    // Saving results
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::cout << "sono il processo " << rank << " finalizzo\n";

	std::string fractal_str = "./fractal" + std::to_string(rank)+".txt";
	calSaveSubstate2Di(multigpu->device_models[0]->host_CA, Q_fractal, (char*)fractal_str.c_str());
	

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
		//setenv("CUDA_CACHE_DISABLE", "1", 1);
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

		mn.run(1);

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
