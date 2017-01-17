#include <mpi.h>
#include <stdio.h>
#include <vector>
#include <string>
#include <iostream>
#include <utility>
#include <time.h>
#include <OpenCAL-CL/calclMultiNode.h>

#define ROWS (64)
#define COLS (16)

//#define ACTIVE_CELLS

#ifdef ACTIVE_CELLS
#define KERNEL_SRC "/home/mpiuser/git/mpi/kernel_life_active/source/"
#define KERNEL_INC "/home/mpiuser/git/mpi/kernel_life_active/include/"
#else
#define KERNEL_SRC "/home/mpiuser/git/mpi/kernel_life/source/"
#define KERNEL_INC "/home/mpiuser/git/mpi/kernel_life/include/"
#endif

#define P_R 0.5
#define P_EPSILON 0.001
// Defining kernels' names(struct CALCLMultiGPU*)
#define KERNEL_LIFE_TRANSITION_FUNCTION "lifeTransitionFunction"

// Declare XCA model (host_CA), substates (Q), parameters (P)
struct CALSubstate2Di* Q;

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

#ifdef ACTIVE_CELLS
    struct CALModel2D* host_CA = calCADef2D(
        mynode.workload, mynode.columns, CAL_MOORE_NEIGHBORHOOD_2D, CAL_SPACE_TOROIDAL, CAL_OPT_ACTIVE_CELLS_NAIVE);
#else
    struct CALModel2D* host_CA =
        calCADef2D(mynode.workload, mynode.columns, CAL_MOORE_NEIGHBORHOOD_2D, CAL_SPACE_TOROIDAL, CAL_NO_OPT);
#endif
    // Register the substate to the host CA
    Q = calAddSubstate2Di(host_CA);

    // Initialize the substate to 0 everywhere
    calInitSubstate2Di(host_CA, Q, 0.0);

    if (rank >= 0) {
	int off = rank;
	// Set a glider
	calInit2Di(host_CA, Q, 0 + off, 2, 1);
	calInit2Di(host_CA, Q, 1 + off, 0, 1);
	calInit2Di(host_CA, Q, 1 + off, 2, 1);
	calInit2Di(host_CA, Q, 2 + off, 1, 1);
	calInit2Di(host_CA, Q, 2 + off, 2, 1);

	
#ifdef ACTIVE_CELLS
	for (int i = mynode.offset; i < mynode.workload; i++)
	    for (int j = 0; j < mynode.columns; j++) {
		calAddActiveCell2D(host_CA, i, j);
	    }
#endif

	printf("KERNEL_SRC = %s	 \n", KERNEL_SRC);
    }
    // calUpdate2D(host_CA);

    int borderSize = 1;

    // Define a device-side CAs
    calclMultiGPUDef2D(multigpu, host_CA, KERNEL_SRC, KERNEL_INC, borderSize, c->is_full_exchange());

    calclAddElementaryProcessMultiGPU2D(multigpu, KERNEL_LIFE_TRANSITION_FUNCTION);
    std::string s = "./life" + std::to_string(rank);
    calSaveSubstate2Di(multigpu->device_models[0]->host_CA, Q, (char*)s.c_str());
}

void finalize(struct CALCLMultiGPU* multigpu)
{
    // Saving results
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::cout << "sono il processo " << rank << " finalizzo\n";
    std::string s = "./lifeLAST" + std::to_string(rank);
    calSaveSubstate2Di(multigpu->device_models[0]->host_CA, Q, (char*)s.c_str());
}

void setUpParallelWork_OneNodeTwoGPU(Cluster& mn)
{
    struct Node n1;
    struct Device d1_0 = { 0, 0, ROWS / 2 };
    struct Device d1_1 = { 0, 1, ROWS / 2 };
    n1.devices.push_back(d1_0);
    n1.devices.push_back(d1_1);

    n1.workload = d1_0.workload + d1_1.workload;
    n1.columns = COLS;
    n1.offset = 0;
    mn.nodes.push_back(n1);
}

void setUpParallelWork_TwoNode(Cluster& mn)
{
    struct Node n1;
    struct Device d1_0 = { 0, 0, ROWS / 3 };
    struct Device d1_1 = { 0, 1, ROWS / 3 };
    n1.devices.push_back(d1_0);
    n1.devices.push_back(d1_1);

    n1.workload = d1_0.workload + d1_1.workload;
    n1.columns = COLS;
    n1.offset = 0;
    mn.nodes.push_back(n1);
	 //------node 2
    struct Node n2;

    struct Device d2_0 = { 0, 0, ROWS / 3  };

    n2.devices.push_back(d2_0);

    n2.workload = d2_0.workload;
    n2.columns = COLS;
    n2.offset = n1.workload;

    mn.nodes.push_back(n2);
	
}

void setUpParallelWork_ThreeNode(Cluster& mn)
{
    struct Node n1;
    struct Device d1_0 = { 0, 0, ROWS / 4 };
    struct Device d1_1 = { 0, 1, ROWS / 4 };
    n1.devices.push_back(d1_0);
    n1.devices.push_back(d1_1);

    n1.workload = d1_0.workload + d1_1.workload;
    n1.columns = COLS;
    n1.offset = 0;
    mn.nodes.push_back(n1);
	 //------node 2
    struct Node n2;

    struct Device d2_0 = { 0, 0, ROWS / 4  };

    n2.devices.push_back(d2_0);

    n2.workload = d2_0.workload;
    n2.columns = COLS;
    n2.offset = n1.workload;

    mn.nodes.push_back(n2);
	
	//------node 3
    struct Node n3;

    struct Device d3_0 = { 0, 0, ROWS / 4  };

    n3.devices.push_back(d3_0);

    n3.workload = d3_0.workload;
    n3.columns = COLS;
    n3.offset = n1.workload+n2.workload;

    mn.nodes.push_back(n3);
	
}


void setUpParallelWork(Cluster& mn)
{
    int R = 8192 * 2;
    int C = 8192 * 2;
    // int R = 16;
    //  int C = 16;

    //------node 1
    struct Node n1;
    struct Device d1_0 = { 0, 0, R / 3 };
    struct Device d1_1 = { 0, 1, R / 3 };

    n1.devices.push_back(d1_0);
    n1.devices.push_back(d1_1);

    n1.workload = d1_0.workload + d1_1.workload;
    n1.columns = C;
    n1.offset = 0;
    mn.nodes.push_back(n1);

    //------node 2
    struct Node n2;

    struct Device d2_0 = { 0, 0, R / 3 + 2 };

    n2.devices.push_back(d2_0);

    n2.workload = d2_0.workload;
    n2.columns = C;
    n2.offset = n1.workload;

    mn.nodes.push_back(n2);

    //------node 2
    struct Node n3;

    struct Device d3_0 = { 0, 2, 204 };

    n3.devices.push_back(d3_0);

    n3.workload = d3_0.workload;
    n3.columns = C;
    n3.offset = n1.workload + n2.workload;

    //  mn.nodes.push_back(n3);
}
void setUpParallelWorkOneNode(Cluster& mn)
{
    int R = ROWS;
    int C = COLS;
    //        int R =600;
    //        int C =200;

    //------node 1
    struct Node n1;

    struct Device d1_0 = { 0, 0, R };
    // struct Device d2_0 = {0,1,R/2};

    n1.devices.push_back(d1_0);
    // n1.devices.push_back(d2_0);

    n1.workload = d1_0.workload; //+d2_0.workload;
    n1.columns = C;
    n1.offset = 0;

    mn.nodes.push_back(n1);
}

int main()
{

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

    // setUpParallelWork(cluster);
    // setUpParallelWorkOneNode(cluster);
    //setUpParallelWork_OneNodeTwoGPU(cluster);
//setUpParallelWork_TwoNode(cluster);
setUpParallelWork_ThreeNode(cluster);
    MPI_Barrier(MPI_COMM_WORLD);

    MultiNode<decltype(init), decltype(finalize)> mn(cluster, world_rank, init, finalize);

    mn.allocateAndInit();

    MPI_Barrier(MPI_COMM_WORLD);

    mn.run(2500);

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
