#include <mpi.h>
#include <stdio.h>
#include <vector>
#include<string>
#include<iostream>
#include <utility>
#include<OpenCAL-CL/calclMultiNode.h>
//#define ACTIVE_CELLS

#define OUTPUT_PATH "./data/width_final_"
#define NUMBER_OF_OUTFLOWS 4


//test local
// #define DEM_PATH "/home/mpiuser/git/sciddicaT-local/testData/testLocal/dem.txt"
// #define SOURCE_PATH "/home/mpiuser/git/sciddicaT-local/testData/testLocal/source.txt"


//STANDARD
#define DEM_PATH "./data/dem_standard.txt"
#define SOURCE_PATH "./data/source_standard.txt"
//STRESS TEST R
//#define DEM_PATH "./testData/stress_test_R/dem_stress_test_R.txt"
//#define SOURCE_PATH "./testData/stress_test_R/source_stress_test_R.txt"

//STRESS TEST RD Rick
//#define DEM_PATH "/home/mpiuser/git/sciddicaT/testData/stress_test_RD/etnad.txt"
//#define SOURCE_PATH "/home/mpiuser/git/sciddicaT/testData/stress_test_RD/sourced.txt"

//STRESS TEST RD 
//#define DEM_PATH "/home/mpiuser/git/sciddicaT/testData/stress_test_R/dem_stress_test_RD.txt"
//#define SOURCE_PATH "/home/mpiuser/git/sciddicaT/testData/stress_test_R/source_stress_test_RD.txt"


#ifdef ACTIVE_CELLS
#define KERNEL_SRC "./kernelActive/source/"
#define KERNEL_INC "./kernelActive/include/"
#else
#define KERNEL_SRC "./kernel_sciddicaT/source/"
#define KERNEL_INC "./kernel_sciddicaT/include/"

#endif


#define P_R 0.5
#define P_EPSILON 0.001
// Defining kernels' names(struct CALCLMultiGPU*)
#define KERNEL_ELEM_PROC_FLOW_COMPUTATION "flowsComputation"
#define KERNEL_ELEM_PROC_WIDTH_UPDATE "widthUpdate"
#define KERNEL_STEERING  "steering"


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
CALCLmem * buffersKernelFlowComp;
CALCLmem bufferEpsilonParameter;
CALCLmem bufferRParameter;


// SciddicaT simulation init function
void sciddicaTSimulationInit(struct CALModel2D* host_CA) {
    CALreal z, h;
    CALint i, j;

    //initializing substates to 0
    calInitSubstate2Dr(host_CA, Q.f[0], 0);
    calInitSubstate2Dr(host_CA, Q.f[1], 0);
    calInitSubstate2Dr(host_CA, Q.f[2], 0);
    calInitSubstate2Dr(host_CA, Q.f[3], 0);

    //sciddicaT parameters sett1ing
    P.r = P_R;
    P.epsilon = P_EPSILON;

    //sciddicaT source initialization
    for (i = 0; i < host_CA->rows; i++)
        for (j = 0; j < host_CA->columns; j++) {
            h = calGet2Dr(host_CA, Q.h, i, j);

            if (h > 0.0) {
                z = calGet2Dr(host_CA, Q.z, i, j);
                calSet2Dr(host_CA, Q.z, i, j, z - h);

#ifdef ACTIVE_CELLS
                //adds the cell (i, j) to the set of active ones
                calAddActiveCell2D(host_CA, i, j);
#endif
            }
        }
}



void init( struct CALCLMultiGPU* multigpu , const Cluster* c){

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    Node mynode = c->nodes[rank];
    auto devices = mynode.devices;


    // calclPrintPlatformsAndDevices(calcl_device_manager);
    struct CALCLDeviceManager * calcl_device_manager = calclCreateManager();

    calclSetNumDevice(multigpu,devices.size());
    for(auto& d : devices){
        calclAddDevice(multigpu,calclGetDevice(calcl_device_manager, d.num_platform , d.num_device) ,  d.workload);
    }

CALModel2D* host_CA;
#ifdef ACTIVE_CELLS
    host_CA = calCADef2D(mynode.workload, mynode.columns, CAL_VON_NEUMANN_NEIGHBORHOOD_2D, CAL_SPACE_TOROIDAL, CAL_OPT_ACTIVE_CELLS_NAIVE);
#else
    host_CA = calCADef2D(mynode.workload, mynode.columns, CAL_VON_NEUMANN_NEIGHBORHOOD_2D, CAL_SPACE_TOROIDAL, CAL_NO_OPT);
#endif
    // Add substates
    Q.f[0] = calAddSubstate2Dr(host_CA);
    Q.f[1] = calAddSubstate2Dr(host_CA);
    Q.f[2] = calAddSubstate2Dr(host_CA);
    Q.f[3] = calAddSubstate2Dr(host_CA);
    Q.z = calAddSubstate2Dr(host_CA);
    Q.h = calAddSubstate2Dr(host_CA);


    int my_readoffset, my_writeoffset=0;
    my_readoffset = mynode.offset;
    // Load configuration
    std::cout << "-------------------------------------------------------------PRIMA DI DOPO" << std::endl;
    calLoadSubstate2DrMulti(host_CA, Q.z, DEM_PATH,my_readoffset,my_writeoffset);//TODO offset e workload
    std::cout << "-------------------------------------------------------------DOPO DI PRIMA" << std::endl;
    calLoadSubstate2DrMulti(host_CA, Q.h, SOURCE_PATH,my_readoffset,my_writeoffset);//TODO offset e workload

    // Initialization
    sciddicaTSimulationInit(host_CA);
    calUpdate2D(host_CA);

    int borderSize=1;

    // Define a device-side CAs
    calclMultiGPUDef2D(multigpu,host_CA,KERNEL_SRC,KERNEL_INC, borderSize,mynode.devices, c->is_full_exchange());

    // Extract kernels from program
    calclAddElementaryProcessMultiGPU2D(multigpu, KERNEL_ELEM_PROC_FLOW_COMPUTATION);
    calclAddElementaryProcessMultiGPU2D(multigpu, KERNEL_ELEM_PROC_WIDTH_UPDATE);



    bufferEpsilonParameter = calclCreateBuffer(multigpu->context, &P.epsilon, sizeof(CALParameterr));
    bufferRParameter = calclCreateBuffer(multigpu->context, &P.r, sizeof(CALParameterr));


    //    calclSetKernelArg2D(KERNEL_ELEM_PROC_FLOW_COMPUTATION, 0, sizeof(CALCLmem), &bufferEpsilonParameter);
    //    calclSetKernelArg2D(KERNEL_ELEM_PROC_FLOW_COMPUTATION, 1, sizeof(CALCLmem), &bufferRParameter);

    calclAddSteeringFuncMultiGPU2D(multigpu,KERNEL_STEERING);
    calclSetKernelArgMultiGPU2D(multigpu,KERNEL_ELEM_PROC_FLOW_COMPUTATION, 0,sizeof(CALCLmem), &bufferEpsilonParameter);
    calclSetKernelArgMultiGPU2D(multigpu,KERNEL_ELEM_PROC_FLOW_COMPUTATION, 1, sizeof(CALCLmem), &bufferRParameter);

    //std::string s = OUTPUT_PATH;
    //s+= "init";
//calSaveSubstate2Dr(multigpu->device_models[0]->host_CA, Q.h, (char*)s.c_str());

}

void finalize(struct CALCLMultiGPU* multigpu){
    // Saving results
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::cout<<"sono il processo "<<rank<<" finalizzo\n";
    std::string s = OUTPUT_PATH + std::to_string(rank) + ".txt";
    calSaveSubstate2Dr(multigpu->device_models[0]->host_CA, Q.h, (char*)s.c_str());
}

/*void setUpParallelWork(Cluster& mn){
    int R =3593;
    int C =3730;

    //------node 1
    struct Node n1;
    struct Device d1_0 = {0,0,R/4};
    struct Device d1_1 = {0,1,R/4};
    struct Device d1_2 = {0,2,R/4};

    n1.devices.push_back(d1_0);
    n1.devices.push_back(d1_1);
    n1.devices.push_back(d1_2);


    n1.workload = d1_0.workload+d1_1.workload+d1_2.workload;
    n1.columns=C;
    n1.offset = 0;
    mn.nodes.push_back(n1);

    //------node 2
    struct Node n2;

    struct Device d2_0 = {0,0,R/4+1};

    n2.devices.push_back(d2_0);

    n2.workload = d2_0.workload;
    n2.columns=C;
    n2.offset = n1.workload;

    mn.nodes.push_back(n2);


}
void setUpParallelWorkOneNode(Cluster& mn){
    int R =3593;
    int C =3730;

    //------node 1
    struct Node n1;

    struct Device d0_0 = {0,0,R/2};
    struct Device d0_1 = {0,1,R/2+1};
//    struct Device d0_2 = {0,2,R/3+2};

    n1.devices.push_back(d0_0);
   n1.devices.push_back(d0_1);
 //   n1.devices.push_back(d0_2);

    n1.workload = d0_0.workload+d0_1.workload;//+d0_2.workload;
    n1.columns=C;
    n1.offset = 0;

    mn.nodes.push_back(n1);
}*/


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

int main(int argc, char** argv){


    try{
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

    //TODO registrare funzioni di init e finalize all'interno di OpenCALMPI

    //setUpParallelWork(cluster);
    //setUpParallelWorkOneNode(cluster);
    cluster.fromClusterFile(clusterfile);

    MPI_Barrier(MPI_COMM_WORLD);


    MultiNode<decltype (init),decltype (finalize)> mn(cluster, world_rank, init , finalize);

    mn.allocateAndInit();
    MPI_Barrier(MPI_COMM_WORLD);
    mn.run(4000);
    MPI_Barrier(MPI_COMM_WORLD);

    // Finalize the MPI environment.
    MPI_Finalize();
    // Print off a hello world message


    return 0;
    }
    catch (const std::exception& e){
    return -1;
    }

}
