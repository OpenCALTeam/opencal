#include <mpi.h>
#include <stdio.h>
#include <vector>
#include <string>
#include <iostream>
#include <utility>
#include <time.h>
#include<SOIL.h>

#include <OpenCAL-CL/calclMultiNode.h>





#define KERNEL_SRC "/home/mpiuser/git/sobel2D/kernel_sobel2D/source/"
#define KERNEL_INC "/home/mpiuser/git/sobel2D/kernel_sobel2D/include/"


// Defining kernels' names(struct CALCLMultiGPU*)
#define KERNEL_LIFE_TRANSITION_FUNCTION "sobel2D_transitionFunction"

const std::string image_path="sobel_image-test2.jpg";

typedef struct{
	
	struct CALSubstate2Di *red;		//red channel
	struct CALSubstate2Di *green;		//green channel
	struct CALSubstate2Di *blue;		//blue substate Q

} Q_RGB;

Q_RGB image;


void readImage( CALModel2D* const host_CA, const std::string& path , const uint offset, const int ROWS, const int COLS){
	int _w,_h;
	unsigned char* soil_image = SOIL_load_image(path.c_str() , &_w,&_h , 0 , 	SOIL_LOAD_RGB);
	
	
	int start = 3*offset;	
	for(int i=0; i<ROWS; i++){
		for(int j=0; j<COLS; j++){
			
			calInit2Di(host_CA, image.red  , i, j, soil_image[start]);
			calInit2Di(host_CA, image.green, i, j, soil_image[start+1]);
			calInit2Di(host_CA, image.blue , i, j, soil_image[start+2]);
			
			start+=3;
		}
	}
		
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
		calclAddDevice(multigpu, 
						calclGetDevice(calcl_device_manager, d.num_platform, d.num_device),
						d.workload);
    }


    struct CALModel2D* host_CA =
        calCADef2D(mynode.workload, mynode.columns, CAL_MOORE_NEIGHBORHOOD_2D, CAL_SPACE_TOROIDAL, CAL_NO_OPT);

    // Register the substate to the host CA
    image.red = calAddSubstate2Di(host_CA);
	image.green = calAddSubstate2Di(host_CA);
	image.blue = calAddSubstate2Di(host_CA);


	//this read the image and update the substate using the right part of the image
	readImage(host_CA,image_path , mynode.offset, mynode.workload, mynode.columns);	


    int borderSize = 1;

    // Define a device-side CAs
    calclMultiGPUDef2D(multigpu, host_CA, KERNEL_SRC, KERNEL_INC, borderSize, mynode.devices, c->is_full_exchange());
	calclAddElementaryProcessMultiGPU2D(multigpu, KERNEL_LIFE_TRANSITION_FUNCTION);
    
	
	std::string fractal_str = "./sobel_red_initial" + std::to_string(rank)+".txt";
	calSaveSubstate2Di(multigpu->device_models[0]->host_CA, image.red, (char*)fractal_str.c_str());
	
	
}

void finalize(struct CALCLMultiGPU* multigpu)
{
    // Saving results
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::cout << "sono il processo " << rank << " finalizzo\n";

	std::string fractal_str = "./sobel_red" + std::to_string(rank)+".txt";
	calSaveSubstate2Di(multigpu->device_models[0]->host_CA, image.red, (char*)fractal_str.c_str());
	

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
