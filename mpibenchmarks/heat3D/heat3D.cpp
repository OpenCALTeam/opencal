#include <mpi.h>
#include <stdio.h>
#include <vector>
#include <string>
#include <iostream>
#include <utility>
#include <time.h>
#include <OpenCAL-CL/calclMultiNode3D.h>

#define SIZE (50)
#define ROWS (SIZE)
#define COLS (SIZE)
#define LAYERS (SIZE)
#define EPSILON (0.01)

#define DELTA_X (0.001)
#define DELTA_Y (0.001)
#define DELTA_Z (0.001)
#define DELTA_T (0.001)
#define THERMAL_CONDUCTIVITY (1)
#define MASS_DENSITY (1)
#define SPECIFIC_HEAT_CAPACITY (1)
#define THERMAL_DIFFUSIVITY ( (THERMAL_CONDUCTIVITY)/(SPECIFIC_HEAT_CAPACITY)*(MASS_DENSITY) )
#define THERMAL_DIFFUSIVITY_WATER (1.4563e-4) //C/m^2
#define INIT_TEMP (1200)


#define KERNEL_SRC "./kernel_heat3D/source/"
#define KERNEL_INC "./kernel_heat3D/include/"


// Defining kernels' names(struct CALCLMultiGPU*)
#define KERNEL_LIFE_TRANSITION_FUNCTION "heat3D_transitionFunction"

// Declare XCA model (host_CA), substates (Q), parameters (P)
struct CALModel3D* host_CA;							//the cellular automaton
struct CALSubstate3Dr *Q_temperature;							//the substate Q
struct CALSubstate3Db *Q_material;							//

void heatModel_SimulationInit(struct CALModel3D* host_CA)
{
    //int i;
    int j, z;
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);


    calInitSubstate3Dr(host_CA, Q_temperature, (CALreal)0);
    calInitSubstate3Db(host_CA, Q_material, CAL_FALSE);


    //for(int i=1 ; i < ROWS ; ++i){
        for (j = 1; j < host_CA->columns; ++j) {
            if(rank == 0){
                z = 1;
            }else
                z =0;
            for (z; z < host_CA->slices; ++z) {

                CALreal _i, _j,_z;
                CALreal chunk = host_CA->rows/2;
                /*for(int l =2 ; l < 4; l++){
                    _i = i -(ROWS/l);
                    _j = i -(COLS/l);
                    _z = z -(LAYERS/l);
                    if(_i *_i + _j*_j + _z * _z <= radius){*/
                        calInit3Dr(host_CA, Q_temperature, chunk, j, z, INIT_TEMP);
                        calInit3Dr(host_CA, Q_temperature, chunk+1, j, z, INIT_TEMP);
                        calInit3Dr(host_CA, Q_temperature, chunk-1, j, z, INIT_TEMP);
                        //calSet3Dr(host_CA, Q_temperature, chunk*2, j, z, INIT_TEMP);
                        //calSet3Dr(host_CA, Q_temperature, chunk*3, j, z, INIT_TEMP);
                        //calSet3Db(host_CA, Q_heat_source, i, j, z, 1);


                }
            }
        }
//}


void init(struct CALCLMultiDevice3D* multidevice, const Node& mynode)
{
    struct CALModel3D* host_CA =
        calCADef3D(mynode.rows, mynode.columns, mynode.workload, CAL_MOORE_NEIGHBORHOOD_3D, CAL_SPACE_TOROIDAL, CAL_NO_OPT);

//    // Register the substate to the host CA
    Q_temperature = calAddSubstate3Dr(host_CA);
    Q_material = calAddSubstate3Db(host_CA);

//    // Initialize the substate to 0 everywhere
    calInitSubstate3Dr(host_CA, Q_temperature, (CALreal)0.0f);
    calInitSubstate3Db(host_CA, Q_material, CAL_FALSE);


//    //calApplyElementaryProcess3D(host_CA, heatModel_SimulationInit);
    heatModel_SimulationInit(host_CA);
    calUpdate3D(host_CA);

    int borderSize = 1;

    //Define a device-side CAs
    calclMultiDeviceCADef3D(multidevice, host_CA, KERNEL_SRC, KERNEL_INC, borderSize, mynode.devices);

    calclMultiDeviceAddElementaryProcess3D(multidevice, KERNEL_LIFE_TRANSITION_FUNCTION);
    
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::string temperature_str = "./heat_" + std::to_string(rank)+"_initial.txt";
    std::string material_str = "./material_" + std::to_string(rank)+"_initial.txt";
    calSaveSubstate3Dr(multidevice->device_models[0]->host_CA, Q_temperature, (char*)temperature_str.c_str());
    calSaveSubstate3Db(multidevice->device_models[0]->host_CA, Q_material, (char*)material_str.c_str());
}

void finalize(struct CALCLMultiDevice3D* multigpu)
{
    // Saving results
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::cout << "sono il processo " << rank << " finalizzo\n";

    std::string temperature_str = "./heat_" + std::to_string(rank)+".txt";
    std::string material_str = "./material_" + std::to_string(rank)+".txt";
  
    calSaveSubstate3Dr(multigpu->device_models[0]->host_CA, Q_temperature, (char*)temperature_str.c_str());
    calSaveSubstate3Db(multigpu->device_models[0]->host_CA, Q_material, (char*)material_str.c_str());

}

int main(int argc, char** argv)
{

    CALDistributedDomain3D domain = calDomainPartition3D(argc,argv);
    MultiNode3D mn(domain, init, finalize);
    mn.allocateAndInit();
    mn.run(10);
    return 0;
}
