#include <mpi.h>
#include <stdio.h>
#include <vector>
#include <string>
#include <iostream>
#include <utility>
#include <time.h>
#include <OpenCAL-OMP/cal3DMultiNode.h>

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

// Declare XCA model (host_CA), substates (Q), parameters (P)
struct CALModel3D* host_CA;							//the cellular automaton
struct CALSubstate3Dr *Q_temperature;							//the substate Q
struct CALSubstate3Db *Q_material;							//

Node node;
const CALreal radius = 5;
int offset = 1;

// The cell's transition function (first and only elementary process)
void heatModel_TransitionFunction(struct CALModel3D* heatModel, int i, int j, int k)
{
        // printf("node->totalnumbersoflayers = %d \n",node.totalnumbersoflayers);
        if(i > 1 && i < heatModel->rows-1 && j > 1 && j < heatModel->columns-1 && k+(offset-1) > 0 && k+(offset-1) < (node.totalnumbersoflayers)-1){
            // if(heatModel->slices == 4) 
        
		CALreal currValue =calGet3Dr(heatModel, Q_temperature , i , j , k );

		CALreal dx2 = (calGet3Dr(heatModel, Q_temperature , i+1,j,k) + calGet3Dr(heatModel,Q_temperature ,i-1,j,k) - (2*currValue))/(DELTA_X*DELTA_X);


		CALreal dy2 = (calGet3Dr(heatModel,Q_temperature ,i,j+1,k) + calGet3Dr(heatModel,Q_temperature ,i,j-1,k) - (2*currValue))/(DELTA_Y*DELTA_Y);


		CALreal dz2 = (calGet3Dr(heatModel,Q_temperature ,i,j,k+1) + calGet3Dr(heatModel,Q_temperature ,i,j,k-1) - (2*currValue))/(DELTA_Z*DELTA_Z);

		CALreal newValue = currValue + DELTA_T*THERMAL_DIFFUSIVITY_WATER * (dx2 + dy2 +dz2);


		//||
		if(newValue > EPSILON  && newValue < 10000){
             //printf(" heatModel->slices = %d \n", heatModel->slices); 
            // if(i == 2 && j == 2 && k == 2 && heatModel->slices == 5)
            // {
            //      printf(" currValue = %f, newValue = %f, %f %f \n", currValue, newValue, calGet3Dr(heatModel,Q_temperature ,i,j,k+1), calGet3Dr(heatModel,Q_temperature ,i,j,k-1));
            // }
                       // printf("newVal i,j,k = %i, %i, %i -> dx2=%.15f , dy2=%.15f , dz2=%.15f , val =%.15f \n" ,i,j,k, dx2,dy2,dz2,newValue);
			calSet3Dr(heatModel, Q_temperature, i, j, k, newValue);
			newValue = currValue;

		}

		// CALint _i = i -((heatModel->rows-1)/2);
		// CALint	_j = j -(heatModel->columns/2);
		// CALint	_z = k -(heatModel->slices/2);
		// if(_i *_i + _j*_j + _z * _z <= radius){
        //                // printf("Temp at source is %f  ->>>>>",newValue);
        //                // printf("newVal i,j,k = %i, %i, %i -> dx2=%.15f , dy2=%.15f , dz2=%.15f , val =%.15f \n" ,i,j,k, dx2,dy2,dz2,newValue);
		// 	//calSet3Dr(heatModel, Q_temperature, i, j, k, newValue-0.1);
		// 	//calSet3Dr(heatModel, Q_temperature, i, j, k, newValue+0.00001);
		// }

	}else{
		calSet3Dr(heatModel, Q_temperature, i, j, k, 0);
	}


}

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
            
            for (z=host_CA->offset; z < host_CA->slices-host_CA->offset; ++z) {

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


void init(MultiNode * multinode, Node& mynode)
{
    node = mynode;
    //printf("node->totalnumbersoflayers = %d \n",node.totalnumbersoflayers);
    int borderSize = 1;
    struct CALModel3D* host_CA =
        calCADef3DMN(mynode.rows, mynode.columns, mynode.workload, CAL_MOORE_NEIGHBORHOOD_3D, CAL_SPACE_TOROIDAL, CAL_NO_OPT, borderSize);

    offset = mynode.offset;
//    // Register the substate to the host CA
    Q_temperature = calAddSubstate3Dr(host_CA);
    Q_material    = calAddSubstate3Db(host_CA);

//    // Initialize the substate to 0 everywhere
    calInitSubstate3Dr(host_CA, Q_temperature, (CALreal)0.0f);
    calInitSubstate3Db(host_CA, Q_material, CAL_FALSE);

//    //calApplyElementaryProcess3D(host_CA, heatModel_SimulationInit);
    heatModel_SimulationInit(host_CA);
    calUpdate3D(host_CA);

    calAddElementaryProcess3D(host_CA, heatModel_TransitionFunction);

    struct CALRun3D * host_simulation = calRunDef3D(host_CA, 1, 1, CAL_UPDATE_IMPLICIT);
    

    multinode->setRunSimulation(host_simulation);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::string temperature_str = "./heat_" + std::to_string(rank)+"_initial.txt";
    std::string material_str = "./material_" + std::to_string(rank)+"_initial.txt";
    calNodeSaveSubstate3Dr(multinode->host_CA, Q_temperature, (char*)temperature_str.c_str(), mynode);
    calNodeSaveSubstate3Db(multinode->host_CA, Q_material, (char*)material_str.c_str(), mynode);

    
}

void finalize(MultiNode * multinode, Node& mynode)
{
    // Saving results
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::cout << "sono il processo " << rank << " finalizzo\n";

    char temperature_str[256];
    char material_str[256];
    char rank_buffer[256];
    sprintf(rank_buffer,"%d",rank);

    strcpy(temperature_str, "./heat_");
    strcat(temperature_str, rank_buffer);
    strcat(temperature_str, ".txt");

    strcpy(material_str, "./material_");
    strcat(material_str, rank_buffer);
    strcat(material_str, ".txt");

    // std::string temperature_str = "./heat_" + std::to_string(rank)+".txt";
    // std::string material_str = "./material_" + std::to_string(rank)+".txt";
  
    // calNodeSaveSubstate3Dr(multinode->host_CA, Q_temperature, (char*)temperature_str.c_str(), mynode);
    // calNodeSaveSubstate3Db(multinode->host_CA, Q_material, (char*)material_str.c_str(), mynode);

    calNodeSaveSubstate3Dr(multinode->host_CA, Q_temperature, (char*)temperature_str, mynode);
    calNodeSaveSubstate3Db(multinode->host_CA, Q_material, (char*)material_str, mynode);
}

int main(int argc, char** argv)
{

    CALDistributedDomain3D domain = calDomainPartition3D(argc,argv);
    MultiNode mn(domain, init, finalize);
    mn.allocateAndInit();
    mn.run(100);
    return 0;
}
