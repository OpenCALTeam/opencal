#ifndef CALCLMULTINODE_H_
#define CALCLMULTINODE_H_

#include <mpi.h>
#include <stdio.h>
#include <vector>
#include<string>
#include<iostream>
#include <utility>
#include <arpa/inet.h> //inep_pton 
#include<stdexcept> //exception handling
#include<fstream> //ifstream and file handling
extern "C"{
#include <OpenCAL-CL/calclMultiGPU2D.h>
#include <OpenCAL/cal2DIO.h>
#include <OpenCAL/cal2DBuffer.h>
#include <OpenCAL/cal2DBufferIO.h>
}

using std::string;
using std::stoi;
using std::stoul;
using std::cin;
using std::ifstream;
typedef unsigned int uint;

//----------------------------------------------------------------------

bool is_ipv4_address(const string& str)
{
    struct sockaddr_in sa;
    const bool ret = inet_pton(AF_INET, str.c_str(), &(sa.sin_addr))!=0;
	if(!ret) //invalid ip address
		throw std::runtime_error( str+" is an invalid IPV4 address." );
	
	return true; 
}

bool is_ipv6_address(const string& str)
{
    struct sockaddr_in6 sa;
    const bool ret = inet_pton(AF_INET6, str.c_str(), &(sa.sin6_addr))!=0;
	if(!ret) //invalid ip address
		throw std::runtime_error( str+" is an invalid IPV6 address." );
	
	return true; 
}

//if is not a valid IP address it throws a runtime_error exception
void validate_ip_address(const string& str){
	 is_ipv4_address(str); //|| is_ipv6_address(str);
}
//----------------------------------------------------------------------



class Device{
public:
Device(){};
Device(const uint _np, const uint _nd, const uint _w) : 
	num_platform(_np), num_device(_nd), workload(_w) {};
	
    uint num_platform;
    uint num_device;
    uint workload;

};

class Node {
public:
Node(){};
Node(const uint _c , const uint _off,  const uint _nd, const string& _ip)
: columns(_c) , offset(_off) , devices(_nd), ip(_ip) {}

    std::vector<Device> devices;
    int workload;
    int columns;
    int offset;
	string ip;
};

class Cluster{
public:
    std::vector<Node> nodes; // quali nodi usiamo? Internamente ogni nodo
    //ha una descrizione dei device da utklizzare e relativi workloads
    inline bool is_full_exchange() const {return nodes.size()==1;}
	
	void fromClusterFile(const std::string& file){
		
		ifstream clusterfile;
		clusterfile.exceptions ( ifstream::failbit | ifstream::badbit );
		
		
		ulong R=0,C=0;
		ulong NNODES=0;
		string buf;
		ulong OFF=0;
		try{
			clusterfile.open(file.c_str());
			clusterfile>>buf; R = stoul(buf.c_str());
			clusterfile>>buf; C = stoul(buf.c_str());
			clusterfile>>buf; NNODES = stoul(buf);
			
			nodes.resize(NNODES);
			
			for(int i=0 ; i<NNODES ; i++){
				
				ulong NDEVICES;
				//parse node i
				clusterfile>>buf; //read IP of node 1
				validate_ip_address(buf);
				string ip = buf;
				//num devices for node i
				clusterfile>>buf; NDEVICES = stoul(buf);
				Node ni (C , OFF , NDEVICES , ip);
				uint node_workload=0;
				for(int j = 0 ; j < NDEVICES ; j++){
					
					//parse device j for node i
					//each device is identified by two uint: platform and device number
					ulong P, D;
					ulong W;
					clusterfile>>buf; P = stoul(buf);
					clusterfile>>buf; D = stoul(buf);
					//read workload for device j					
					clusterfile>>buf; W = stoul(buf);
					
					//add this device to the list of devices of node i
					Device d_i_j (P,D,W);
					ni.devices[j]=d_i_j;
					
					node_workload+=W;
				}
				ni.workload = node_workload;
				//add the just created node to the list of nodes of the cluster
				this->nodes[i] = ni;
				
				OFF+=node_workload; //offset for the next node
			}
			//close clusterfile
			clusterfile.close();
			
			//consistency check on the workload.
			if(OFF!=R){
				string err = "Nodes total workload is not consistent with the  specified number of rows\n";
				throw  std::runtime_error(err.c_str());
			}
			
		} catch (const ifstream::failure& e) {
			std::cout << "Exception opening/reading file: "<<file<<std::endl;
			std::cout<<e.what()<<std::endl;
			throw;
		} catch ( const std::exception& e) { // cautch all clause
			std::cout << "A standard exception was caught, with message '"
                 << e.what() << "'\n";
				throw;
				 
		}
	}
};


CALbyte calLoadMatrix2DrMulti(CALreal* M, const int rows, const int columns, const char* path,  int read_offset = 0, const int write_offset = 0)
{
    FILE *f = NULL;
    f = fopen(path, "r");

    if ( !f )
        return CAL_FALSE;

    //skip #offset rows
    const int _s = 20*2*columns; //assuming 20char per number + spaces
    char tmp[_s];
    while(read_offset--)
        fgets(tmp,sizeof(char)*_s,f);

    calfLoadMatrix2Dr(M+write_offset, rows, columns, f);

    fclose(f);

    return CAL_TRUE;
}


CALbyte calLoadSubstate2DrMulti(CALModel2D* ca2D, struct CALSubstate2Dr* Q, char* path,int read_offset = 0, const int write_offset = 0) {
    CALbyte return_state = calLoadMatrix2DrMulti(Q->current, ca2D->rows, ca2D->columns, path,read_offset,write_offset);
    if (Q->next)
        calCopyBuffer2Dr(Q->current, Q->next, ca2D->rows, ca2D->columns);
    return return_state;
}

//MULTINODE---------------
template <class F_INIT,class F_FINALIZE>
class MultiNode{
public:
    Cluster c;
    F_INIT *init;
    F_FINALIZE *finalize;
    int rank;
    CALCLMultiGPU* multigpu;

    CALreal* realNodeGhosts=0;
    CALint * intNodeGhosts=0;
    CALbyte* byteNodeGhosts=0;


    MultiNode(Cluster _c, int _rank,F_INIT *i, F_FINALIZE *f):c(_c), rank(_rank),
        init(i), finalize(f) {
        multigpu=nullptr;
        multigpu = (CALCLMultiGPU*)malloc(sizeof(CALCLMultiGPU));

    }

    bool checkWorkloads(){return true;};

    void allocateAndInit(){

        init(multigpu,&c);

        CALCLModel2D* last_gpu = multigpu->device_models[multigpu->num_devices-1];
        const CALint sizeBorder = last_gpu->borderSize*last_gpu->columns;
        const int rnumSubstate = last_gpu->host_CA->sizeof_pQr_array;
        const int inumSubstate = last_gpu->host_CA->sizeof_pQi_array;
        const int bnumSubstate = last_gpu->host_CA->sizeof_pQb_array;
        realNodeGhosts=0;
        intNodeGhosts=0;
        byteNodeGhosts=0;
        realNodeGhosts = (CALreal*)calloc(rnumSubstate*sizeBorder*2,sizeof(CALreal));
        intNodeGhosts  = (CALint*)calloc(inumSubstate*sizeBorder*2,sizeof(CALint));
        byteNodeGhosts = (CALbyte*)calloc(bnumSubstate*sizeBorder*2,sizeof(CALbyte));
    }

    void _finalize(){
        free(realNodeGhosts);
        free(byteNodeGhosts);
        free(intNodeGhosts);
        finalize(multigpu);
    }

	
    void run(int STEPS){

	int rank;
	time_t start, end;
   	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	if(rank == 0)
		start = time(NULL);

        MPI_Barrier(MPI_COMM_WORLD);

        size_t* singleStepThreadNum;
        int dimNum;
        computekernelLaunchParams(multigpu, &singleStepThreadNum ,&dimNum );
        handleBorderNodes();
        while(STEPS--){
            //debug("step \n");
	    for (int j = 0; j < multigpu->device_models[0]->elementaryProcessesNum; j++) {
                MPI_Barrier(MPI_COMM_WORLD);
                calcl_executeElementaryProcess(multigpu, j, singleStepThreadNum, dimNum/*elementary process*/);
                MPI_Barrier(MPI_COMM_WORLD);
                handleBorderNodes();
            }


        }
        //handleBorderNodes();
        calclDevicesToNode(multigpu);

	if(rank == 0){
		end = time(NULL);
		printf("Elapsed time: %lds\n", end - start);
	}

        _finalize();
    }//run function

    void handleBorderNodes(){
        handleBorderNodesR();
        handleBorderNodesI();
        handleBorderNodesB();
    }


    void handleBorderNodesR(){
        const MPI_Datatype DATATYPE = MPI_DOUBLE;
        if(!c.is_full_exchange()){

            CALint prev,next;
            CALCLModel2D* gpu_to_use = multigpu->device_models[0];
            CALreal* send_offset;
            CALreal* recv_offset = realNodeGhosts;

            const CALint sizeBorder = gpu_to_use->borderSize*gpu_to_use->columns;
            const int numSubstates = gpu_to_use->host_CA->sizeof_pQr_array;
            const CALint count = (numSubstates*sizeBorder);

            if(numSubstates <= 0)
                return;

            for(int i=0;i<2;i++){

                next=((rank+1)+c.nodes.size())%c.nodes.size();
                prev=((rank-1)+c.nodes.size())%c.nodes.size();
                if(i==1)
                    std::swap(next,prev);


                //this should be multigpu->num_devices-1 and 0 during the two iterations
                CALint indexgpu = (i+multigpu->num_devices-1)%multigpu->num_devices;
                gpu_to_use = multigpu->device_models[indexgpu];


                send_offset = gpu_to_use->borderMapper.realBorder_OUT;
                send_offset+=(i==0 ? 1 : 0)*count;

                recv_offset = realNodeGhosts;
                recv_offset+= (i==0 ? 0: 1)*count;

                if(rank % 2 == 0){
                    //MPI send


                    // printf("I'm %d:  sedning to %d \n" ,  rank , next);
                    //cerca convenzione per i nomi dei tags
                    MPI_Send(send_offset,count, DATATYPE , next , i ,MPI_COMM_WORLD);

                    // printf("I'm %d:  receiving from  %d \n" ,  rank , prev);
                    MPI_Recv(recv_offset , count , DATATYPE, prev, i, MPI_COMM_WORLD,MPI_STATUS_IGNORE);


//                    if(rank==0){
//                        std::cout<<std::endl;
//                        for (int i = 0; i < 2*count; ++i) {
//                            if(i%gpu_to_use->columns==0)
//                                std::cout<<std::endl;
//                            std::cout <<  gpu_to_use->borderMapper.realBorder_OUT[i] << " ";
//                        }
//                        std::cout<<std::endl;
//                    }

                    //send to rank+1
                    //receive rank-1
                }else{

                    // printf("I'm %d:  receiving from  %d \n" ,  rank , prev);
                    MPI_Recv(recv_offset , count , DATATYPE, prev, i ,MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    // printf("I'm %d:  sedning to %d \n" ,  rank , next);
                    MPI_Send(send_offset , count, DATATYPE , next , i ,MPI_COMM_WORLD);

                    //                    if(rank==1){
                    //                        std::cout<<std::endl;
                    //                        for (int i = 0; i < 2*count; ++i) {
                    //                            if(i%gpu_to_use->columns==0)
                    //                                std::cout<<std::endl;
                    //                            std::cout << realNodeGhosts[i] << " ";
                    //                        }
                    //                        std::cout<<std::endl;
                    //                    }
                    //receice from rank-1;
                    //send to rank+1
                }

                MPI_Barrier(MPI_COMM_WORLD);


            }




            //memory has been exchanged between mpi processes. Now it'0s time to foward that memory to
            //the right GPUs
            cl_int err;
            for(int i =0; i < numSubstates; i++){
                CALCLModel2D* m = multigpu->device_models[0];
                //upper ghost
                err = clEnqueueWriteBuffer(m->queue,
                                           m->bufferCurrentRealSubstate,
                                           CL_TRUE,
                                           (i*m->fullSize)*sizeof(CALreal),
                                           sizeof(CALreal)*sizeBorder,
                                           realNodeGhosts +(i*sizeBorder),
                                           0,
                                           NULL,
                                           NULL);
                calclHandleError(err);
                m = multigpu->device_models[multigpu->num_devices-1];
                //lower ghost
                err = clEnqueueWriteBuffer(m->queue,
                                           m->bufferCurrentRealSubstate,
                                           CL_TRUE,
                                           ((i+1)*m->fullSize-sizeBorder)*sizeof(CALreal),
                                           sizeof(CALreal)*sizeBorder,
                                           realNodeGhosts + numSubstates*sizeBorder + (i*sizeBorder),
                                           0,
                                           NULL,
                                           NULL);
                calclHandleError(err);

            }


        }
    }





    void handleBorderNodesI(){
        const MPI_Datatype DATATYPE = MPI_INT;
        if(!c.is_full_exchange()){

            CALint prev,next;
            CALCLModel2D* gpu_to_use = multigpu->device_models[0];
            CALint* send_offset;
            CALint* recv_offset = intNodeGhosts;

            const CALint sizeBorder = gpu_to_use->borderSize*gpu_to_use->columns;
            const int numSubstates = gpu_to_use->host_CA->sizeof_pQi_array;
            const CALint count = (numSubstates*sizeBorder);

            if(numSubstates <= 0)
                return;

            for(int i=0;i<2;i++){

                next=((rank+1)+c.nodes.size())%c.nodes.size();
                prev=((rank-1)+c.nodes.size())%c.nodes.size();
                if(i==1)
                    std::swap(next,prev);


                //this should be multigpu->num_devices-1 and 0 during the two iterations
                CALint indexgpu = (i+multigpu->num_devices-1)%multigpu->num_devices;
                gpu_to_use = multigpu->device_models[indexgpu];


                send_offset = gpu_to_use->borderMapper.intBorder_OUT;
                send_offset+=(i==0 ? 1 : 0)*count;

                recv_offset = intNodeGhosts;
                recv_offset+= (i==0 ? 0: 1)*count;

                if(rank % 2 == 0){
                    //MPI send


                    // printf("I'm %d:  sedning to %d \n" ,  rank , next);
                    //cerca convenzione per i nomi dei tags
                    MPI_Send(send_offset,count, DATATYPE , next , i ,MPI_COMM_WORLD);

                    // printf("I'm %d:  receiving from  %d \n" ,  rank , prev);
                    MPI_Recv(recv_offset , count , DATATYPE, prev, i, MPI_COMM_WORLD,MPI_STATUS_IGNORE);



                    //send to rank+1
                    //receive rank-1
                }else{

                    // printf("I'm %d:  receiving from  %d \n" ,  rank , prev);
                    MPI_Recv(recv_offset , count , DATATYPE, prev, i ,MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    // printf("I'm %d:  sedning to %d \n" ,  rank , next);
                    MPI_Send(send_offset , count, DATATYPE , next , i ,MPI_COMM_WORLD);

                    /*  if(rank==1){
                    std::cout<<std::endl;
                    for (int i = 0; i < 2*count; ++i) {
                        if(i%gpu_to_use->columns==0)
                            std::cout<<std::endl;
                        std::cout << intNodeGhosts[i] << " ";
                    }
                    std::cout<<std::endl;
                }*/
                    //receice from rank-1;
                    //send to rank+1
                }

                MPI_Barrier(MPI_COMM_WORLD);


            }
            //memory has been exchanged between mpi processes. Now it'0s time to foward that memory to
            //the right GPUs
            cl_int err;
            for(int i =0; i < numSubstates; i++){
                CALCLModel2D* m = multigpu->device_models[0];
                //upper ghost
                err = clEnqueueWriteBuffer(m->queue,
                                           m->bufferCurrentIntSubstate,
                                           CL_TRUE,
                                           (i*m->fullSize)*sizeof(CALint),
                                           sizeof(CALint)*sizeBorder,
                                           intNodeGhosts +(i*sizeBorder),
                                           0,
                                           NULL,
                                           NULL);
                calclHandleError(err);
                m = multigpu->device_models[multigpu->num_devices-1];
                //lower ghost
                err = clEnqueueWriteBuffer(m->queue,
                                           m->bufferCurrentIntSubstate,
                                           CL_TRUE,
                                           ((i+1)*m->fullSize-sizeBorder)*sizeof(CALint),
                                           sizeof(CALint)*sizeBorder,
                                           intNodeGhosts + numSubstates*sizeBorder + (i*sizeBorder),
                                           0,
                                           NULL,
                                           NULL);
                calclHandleError(err);

            }


        }
    }



    void handleBorderNodesB(){
        const MPI_Datatype DATATYPE = MPI_CHAR;
        if(!c.is_full_exchange()){

            CALint prev,next;
            CALCLModel2D* gpu_to_use = multigpu->device_models[0];
            CALbyte* send_offset;
            CALbyte* recv_offset = byteNodeGhosts;

            const CALint sizeBorder = gpu_to_use->borderSize*gpu_to_use->columns;
            const int numSubstates = gpu_to_use->host_CA->sizeof_pQb_array;
            const CALint count = (numSubstates*sizeBorder);

            if(numSubstates <= 0)
                return;

            for(int i=0;i<2;i++){

                next=((rank+1)+c.nodes.size())%c.nodes.size();
                prev=((rank-1)+c.nodes.size())%c.nodes.size();
                if(i==1)
                    std::swap(next,prev);


                //this should be multigpu->num_devices-1 and 0 during the two iterations
                CALint indexgpu = (i+multigpu->num_devices-1)%multigpu->num_devices;
                gpu_to_use = multigpu->device_models[indexgpu];


                send_offset = gpu_to_use->borderMapper.byteBorder_OUT;
                send_offset+=(i==0 ? 1 : 0)*count;

                recv_offset = byteNodeGhosts;
                recv_offset+= (i==0 ? 0: 1)*count;

                if(rank % 2 == 0){
                    //MPI send


                    // printf("I'm %d:  sedning to %d \n" ,  rank , next);
                    //cerca convenzione per i nomi dei tags
                    MPI_Send(send_offset,count, DATATYPE , next , i ,MPI_COMM_WORLD);

                    // printf("I'm %d:  receiving from  %d \n" ,  rank , prev);
                    MPI_Recv(recv_offset , count , DATATYPE, prev, i, MPI_COMM_WORLD,MPI_STATUS_IGNORE);



                    //send to rank+1
                    //receive rank-1
                }else{

                    // printf("I'm %d:  receiving from  %d \n" ,  rank , prev);
                    MPI_Recv(recv_offset , count , DATATYPE, prev, i ,MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    // printf("I'm %d:  sedning to %d \n" ,  rank , next);
                    MPI_Send(send_offset , count, DATATYPE , next , i ,MPI_COMM_WORLD);

                    /*  if(rank==1){
                    std::cout<<std::endl;
                    for (int i = 0; i < 2*count; ++i) {
                        if(i%gpu_to_use->columns==0)
                            std::cout<<std::endl;
                        std::cout << intNodeGhosts[i] << " ";
                    }
                    std::cout<<std::endl;
                }*/
                    //receice from rank-1;
                    //send to rank+1
                }

                MPI_Barrier(MPI_COMM_WORLD);


            }




            //memory has been exchanged between mpi processes. Now it'0s time to foward that memory to
            //the right GPUs
            cl_int err;
            for(int i =0; i < numSubstates; i++){
                CALCLModel2D* m = multigpu->device_models[0];
                //upper ghost
                err = clEnqueueWriteBuffer(m->queue,
                                           m->bufferCurrentByteSubstate,
                                           CL_TRUE,
                                           (i*m->fullSize)*sizeof(CALbyte),
                                           sizeof(CALbyte)*sizeBorder,
                                           byteNodeGhosts +(i*sizeBorder),
                                           0,
                                           NULL,
                                           NULL);
                calclHandleError(err);
                m = multigpu->device_models[multigpu->num_devices-1];
                //lower ghost
                err = clEnqueueWriteBuffer(m->queue,
                                           m->bufferCurrentByteSubstate,
                                           CL_TRUE,
                                           ((i+1)*m->fullSize-sizeBorder)*sizeof(CALbyte),
                                           sizeof(CALbyte)*sizeBorder,
                                           byteNodeGhosts + numSubstates*sizeBorder + (i*sizeBorder),
                                           0,
                                           NULL,
                                           NULL);
                calclHandleError(err);

            }


        }
    }








};
//END MULTINODE--------

#endif /*CALCLMULTINODE_H_*/
