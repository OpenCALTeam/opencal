#ifndef calclcluster_
#define calclcluster_

#include <vector>
#include<string>
#include<iostream>
#include <utility>
#include<stdexcept> //exception handling
#include<fstream> //ifstream and file handling

#include <OpenCAL-CL/caclClusterValidation.h>

using std::string;
using std::stoi;
using std::stoul;
using std::cin;
using std::ifstream;
typedef unsigned int uint;

//2D cluster -------------------------------------------

class Device{
public:
Device(){};
Device(const uint _np, const uint _nd, const uint _w, const uint _o) : 
	num_platform(_np), num_device(_nd), workload(_w) , offset(_o){};
	
    uint num_platform;
    uint num_device;
    uint workload;
	
	uint offset;

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
					Device d_i_j (P,D,W,OFF+node_workload);
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

//3D --------------------------------------

class Device3D : public Device{
public:
Device3D(){};
Device3D(const uint _np, const uint _nd, const uint _w, const uint _o) : Device(_np,_nd,_w,_o) {};
	
    uint num_platform;
    uint num_device;
    uint workload;	
	uint offset;

};

class Node3D : public Node{
public:
Node3D(){};
Node3D(const uint _c , const uint _l ,const uint _off,  const uint _nd, const string& _ip)
: Node(_c  ,_off,_nd,_ip) , layers(_l){};

    std::vector<Device3D> devices;
    int workload;
    int columns;
    int layers;
    int offset;
	string ip;
};

class Cluster3D{
public:
    std::vector<Node3D> nodes; // quali nodi usiamo? Internamente ogni nodo
    //ha una descrizione dei device da utklizzare e relativi workloads
    inline bool is_full_exchange() const {return nodes.size()==1;}
	
	void fromClusterFile(const std::string& file){
		
		ifstream clusterfile;
		clusterfile.exceptions ( ifstream::failbit | ifstream::badbit );
		
		ulong R=0,C=0, L=0;;
		ulong NNODES=0;
		string buf;
		ulong OFF=0;
		try{
			clusterfile.open(file.c_str());
			
            clusterfile>>buf; R = stoul(buf.c_str());
			clusterfile>>buf; C = stoul(buf.c_str());
            clusterfile>>buf; L = stoul(buf.c_str());
			
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
				Node3D ni (C , L, OFF , NDEVICES , ip);
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
					Device3D d_i_j (P,D,W,OFF+node_workload);
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



#endif // calclcluster_