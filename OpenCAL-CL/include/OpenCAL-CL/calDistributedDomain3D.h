#ifndef calcldistributeddomain3D_
#define calcldistributeddomain3D_

#include "calclCommon.h"

class Node {
public:
Node(){}
Node(const uint _c , const uint _r ,const uint _col , const uint _off,  const uint _nd, const string& _ip)
: rowcolumns(_c) ,rows(_r), columns(_col),  offset(_off) , devices(_nd), ip(_ip) {}

    std::vector<Device> devices;
    int workload;   // number of layers 
    int rowcolumns; // first two dimension
    int rows;
    int columns; // first two dimension
    int offset;     // number of layers from the top
	string ip;	    
};

class CALDistributedDomain3D{
public:
    std::vector<Node> nodes; // quali nodi usiamo? Internamente ogni nodo
    //ha una descrizione dei device da utklizzare e relativi workloads
    
    inline bool is_full_exchange() const {return nodes.size()==1;}
	
	void fromDomainFile(const std::string& file){
		
		ifstream domainfile;
		domainfile.exceptions ( ifstream::failbit | ifstream::badbit );
		
		ulong R=0,C=0, L=0; // rows, columns, layers
		ulong NNODES=0;
		string buf;
		ulong OFF=0;
		try{
			domainfile.open(file.c_str());
			domainfile>>buf; R = stoul(buf.c_str());
			domainfile>>buf; C = stoul(buf.c_str());
			domainfile>>buf; L = stoul(buf.c_str());
			domainfile>>buf; NNODES = stoul(buf);
			
			nodes.resize(NNODES);
			
			for(int i=0 ; i<NNODES ; i++){
				
				ulong NDEVICES;
				//parse node i
				domainfile>>buf; //read IP of node 1
				validate_ip_address(buf);
				string ip = buf;
				//num devices for node i
				domainfile>>buf; NDEVICES = stoul(buf);
                Node ni (R*C ,R,C, OFF , NDEVICES , ip);
				uint node_workload=0;
				for(int j = 0 ; j < NDEVICES ; j++){
					
					//parse device j for node i
					//each device is identified by two uint: platform and device number
					ulong P, D;
					ulong W;
					domainfile>>buf; P = stoul(buf);
					domainfile>>buf; D = stoul(buf);
					//read workload for device j					
					domainfile>>buf; W = stoul(buf);
					
					//add this device to the list of devices of node i
                    Device d_i_j (P,D,W,/*OFF+*/node_workload, OFF);
					ni.devices[j]=d_i_j;
					
					node_workload+=W;
				}
				ni.workload = node_workload;
				//add the just created node to the list of nodes of the domain
				this->nodes[i] = ni;
                std::cout <<"OFF = " << OFF << "; L = " << L << std::endl;
				OFF+=node_workload; //offset for the next node
			}
			//close domainfile
			domainfile.close();
			
			//consistency check on the workload.

            if(OFF!=L){
                string err = "Nodes total workload is not consistent with the  specified number of slices\n";
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
	
	
};

CALDistributedDomain3D calDomainPartition3D(int argc, char** argv){
            CALDistributedDomain3D domain;
            string domainfile = domain.parseCommandLineArgs(argc, argv);
			domain.fromDomainFile(domainfile);
			return domain;
}



#endif // calcldistributeddomain3D
