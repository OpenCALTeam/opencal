#ifndef calclclusterValidation_
#define calclclusterValidation_
#include<string>

extern "C"{
	#include <arpa/inet.h> //inep_pton 
}

	//----------------------------------------------------------------------

bool is_ipv4_address(const std::string& str)
{
    struct sockaddr_in sa;
    const bool ret = inet_pton(AF_INET, str.c_str(), &(sa.sin_addr))!=0;
	if(!ret) //invalid ip address
		throw std::runtime_error( str+" is an invalid IPV4 address." );
	
	return true; 
}

bool is_ipv6_address(const std::string& str)
{
    struct sockaddr_in6 sa;
    const bool ret = inet_pton(AF_INET6, str.c_str(), &(sa.sin6_addr))!=0;
	if(!ret) //invalid ip address
		throw std::runtime_error( str+" is an invalid IPV6 address." );
	
	return true; 
}

//if is not a valid IP address it throws a runtime_error exception
void validate_ip_address(const std::string& str){
	 is_ipv4_address(str); //|| is_ipv6_address(str);
}
//----------------------------------------------------------------------

#endif // calclclusterValidation