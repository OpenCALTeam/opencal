#ifndef calclCommon_h
#define calclCommon_h

#include <vector>
#include<string>
#include<iostream>
#include <utility>
extern "C"{
    #include <arpa/inet.h> //inep_pton
}
#include<stdexcept> //exception handling
#include<fstream> //ifstream and file handling

using std::string;
using std::stoi;
using std::stoul;
using std::cin;
using std::ifstream;
typedef unsigned int uint;




class Device{
public:
Device(){}
Device(const uint _np, const uint _nd, const uint _w, const uint _o, const uint _go) :
    num_platform(_np), num_device(_nd), workload(_w) , offset(_o), goffset(_go){}

    uint num_platform;
    uint num_device;
    uint workload;

    uint offset;
    uint goffset;// global offset

};


#endif
