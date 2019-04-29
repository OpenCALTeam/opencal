#ifndef CALCLMULTIDEVICE3D_H_
#define CALCLMULTIDEVICE3D_H_


extern "C"{
#include <OpenCAL-CL/calcl3D.h>
}

class MultiNode;

#include <OpenCAL-CL/calDistributedDomain3D.h>

typedef struct kernelID
{
    char name[MAX_FNAME_SIZE];
    int index;
} kernelID;

struct CALCLMultiDevice3D
{
    struct CALCLModel3D ** device_models;
    CALint num_devices;
    CALCLcontext context;
    CALCLdevice * devices;
    CALCLprogram * programs;
    CALint * workloads;
    cl_event * kernel_events;
    CALint pos_device;
    CALbyte exchange_full_border;
    C_vector kernelsID;
    size_t** singleStepThreadNums;
};


void calclSetNumDevice(struct CALCLMultiDevice3D* multidevice, const CALint _num_devices);

void calclAddDevice(struct CALCLMultiDevice3D* multidevice,const CALCLdevice device, const CALint workload);

int calclCheckWorkload(struct CALCLMultiDevice3D* multidevice);

void calclMultiDeviceCADef3D(struct CALCLMultiDevice3D* mulitgpu,struct CALModel3D *host_CA ,char* kernel_src,char* kernel_inc, const CALint borderSize , const std::vector<Device>& devices);

void calclMultiDeviceRun3D(struct CALCLMultiDevice3D* multidevice, CALint init_step, CALint final_step);

void calclMultiDeviceAddElementaryProcess3D(struct CALCLMultiDevice3D* multidevice, char * kernelName);

void calclMultiDeviceAddSteeringFunc3D(struct CALCLMultiDevice3D* multidevice, char * kernelName);

void calclMultiDeviceFinalize3D(struct CALCLMultiDevice3D* mulitgpu);

size_t* computekernelLaunchParams(struct CALCLMultiDevice3D* , const int, int*);

void calclMultiDeviceSetKernelArg3D(struct CALCLMultiDevice3D * multidevice,const char * kernel, cl_uint arg_index, size_t arg_size, const void *arg_value);

void calclMultiDeviceToNode(struct CALCLMultiDevice3D* multidevice);

void calclMultiDeviceUpdateHalos3D(struct CALCLMultiDevice3D* multidevice,const CALbyte exchange_full_border);

void calclMultiDeviceSetWorkGroupSize3D(struct CALCLMultiDevice3D* multidevice, int m, int n);







#endif /* CALCLMULTIDEVICE3D_H_ */
