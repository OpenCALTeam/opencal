///*
// * Copyright (c) 2016 OpenCALTeam (https://github.com/OpenCALTeam),
// * Telesio Research Group,
// * Department of Mathematics and Computer Science,
// * University of Calabria, Italy.
// *
// * This file is part of OpenCAL (Open Computing Abstraction Layer).
// *
// * OpenCAL is free software: you can redistribute it and/or modify
// * it under the terms of the GNU Lesser General Public License as
// * published by the Free Software Foundation, either version 3 of
// * the License, or (at your option) any later version.
// *
// * OpenCAL is distributed in the hope that it will be useful,
// * but WITHOUT ANY WARRANTY; without even the implied warranty of
// * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// * GNU Lesser General Public License for more details.
// *
// * You should have received a copy of the GNU Lesser General Public
// * License along with OpenCAL. If not, see <http://www.gnu.org/licenses/>.
// */
//
///*! \file calclMultiGPU2D.h
// *\brief calclMultiGPU2D contains structures and functions that allow to run parallel CA simulation using Opencl and OpenCAL.
// *
// *	calclMultiGPU2D contains structures that allows easily to transfer data of a CALModel2D instance from host to GPUs.
// *	It's possible to setup a CA simulation by only defining kernels for elementary processes, and optionally
// *	initialization, steering and stop condition. Moreover, user can avoid to use the simulation cycle provided
// *	by the library and define his own simulation cycle.
// */
//
#ifndef CALCLMULTIGPU_H_
#define CALCLMULTIGPU_H_

extern "C"{
#include <OpenCAL-CL/calcl3D.h>
}
#include <OpenCAL-CL/calclCluster.h>

typedef struct kernelID{
    char name[MAX_FNAME_SIZE];
    int index;
}kernelID;

struct CALCLMultiGPU3D{

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
};


void calclSetNumDevice(struct CALCLMultiGPU3D* multigpu, const CALint _num_devices);

void calclAddDevice(struct CALCLMultiGPU3D* multigpu,const CALCLdevice device, const CALint workload);

int calclCheckWorkload(struct CALCLMultiGPU3D* multigpu);

void calclMultiGPUDef3D(struct CALCLMultiGPU3D* mulitgpu,struct CALModel3D *host_CA ,char* kernel_src,char* kernel_inc, const CALint borderSize , const std::vector<Device3D>& devices, const CALbyte exchange_full_border);

void calclMultiGPURun2D(struct CALCLMultiGPU3D* multigpu, CALint init_step, CALint final_step);

void calclAddElementaryProcessMultiGPU3D(struct CALCLMultiGPU3D* multigpu, char * kernelName);

void calclAddSteeringFuncMultiGPU3D(struct CALCLMultiGPU3D* multigpu, char * kernelName);

void calclMultiGPUFinalize(struct CALCLMultiGPU3D* mulitgpu);


void calcl_executeElementaryProcess(struct CALCLMultiGPU3D* multigpu,const int el_proc, size_t* singleStepThreadNum,int dimNum);

void computekernelLaunchParams(struct CALCLMultiGPU3D* multigpu,  size_t**  singleStepThreadNum, int *dim);

void calclSetKernelArgMultiGPU3D(struct CALCLMultiGPU3D* multigpu,const char * kernel, cl_uint arg_index, size_t arg_size, const void *arg_value);


void calclDevicesToNode(struct CALCLMultiGPU3D* multigpu);

#endif /* CALCLMULTIGPU_H_ */
