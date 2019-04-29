/*
 * Copyright (c) 2016 OpenCALTeam (https://github.com/OpenCALTeam),
 * Telesio Research Group,
 * Department of Mathematics and Computer Science,
 * University of Calabria, Italy.
 *
 * This file is part of OpenCAL (Open Computing Abstraction Layer).
 *
 * OpenCAL is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3 of
 * the License, or (at your option) any later version.
 *
 * OpenCAL is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with OpenCAL. If not, see <http://www.gnu.org/licenses/>.
 */

/*! \file calclMultiDevice2D.h
 *\brief calclMultiDevice2D contains structures and functions that allow to run parallel CA simulation using Opencl and OpenCAL.
 *
 *	calclMultiDevice2D contains structures that allows easily to transfer data of a CALModel2D instance from host to OpenCL devices.
 *	It's possible to setup a CA simulation by only defining kernels for elementary processes, and optionally
 *	initialization, steering and stop condition. Moreover, user can avoid to use the simulation cycle provided
 *	by the library and define his own simulation cycle.
 */

#ifndef CALCLMULTIDEVICE_H_
#define CALCLMULTIDEVICE_H_

extern "C"{
#include <OpenCAL-CL/calcl2D.h>
}

class MultiNode;

#include <OpenCAL-CL/calDistributedDomain2D.h>

typedef struct kernelID
{
    char name[MAX_FNAME_SIZE];
    int index;
} kernelID;

struct CALCLMultiDevice2D
{
    struct CALCLModel2D ** device_models;
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

void calclSetNumDevice(struct CALCLMultiDevice2D* multidevice, const CALint _num_devices);

void calclAddDevice(struct CALCLMultiDevice2D* multidevice,const CALCLdevice device, const CALint workload);

int calclCheckWorkload(struct CALCLMultiDevice2D* multidevice);

void calclMultiDeviceCADef2D(struct CALCLMultiDevice2D* mulitgpu,struct CALModel2D *host_CA ,char* kernel_src,char* kernel_inc, const CALint borderSize , const std::vector<Device>& devices);

void calclMultiDeviceRun2D(struct CALCLMultiDevice2D* multidevice, CALint init_step, CALint final_step);

void calclMultiDeviceAddElementaryProcess2D(struct CALCLMultiDevice2D* multidevice, char * kernelName);

void calclMultiDeviceAddSteeringFunc2D(struct CALCLMultiDevice2D* multidevice, char * kernelName);

void calclMultiDeviceFinalize2D(struct CALCLMultiDevice2D* mulitgpu);

size_t* computekernelLaunchParams(struct CALCLMultiDevice2D* , const int, int*);

void calclMultiDeviceSetKernelArg2D(struct CALCLMultiDevice2D * multidevice,const char * kernel, cl_uint arg_index, size_t arg_size, const void *arg_value);

void calclMultiDeviceToNode(struct CALCLMultiDevice2D* multidevice);

void calclMultiDeviceUpdateHalos2D(struct CALCLMultiDevice2D* multidevice,const CALbyte exchange_full_border);

void calclMultiDeviceSetWorkGroupSize2D(struct CALCLMultiDevice2D* multidevice, int m, int n);

#endif /* CALCLMULTIDEVICE_H_ */
