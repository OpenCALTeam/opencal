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

/*! \file OpenCL_Utility.h
 * \brief OpenCL_Utility contains structures and functions to simplify OpenCL coding
 *
 * OpenCL_Utility contains structures and functions to simplify OpenCL coding. Functions allow
 * to query available platforms and devices. It's also possible to obtain all available platforms
 * and devices with fews functions calls and having them together in the structure CALCLDeviceManager.
 *
 */

#ifndef OPENCL_UTILITY_H_
#define OPENCL_UTILITY_H_

#ifdef _WIN32
#define _CRT_SECURE_NO_WARNINGS 1
#endif

#include <OpenCAL-CL/dllexport.h>

#ifdef _WIN32
#include <OpenCAL-CL/dirent.h>
#else
#include <dirent.h>
#endif
#include <string.h>
#include <stdio.h>

#ifdef MAC
#include <OpenCL/cl.h>
#else
#include "CL/cl.h"
#endif

typedef cl_device_id CALCLdevice;			//!< Redefinition of type cl_device_id
typedef cl_context CALCLcontext;			//!< Redefinition of type cl_context
typedef cl_program CALCLprogram;			//!< Redefinition of type cl_program
typedef cl_platform_id CALCLplatform;		//!< Redefinition of type cl_platform_id
typedef cl_mem CALCLmem; 					//!< Redefinition of type cl_mem
typedef cl_kernel CALCLkernel; 				//!< Redefinition of type cl_kernel
typedef cl_command_queue CALCLqueue; 		//!< Redefinition of type cl_command_queue
typedef cl_uint CALCLuint; 					//!< Redefinition of type cl_uint
typedef cl_int CALCLint; 					//!< Redefinition of type cl_int
typedef cl_device_type CALCLdeviceType; 	//!< Redefinition of type cl_device_type
typedef cl_ulong CALCLulong; 				//!< Redefinition of type cl_ulong
typedef cl_long CALCLlong; 					//!< Redefinition of type cl_long

/*!\brief CALCLDeviceManager contains Opencl platforms and devices */
struct CALCLDeviceManager{
	CALCLplatform *platforms; 				//!< Array of Opencl available platforms
	CALCLuint num_platforms; 				//!< Number of Opencl available platforms
	CALCLdevice ** devices; 				//!< Matrix containing Opencl devices. Each row represents a platform and contains all its devices.
	int * num_platforms_devices; 			//!< Array containing the number of devices for each platform.
};

/*! \brief Allocates and returns an instance of CALOpencCL structure. */
DllExport
struct CALCLDeviceManager * calclCreateManager();

/*! \brief Creates an Opencl context */
DllExport
CALCLcontext calclCreateContext(CALCLdevice * devices,	//!< Array containing the devices used to create the context
                                CALCLint num_devices
		);

/*! \brief Creates an Opencl buffer */
DllExport
CALCLmem calclCreateBuffer(CALCLcontext context, 	//!< Opencl context
		void* host_ptr, 							//!< Pointer to data that initializes the buffer
		size_t size 								//!< Size of data that initializes the buffer
		);

/*! \brief Creates an Opencl command queue for a specific device */
DllExport
CALCLqueue calclCreateCommandQueue(CALCLcontext context, 	//!< Opencl context
		CALCLdevice device 									//!< Device associated to the command queue
		);

/*! \brief Gets one of the allocated device */
DllExport
CALCLdevice calclGetDevice(struct CALCLDeviceManager * calOpenCL, 	//!< Pointer to struct CALCLDeviceManager structure
		int platformIndex, 							//!< Index referring the platform associated to the device (indexes go from zero to the number of available platforms)
		int deviceIndex 							//!< Index referring the device (indexes go from zero to the number of available devices in the selected platform)
		);

/*! \brief Deallocates the structure CALOpencl */
DllExport
void calclFinalizeManager(struct CALCLDeviceManager * opencl 	//!< Pointer to struct CALCLDeviceManager structure
		);

/*! \brief Given some paths returns the number of files in the paths and the names of the files  */
DllExport
void calclGetDirFiles(char ** paths, 	//!< Array of strings. Each string is a path
		int pathsNum, 					//!< Number of paths
		char *** files_names, 			//!< Pointer to an array of strings. The function allocates the array and initializes it with paths files names
		int * num_files 				//!< Pointer to int. The function assigns to it the number of file contained in the paths
		);

/*! \brief Print on standard output informations about all platforms and devices*/
DllExport
void calclPrintPlatformsAndDevices(struct CALCLDeviceManager * opencl);

/*! \brief Print on standard output informations about all platforms and devices*/
DllExport
void calclGetPlatformAndDeviceFromStdIn(struct CALCLDeviceManager * opencl,CALCLdevice * device);

/*! \brief Reads a file and return its content  */
DllExport
void calclReadFile(char * fileName, 	//!< File path
		char ** programBuffer, 			//!< Pointer to a string. The function allocate the string and assigns to it the content of the file
		size_t * program_size 			//!< Pointer to size_t. The function assigns to it the file content size
		);

/*! \brief Builds and returns a Opencl program
 *
 * Given an array of files names, the function reads the files contents and builds an Opencl program.
 * If the build procedure fails the function prints on standard output the building log and closes the application.
 *
 * */
DllExport
CALCLprogram calclGetProgramFromFiles(char** filesNames, 	//!< Array of strings containing the files names
		unsigned filesNum, 									//!< Number of files
		char * compilerOptions, 							//!< String containing compiler options for the Opencl runtime compiler
		CALCLcontext context, 								//!< Opencl context
		CALCLdevice * devices, 								//!< Array of devices for which the program will be compiled
		CALCLuint num_devices 								//!< Number of devices
		);

/*! \brief Gets an Opencl kernel given a compiled Opencl program */
DllExport
CALCLkernel calclGetKernelFromProgram(CALCLprogram program, 	//!< Pointer to an Opencl program
        char * kernelName 										//!< Kernel name
        );

/*! \brief Converts Opencl error codes (cl_int) into error messages (char*)*/
DllExport
const char * calclGetErrorString(CALCLint err 	//!< Integer representing an Opencl error code
		);

/*! \brief Handles Opencl errors codes
 *
 * The function takes in input an error code. If the error is minor then zero (this means there is some runtime error)
 * it prints an error message and closes the application otherwise it returns
 *
 * */
 DllExport
void calclHandleError(CALCLint err 	//!< Integer representing an Opencl error code
		);

/****************************************************************************
 * 		FUNCTIONS TO GET OPENCL STUFF WITHOUT CALOPENCL STRUCT
 * 		FUNCTIONS TO PRINT OPENCL DEVICES AND PLATFORMS INFO
 ****************************************************************************/

//utility file merging
//& removed - need test
/*! \brief Gets Opencl platforms*/
DllExport
void calclGetPlatforms(CALCLplatform** platforms, 	//!< Pointer to an array of platforms. The function allocates it and assigns to it the available platforms
		CALCLuint * platformNum 					//!< Pointer to unsigned int. The function assigns to it the number of available platforms
		);

/*! \brief Get Opencl platforms by choosing a specific vendor*/
DllExport
void calclGetPlatformByVendor(CALCLplatform * platform, 	//!< Pointer to an Opencl platform
		char* vendor 										//!< Platform vendor name
		);

/*! \brief Get Opencl NVIDIA platform*/
DllExport
void calclGetPlatformNVIDIA(CALCLplatform * platform 	//!< Pointer to an Opencl platform
		);

/*! \brief Get Opencl INTEL platform*/
DllExport
void calclGetPlatformINTEL(CALCLplatform * platform 	//!< Pointer to an Opencl platform
		);

/*! \brief Get Opencl AMD platform*/
DllExport
void calclGetPlatformAMD(CALCLplatform * platform 		//!< Pointer to an Opencl platform
		);

/*! \brief Get Opencl platform name*/
DllExport
char* calclGetPlatformName(CALCLplatform platform 		//!< Opencl platform
		);

/*! \brief Get Opencl platform vendor name*/
DllExport
char* calclGetPlatformVendor(CALCLplatform platform 	//!< Opencl platform
		);

/*! \brief Get Opencl platform profile*/
DllExport
char* calclGetPlatformProfile(CALCLplatform platform 	//!< Opencl platform
		);

/*! \brief Get Opencl platform version*/
DllExport
char* calclGetPlatformVersion(CALCLplatform platform 	//!< Opencl platform
		);

/*! \brief Get Opencl platform supported extensions*/
DllExport
char* calclGetPlatformExtensions(CALCLplatform platform //!< Opencl platform
		);

/*! \brief Print on standard output information about the given platform*/
DllExport
void calclPrintAllPlatformInfo(CALCLplatform platform 	//!< Opencl platform
		);

/*! \brief Get Opencl devices*/
DllExport
void calclGetDevices(CALCLplatform platform, 	//!< Opencl platform
		CALCLdeviceType type, 					//!< Platform type (CL_DEVICE_TYPE_CPU, CL_DEVICE_TYPE_GPU, CL_DEVICE_TYPE_ACCELLERATOR, CL_DEVICE_TYPE_ALL)
		CALCLdevice ** devices, 				//!< Pointer to array of Opencl devices. The function allocates it and assigns to it the available devices for the selected platform
		CALCLuint * size 						//!< Number of devices available for the selected platform
		);

/*! \brief Get Opencl devices with a specific name */
DllExport
void calclGetDevicesByName(CALCLplatform plat, 	//!< Opencl platform
		CALCLdevice ** devices, 				//!< Pointer to array of Opencl devices. The function allocates it and assigns to it the available devices for the selected platform
		CALCLuint * size, 						//!< Number of devices available for the selected platform
		char* name 								//!< Device name
		);

/*! \brief Get Opencl device name*/
DllExport
char* calclGetDeviceName(CALCLdevice device 	//!< Opencl device
		);

/*! \brief Get Opencl device vendor name*/
DllExport
char* calclGetDeviceVendor(CALCLdevice device 	//!< Opencl device
		);

/*! \brief Get Opencl device supported extensions*/
DllExport
char* calclGetDeviceExtensions(CALCLdevice device 	//!< Opencl device
		);

/*! \brief Get Opencl device global memory size (in bytes)*/
DllExport
CALCLulong calclGetDeviceGlobalMemSize(CALCLdevice device 	//!< Opencl device
		);

/*! \brief Get Opencl device maximum number of work items dimensions*/
DllExport
CALCLuint calclGetDeviceMaxWorkItemDimensions(CALCLdevice device 	//!< Opencl device
		);

/*! \brief Print on standard output informations about the given device*/
DllExport
void calclPrintAllDeviceInfo(CALCLdevice device 	//!< Opencl device
		);

/*! \brief Print on standard output informations about all platforms and devices*/
DllExport
void calclPrintAllPlatformAndDevices(struct CALCLDeviceManager * opencl);

/*! \brief Print on standard output informations about all platforms and devices*/
DllExport
void calclGetPlatformAndDeviceFromStandardInput(struct CALCLDeviceManager * opencl,CALCLdevice * device);

#endif /* CALCLDeviceManager_H_ */
