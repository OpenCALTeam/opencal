/*! \file OpenCL_Utility.h
 * \brief OpenCL_Utility contains structures and functions to simplify OpenCL coding
 *
 * OpenCL_Utility contains structures and functions to simplify OpenCL coding. Functions allow
 * to query available platforms and devices. It's also possible to obtain all available platforms
 * and devices with fews functions calls and having them together in the structure CALOpenCL.
 *
 * Example:
 *
 * CALOpenCL * calOpenCL = calclCreateCALOpenCL();
 * calclInitializePlatforms(calOpenCL);
 * calclInitializeDevices(calOpenCL);
 *
 *
 */

#ifndef OPENCL_UTILITY_H_
#define OPENCL_UTILITY_H_

#ifdef _WIN32
#define _CRT_SECURE_NO_WARNINGS 1 
#endif

#include <dirent.h>
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

/*!\brief CALOpenCL contains Opencl platforms and devices */
typedef struct {
	CALCLplatform *platforms; 				//!< Array of Opencl available platforms
	CALCLuint num_platforms; 				//!< Number of Opencl available platforms
	CALCLdevice ** devices; 				//!< Matrix containing Opencl devices. Each row represents a platform and contains all its devices.
	int * num_platforms_devices; 			//!< Array containing the number of devices for each platform.
} CALOpenCL;

/*! \brief Allocates and returns an instance of CALOpencCL structure. */
CALOpenCL * calclCreateCALOpenCL();

/*! \brief Allocates and initializes available Opencl platforms */
void calclInitializePlatforms(CALOpenCL * opencl	//!< Pointer to CALOpenCL structure
		);

/*! \brief For each available platform allocates and initializes its Opencl devices */
void calclInitializeDevices(CALOpenCL * opencl //!< Pointer to CALOpenCL structure
		);

/*! \brief Creates an Opencl context */
CALCLcontext calclcreateContext(CALCLdevice * devices,	//!< Array containing the devices used to create the context
		CALCLuint num_devices							//!< Number of devices used to create the context
		);

/*! \brief Creates an Opencl buffer */
CALCLmem calclCreateBuffer(CALCLcontext context, 	//!< Opencl context
		void* host_ptr, 							//!< Pointer to data that initializes the buffer
		size_t size 								//!< Size of data that initializes the buffer
		);

/*! \brief Creates an Opencl command queue for a specific device */
CALCLqueue calclCreateCommandQueue(CALCLcontext context, 	//!< Opencl context
		CALCLdevice device 									//!< Device associated to the command queue
		);

/*! \brief Gets one of the allocated device */
CALCLdevice calclGetDevice(CALOpenCL * calOpenCL, 	//!< Pointer to CALOpenCL structure
		int platformIndex, 							//!< Index referring the platform associated to the device (indexes go from zero to the number of available platforms)
		int deviceIndex 							//!< Index referring the device (indexes go from zero to the number of available devices in the selected platform)
		);

/*! \brief Deallocates the structure CALOpencl */
void calclFinalizeCALOpencl(CALOpenCL * opencl 	//!< Pointer to CALOpenCL structure
		);

/*! \brief Given some paths returns the number of files in the paths and the names of the files  */
void calclGetDirFiles(char ** paths, 	//!< Array of strings. Each string is a path
		int pathsNum, 					//!< Number of paths
		char *** files_names, 			//!< Pointer to an array of strings. The function allocates the array and initializes it with paths files names
		int * num_files 				//!< Pointer to int. The function assigns to it the number of file contained in the paths
		);/*! \brief Print on standard output informations about all platforms and devices*/
void calclPrintAllPlatformAndDevices(CALOpenCL * opencl);

/*! \brief Print on standard output informations about all platforms and devices*/
void calclGetPlatformAndDeviceFromStandardInput(CALOpenCL * opencl,CALCLdevice * device);

/*! \brief Reads a file and return its content  */
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
CALCLprogram calclGetProgramFromFiles(char** filesNames, 	//!< Array of strings containing the files names
		unsigned filesNum, 									//!< Number of files
		char * compilerOptions, 							//!< String containing compiler options for the Opencl runtime compiler
		CALCLcontext context, 								//!< Opencl context
		CALCLdevice * devices, 								//!< Array of devices for which the program will be compiled
		CALCLuint num_devices 								//!< Number of devices
		);

/*! \brief Gets an Opencl kernel given a compiled Opencl program */
CALCLkernel calclGetKernelFromProgram(CALCLprogram * program, 	//!< Pointer to an Opencl program
		char * kernelName 										//!< Kernel name
		);

/*! \brief Converts Opencl error codes (cl_int) into error messages (char*)*/
const char * calclGetErrorString(CALCLint err 	//!< Integer representing an Opencl error code
		);

/*! \brief Handles Opencl errors codes
 *
 * The function takes in input an error code. If the error is minor then zero (this means there is some runtime error)
 * it prints an error message and closes the application otherwise it returns
 *
 * */
void calclHandleError(CALCLint err 	//!< Integer representing an Opencl error code
		);

/****************************************************************************
 * 		FUNCTIONS TO GET OPENCL STUFF WITHOUT CALOPENCL STRUCT
 * 		FUNCTIONS TO PRINT OPENCL DEVICES AND PLATFORMS INFO
 ****************************************************************************/

//utility file merging
//& removed - need test
/*! \brief Gets Opencl platforms*/
void calclGetPlatforms(CALCLplatform** platforms, 	//!< Pointer to an array of platforms. The function allocates it and assigns to it the available platforms
		CALCLuint * platformNum 					//!< Pointer to unsigned int. The function assigns to it the number of available platforms
		);

/*! \brief Get Opencl platforms by choosing a specific vendor*/
void calclGetPlatformByVendor(CALCLplatform * platform, 	//!< Pointer to an Opencl platform
		char* vendor 										//!< Platform vendor name
		);

/*! \brief Get Opencl NVIDIA platform*/
void calclGetPlatformNVIDIA(CALCLplatform * platform 	//!< Pointer to an Opencl platform
		);

/*! \brief Get Opencl INTEL platform*/
void calclGetPlatformINTEL(CALCLplatform * platform 	//!< Pointer to an Opencl platform
		);

/*! \brief Get Opencl AMD platform*/
void calclGetPlatformAMD(CALCLplatform * platform 		//!< Pointer to an Opencl platform
		);

/*! \brief Get Opencl platform name*/
char* calclGetPlatformName(CALCLplatform platform 		//!< Opencl platform
		);

/*! \brief Get Opencl platform vendor name*/
char* calclGetPlatformVendor(CALCLplatform platform 	//!< Opencl platform
		);

/*! \brief Get Opencl platform profile*/
char* calclGetPlatformProfile(CALCLplatform platform 	//!< Opencl platform
		);

/*! \brief Get Opencl platform version*/
char* calclGetPlatformVersion(CALCLplatform platform 	//!< Opencl platform
		);

/*! \brief Get Opencl platform supported extensions*/
char* calclGetPlatformExtensions(CALCLplatform platform //!< Opencl platform
		);

/*! \brief Print on standard output information about the given platform*/
void calclPrintAllPlatformInfo(CALCLplatform platform 	//!< Opencl platform
		);

/*! \brief Get Opencl devices*/
void calclGetDevices(CALCLplatform platform, 	//!< Opencl platform
		CALCLdeviceType type, 					//!< Platform type (CL_DEVICE_TYPE_CPU, CL_DEVICE_TYPE_GPU, CL_DEVICE_TYPE_ACCELLERATOR, CL_DEVICE_TYPE_ALL)
		CALCLdevice ** devices, 				//!< Pointer to array of Opencl devices. The function allocates it and assigns to it the available devices for the selected platform
		CALCLuint * size 						//!< Number of devices available for the selected platform
		);

/*! \brief Get Opencl devices with a specific name */
void calclGetDevicesByName(CALCLplatform plat, 	//!< Opencl platform
		CALCLdevice ** devices, 				//!< Pointer to array of Opencl devices. The function allocates it and assigns to it the available devices for the selected platform
		CALCLuint * size, 						//!< Number of devices available for the selected platform
		char* name 								//!< Device name
		);

/*! \brief Get Opencl device name*/
char* calclGetDeviceName(CALCLdevice device 	//!< Opencl device
		);

/*! \brief Get Opencl device vendor name*/
char* calclGetDeviceVendor(CALCLdevice device 	//!< Opencl device
		);

/*! \brief Get Opencl device supported extensions*/
char* calclGetDeviceExtensions(CALCLdevice device 	//!< Opencl device
		);

/*! \brief Get Opencl device global memory size (in bytes)*/
CALCLulong calclGetDeviceGlobalMemSize(CALCLdevice device 	//!< Opencl device
		);

/*! \brief Get Opencl device maximum number of work items dimensions*/
CALCLuint calclGetDeviceMaxWorkItemDimensions(CALCLdevice device 	//!< Opencl device
		);

/*! \brief Print on standard output informations about the given device*/
void calclPrintAllDeviceInfo(CALCLdevice device 	//!< Opencl device
		);

/*! \brief Print on standard output informations about all platforms and devices*/
void calclPrintAllPlatformAndDevices(CALOpenCL * opencl);

/*! \brief Print on standard output informations about all platforms and devices*/
void calclGetPlatformAndDeviceFromStandardInput(CALOpenCL * opencl,CALCLdevice * device);

#endif /* CALOpenCL_H_ */
