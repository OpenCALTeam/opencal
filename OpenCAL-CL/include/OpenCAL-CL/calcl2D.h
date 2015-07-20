// (C) Copyright University of Calabria and others.
// All rights reserved. This program and the accompanying materials
// are made available under the terms of the GNU Lesser General Public License
// (LGPL) version 2.1 which accompanies this distribution, and is available at
// http://www.gnu.org/licenses/lgpl-2.1.html
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
// Lesser General Public License for more details.

/*! \file calCL2D.h
 *\brief calCL2D contains structures and functions that allow to run parallel CA simulation using Opencl and OpenCAL.
 *
 *	calCL2D contains structures that allows easily to transfer data of a CALModel2D instance from host to GPU.
 *	It's possible to setup a CA simulation by only defining kernels for elementary processes, and optionally
 *	initialization, steering and stop condition. Moreover, user can avoid to use the simulation cycle provided
 *	by the library and define his own simulation cycle.
 */

#ifndef CALCL_H_
#define CALCL_H_

#include <OpenCAL/cal2D.h>
#include "math.h"
#include "OpenCL_Utility.h"

#ifdef _WIN32
#define ROOT_DIR ".."
#else
#define ROOT_DIR ".."
#endif // _WIN32

#define KERNEL_SOURCE_DIR "/kernel/source/" 	//!< Library kernel source file
#define KERNEL_INCLUDE_DIR "/kernel/include"	//!< Library kernel include file

#define KER_UPDATESUBSTATES "calclkernelUpdateSubstates2D"

//stream compaction kernels
#define KER_STC_COMPACT "calclkernelCompact2D"
#define KER_STC_COMPUTE_COUNTS "calclkernelComputeCounts2D"
#define KER_STC_UP_SWEEP "calclkernelUpSweep2D"
#define KER_STC_DOWN_SWEEP "calclkernelDownSweep2D"

#define MODEL_ARGS_NUM 21	//!< Number of default arguments for each kernel defined by the user
#define CALCL_RUN_LOOP 0	//!< Define used by the user to specify an infinite loop simulation

/*! \brief CALCLSubstateMapper contains arrays used to retrieve substates from GPU
 *
 * CALCLSubstateMapper contains arrays used to retrieve all substates from GPU. There is an
 * array for each type of substate defined in the library OpenCAL.
 *
 */
typedef struct CALCLSubstateMapper {

	size_t bufDIMreal;							//!< Number of CALreal substates
	size_t bufDIMint;							//!< Number of CALint substates
	size_t bufDIMbyte;							//!< Number of CALbyte substates

	CALreal * realSubstate_current_OUT;			//!< Array containing all the CALreal substates
	CALint * intSubstate_current_OUT;			//!< Array containing all the CALint substates
	CALbyte * byteSubstate_current_OUT;			//!< Array containing all the CALbyte substates

} CALCLSubstateMapper;

/*! \brief CALCLToolkit2D contains necessary data to run 2D cellular automaton elementary processes on gpu.
 *
 * CALCLToolkit2D contains necessary data to run 2D cellular automata elementary processes on GPU. In particular,
 * it contains Opencl buffers to transfer CA data to the gpu, and Opencl kernels to setup a CA simulation.
 *
 */
typedef struct CALCLToolkit2D {
	enum CALOptimization opt;								//!< Enumeration used for optimization strategies (CAL_NO_OPT, CAL_OPT_ACTIVE_CELL).
	int callbackSteps;									//!< Define how many steps must be executed before call the function cl_update_substates.
	int steps;											//!< Simulation current step.
	void (*cl_update_substates)(struct CALModel2D*); 	//!< Callback function defined by the user. It allows to access data during a simulation.

	CALCLkernel kernelUpdateSubstate;					//!< Opencl kernel that updates substates GPU side (CALCL_OPT_ACTIVE_CELL strategy only)
	//user kernels
	CALCLkernel kernelInitSubstates;					//!< Opencl kernel defined by the user to initialize substates (optionally)
	CALCLkernel kernelSteering;							//!< Opencl kernel defined by the user to perform steering (optionally)
	CALCLkernel kernelStopCondition;					//!< Opencl kernel defined by the user to define a stop condition (optionally)
	CALCLkernel * elementaryProcesses;					//!< Array of Opencl kernels defined by the user. They represents CA elementary processes.

	CALint elementaryProcessesNum;						//!< Number of elementary processes defined by the user

	CALCLmem bufferColumns;								//!< Opencl buffer used to transfer GPU side the number of CA columns
	CALCLmem bufferRows;								//!< Opencl buffer used to transfer GPU side the number of CA rows

	CALCLmem bufferByteSubstateNum;						//!< Opencl buffer used to transfer GPU side the number of CA CALbyte substates
	CALCLmem bufferIntSubstateNum;						//!< Opencl buffer used to transfer GPU side the number of CA CALint substates
	CALCLmem bufferRealSubstateNum;						//!< Opencl buffer used to transfer GPU side the number of CA CALreal substates

	CALCLmem bufferCurrentByteSubstate;					//!< Opencl buffer used to transfer GPU side CALbyte current substates used for reading purposes
	CALCLmem bufferCurrentIntSubstate;					//!< Opencl buffer used to transfer GPU side CALint current substates used for reading purposes
	CALCLmem bufferCurrentRealSubstate;					//!< Opencl buffer used to transfer GPU side CALreal current substates used for reading purposes

	CALCLmem bufferNextByteSubstate;					//!< Opencl buffer used to transfer GPU side CALbyte next substates used for writing purposes
	CALCLmem bufferNextIntSubstate;						//!< Opencl buffer used to transfer GPU side CALint next substates used for writing purposes
	CALCLmem bufferNextRealSubstate;					//!< Opencl buffer used to transfer GPU side CALreal next substates used for writing purposes

	CALCLmem bufferActiveCells;							//!< Opencl buffer used to transfer GPU side CA active cells array
	CALCLmem bufferActiveCellsNum;						//!< Opencl buffer used to transfer GPU side the number of CA active cells
	CALCLmem bufferActiveCellsFlags;					//!< Opencl buffer used to transfer GPU side CA active cells flags array (CALbyte*)

	//stop condition
	CALCLmem bufferStop;								//!< Opencl buffer used to transfer GPU side CALbyte stop flag. The user can set it to CAL_TRUE to stop the CA simulation

	CALCLSubstateMapper substateMapper;					//!< Structure used to retrieve substates from GPU

	//user kernels buffers args
	CALCLmem bufferNeighborhood;						//!< Opencl buffer used to transfer GPU side the array representing the CA neighborhood
	CALCLmem bufferNeighborhoodID;						//!< Opencl buffer used to transfer GPU side CA neighborhood ID
	CALCLmem bufferNeighborhoodSize;					//!< Opencl buffer used to transfer GPU side CA neighborhood size
	CALCLmem bufferBoundaryCondition;					//!< Opencl buffer used to transfer GPU side CA boundary conditions

	//stream compaction kernel
	CALCLkernel kernelComputeCounts;					//!< Opencl kernel used to compute stream compaction
	CALCLkernel kernelUpSweep;							//!< Opencl kernel used to compute stream compaction
	CALCLkernel kernelDownSweep;						//!< Opencl kernel used to compute stream compaction
	CALCLkernel kernelCompact;							//!< Opencl kernel used to compute stream compaction

	CALCLmem bufferSTCounts;							//!< Opencl buffer used by stream compaction algorithm
	CALCLmem bufferSTOffsets1;							//!< Opencl buffer used by stream compaction algorithm
	CALCLmem bufferSTCountsDiff;						//!< Opencl buffer used by stream compaction algorithm
	size_t streamCompactionThreadsNum;					//!< Number of threads used to compute stream compaction

	CALCLqueue queue;									//!< Opencl command queue

} CALCLToolkit2D;

/*! \brief Allocate, initialize and return a pointer to a CALCLToolkit2D.
 *
 * Allocate, initialize and return a pointer to a CALCLToolkit2D. Opencl buffers are initialized using data from a CALModel2D instance.
 * Moreover, the function receive an Opencl program used to initialize library kernels.
 */
CALCLToolkit2D * calclCreateToolkit2D(struct CALModel2D *model,		//!< Pointer to a CALModel2D
		CALCLcontext context,										//!< Opencl context
		CALCLprogram program,										//!< Opencl program containing library source and user defined source
		CALCLdevice device,											//!< Opencl device
		enum CALOptimization opt											//!< Optimization strategies (CAL_NO_OPT, CAL_OPT_ACTIVE_CELL)
		);

/*! \brief Main simulation cycle. It can become a loop if maxStep == CALCL_RUN_LOOP */
void calclRun2D(CALCLToolkit2D* toolkit2d, 		//!< Pointer to a CALCLToolkit2D
		struct CALModel2D * model,				//!< Pointer to a CALModel2D
		unsigned maxStep						//!< Maximum number of CA steps. Simulation can become a loop if maxStep == CALCL_RUN_LOOP
		);

/*! \brief A single step of CA. It executes the transition function, the steering and check the stop condition */
CALbyte calclSingleStep2D(CALCLToolkit2D* toolkit2d,		//!< Pointer to a CALCLToolkit2D
		struct CALModel2D * model,							//!< Pointer to a CALModel2D
		size_t * dimSize,									//!< Array of size_t containing the number of threads for each used Opencl dimension (CALCL_NO_OPT 2 dimensions, CALCL_OPT_ACTIVE_CELL 1 dimension)
		int dimNum											//!< Number of Opencl dimensions (CALCL_NO_OPT 2 dimensions, CALCL_OPT_ACTIVE_CELL 1 dimension)
		);

/*! \brief Execute an Opencl kernel */
void calclKernelCall2D(CALCLToolkit2D* toolkit2d,		//!< Pointer to a CALCLToolkit2D
		CALCLkernel ker,								//!< Opencl kernel
		int dimNum,										//!< Number of Opencl dimensions (CALCL_NO_OPT 2 dimensions, CALCL_OPT_ACTIVE_CELL 1 dimension)
		size_t * dimSize,								//!< Array of size_t containing the number of threads for each used Opencl dimension (CALCL_NO_OPT 2 dimensions, CALCL_OPT_ACTIVE_CELL 1 dimension)
		size_t * localDimSize							//!< Array of size_t containing the number of threads for each used Opencl local dimension
		);

/*! \brief Execute stream compaction kernels to compact and order CA active cells */
void calclComputeStreamCompaction2D(CALCLToolkit2D * toolkit		//!< Pointer to a CALCLToolkit2D
		);

/*! \brief Add arguments to the given Opencl kernel defined by the user
 *
 * Add arguments to the given Opencl kernel defined by the user. Kernel arguments are added
 * after the default argument provided by the library.
 *
 *  */
void calclSetCALKernelArgs2D(CALCLkernel * kernel,		//!< Pointer to Opencl kernel
		CALCLmem * args,								//!< Array of Opencl buffers that represents kernel additional arguments
		cl_uint numArgs									//!< Number of Opencl kernel additional arguments
		);

/*! \brief Set the stop condition Opencl kernel
 *
 * Set the stop condition Opencl kernel. If defined, the stop condition kernel is executed
 * each time the function calclSingleStep2D is called. Set the kernel argument stop to CAL_TRUE
 * to stop the simulation.
 *
 *  */
void calclSetStopConditionKernel2D(CALCLToolkit2D * toolkit2d,		//!< Pointer to a CALCLToolkit2D
		struct CALModel2D *model,									//!< Pointer to a CALModel2D
		CALCLkernel * kernel										//!< Pointer to Opencl kernel
		);

/*! \brief Set the Opencl kernel used to initialize substates
 *
 * Set the Opencl kernel used to initialize substates. If defined, the kernel is executed
 * at the beginning of the simulation
 *
 *  */
void calclSetInitSubstatesKernel2D(CALCLToolkit2D * toolkit2d,		//!< Pointer to a CALCLToolkit2D
		struct CALModel2D *model,									//!< Pointer to a CALModel2D
		CALCLkernel * kernel										//!< Pointer to Opencl kernel
		);

/*! \brief Set the steering Opencl kernel
 *
 * Set the steering Opencl kernel. If defined, the stop condition kernel is executed
 * each time the function calclSingleStep2D is called.
 *
 *  */
void calclSetSteeringKernel2D(CALCLToolkit2D * toolkit2d,		//!< Pointer to a CALCLToolkit2D
		struct CALModel2D *model,								//!< Pointer to a CALModel2D
		CALCLkernel * kernel									//!< Pointer to Opencl kernel
		);

/*! \brief Set the function used to access substate on the GPU every callbackSteps steps.
 *
 *	Set the function used to access substate on the GPU every callbackSteps steps. This function
 *	could decrease the performance because of the transfer of data between host and GPU.
 *
 *  */
void calclSetUpdateSubstatesFunction2D(CALCLToolkit2D* toolkit2d,		//!< Pointer to a CALCLToolkit2D
		void (*cl_update_substates)(struct CALModel2D*),				//!< Callback function executed each callbackSteps steps
		int callbackSteps												//!< Define how many steps must be executed before call the callback functions
		);

/*! \brief Add an Opencl kernel to the elementary processes kernels.
 *
 *	Add an Opencl kernel to the elementary processes kernels. Each elementary process kernel
 *	is executed each time the function calclSingleStep2D is called.
 *
 *  */
void calclAddElementaryProcessKernel2D(CALCLToolkit2D * toolkit2d,		//!< Pointer to a CALCLToolkit2D
		struct CALModel2D *model,										//!< Pointer to a CALModel2D
		CALCLkernel * kernel											//!< Pointer to Opencl kernel
		);

/*! \brief Deallcate a CALCLToolkit2D instance */
void calclFinalizeToolkit2D(CALCLToolkit2D * toolkit2d		//!< Pointer to a CALCLToolkit2D
		);

/*! \brief Allocate, initialize and return an Opencl program
 *
 *	Allocate, initialize and return an Opencl program. The program returned
 *	is compiled using library source files and user defined source files.
 *
 *  */
CALCLprogram calclLoadProgramLib2D(CALCLcontext context,		//!< Opencl context
		CALCLdevice device,										//!< Opencl device
		char* path_user_kernel,									//!< Kernel source files path
		char* path_user_include								//!< Kernel include files path
		);


#endif /* CALCL_H_ */
