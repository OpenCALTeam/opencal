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

/*! \file calcl2D.h
 *\brief calcl2D contains structures and functions that allow to run parallel CA simulation using Opencl and OpenCAL.
 *
 *	calcl2D contains structures that allows easily to transfer data of a CALModel2D instance from host to GPU.
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

#define MODEL_ARGS_NUM 51	//!< Number of default arguments for each kernel defined by the user
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

/*! \brief CALCLModel2D contains necessary data to run 2D cellular automaton elementary processes on gpu.
 *
 * CALCLModel2D contains necessary data to run 2D cellular automata elementary processes on GPU. In particular,
 * it contains Opencl buffers to transfer CA data to the gpu, and Opencl kernels to setup a CA simulation.
 *
 */
struct CALCLModel2D {
	struct CALModel2D * host_CA;								//!< Pointer to a host-side CA
	enum CALOptimization opt;								//!< Enumeration used for optimization strategies (CAL_NO_OPT, CAL_OPT_ACTIVE_CELL).
	int callbackSteps;									//!< Define how many steps must be executed before call the function cl_update_substates.
	int steps;											//!< Simulation current step.
	void (*cl_update_substates)(struct CALModel2D*); 	//!< Callback function defined by the user. It allows to access data during a simulation.

	CALCLkernel kernelUpdateSubstate;					//!< Opencl kernel that updates substates GPU side (CALCL_OPT_ACTIVE_CELL strategy only)
	//user kernels
	CALCLkernel kernelInitSubstates;					//!< Opencl kernel defined by the user to initialize substates (optionally)
	CALCLkernel kernelSteering;							//!< Opencl kernel defined by the user to perform steering (optionally)
	CALCLkernel kernelStopCondition;					//!< Opencl kernel defined by the user to define a stop condition (optionally)
	CALCLkernel *elementaryProcesses;					//!< Array of Opencl kernels defined by the user. They represents CA elementary processes.

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

	//Reduction


	CALCLmem  bufferPartialMini;
	CALCLmem  bufferPartialMaxi;
	CALCLmem  bufferPartialSumi;
	CALCLmem  bufferPartialProdi;
	CALCLmem  bufferPartialLogicalAndi;
	CALCLmem  bufferPartialLogicalOri;
	CALCLmem  bufferPartialLogicalXOri;
	CALCLmem  bufferPartialBinaryAndi;
	CALCLmem  bufferPartialBinaryOri;
	CALCLmem  bufferPartialBinaryXOri;

	CALCLmem  bufferPartialMinb;
	CALCLmem  bufferPartialMaxb;
	CALCLmem  bufferPartialSumb;
	CALCLmem  bufferPartialProdb;
	CALCLmem  bufferPartialLogicalAndb;
	CALCLmem  bufferPartialLogicalOrb;
	CALCLmem  bufferPartialLogicalXOrb;
	CALCLmem  bufferPartialBinaryAndb;
	CALCLmem  bufferPartialBinaryOrb;
	CALCLmem  bufferPartialBinaryXOrb;

	CALCLmem  bufferPartialMinr;
	CALCLmem  bufferPartialMaxr;
	CALCLmem  bufferPartialSumr;
	CALCLmem  bufferPartialProdr;
	CALCLmem  bufferPartialLogicalAndr;
	CALCLmem  bufferPartialLogicalOrr;
	CALCLmem  bufferPartialLogicalXOrr;
	CALCLmem  bufferPartialBinaryAndr;
	CALCLmem  bufferPartialBinaryOrr;
	CALCLmem  bufferPartialBinaryXOrr;

	CALbyte * reductionFlagsMinb; 				//!< Pointer to array of flags that determine if a min reduction have to be compute
	CALbyte * reductionFlagsMini; 				//!< Pointer to array of flags that determine if a min reduction have to be compute
	CALbyte * reductionFlagsMinr; 				//!< Pointer to array of flags that determine if a min reduction have to be compute
	CALreal * minimab;										//!< Array of CALreal that contains the min results
	CALreal * minimai;										//!< Array of CALreal that contains the min results
	CALreal * minimar;										//!< Array of CALreal that contains the min results

	CALbyte * reductionFlagsMaxb;      	//!< Pointer to array of flags that determine if a max reduction have to be compute
	CALbyte * reductionFlagsMaxi;      	//!< Pointer to array of flags that determine if a max reduction have to be compute
	CALbyte * reductionFlagsMaxr;      	//!< Pointer to array of flags that determine if a max reduction have to be compute
	CALreal * maximab;										//!< Array of CALreal that contains the max results
	CALreal * maximai;										//!< Array of CALreal that contains the max results
	CALreal * maximar;										//!< Array of CALreal that contains the max results

	CALbyte * reductionFlagsSumb;				//!< Pointer to array of flags that determine if a sum reduction have to be compute
	CALbyte * reductionFlagsSumi;				//!< Pointer to array of flags that determine if a sum reduction have to be compute
	CALbyte * reductionFlagsSumr;				//!< Pointer to array of flags that determine if a sum reduction have to be compute
	CALreal * sumsb;											//!< Array of CALreal that contains the sum results
	CALreal * sumsi;											//!< Array of CALreal that contains the sum results
	CALreal * sumsr;											//!< Array of CALreal that contains the sum results

	CALbyte * reductionFlagsProdb;				//!< Pointer to array of flags that determine if a sum reduction have to be compute
	CALbyte * reductionFlagsProdi;				//!< Pointer to array of flags that determine if a sum reduction have to be compute
	CALbyte * reductionFlagsProdr;				//!< Pointer to array of flags that determine if a sum reduction have to be compute
	CALreal * prodsb;											//!< Array of CALreal that contains the sum results
	CALreal * prodsi;											//!< Array of CALreal that contains the sum results
	CALreal * prodsr;											//!< Array of CALreal that contains the sum results

	CALbyte * reductionFlagsLogicalAndb; //!< Pointer to array of flags that determine if a logical and reduction have to be compute
	CALbyte * reductionFlagsLogicalAndi; //!< Pointer to array of flags that determine if a logical and reduction have to be compute
	CALbyte * reductionFlagsLogicalAndr; //!< Pointer to array of flags that determine if a logical and reduction have to be compute
	CALint * logicalAndsb;							//!< Array of CALreal that contains the logical and results
	CALint * logicalAndsi;							//!< Array of CALreal that contains the logical and results
	CALint * logicalAndsr;							//!< Array of CALreal that contains the logical and results

	CALbyte * reductionFlagsLogicalOrb;	//!< Pointer to array of flags that determine if a logical or reduction have to be compute
	CALbyte * reductionFlagsLogicalOri;	//!< Pointer to array of flags that determine if a logical or reduction have to be compute
	CALbyte * reductionFlagsLogicalOrr;	//!< Pointer to array of flags that determine if a logical or reduction have to be compute
	CALint * logicalOrsb;								//!< Array of CALreal that contains the logical or results
	CALint * logicalOrsi;								//!< Array of CALreal that contains the logical or results
	CALint * logicalOrsr;								//!< Array of CALreal that contains the logical or results

	CALbyte * reductionFlagsLogicalXOrb;	//!< Pointer to array of flags that determine if a logical xor reduction have to be compute
	CALbyte * reductionFlagsLogicalXOri;	//!< Pointer to array of flags that determine if a logical xor reduction have to be compute
	CALbyte * reductionFlagsLogicalXOrr;	//!< Pointer to array of flags that determine if a logical xor reduction have to be compute
	CALint * logicalXOrsb;							//!< Array of CALreal that contains the logical xor results
	CALint * logicalXOrsi;							//!< Array of CALreal that contains the logical xor results
	CALint * logicalXOrsr;							//!< Array of CALreal that contains the logical xor results

	CALbyte * reductionFlagsBinaryAndb;	//!< Pointer to array of flags that determine if a binary and reduction have to be compute
	CALbyte * reductionFlagsBinaryAndi;	//!< Pointer to array of flags that determine if a binary and reduction have to be compute
	CALbyte * reductionFlagsBinaryAndr;	//!< Pointer to array of flags that determine if a binary and reduction have to be compute
	CALint * binaryAndsb;								//!< Array of CALreal that contains the binary amd results
	CALint * binaryAndsi;								//!< Array of CALreal that contains the binary amd results
	CALint * binaryAndsr;								//!< Array of CALreal that contains the binary amd results

	CALbyte * reductionFlagsBinaryOrb;		//!< Pointer to array of flags that determine if a binary or reduction have to be compute
	CALbyte * reductionFlagsBinaryOri;		//!< Pointer to array of flags that determine if a binary or reduction have to be compute
	CALbyte * reductionFlagsBinaryOrr;		//!< Pointer to array of flags that determine if a binary or reduction have to be compute
	CALint * binaryOrsb;								//!< Array of CALreal that contains the binary or results
	CALint * binaryOrsi;								//!< Array of CALreal that contains the binary or results
	CALint * binaryOrsr;								//!< Array of CALreal that contains the binary or results

	CALbyte * reductionFlagsBinaryXOrb;  //!< Pointer to array of flags that determine if a binary xor reduction have to be compute
	CALbyte * reductionFlagsBinaryXOri;  //!< Pointer to array of flags that determine if a binary xor reduction have to be compute
	CALbyte * reductionFlagsBinaryXOrr;  //!< Pointer to array of flags that determine if a binary xor reduction have to be compute
	CALint * binaryXOrsb;								//!< Array of CALreal that contains the binary xor results
	CALint * binaryXOrsi;								//!< Array of CALreal that contains the binary xor results
	CALint * binaryXOrsr;								//!< Array of CALreal that contains the binary xor results

	CALCLkernel kernelMinReductionb;
	CALCLkernel kernelMinReductioni;
	CALCLkernel kernelMinReductionr;

	CALCLkernel kernelMaxReductionb;
	CALCLkernel kernelMaxReductioni;
	CALCLkernel kernelMaxReductionr;

	CALCLkernel kernelSumReductionb;
	CALCLkernel kernelSumReductioni;
	CALCLkernel kernelSumReductionr;

	CALCLkernel kernelProdReductionb;
	CALCLkernel kernelProdReductioni;
	CALCLkernel kernelProdReductionr;

	CALCLkernel kernelLogicalAndReductionb;
	CALCLkernel kernelLogicalAndReductioni;
	CALCLkernel kernelLogicalAndReductionr;

	CALCLkernel kernelBinaryAndReductionb;
	CALCLkernel kernelBinaryAndReductioni;
	CALCLkernel kernelBinaryAndReductionr;

	CALCLkernel kernelLogicalOrReductionb;
	CALCLkernel kernelLogicalOrReductioni;
	CALCLkernel kernelLogicalOrReductionr;

	CALCLkernel kernelBinaryOrReductionb;
	CALCLkernel kernelBinaryOrReductioni;
	CALCLkernel kernelBinaryOrReductionr;

	CALCLkernel kernelLogicalXOrReductionb;
	CALCLkernel kernelLogicalXOrReductioni;
	CALCLkernel kernelLogicalXOrReductionr;

	CALCLkernel kernelBinaryXorReductionb;
	CALCLkernel kernelBinaryXorReductioni;
	CALCLkernel kernelBinaryXorReductionr;

	CALCLkernel kernelMinCopyb;
	CALCLkernel kernelMinCopyi;
	CALCLkernel kernelMinCopyr;

	CALCLkernel kernelSumCopyb;
	CALCLkernel kernelSumCopyi;
	CALCLkernel kernelSumCopyr;

	CALCLkernel kernelProdCopyb;
	CALCLkernel kernelProdCopyi;
	CALCLkernel kernelProdCopyr;


	CALCLkernel kernelMaxCopyb;
	CALCLkernel kernelMaxCopyi;
	CALCLkernel kernelMaxCopyr;

	CALCLkernel kernelLogicalAndCopyb;
	CALCLkernel kernelLogicalAndCopyi;
	CALCLkernel kernelLogicalAndCopyr;

	CALCLkernel kernelLogicalOrCopyb;
	CALCLkernel kernelLogicalOrCopyi;
	CALCLkernel kernelLogicalOrCopyr;

	CALCLkernel kernelLogicalXOrCopyb;
	CALCLkernel kernelLogicalXOrCopyi;
	CALCLkernel kernelLogicalXOrCopyr;

	CALCLkernel kernelBinaryAndCopyb;
	CALCLkernel kernelBinaryAndCopyi;
	CALCLkernel kernelBinaryAndCopyr;

	CALCLkernel kernelBinaryOrCopyb;
	CALCLkernel kernelBinaryOrCopyi;
	CALCLkernel kernelBinaryOrCopyr;

	CALCLkernel kernelBinaryXOrCopyb;
	CALCLkernel kernelBinaryXOrCopyi;
	CALCLkernel kernelBinaryXOrCopyr;


	CALCLmem bufferMinimab;
	CALCLmem bufferMinimai;
	CALCLmem bufferMinimar;

	CALCLmem bufferMiximab;
	CALCLmem bufferMiximai;
	CALCLmem bufferMiximar;

	CALCLmem bufferSumb;
	CALCLmem bufferSumi;
	CALCLmem bufferSumr;

	CALCLmem bufferProdb;
	CALCLmem bufferProdi;
	CALCLmem bufferProdr;

	CALCLmem bufferLogicalAndsb;
	CALCLmem bufferLogicalAndsi;
	CALCLmem bufferLogicalAndsr;

	CALCLmem bufferLogicalOrsb;
	CALCLmem bufferLogicalOrsi;
	CALCLmem bufferLogicalOrsr;

	CALCLmem bufferLogicalXOrsb;
	CALCLmem bufferLogicalXOrsi;
	CALCLmem bufferLogicalXOrsr;

	CALCLmem bufferBinaryAndsb;
	CALCLmem bufferBinaryAndsi;
	CALCLmem bufferBinaryAndsr;

	CALCLmem bufferBinaryOrsb;
	CALCLmem bufferBinaryOrsi;
	CALCLmem bufferBinaryOrsr;

	CALCLmem bufferBinaryXOrsb;
	CALCLmem bufferBinaryXOrsi;
	CALCLmem bufferBinaryXOrsr;

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

	CALCLcontext context;
	CALCLqueue queue;									//!< Opencl command queue
	int roundedDimensions;

};

/*! \brief Allocate, initialize and return a pointer to a struct CALCLModel2D.
 *
 * Allocate, initialize and return a pointer to a struct CALCLModel2D. Opencl buffers are initialized using data from a CALModel2D instance.
 * Moreover, the function receive an Opencl program used to initialize library kernels.
 */
struct CALCLModel2D * calclCADef2D(struct CALModel2D *host_CA,		//!< Pointer to a CALModel2D
		CALCLcontext context,										//!< Opencl context
		CALCLprogram program,										//!< Opencl program containing library source and user defined source
		CALCLdevice device											//!< Opencl device
		);

/*! \brief Main simulation cycle. It can become a loop if maxStep == CALCL_RUN_LOOP */
void calclRun2D(struct CALCLModel2D* calclmodel2D, 		//!< Pointer to a struct CALCLModel2D
		unsigned int initialStep,				//!< Initial simulation step
		unsigned maxStep						//!< Maximum number of CA steps. Simulation can become a loop if maxStep == CALCL_RUN_LOOP
		);

/*! \brief A single step of CA. It executes the transition function, the steering and check the stop condition */
CALbyte calclSingleStep2D(struct CALCLModel2D* calclmodel2D,		//!< Pointer to a struct CALCLModel2D
		size_t * dimSize,							//!< Array of size_t containing the number of threads for each used Opencl dimension (CALCL_NO_OPT 2 dimensions, CALCL_OPT_ACTIVE_CELL 1 dimension)
		int dimNum											//!< Number of Opencl dimensions (CALCL_NO_OPT 2 dimensions, CALCL_OPT_ACTIVE_CELL 1 dimension)
		);

/*! \brief Execute an Opencl kernel */
void calclKernelCall2D(struct CALCLModel2D* calclmodel2D,		//!< Pointer to a struct CALCLModel2D
		CALCLkernel ker,								//!< Opencl kernel
		int dimNum,										//!< Number of Opencl dimensions (CALCL_NO_OPT 2 dimensions, CALCL_OPT_ACTIVE_CELL 1 dimension)
		size_t * dimSize,							//!< Array of size_t containing the number of threads for each used Opencl dimension (CALCL_NO_OPT 2 dimensions, CALCL_OPT_ACTIVE_CELL 1 dimension)
		size_t * localDimSize							//!< Array of size_t containing the number of threads for each used Opencl local dimension
		);

/*! \brief Execute stream compaction kernels to compact and order CA active cells */
void calclComputeStreamCompaction2D(struct CALCLModel2D * calclmodel2D		//!< Pointer to a struct CALCLModel2D
		);

/*! \brief Add arguments to the given Opencl kernel defined by the user
 *
 * Add arguments to the given Opencl kernel defined by the user. Kernel arguments are added
 * after the default argument provided by the library.
 *
 *  */
void calclSetKernelArgs2D(CALCLkernel * kernel,		//!< Pointer to Opencl kernel
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
void calclAddStopConditionFunc2D(struct CALCLModel2D * calclmodel2D,		//!< Pointer to a struct CALCLModel2D
		CALCLkernel * kernel										//!< Pointer to Opencl kernel
		);

/*! \brief Set the Opencl kernel used to initialize substates
 *
 * Set the Opencl kernel used to initialize substates. If defined, the kernel is executed
 * at the beginning of the simulation
 *
 *  */
void calclAddInitFunc2D(struct CALCLModel2D * calclmodel2D,		//!< Pointer to a struct CALCLModel2D
		CALCLkernel * kernel										//!< Pointer to Opencl kernel
		);

/*! \brief Set the steering Opencl kernel
 *
 * Set the steering Opencl kernel. If defined, the stop condition kernel is executed
 * each time the function calclSingleStep2D is called.
 *
 *  */
void calclAddSteeringFunc2D(struct CALCLModel2D * calclmodel2D,		//!< Pointer to a struct CALCLModel2D
		CALCLkernel * kernel									//!< Pointer to Opencl kernel
		);

/*! \brief Set the function used to access substate on the GPU every callbackSteps steps.
 *
 *	Set the function used to access substate on the GPU every callbackSteps steps. This function
 *	could decrease the performance because of the transfer of data between host and GPU.
 *
 *  */
void calclBackToHostFunc2D(struct CALCLModel2D* calclmodel2D,		//!< Pointer to a struct CALCLModel2D
		void (*cl_update_substates)(struct CALModel2D*),				//!< Callback function executed each callbackSteps steps
		int callbackSteps												//!< Define how many steps must be executed before call the callback functions
		);

/*! \brief Add an Opencl kernel to the elementary processes kernels.
 *
 *	Add an Opencl kernel to the elementary processes kernels. Each elementary process kernel
 *	is executed each time the function calclSingleStep2D is called.
 *
 *  */
void calclAddElementaryProcess2D(struct CALCLModel2D * calclmodel2D,		//!< Pointer to a struct CALCLModel2D
		CALCLkernel * kernel											//!< Pointer to Opencl kernel
		);

void calclAddMinReduction2Db(struct CALCLModel2D * calclmodel2D,					//!< Pointer to a struct CALCLModel2D
		int numSubstates													//!< Number of the substate
		);
void calclAddMinReduction2Di(struct CALCLModel2D * calclmodel2D,					//!< Pointer to a struct CALCLModel2D
		int numSubstates													//!< Number of the substate
		);
void calclAddMinReduction2Dr(struct CALCLModel2D * calclmodel2D,					//!< Pointer to a struct CALCLModel2D
		int numSubstates													//!< Number of the substate
		);

void calclAddMaxReduction2Db(struct CALCLModel2D * calclmodel2D, int numSubstate);
void calclAddMaxReduction2Di(struct CALCLModel2D * calclmodel2D, int numSubstate);
void calclAddMaxReduction2Dr(struct CALCLModel2D * calclmodel2D, int numSubstate);

void calclAddSumReduction2Db(struct CALCLModel2D * calclmodel2D, int numSubstate);
void calclAddSumReduction2Di(struct CALCLModel2D * calclmodel2D, int numSubstate);
void calclAddSumReduction2Dr(struct CALCLModel2D * calclmodel2D, int numSubstate);

void calclAddProdReduction2Db(struct CALCLModel2D * calclmodel2D, int numSubstate);
void calclAddProdReduction2Di(struct CALCLModel2D * calclmodel2D, int numSubstate);
void calclAddProdReduction2Dr(struct CALCLModel2D * calclmodel2D, int numSubstate);

void calclAddLogicalAndReduction2Db(struct CALCLModel2D * calclmodel2D, int numSubstate);
void calclAddLogicalAndReduction2Di(struct CALCLModel2D * calclmodel2D, int numSubstate);
void calclAddLogicalAndReduction2Dr(struct CALCLModel2D * calclmodel2D, int numSubstate);

void calclAddLogicalOrReduction2Db(struct CALCLModel2D * calclmodel2D, int numSubstate);
void calclAddLogicalOrReduction2Di(struct CALCLModel2D * calclmodel2D, int numSubstate);
void calclAddLogicalOrReduction2Dr(struct CALCLModel2D * calclmodel2D, int numSubstate);

void calclAddLogicalXOrReduction2Db(struct CALCLModel2D * calclmodel2D, int numSubstate);
void calclAddLogicalXOrReduction2Di(struct CALCLModel2D * calclmodel2D, int numSubstate);
void calclAddLogicalXOrReduction2Dr(struct CALCLModel2D * calclmodel2D, int numSubstate);

void calclAddBinaryAndReduction2Db(struct CALCLModel2D * calclmodel2D, int numSubstate);
void calclAddBinaryAndReduction2Di(struct CALCLModel2D * calclmodel2D, int numSubstate);
void calclAddBinaryAndReduction2Dr(struct CALCLModel2D * calclmodel2D, int numSubstate);

void calclAddBinaryOrReduction2Db(struct CALCLModel2D * calclmodel2D, int numSubstate);
void calclAddBinaryOrReduction2Di(struct CALCLModel2D * calclmodel2D, int numSubstate);
void calclAddBinaryOrReduction2Dr(struct CALCLModel2D * calclmodel2D, int numSubstate);

void calclAddBinaryXOrReduction2Db(struct CALCLModel2D * calclmodel2D, int numSubstate);
void calclAddBinaryXOrReduction2Di(struct CALCLModel2D * calclmodel2D, int numSubstate);
void calclAddBinaryXOrReduction2Dr(struct CALCLModel2D * calclmodel2D, int numSubstate);

/*! \brief Deallcate a struct CALCLModel2D instance */
void calclFinalize2D(struct CALCLModel2D * calclmodel2D	//!< Pointer to a struct CALCLModel2D
		);

/*! \brief Allocate, initialize and return an Opencl program
 *
 *	Allocate, initialize and return an Opencl program. The program returned
 *	is compiled using library source files and user defined source files.
 *
 *  */
CALCLprogram calclLoadProgram2D(CALCLcontext context,		//!< Opencl context
		CALCLdevice device,										//!< Opencl device
		char* path_user_kernel,									//!< Kernel source files path
		char* path_user_include								//!< Kernel include files path
		);

/*! \brief Set a kernel argument   */
int calclSetKernelArg2D(CALCLkernel* kernel,			//!< Opencl kernel
		cl_uint arg_index,			//!< Index argument
		size_t arg_size,			//!< Size argument
		const void *arg_value                   //!< Value argument
		);

/*! \brief Copy all the substates device memory to host memory   */
void calclGetSubstatesDeviceToHost2D(struct CALCLModel2D* calclmodel2D //!< Pointer to a CALCLModel3D
		);

void calclSetReductionParameters2D(struct CALCLModel2D* calclmodel2D, CALCLkernel * kernel);

#endif /* CALCL_H_ */
