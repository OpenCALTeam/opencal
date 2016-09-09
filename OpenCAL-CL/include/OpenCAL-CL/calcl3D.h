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

/*! \file calcl3D.h
 *\brief calcl3D contains structures and functions that allow to run parallel CA simulation using Opencl and OpenCAL.
 *
 *	calcl3D contains structures that allows easily to transfer data of a CALModel3D instance from host to GPU.
 *	It's possible to setup a CA simulation by only defining kernels for elementary processes, and optionally
 *	initialization, steering and stop condition. Moreover, user can avoid to use the simulation cycle provided
 *	by the library and define his own simulation cycle.
 */
#include <math.h>
#include <OpenCAL/cal3D.h>
#include <OpenCAL-CL/clUtility.h>
#include <OpenCAL-CL/dllexport.h>

#ifdef _WIN32
#define ROOT_DIR ".."
#else
#define ROOT_DIR "../../.."
#endif // _WIN32

#ifndef CALCL_H_
#define CALCL_H_
#define KERNEL_SOURCE_DIR "/kernel/source/"     //!< Library kernel source file
#define KERNEL_INCLUDE_DIR "/kernel/include"	//!< Library kernel include file
#define PARAMETERS_FILE ROOT_DIR"/OpenCAL-CL/parameters"

//#define KERNEL_COMPILER_OPTIONS " -I "KERNEL_INCLUDE_DIR

#define KER_UPDATESUBSTATES "calclkernelUpdateSubstates3D"

//stream compaction kernels
#define KER_STC_COMPACT "calclkernelCompact3D"
#define KER_STC_COMPUTE_COUNTS "calclkernelComputeCounts3D"
#define KER_STC_UP_SWEEP "calclkernelUpSweep3D"
#define KER_STC_DOWN_SWEEP "calclkernelDownSweep3D"

#define MODEL_ARGS_NUM 58		//!< Number of default arguments for each kernel defined by the user
#define CALCL_RUN_LOOP 0		//!< Define used by the user to specify an infinite loop simulation

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

/*! \brief CALCLModel3D contains necessary data to run 3D cellular automaton elementary processes on gpu.
 *
 * CALCLModel3D contains necessary data to run 3D cellular automata elementary processes on GPU. In particular,
 * it contains Opencl buffers to transfer CA data to the gpu, and Opencl kernels to setup a CA simulation.
 *
 */
struct CALCLModel3D {
	struct CALModel3D * host_CA;						//!< Pointer to a host-side CA
	enum CALOptimization opt;							//!< Enumeration used for optimization strategies (CAL_NO_OPT, CAL_OPT_ACTIVE_CELL).
	int callbackSteps;									//!< Define how many steps must be executed before call the function cl_update_substates.
	int steps;											//!< Simulation current step.
	void (*cl_update_substates)(struct CALModel3D*);	//!< Callback function defined by the user. It allows to access data during a simulation.

	CALCLkernel kernelUpdateSubstate;					//!< Opencl kernel that updates substates GPU side (CALCL_OPT_ACTIVE_CELL strategy only)
	//user kernels
	CALCLkernel kernelInitSubstates;					//!< Opencl kernel defined by the user to initialize substates (optionally)
	CALCLkernel kernelSteering;							//!< Opencl kernel defined by the user to perform steering (optionally)
	CALCLkernel kernelStopCondition;					//!< Opencl kernel defined by the user to define a stop condition (optionally)
	CALCLkernel * elementaryProcesses;					//!< Array of Opencl kernels defined by the user. They represents CA elementary processes.

	CALint elementaryProcessesNum;						//!< Number of elementary processes defined by the user

	CALCLmem bufferColumns;								//!< Opencl buffer used to transfer GPU side the number of CA columns
	CALCLmem bufferRows;								//!< Opencl buffer used to transfer GPU side the number of CA rows
	CALCLmem bufferSlices;								//!< Opencl buffer used to transfer GPU side the number of CA slices

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

    CALCLmem bufferSingleLayerByteSubstateNum;			//!< Opencl buffer used to transfer GPU side the number of CA CALbyte single layer substates
    CALCLmem bufferSingleLayerIntSubstateNum;			//!< Opencl buffer used to transfer GPU side the number of CA CALint single layer substates
    CALCLmem bufferSingleLayerRealSubstateNum;			//!< Opencl buffer used to transfer GPU side the number of CA CALreal single layer substates

    CALCLmem bufferSingleLayerByteSubstate;				//!< Opencl buffer used to transfer GPU side CALbyte single layer substates used for reading purposes
    CALCLmem bufferSingleLayerIntSubstate;				//!< Opencl buffer used to transfer GPU side CALint single layer substates used for reading purposes
    CALCLmem bufferSingleLayerRealSubstate;				//!< Opencl buffer used to transfer GPU side CALreal single layer substates used for reading purposes

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

	CALCLmem  bufferPartialMini;                        //!< Opencl buffer used as auxiliary array device side for min CALint substates reduction
	CALCLmem  bufferPartialMaxi;                        //!< Opencl buffer used as auxiliary array device side for max CALint substates reduction
	CALCLmem  bufferPartialSumi;                        //!< Opencl buffer used as auxiliary array device side for sum CALint substates reduction
	CALCLmem  bufferPartialProdi;                       //!< Opencl buffer used as auxiliary array device side for prod CALint substates reduction
	CALCLmem  bufferPartialLogicalAndi;                 //!< Opencl buffer used as auxiliary array device side for logical and CALint substates reduction
	CALCLmem  bufferPartialLogicalOri;                  //!< Opencl buffer used as auxiliary array device side for logical or CALint substates reduction
	CALCLmem  bufferPartialLogicalXOri;                 //!< Opencl buffer used as auxiliary array device side for logical xor CALint substates reduction
	CALCLmem  bufferPartialBinaryAndi;                  //!< Opencl buffer used as auxiliary array device side for binary and CALint substates reduction
	CALCLmem  bufferPartialBinaryOri;                   //!< Opencl buffer used as auxiliary array device side for binary or CALint substates reduction
	CALCLmem  bufferPartialBinaryXOri;                  //!< Opencl buffer used as auxiliary array device side for binary xor CALint substates reduction

	CALCLmem  bufferPartialMinb;                        //!< Opencl buffer used as auxiliary array device side for min CALbyte substates reduction
	CALCLmem  bufferPartialMaxb;                        //!< Opencl buffer used as auxiliary array device side for max CALbyte substates reduction
	CALCLmem  bufferPartialSumb;                        //!< Opencl buffer used as auxiliary array device side for sum CALbyte substates reduction
	CALCLmem  bufferPartialProdb;                       //!< Opencl buffer used as auxiliary array device side for prod CALbyte substates reduction
	CALCLmem  bufferPartialLogicalAndb;                 //!< Opencl buffer used as auxiliary array device side for logical and CALbyte substates reduction
	CALCLmem  bufferPartialLogicalOrb;                  //!< Opencl buffer used as auxiliary array device side for logical or CALbyte substates reduction
	CALCLmem  bufferPartialLogicalXOrb;                 //!< Opencl buffer used as auxiliary array device side for logical xor CALbyte substates reduction
	CALCLmem  bufferPartialBinaryAndb;                  //!< Opencl buffer used as auxiliary array device side for binary and CALbyte substates reduction
	CALCLmem  bufferPartialBinaryOrb;                   //!< Opencl buffer used as auxiliary array device side for binary or CALbyte substates reduction
	CALCLmem  bufferPartialBinaryXOrb;                  //!< Opencl buffer used as auxiliary array device side for binary xor CALbyte substates reduction

	CALCLmem  bufferPartialMinr;                        //!< Opencl buffer used as auxiliary array device side for min CALreal substates reduction
	CALCLmem  bufferPartialMaxr;                        //!< Opencl buffer used as auxiliary array device side for max CALreal substates reduction
	CALCLmem  bufferPartialSumr;                        //!< Opencl buffer used as auxiliary array device side for sum CALreal substates reduction
	CALCLmem  bufferPartialProdr;                       //!< Opencl buffer used as auxiliary array device side for prod CALreal substates reduction
	CALCLmem  bufferPartialLogicalAndr;                 //!< Opencl buffer used as auxiliary array device side for logical and CALreal substates reduction
	CALCLmem  bufferPartialLogicalOrr;                  //!< Opencl buffer used as auxiliary array device side for logical or CALreal substates reduction
	CALCLmem  bufferPartialLogicalXOrr;                 //!< Opencl buffer used as auxiliary array device side for logical xor CALreal substates reduction
	CALCLmem  bufferPartialBinaryAndr;                  //!< Opencl buffer used as auxiliary array device side for binary and CALreal substates reduction
	CALCLmem  bufferPartialBinaryOrr;                   //!< Opencl buffer used as auxiliary array device side for binary or CALreal substates reduction
	CALCLmem  bufferPartialBinaryXOrr;                  //!< Opencl buffer used as auxiliary array device side for binary xor CALreal substates reduction

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

	CALCLkernel kernelMinReductionb;                    //!< Opencl kernel that compute min reduction for CALbyte substates GPU side
	CALCLkernel kernelMinReductioni;                    //!< Opencl kernel that compute min reduction for CALint substates GPU side
	CALCLkernel kernelMinReductionr;                    //!< Opencl kernel that compute min reduction for CALreal substates GPU side

	CALCLkernel kernelMaxReductionb;                    //!< Opencl kernel that compute max reduction for CALbyte substates GPU side
	CALCLkernel kernelMaxReductioni;                    //!< Opencl kernel that compute max reduction for CALint substates GPU side
	CALCLkernel kernelMaxReductionr;                    //!< Opencl kernel that compute max reduction for CALreal substates GPU side

	CALCLkernel kernelSumReductionb;                    //!< Opencl kernel that compute sum reduction for CALbyte substates GPU side
	CALCLkernel kernelSumReductioni;                    //!< Opencl kernel that compute sum reduction for CALint substates GPU side
	CALCLkernel kernelSumReductionr;                    //!< Opencl kernel that compute sum reduction for CALreal substates GPU side

	CALCLkernel kernelProdReductionb;                   //!< Opencl kernel that compute prod reduction for CALbyte substates GPU side
	CALCLkernel kernelProdReductioni;                   //!< Opencl kernel that compute prod reduction for CALint substates GPU side
	CALCLkernel kernelProdReductionr;                   //!< Opencl kernel that compute prod reduction for CALreal substates GPU side

	CALCLkernel kernelLogicalAndReductionb;             //!< Opencl kernel that compute logical and reduction for CALbyte substates GPU side
	CALCLkernel kernelLogicalAndReductioni;             //!< Opencl kernel that compute logical and reduction for CALint substates GPU side
	CALCLkernel kernelLogicalAndReductionr;             //!< Opencl kernel that compute logical and reduction for CALreal substates GPU side

	CALCLkernel kernelBinaryAndReductionb;              //!< Opencl kernel that compute binary and reduction for CALbyte substates GPU side
	CALCLkernel kernelBinaryAndReductioni;              //!< Opencl kernel that compute binary and reduction for CALint substates GPU side
	CALCLkernel kernelBinaryAndReductionr;              //!< Opencl kernel that compute binary and reduction for CALreal substates GPU side

	CALCLkernel kernelLogicalOrReductionb;              //!< Opencl kernel that compute logical or reduction for CALbyte substates GPU side
	CALCLkernel kernelLogicalOrReductioni;              //!< Opencl kernel that compute logical or reduction for CALint substates GPU side
	CALCLkernel kernelLogicalOrReductionr;              //!< Opencl kernel that compute logical or reduction for CALreal substates GPU side

	CALCLkernel kernelBinaryOrReductionb;               //!< Opencl kernel that compute binary or reduction for CALbyte substates GPU side
	CALCLkernel kernelBinaryOrReductioni;               //!< Opencl kernel that compute binary or reduction for CALint substates GPU side
	CALCLkernel kernelBinaryOrReductionr;               //!< Opencl kernel that compute binary or reduction for CALreal substates GPU side

	CALCLkernel kernelLogicalXOrReductionb;             //!< Opencl kernel that compute logical xor reduction for CALbyte substates GPU side
	CALCLkernel kernelLogicalXOrReductioni;             //!< Opencl kernel that compute logical xor reduction for CALint substates GPU side
	CALCLkernel kernelLogicalXOrReductionr;             //!< Opencl kernel that compute logical xor reduction for CALreal substates GPU side

	CALCLkernel kernelBinaryXorReductionb;              //!< Opencl kernel that compute binary xor reduction for CALbyte substates GPU side
	CALCLkernel kernelBinaryXorReductioni;              //!< Opencl kernel that compute binary xor reduction for CALint substates GPU side
	CALCLkernel kernelBinaryXorReductionr;              //!< Opencl kernel that compute binary xor reduction for CALreal substates GPU side

	CALCLkernel kernelMinCopyb;                         //!< Opencl kernel that compute min copy buffer for CALbyte substates GPU side
	CALCLkernel kernelMinCopyi;                         //!< Opencl kernel that compute min copy buffer for CALint substates GPU side
	CALCLkernel kernelMinCopyr;                         //!< Opencl kernel that compute min copy buffer for CALreal substates GPU side

	CALCLkernel kernelSumCopyb;                         //!< Opencl kernel that compute max copy buffer for CALbyte substates GPU side
	CALCLkernel kernelSumCopyi;                         //!< Opencl kernel that compute max copy buffer for CALint substates GPU side
	CALCLkernel kernelSumCopyr;                         //!< Opencl kernel that compute max copy buffer for CALreal substates GPU side

	CALCLkernel kernelProdCopyb;                        //!< Opencl kernel that compute sum copy buffer for CALbyte substates GPU side
	CALCLkernel kernelProdCopyi;                        //!< Opencl kernel that compute sum copy buffer for CALint substates GPU side
	CALCLkernel kernelProdCopyr;                        //!< Opencl kernel that compute sum copy buffer for CALreal substates GPU side

	CALCLkernel kernelMaxCopyb;                         //!< Opencl kernel that compute prod copy buffer for CALbyte substates GPU side
	CALCLkernel kernelMaxCopyi;                         //!< Opencl kernel that compute prod copy buffer for CALint substates GPU side
	CALCLkernel kernelMaxCopyr;                         //!< Opencl kernel that compute prod copy buffer for CALreal substates GPU side

	CALCLkernel kernelLogicalAndCopyb;                  //!< Opencl kernel that compute logical and copy buffer for CALbyte substates GPU side
	CALCLkernel kernelLogicalAndCopyi;                  //!< Opencl kernel that compute logical and copy buffer for CALint substates GPU side
	CALCLkernel kernelLogicalAndCopyr;                  //!< Opencl kernel that compute logical and copy buffer for CALreal substates GPU side

	CALCLkernel kernelLogicalOrCopyb;                   //!< Opencl kernel that compute binary and copy buffer for CALbyte substates GPU side
	CALCLkernel kernelLogicalOrCopyi;                   //!< Opencl kernel that compute binary and copy buffer for CALint substates GPU side
	CALCLkernel kernelLogicalOrCopyr;                   //!< Opencl kernel that compute binary and copy buffer for CALreal substates GPU side

	CALCLkernel kernelLogicalXOrCopyb;                  //!< Opencl kernel that compute logical or copy buffer for CALbyte substates GPU side
	CALCLkernel kernelLogicalXOrCopyi;                  //!< Opencl kernel that compute logical or copy buffer for CALint substates GPU side
	CALCLkernel kernelLogicalXOrCopyr;                  //!< Opencl kernel that compute logical or copy buffer for CALreal substates GPU side

	CALCLkernel kernelBinaryAndCopyb;                   //!< Opencl kernel that compute binary or copy buffer for CALbyte substates GPU side
	CALCLkernel kernelBinaryAndCopyi;                   //!< Opencl kernel that compute binary or copy buffer for CALint substates GPU side
	CALCLkernel kernelBinaryAndCopyr;                   //!< Opencl kernel that compute binary or copy buffer for CALreal substates GPU side

	CALCLkernel kernelBinaryOrCopyb;                    //!< Opencl kernel that compute logical xor copy buffer for CALbyte substates GPU side
	CALCLkernel kernelBinaryOrCopyi;                    //!< Opencl kernel that compute logical xor copy buffer for CALint substates GPU side
	CALCLkernel kernelBinaryOrCopyr;                    //!< Opencl kernel that compute logical xor copy buffer for CALreal substates GPU side

	CALCLkernel kernelBinaryXOrCopyb;                   //!< Opencl kernel that compute binary xor copy buffer for CALbyte substates GPU side
	CALCLkernel kernelBinaryXOrCopyi;                   //!< Opencl kernel that compute binary xor copy buffer for CALint substates GPU side
	CALCLkernel kernelBinaryXOrCopyr;                   //!< Opencl kernel that compute binary xor copy buffer for CALreal substates GPU side

	CALCLmem bufferMinimab;                             //!< Opencl buffer that contains min results for CALbyte substates GPU side
	CALCLmem bufferMinimai;                             //!< Opencl kernel that contains min results for CALint substates GPU side
	CALCLmem bufferMinimar;                             //!< Opencl kernel that contains min results for CALreal substates GPU side

	CALCLmem bufferMaximab;                             //!< Opencl kernel that contains max results for CALbyte substates GPU side
	CALCLmem bufferMaximai;                             //!< Opencl kernel that contains max results for CALint substates GPU side
	CALCLmem bufferMaximar;                             //!< Opencl kernel that contains max results for CALreal substates GPU side

	CALCLmem bufferSumb;                                //!< Opencl kernel that contains sum results for CALbyte substates GPU side
	CALCLmem bufferSumi;                                //!< Opencl kernel that contains sum results for CALint substates GPU side
	CALCLmem bufferSumr;                                //!< Opencl kernel that contains sum results for CALreal substates GPU side

	CALCLmem bufferProdb;                               //!< Opencl kernel that contains prod results for CALbyte substates GPU side
	CALCLmem bufferProdi;                               //!< Opencl kernel that contains prod results for CALint substates GPU side
	CALCLmem bufferProdr;                               //!< Opencl kernel that contains prod results for CALreal substates GPU side

	CALCLmem bufferLogicalAndsb;                        //!< Opencl kernel that contains logical and results for CALbyte substates GPU side
	CALCLmem bufferLogicalAndsi;                        //!< Opencl kernel that contains logical and results for CALint substates GPU side
	CALCLmem bufferLogicalAndsr;                        //!< Opencl kernel that contains logical and results for CALreal substates GPU side

	CALCLmem bufferLogicalOrsb;                         //!< Opencl kernel that contains logical or results for CALbyte substates GPU side
	CALCLmem bufferLogicalOrsi;                         //!< Opencl kernel that contains logical or results for CALint substates GPU side
	CALCLmem bufferLogicalOrsr;                         //!< Opencl kernel that contains logical or results for CALreal substates GPU side

	CALCLmem bufferLogicalXOrsb;                        //!< Opencl kernel that contains logical xor results for CALbyte substates GPU side
	CALCLmem bufferLogicalXOrsi;                        //!< Opencl kernel that contains logical xor results for CALint substates GPU side
	CALCLmem bufferLogicalXOrsr;                        //!< Opencl kernel that contains logical xor results for CALreal substates GPU side

	CALCLmem bufferBinaryAndsb;                         //!< Opencl kernel that contains binary and results for CALbyte substates GPU side
	CALCLmem bufferBinaryAndsi;                         //!< Opencl kernel that contains binary and results for CALint substates GPU side
	CALCLmem bufferBinaryAndsr;                         //!< Opencl kernel that contains binary and results for CALreal substates GPU side

	CALCLmem bufferBinaryOrsb;                          //!< Opencl kernel that contains binary or results for CALbyte substates GPU side
	CALCLmem bufferBinaryOrsi;                          //!< Opencl kernel that contains binary or results for CALint substates GPU side
	CALCLmem bufferBinaryOrsr;                          //!< Opencl kernel that contains binary or results for CALreal substates GPU side

	CALCLmem bufferBinaryXOrsb;                         //!< Opencl kernel that contains binary xor results for CALbyte substates GPU side
	CALCLmem bufferBinaryXOrsi;                         //!< Opencl kernel that contains binary xor results for CALint substates GPU side
	CALCLmem bufferBinaryXOrsr;                         //!< Opencl kernel that contains binary xor results for CALreal substates GPU side

	int roundedDimensions;
	CALCLcontext context;
};

/*! \brief Allocate, initialize and return a pointer to a struct CALCLModel3D.
 *
 * Allocate, initialize and return a pointer to a struct CALCLModel3D. Opencl buffers are initialized using data from a CALModel3D instance.
 * Moreover, the function receive an Opencl program used to initialize library kernels.
 */
DllExport
struct CALCLModel3D * calclCADef3D(struct CALModel3D *model,		//!< Pointer to a CALModel3D
		CALCLcontext context,										//!< Opencl context
		CALCLprogram program,										//!< Opencl program containing library source and user defined source
		CALCLdevice device 											//!< Opencl device
		);

/*! \brief Main simulation cycle. It can become a loop if maxStep == CALCL_RUN_LOOP */
DllExport
void calclRun3D(struct CALCLModel3D* calclmodel3D,			//!< Pointer to a struct CALCLModel3D
		unsigned int initialStep,				//!< Initial simulation step
		unsigned maxStep							//!< Maximum number of CA steps. Simulation can become a loop if maxStep == CALCL_RUN_LOOP
		);

/*! \brief A single step of CA. It executes the transition function, the steering and check the stop condition */
DllExport
CALbyte calclSingleStep3D(struct CALCLModel3D* calclmodel3D, 		//!< Pointer to a struct CALCLModel3D
		size_t * dimSize,									//!< Array of size_t containing the number of threads for each used Opencl dimension (CALCL_NO_OPT 3 dimensions, CALCL_OPT_ACTIVE_CELL 1 dimension)
		int dimNum											//!< Number of Opencl dimensions (CALCL_NO_OPT 3 dimensions, CALCL_OPT_ACTIVE_CELL 1 dimension)
		);

/*! \brief Execute an Opencl kernel */
DllExport
void calclKernelCall3D(struct CALCLModel3D * calclmodel3D,		//!< Pointer to a struct CALCLModel3D
		CALCLkernel ker,								//!< Opencl kernel
		int numDim,										//!< Number of Opencl dimensions (CALCL_NO_OPT 3 dimensions, CALCL_OPT_ACTIVE_CELL 1 dimension)
		size_t * dimSize,								//!< Array of size_t containing the number of threads for each used Opencl dimension (CALCL_NO_OPT 3 dimensions, CALCL_OPT_ACTIVE_CELL 1 dimension)
		size_t * localDimSize							//!< Array of size_t containing the number of threads for each used Opencl local dimension
		);

/*! \brief Execute stream compaction kernels to compact and order CA active cells */
DllExport
void calclComputeStreamCompaction3D(struct CALCLModel3D * calclmodel3D		//!< Pointer to a struct CALCLModel3D
		);

/*! \brief Add arguments to the given Opencl kernel defined by the user
 *
 * Add arguments to the given Opencl kernel defined by the user. Kernel arguments are added
 * after the default argument provided by the library.
 *
 *  */
DllExport
void calclSetKernelArgs3D(CALCLkernel * kernel,			//!< Opencl kernel
		CALCLmem * args,									//!< Array of Opencl buffers that represents kernel additional arguments
		cl_uint numArgs										//!< Number of Opencl kernel additional arguments
		);

/*! \brief Set the stop condition Opencl kernel
 *
 * Set the stop condition Opencl kernel. If defined, the stop condition kernel is executed
 * each time the function calclSingleStep3D is called. Set the kernel argument stop to CAL_TRUE
 * to stop the simulation.
 *
 *  */
DllExport
void calclAddStopConditionFunc3D(struct CALCLModel3D * calclmodel3D,		//!< Pointer to a struct CALCLModel3D
		CALCLkernel * kernel										//!< Opencl kernel
		);

/*! \brief Set the Opencl kernel used to initialize substates
 *
 * Set the Opencl kernel used to initialize substates. If defined, the kernel is executed
 * at the beginning of the simulation
 *
 *  */
DllExport
void calclAddInitFunc3D(struct CALCLModel3D * calclmodel3D, 		//!< Pointer to a struct CALCLModel3D
		CALCLkernel * kernel										//!< Opencl kernel
		);

/*! \brief Set the steering Opencl kernel
 *
 * Set the steering Opencl kernel. If defined, the stop condition kernel is executed
 * each time the function calclSingleStep3D is called.
 *
 *  */
DllExport
void calclAddSteeringFunc3D(struct CALCLModel3D * calclmodel3D,			//!< Pointer to a struct CALCLModel3D
		CALCLkernel * kernel										//!< Opencl kernel
		);

/*! \brief Set the function used to access substate on the GPU every callbackSteps steps.
 *
 *	Set the function used to access substate on the GPU every callbackSteps steps. This function
 *	could decrease the performance because of the transfer of data between host and GPU.
 *
 *  */
DllExport
void calclBackToHostFunc3D(struct CALCLModel3D* calclmodel3D,		//!< Pointer to a struct CALCLModel3D
		void (*cl_update_substates)(struct CALModel3D*), 				//!< Callback function executed each callbackSteps steps
		int callbackSteps												//!< Define how many steps must be executed before call the callback functions
		);


/*! \brief Add an Opencl kernel to the elementary processes kernels.
 *
 *	Add an Opencl kernel to the elementary processes kernels. Each elementary process kernel
 *	is executed each time the function calclSingleStep3D is called.
 *
 *  */
DllExport
void calclAddElementaryProcess3D(struct CALCLModel3D* calclmodel3D, 		//!< Pointer to a struct CALCLModel3D
		CALCLkernel * kernel											//!< Pointer to Opencl kernel
		);

/*! \brief Deallcate a struct CALCLModel3D instance */
DllExport
void calclFinalize3D(struct CALCLModel3D * calclmodel3D			//!< Pointer to a struct CALCLModel3D
		);

/*! \brief Allocate, initialize and return an Opencl program
 *
 *	Allocate, initialize and return an Opencl program. The program returned
 *	is compiled using library source files and user defined source files.
 *
 *  */
DllExport
CALCLprogram calclLoadProgram3D(CALCLcontext context, 		//!< Opencl context
		CALCLdevice device, 									//!< Opencl device
		char* path_user_kernel,									//!< Kernel source files path
		char* path_user_include								//!< Kernel include files path
		);

/*! \brief Set a kernel argument   */
DllExport
int calclSetKernelArg3D(CALCLkernel kernel,			//!< Opencl kernel
			cl_uint arg_index,			//!< Index argument
			size_t arg_size,			//!< Size argument
			const void *arg_value                   //!< Value argument
			);

/*! \brief Copy all the substates device memory to host memory   */
DllExport
void calclGetSubstatesDeviceToHost3D(struct CALCLModel3D* calclmodel3D //!< Pointer to a struct CALCLModel3D
			);

/*! \brief Set reduction arguments to Opencl kernel   */
DllExport
void calclSetReductionParameters3D(struct CALCLModel3D* calclmodel3D,		//!< Pointer to a struct CALCLModel3D
		CALCLkernel * kernel		//!< Pointer to Opencl kernel
		);



#endif /* CALCL_H_ */
