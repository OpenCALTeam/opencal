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

#ifndef cal2D_h
#define cal2D_h

#include "calCommon.cuh"
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cal2DToolkit.cuh"
#include "cal2DRun.cuh"
#include "chag\pp\compact.cuh"

namespace pp = chag::pp;

/*****************************************************************************
DEFINITIONS OF NEW DATA TYPES

*****************************************************************************/

/*! \brief Enumeration of 2D neighbourhood.
Enumeration that identifies the cellular automaton's 2D neighbourhood.
*/
enum CALNeighborhood2D {
	CAL_CUSTOM_NEIGHBORHOOD_2D,			//!< Enumerator used for the definition of a custom 2D neighbourhood; this is built by calling the function calAddNeighbor2D.
	CAL_VON_NEUMANN_NEIGHBORHOOD_2D,	//!< Enumerator used for specifying the 2D von Neumann neighbourhood; no calls to calAddNeighbor2D are needed.
	CAL_MOORE_NEIGHBORHOOD_2D,			//!< Enumerator used for specifying the 2D Moore neighbourhood; no calls to calAddNeighbor2D are needed.
	CAL_HEXAGONAL_NEIGHBORHOOD_2D,		//!< Enumerator used for specifying the 2D Moore Hexagonal neighbourhood; no calls to calAddNeighbor2D are needed.
	CAL_HEXAGONAL_NEIGHBORHOOD_ALT_2D	//!< Enumerator used for specifying the alternative 90° rotated 2D Moore Hexagonal neighbourhood; no calls to calAddNeighbor2D are needed.
};

#define CAL_HEXAGONAL_SHIFT 7			//<! Shif used for accessing to the correct neighbor in case hexagonal heighbourhood and odd column cell


/*! \brief Structure defining the 2D cellular automaton for CUDA version.
*/
struct CudaCALModel2D {
	int rows;							//!< Number of rows of the 2D cellular space.
	int columns;						//!< Number of columns of the 2D cellular space.
	enum CALSpaceBoundaryCondition T;	//!< Type of cellular space: toroidal or non-toroidal.

	enum CALOptimization OPTIMIZATION;	//!< Type of optimization used. It can be CAL_NO_OPT or CAL_OPT_ACTIVE_CELLS.

	//struct CALActiveCells2D A;		
	CALbyte* activecell_flags;			//!< Array of flags having the substates' dimension: flag is CAL_TRUE if the corresponding cell is active, CAL_FALSE otherwise.
	int activecell_size_next;			//!< Number of CAL_TRUE flags.
	int activecell_size_current;		//!< Number of active cells in the current step.	
	unsigned int* activecell_index;			//!< Set of index where there are active cells.
	unsigned int * array_of_index_result;	//!< Support set for stream compaction. This array is populated by the index of active cells
	
	CALbyte stop; //!< Variable to check if the simulation have to be stopped. CAL_TRUE if the simulation have to be stopped, CAL_FALSE otherwise.

	//struct CALCell2D* X;				
	int *i;			//!< Array of coordinate "i" defining the cellular automaton neighbourhood relation.
	int *j;			//!< Array of coordinate "j" defining the cellular automaton neighbourhood relation.

	int sizeof_X;						//!< Number of cells belonging to the neighbourhood. Note that predefined neighbourhoods include the central cell.
	enum CALNeighborhood2D X_id;		//!< Neighbourhood relation's id.

	CALbyte* pQb_array_current;			//!< Current linearised matrix of the BYTE substates, used for reading purposes.
	CALbyte* pQb_array_next;			//!< Next linearised matrix of the BYTE substates, used for writing purposes.
	CALint* pQi_array_current;			//!< Current linearised matrix of the INTEGER substates, used for reading purposes.
	CALint* pQi_array_next;				//!< Next linearised matrix of the INTEGER substates, used for writing purposes.
	CALreal* pQr_array_current;			//!< Current linearised matrix of the REAL (FLOATING POINT) substateS, used for reading purposes.
	CALreal* pQr_array_next;			//!< Next linearised matrix of the REAL (FLOATING POINT) substateS, used for writing purposes.

	int sizeof_pQb_array;				//!< Number of substates of type byte.
	int sizeof_pQi_array;				//!< Number of substates of type int.
	int sizeof_pQr_array;				//!< Number of substates of type real (floating point).

	void (**elementary_processes)(struct CudaCALModel2D* ca2D); //!< Array of function pointers to the transition function's elementary processes callback functions. Note that a substates' update must be performed after each elementary process has been applied to each cell of the cellular space (see calGlobalTransitionFunction2D).
	int num_of_elementary_processes; //!< Number of function pointers to the transition functions's elementary processes callbacks.
};



/*! \brief Fake function pointer type.
*/
typedef void (* CALCallbackFunc2D)(struct CALModel2D* ca2D, int i, int j);
typedef void (* CALCudaCallbackFunc2D)(struct CudaCALModel2D* ca2D);



/******************************************************************************
DEFINITIONS OF FUNCTIONS PROTOTYPES

*******************************************************************************/

/*! \brief Creates an object of type CALModel2D for CUDA version, sets its records and returns it as a pointer; it defines the cellular automaton structure.
*/
struct CudaCALModel2D* calCudaCADef2D(int rows, //!< Number of rows of the 2D cellular space.
	int columns, //!< Number of columns of the 2D cellular space.
	enum CALNeighborhood2D CAL_NEIGHBORHOOD_2D, //!< Enumerator that identifies the type of neighbourhood relation to be used.
	enum CALSpaceBoundaryCondition CAL_TOROIDALITY, //!< Enumerator that identifies the type of cellular space: toroidal or non-toroidal.
	enum CALOptimization CAL_OPTIMIZATION //!< Enumerator used for specifying the active cells optimization or no optimization.
	);

/*! \brief Sets the cell (offset) of the matrix flags to CAL_TRUE and increments the 
couter sizeof_active_flags.
*/
__device__
	void calCudaAddActiveCell2D(struct CudaCALModel2D* ca2D,	//!< Pointer to the cellular automaton structure.
	int offset //!< Offset of threads
	);

/*! \brief Sets the n-th neighbor of the cell (offset) of the matrix flags to 
CAL_TRUE and increments the couter sizeof_active_flags.
*/
__device__
	void calCudaAddActiveCellX2D(struct CudaCALModel2D* ca2D,	//!< Pointer to the cellular automaton structure.
	int offset, //!< Offset of threads
	int n	//!< Index of the n-th neighbor to be added.
	);

/*! \brief Sets the cell (offset) of the matrix flags to CAL_FALSE and decrements the 
couter sizeof_active_flags.
*/
__device__
	void calCudaRemoveActiveCell2D(struct CudaCALModel2D* ca2D,	//!< Pointer to the cellular automaton structure.
	int offset //!< Offset of threads
	);

__global__ void generateSetOfIndex(CudaCALModel2D *device_ca2D);
void calCudaApplyStreamCompaction(struct CudaCALRun2D* simulation, dim3 grid, dim3 block);

/*! \brief Puts the cells marked as actives in activecells_flags into the arrays of active cells 
and sets its dimension, activecells_size, to activecells_size_of_actives, i.e. the actual 
number of active cells.
*/
void calCudaUpdateActiveCells2D(struct CudaCALRun2D* simulation	//!< Pointer to the cellular automaton structure.
								);



/*! \brief 
Adds a neighbour to i and j.
i and j are two vectors that contain value of neighborhoods.
The value of CudaCALModel2D::sizeof_X increase after this operation.
*/
void  calCudaAddNeighbor2D(struct CudaCALModel2D* ca2D, //!< Pointer to the cellular automaton structure.
						   int i,	//!< Relative row coordinate with respect to the central cell (the north neighbour has i = -1, the south i = +1, etc.).
						   int j	//!< Relative column coordinate with respect to the central cell (the east neighbour has j = -1, the west i = +1, etc.).
						   );

/*! \brief Creates and adds a new byte substate to CudaCALModel2D::pQb_array_current and CudaCALModel2D::pQb_array_next
and return an object that point to last substate. 
*/
cudaError_t calCudaAddSubstate2Db(struct CudaCALModel2D* ca2D,	//!< Pointer to the cellular automaton structure.
								  CALint NUMBER_OF_SUBSTATE	//!< Number of substate (byte) to alloc.
								  );

/*! \brief Creates and adds a new int substate to CudaCALModel2D::pQb_array_current and CudaCALModel2D::pQb_array_next
and return an object that point to last substate. 
*/
cudaError_t calCudaAddSubstate2Di(struct CudaCALModel2D* ca2D,	//!< Pointer to the cellular automaton structure.
								  CALint NUMBER_OF_SUBSTATE	//!< Number of substate (integer) to alloc.
								  );

/*! \brief Creates and adds a new real substate to CudaCALModel2D::pQb_array_current and CudaCALModel2D::pQb_array_next
and return an object that point to last substate. 
*/
cudaError_t calCudaAddSubstate2Dr(struct CudaCALModel2D* ca2D,	//!< Pointer to the cellular automaton structure.
								  CALint NUMBER_OF_SUBSTATE	//!< Number of substate (real - floating point) to alloc.
								  );


/*! \brief Creates a new single-layer byte substate and returns a pointer to it.
Note that sinlgle-layer substates are not added to CALModel2D::pQ*_array because
they do not nedd to be updated.
*/
struct CALSubstate2Db* calAddSingleLayerSubstate2Db(struct CALModel2D* ca2D	//!< Pointer to the cellular automaton structure.
	);

/*! \brief Creates a new single-layer int substate and returns a pointer to it.
Note that sinlgle-layer substates are not added to CALModel2D::pQ*_array because
they do not nedd to be updated.
*/
struct CALSubstate2Di* calAddSingleLayerSubstate2Di(struct CALModel2D* ca2D	//!< Pointer to the cellular automaton structure.
	);

/*! \brief Creates a new single-layer real (floating point) substate returns a pointer to it.
Note that sinlgle-layer substates are not added to CALModel2D::pQ*_array because
they do not nedd to be updated.
*/
struct CALSubstate2Dr* calAddSingleLayerSubstate2Dr(struct CALModel2D* ca2D	//!< Pointer to the cellular automaton structure.
	);

/*! \brief Adds a transition function's elementary process to the CALModel2D::elementary_processes array of callbacks pointers.
Call this function even if you're using CUDA version
Note that the function calCudaGlobalTransitionFunction2D calls a substates' update after each elementary process.
*/
CALCudaCallbackFunc2D* calCudaAddElementaryProcess2D(struct CudaCALModel2D* ca2D,	//!< Pointer to the cellular automaton structure.
													 void (* elementary_process)(struct CudaCALModel2D* ca2D) //!< Pointer to a transition function's elementary process.
													 );

/*! \brief Apply an elementary process to all the cellular space.
*/
void calCudaApplyElementaryProcess2D(struct CudaCALRun2D* simulation,	//!< Pointer to the cellular automaton structure.
									 void (* elementary_process)(struct CudaCALModel2D* ca2D), //!< Pointer to a transition function's elementary process.
									 dim3 grid, dim3 block
									 );

/*! \brief The cellular automaton global transition function.
It applies the transition function to each cell of the cellular space.
After each elementary process, a global substates update is performed.
Use this one even if you're using the CUDA version
*/
void calCudaGlobalTransitionFunction2D(struct CudaCALRun2D* simulation,	//!< Pointer to the cellular automaton structure.
									   dim3 grid, //!< grid of blocks for GPGPU instruction.
									   dim3 block //!< block of threads for GPGPU instruction.
									   );

/*! \brief Updates all the substates registered in CudaCALModel2D (pQb_array, pQi_array and pQr_array). 
It is called by the global transition function.
Updates the substates allocated in device.
*/
void calCudaUpdate2D(struct CudaCALRun2D* simulation);

/*! \brief Inits the value of a byte substate in the cell (offset) to value; it updates both the current and next matrices at the position (offset).  
it's for CUDA version
*/
__device__
	void calCudaInit2Db(struct CudaCALModel2D* ca2D,	//!< Pointer to the cellular automaton structure.
	int offset,						//!< offset of the cell to be initialized.
	CALbyte value,				//!< initializing value for the substate at the cell (offset).
	CALint substate_index		//!< Index of substate that you want to initialize.
	);

/*! \brief Inits the value of a integer substate in the cell (offset) to value; it updates both the current and next matrices at the position (offset).  
it's for CUDA version
*/
__device__
	void calCudaInit2Di(struct CudaCALModel2D* ca2D,	//!< Pointer to the cellular automaton structure.
	int offset,						//!< offset of the cell to be initialized.
	CALint value,				//!< initializing value for the substate at the cell (offset).
	CALint substate_index		//!< Index of substate that you want to initialize.
	);

/*! \brief Inits the value of a real (floating point) substate in the cell (offset) to value; it updates both the current and next matrices at the position (offset).  
it's for CUDA version
*/
__device__
	void calCudaInit2Dr(struct CudaCALModel2D* ca2D,	//!< Pointer to the cellular automaton structure.
	int offset,						//!< offset of the cell to be initialized.
	CALreal value,				//!< initializing value for the substate at the cell (offset).
	CALint substate_index		//!< Index of substate that you want to initialize.
	);

/*! \brief Returns the cell (offset) value of a byte substate.  
*/
__device__
	CALbyte calCudaGet2Db(struct CudaCALModel2D* ca2D,	//!< Pointer to the cellular automaton structure.
	int offset, //!< Thread id
	CALint substate_index		//!< Index of substate.
	);

/*! \brief Returns the cell (offset) value of a integer substate.  
*/
__device__
	CALint calCudaGet2Di(struct CudaCALModel2D* ca2D,	//!< Pointer to the cellular automaton structure.
	int offset, //!< Thread id
	CALint substate_index		//!< Index of substate.
	);

/*! \brief Returns the cell (offset) value of a real (floating point) substate.  
*/
__device__
	CALreal calCudaGet2Dr(struct CudaCALModel2D* ca2D,	//!< Pointer to the cellular automaton structure.
	int offset, //!< Thread id
	CALint substate_index		//!< Index of substate.
	);


/*! \brief Return indexes in case of flat cellular spaces.
return -1 if index is not in matrix, number else.	
*/
__device__ CALint calGetLinearIndex(int offset, CALint columns, CALint rows, CALint in, CALint jn, CALint substate_index);

/*! \brief Return indexes in case of flat cellular spaces.
return the index in the matrix.	
*/
__device__ CALint calGetToroidalLinearIndex(int offset, CALint columns, CALint rows, CALint in, CALint jn, CALint substate_index);

/*! \brief Returns the n-th neighbor of the cell (offset) value of a byte substate.
*/
__device__ CALbyte calCudaGetX2Db(struct CudaCALModel2D* ca2D, int offset, int n, CALint substate_index);

/*! \brief Returns the n-th neighbor of the cell (offset) value of a integer substate.
*/
__device__ CALint calCudaGetX2Di(struct CudaCALModel2D* ca2D, int offset, int n, CALint substate_index);

/*! \brief Returns the n-th neighbor of the cell (offset) value of a real (floating point) substate.
*/
__device__ CALreal calCudaGetX2Dr(struct CudaCALModel2D* ca2D, int offset, int n, CALint substate_index);

/*! \brief Sets the cell (offset) value of a byte substate for Cuda version.  
*/
__device__ 
	void calCudaSet2Db(struct CudaCALModel2D* ca2D,		//!< Pointer to the cellular automaton structure.
	int index,					//!< Current thread of index.
	CALbyte value,				//!< initializing value.
	CALint substate_index		//!< Index of substate.
	);

/*! \brief Sets the cell (offset) value of a integer substate for Cuda version.  
*/
__device__ 
	void calCudaSet2Di(struct CudaCALModel2D* ca2D,		//!< Pointer to the cellular automaton structure.
	int index,					//!< Current thread of index.
	CALint value,				//!< initializing value.
	CALint substate_index		//!< Index of substate.
	);

/*! \brief Sets the cell (offset) value of a real (floating point) substate for Cuda version.  
*/
__device__ 
	void calCudaSet2Dr(struct CudaCALModel2D* ca2D,		//!< Pointer to the cellular automaton structure.
	int index,					//!< Current thread of index.
	CALreal value,				//!< initializing value.
	CALint substate_index		//!< Index of substate.
	);

/*! \brief Sets the value of the cell (offset) of a byte substate of the CURRENT matrix.
This operation is unsafe since it writes a value directly to the current matrix.
*/
__device__
	void calCudaSetCurrent2Db(struct CudaCALModel2D* ca2D,	//!< Pointer to the cellular automaton structure.
	int index,					//!< Column coordinate of the central cell.
	CALbyte value,			//!< initializing value.
	CALint substate_index		//!< Index of substate.
	);

/*! \brief Sets the value of the cell (offset) of a integer substate of the CURRENT matrix.
This operation is unsafe since it writes a value directly to the current matrix.
*/
__device__
	void calCudaSetCurrent2Di(struct CudaCALModel2D* ca2D,	//!< Pointer to the cellular automaton structure.
	int index,					//!< Column coordinate of the central cell.
	CALint value,			//!< initializing value.
	CALint substate_index		//!< Index of substate.
	);

/*! \brief Sets the value of the cell (offset) of a real (floating point) substate of the CURRENT matrix.
This operation is unsafe since it writes a value directly to the current matrix.
*/
__device__
	void calCudaSetCurrent2Dr(struct CudaCALModel2D* ca2D,	//!< Pointer to the cellular automaton structure.
	int index,					//!< Column coordinate of the central cell.
	CALreal value,			//!< initializing value.
	CALint substate_index		//!< Index of substate.
	);

/*! \brief Finalization function: it releases the memory allocated.
this is a finalize function for CUDA version, so use even if you want use CUDA version.
*/
void calCudaFinalize2D(struct CudaCALModel2D* ca2D,	//!< Pointer to the host cellular automaton structure.
struct CudaCALModel2D* device_ca2D	//!< Pointer to the device cellular automaton structure.
	);

__device__
void calCudaStop(struct CudaCALModel2D* ca2D);
__device__
void calCudaSetStop(struct CudaCALModel2D* ca2D, CALbyte flag);
#endif
