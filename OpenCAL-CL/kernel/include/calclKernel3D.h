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

#ifndef CALCLKERNEL3D_H_
#define CALCLKERNEL3D_H_

/*! \brief Terminates threads that exceed matrix sizes */
#define initThreads3D() if(get_global_id(0)>=get_rows() || get_global_id(1)>=get_columns()|| get_global_id(2)>=get_slices()) return

/*! \brief Terminates threads that exceed active cells number */
#define initActiveThreads3D() if(get_global_id(0)>=get_active_cellsNum()) return

/*! \brief Defines model parameters (This define is used in function declaration) */
#define __CALCL_MODEL_3D __global CALint * CALCLrows,__global CALint * CALCLcolumns,__global CALint * CALCLslices,__global CALint * CALCLbyteSubstatesNum,__global CALint * CALCLintSubstatesNum,	__global CALint * CALCLrealSubstatesNum,__global CALbyte * CALCLcurrentByteSubstates,__global CALint * CALCLcurrentIntSubstates,__global CALreal * CALCLcurrentRealSubstates,__global CALbyte * CALCLnextByteSubstates,__global CALint * CALCLnextIntSubstates,__global CALreal * CALCLnextRealSubstates,__global struct CALCell3D * CALCLactiveCells,__global CALint * CALCLactiveCellsNum,__global CALbyte * CALCLactiveCellsFlags,__global struct CALCell3D * CALCLneighborhood,__global enum CALNeighborhood3D * CALCLneighborhoodID,__global CALint * CALCLneighborhoodSize,__global enum CALSpaceBoundaryCondition * CALCLboundaryCondition, __global CALbyte * CALCLstop,__global CALbyte * CALCLdiff,CALint CALCLchunk

/*! \brief Defines model parameters (This define is used in function calls) */
#define MODEL_3D CALCLrows,CALCLcolumns,CALCLslices,CALCLbyteSubstatesNum,CALCLintSubstatesNum,CALCLrealSubstatesNum,CALCLcurrentByteSubstates,CALCLcurrentIntSubstates,CALCLcurrentRealSubstates,CALCLnextByteSubstates,CALCLnextIntSubstates,CALCLnextRealSubstates,CALCLactiveCells,CALCLactiveCellsNum,CALCLactiveCellsFlags,CALCLneighborhood,CALCLneighborhoodID,CALCLneighborhoodSize,CALCLboundaryCondition, CALCLstop,CALCLdiff,CALCLchunk

#define get_rows() *CALCLrows
#define get_columns() *CALCLcolumns
#define get_slices() *CALCLslices
#define get_byte_substates_num() *CALCLbyteSubstatesNum
#define get_int_substates_num() *CALCLintSubstatesNum
#define get_real_substates_num() *CALCLrealSubstatesNum
#define get_current_byte_substates() CALCLcurrentByteSubstates
#define get_current_int_substates() CALCLcurrentIntSubstates
#define get_current_real_substates() CALCLcurrentRealSubstates
#define get_next_byte_substates() CALCLnextByteSubstates
#define get_next_int_substates() CALCLnextIntSubstates
#define get_next_real_substates() CALCLnextRealSubstates
#define get_active_cells() CALCLactiveCells
#define get_active_cellsNum() *CALCLactiveCellsNum
#define get_active_cells_flags() CALCLactiveCellsFlags
#define get_neighborhood() CALCLneighborhood
#define get_neighborhood_id() *CALCLneighborhoodID
#define get_neighborhoods_size() *CALCLneighborhoodSize
#define get_boundary_condition() *CALCLboundaryCondition
#define stopExecution() *CALCLstop = CAL_TRUE

/*! \brief Gets the thread id in the first global dimension */
#define getRow() get_global_id(0)

/*! \brief Gets the thread id in the second global dimension */
#define getCol() get_global_id(1)

/*! \brief Gets the thread id in the third global dimension */
#define getSlice() get_global_id(2)

/*! \brief Gets the thread id in the first local dimension */
#define getLocalRow() get_local_id(0)

/*! \brief Gets the thread id in the second local dimension */
#define getLocalCol() get_local_id(1)

/*! \brief Gets the thread id in the first local dimension */
#define getLocalSlice() get_local_id(2)

/*! \brief Gets the active cell row coordinate relative to the given thread id */
#define getActiveCellRow(threadID) get_active_cells()[threadID].i

/*! \brief Gets the active cell column coordinate relative to the given thread id */
#define getActiveCellCol(threadID) get_active_cells()[threadID].j

/*! \brief Gets the active cell slice coordinate relative to the given thread id */
#define getActiveCellSlice(threadID) get_active_cells()[threadID].k



#endif /* CALCLKERNEL3D_H_ */





