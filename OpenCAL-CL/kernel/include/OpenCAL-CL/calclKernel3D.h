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

#ifndef CALCLKERNEL3D_H_
#define CALCLKERNEL3D_H_

/*! \brief Terminates threads that exceed matrix sizes */
#define calclThreadCheck3D() if(get_global_id(0)>=calclGetRows() || get_global_id(1)>=calclGetColumns()|| get_global_id(2)>=calclGetSlices()) return

/*! \brief Terminates threads that exceed active cells number */
#define calclActiveThreadCheck3D() if(get_global_id(0)>=calclGetActiveCellsNum()) return

/*! \brief Defines model parameters (This define is used in function declaration) */
#define __CALCL_MODEL_3D __global CALint * CALCLrows,__global CALint * CALCLcolumns,__global CALint * CALCLslices,__global CALint * CALCLbyteSubstatesNum,__global CALint * CALCLintSubstatesNum,	__global CALint * CALCLrealSubstatesNum,__global CALbyte * CALCLcurrentByteSubstates,__global CALint * CALCLcurrentIntSubstates,__global CALreal * CALCLcurrentRealSubstates,__global CALbyte * CALCLnextByteSubstates,__global CALint * CALCLnextIntSubstates,__global CALreal * CALCLnextRealSubstates,__global struct CALCell3D * CALCLactiveCells,__global CALint * CALCLactiveCellsNum,__global CALbyte * CALCLactiveCellsFlags,__global struct CALCell3D * CALCLneighborhood,__global enum CALNeighborhood3D * CALCLneighborhoodID,__global CALint * CALCLneighborhoodSize,__global enum CALSpaceBoundaryCondition * CALCLboundaryCondition, __global CALbyte * CALCLstop,__global CALbyte * CALCLdiff,CALint CALCLchunk, __global CALreal * minimab, __global CALreal * minimai, __global CALreal * minimar, __global CALreal * maximab,  __global CALreal * maximai,  __global CALreal * maximar,  __global CALreal * sumb, __global CALreal * sumi, __global CALreal * sumr, __global CALint * logicalAndb,__global CALint * logicalAndi,__global CALint * logicalAndr, __global CALint * logicalOrb,__global CALint * logicalOri,__global CALint * logicalOrr, __global CALint * logicalXOrb,__global CALint * logicalXOri, __global CALint * logicalXOrr, __global CALint * binaryAndb,__global CALint * binaryAndi,__global CALint * binaryAndr ,__global CALint * binaryOrb, __global CALint * binaryOri, __global CALint * binaryOrr, __global CALint * binaryXOrb,  __global CALint * binaryXOri,  __global CALint * binaryXOrr,  __global CALreal * prodb , __global CALreal * prodi , __global CALreal * prodr

/*! \brief Defines model parameters (This define is used in function calls) */
#define MODEL_3D CALCLrows,CALCLcolumns,CALCLslices,CALCLbyteSubstatesNum,CALCLintSubstatesNum,CALCLrealSubstatesNum,CALCLcurrentByteSubstates,CALCLcurrentIntSubstates,CALCLcurrentRealSubstates,CALCLnextByteSubstates,CALCLnextIntSubstates,CALCLnextRealSubstates,CALCLactiveCells,CALCLactiveCellsNum,CALCLactiveCellsFlags,CALCLneighborhood,CALCLneighborhoodID,CALCLneighborhoodSize,CALCLboundaryCondition, CALCLstop,CALCLdiff,CALCLchunk, minimab, minimai, minimar, maximab, maximai, maximar, sumb, sumi, sumr, logicalAndb, logicalAndi, logicalAndr, logicalOrb, logicalOri, logicalOrb, logicalXOrb,logicalXOri, logicalXOrr, binaryAndb, binaryAndi, binaryAndr, binaryOrb, binaryOri, binaryOrr, binaryXOrb, binaryXOri, binaryXOrr, prodb, prodi, prodr

#define calclGetRows() *CALCLrows
#define calclGetColumns() *CALCLcolumns
#define calclGetSlices() *CALCLslices
#define calclGetByteSubstatesNum() *CALCLbyteSubstatesNum
#define calclGetIntSubstatesNum() *CALCLintSubstatesNum
#define calclGetRealSubstatesNum() *CALCLrealSubstatesNum
#define calclGetCurrentByteSubstates() CALCLcurrentByteSubstates
#define calclGetCurrentIntSubstates() CALCLcurrentIntSubstates
#define calclGetCurrentRealSubstates() CALCLcurrentRealSubstates
#define calclGetNextByteSubstates() CALCLnextByteSubstates
#define calclGetNextIntSubstates() CALCLnextIntSubstates
#define calclGetNextRealSubstates() CALCLnextRealSubstates
#define calclGetActiveCells() CALCLactiveCells
#define calclGetActiveCellsNum() *CALCLactiveCellsNum
#define calclGetActiveCellsFlags() CALCLactiveCellsFlags
#define calclGetNeighborhood() CALCLneighborhood
#define calclGetNeighborhoodId() *CALCLneighborhoodID
#define calclGetNeighborhoodSize() *CALCLneighborhoodSize
#define calclGetBoundaryCondition() *CALCLboundaryCondition
#define calclRunStop() *CALCLstop = CAL_TRUE

/*! \brief Gets the thread id in the first global dimension */
#define calclGlobalRow() get_global_id(0)

/*! \brief Gets the thread id in the second global dimension */
#define calclGlobalColumn() get_global_id(1)

/*! \brief Gets the thread id in the third global dimension */
#define calclGlobalSlice() get_global_id(2)

/*! \brief Gets the thread id in the first local dimension */
#define calclLocalRow() get_local_id(0)

/*! \brief Gets the thread id in the second local dimension */
#define calclLocalColumn() get_local_id(1)

/*! \brief Gets the thread id in the first local dimension */
#define calclLocalSlice() get_local_id(2)

/*! \brief Gets the active cell row coordinate relative to the given thread id */
#define calclActiveCellRow(threadID) get_active_cells()[threadID].i

/*! \brief Gets the active cell column coordinate relative to the given thread id */
#define calclActiveCellColumn(threadID) get_active_cells()[threadID].j

/*! \brief Gets the active cell slice coordinate relative to the given thread id */
#define getActiveCellSlice(threadID) get_active_cells()[threadID].k



#endif /* CALCLKERNEL3D_H_ */
