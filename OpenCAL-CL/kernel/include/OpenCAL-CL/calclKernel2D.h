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

#ifndef CALCLKERNEL2D_H_
#define CALCLKERNEL2D_H_

/*! \brief Terminates threads that exceed matrix sizes */
#define calclThreadCheck2D() if(get_global_id(0)>=calclGetRows() || get_global_id(1)>=calclGetColumns()) return

/*! \brief Terminates threads that exceed active cells number */
#define calclActiveThreadCheck2D() if(get_global_id(0)>=calclGetActiveCellsNum()) return

/*! \brief Defines model parameters (This define is used in function declaration) */
#define __CALCL_MODEL_2D  __global CALint * CALCLrows,__global CALint * CALCLcolumns,__global CALint * CALCLbyteSubstatesNum,__global CALint * CALCLintSubstatesNum,	__global CALint * CALCLrealSubstatesNum,__global CALbyte * CALCLcurrentByteSubstates,__global CALint * CALCLcurrentIntSubstates,__global CALreal * CALCLcurrentRealSubstates,__global CALbyte * CALCLnextByteSubstates,__global CALint * CALCLnextIntSubstates,__global CALreal * CALCLnextRealSubstates,__global struct CALCell2D * CALCLactiveCells,__global CALint * CALCLactiveCellsNum,__global CALbyte * CALCLactiveCellsFlags,__global struct CALCell2D * CALCLneighborhood,__global enum CALNeighborhood2D * CALCLneighborhoodID,__global CALint * CALCLneighborhoodSize,__global enum CALSpaceBoundaryCondition * CALCLboundaryCondition,__global CALbyte * CALCLstop,__global CALbyte * CALCLdiff,CALint CALCLchunk, __global CALreal * minimab, __global CALreal * minimai, __global CALreal * minimar, __global CALreal * maximab,  __global CALreal * maximai,  __global CALreal * maximar,  __global CALreal * sumb, __global CALreal * sumi, __global CALreal * sumr, __global CALreal * logicalAndb,__global CALreal * logicalAndi,__global CALreal * logicalAndr, __global CALreal * logicalOrb,__global CALreal * logicalOri,__global CALreal * logicalOrr, __global CALreal * logicalXOrb,__global CALreal * logicalXOri, __global CALreal * logicalXOrr, __global CALreal * binaryAndb,__global CALreal * binaryAndi,__global CALreal * binaryAndr ,__global CALreal * binaryOrb, __global CALreal * binaryOri, __global CALreal * binaryOrr, __global CALreal * binaryXOrb,  __global CALreal * binaryXOri,  __global CALreal * binaryXOrr

/*! \brief Defines model parameters (This define is used in function calls) */
#define MODEL_2D CALCLrows,CALCLcolumns,CALCLbyteSubstatesNum,CALCLintSubstatesNum,CALCLrealSubstatesNum,CALCLcurrentByteSubstates,CALCLcurrentIntSubstates,CALCLcurrentRealSubstates,CALCLnextByteSubstates,CALCLnextIntSubstates,CALCLnextRealSubstates,CALCLactiveCells,CALCLactiveCellsNum,CALCLactiveCellsFlags,CALCLneighborhood,CALCLneighborhoodID,CALCLneighborhoodSize,CALCLboundaryCondition, CALCLstop, CALCLdiff, CALCLchunk, minimab, minimai, minimar, maximab, maximai, maximar, sumb, sumi, sumr, logicalAndb, logicalAndi, logicalAndr, logicalOrb, logicalOri, logicalOrb, logicalXOrb,logicalXOri, logicalXOrr, binaryAndb, binaryAndi, binaryAndr, binaryOrb, binaryOri, binaryOrr, binaryXOrb, binaryXOri, binaryXOrr

#define calclGetRows() *CALCLrows
#define calclGetColumns() *CALCLcolumns
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
#define calclGlobalColumns() get_global_id(1)

/*! \brief Gets the thread id in the second local dimension */
#define calclLocalRow() get_local_id(0)

/*! \brief Gets the thread id in the second local dimension */
#define calclLocalColumns() get_local_id(1)

/*! \brief Gets the active cell row coordinate relative to the given thread id */
#define calclActiveCellRow(threadID) calclGetActiveCells()[threadID].i

/*! \brief Gets the active cell column coordinate relative to the given thread id */
#define calclActiveCellColumns(threadID) calclGetActiveCells()[threadID].j

#endif /* CALCLKERNEL2D_H_ */
