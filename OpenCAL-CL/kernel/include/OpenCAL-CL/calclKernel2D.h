/*
 * Copyright (c) 2016 OpenCALTeam (https://github.com/OpenCALTeam),
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

#ifndef CALCLKERNEL2D_H_
#define CALCLKERNEL2D_H_

/*! \brief Terminates threads that exceed matrix sizes */
#define calclThreadCheck2D() if(get_global_id(0)>=calclGetRows() || get_global_id(1)>=calclGetColumns()) return

/*! \brief Terminates threads that exceed active cells number */
#define calclActiveThreadCheck2D() if(get_global_id(0)>=calclGetActiveCellsNum()) return

/*! \brief Defines model parameters (This define is used in function declaration) */
#define __CALCL_MODEL_2D  __global CALint * CALCLrows,__global CALint * CALCLcolumns,__global CALint * CALCLbyteSubstatesNum,__global CALint * CALCLintSubstatesNum,	__global CALint * CALCLrealSubstatesNum,__global CALbyte * CALCLcurrentByteSubstates,__global CALint * CALCLcurrentIntSubstates,__global CALreal * CALCLcurrentRealSubstates,__global CALbyte * CALCLnextByteSubstates,__global CALint * CALCLnextIntSubstates,__global CALreal * CALCLnextRealSubstates,__global struct CALCell2D * CALCLactiveCells,__global CALint * CALCLactiveCellsNum,__global CALbyte * CALCLactiveCellsFlags,__global struct CALCell2D * CALCLneighborhood,__global enum CALNeighborhood2D * CALCLneighborhoodID,__global CALint * CALCLneighborhoodSize,__global enum CALSpaceBoundaryCondition * CALCLboundaryCondition,__global CALbyte * CALCLstop,__global CALbyte * CALCLdiff,CALint CALCLchunk, __global CALreal * minimab, __global CALreal * minimai, __global CALreal * minimar, __global CALreal * maximab,  __global CALreal * maximai,  __global CALreal * maximar,  __global CALreal * sumb, __global CALreal * sumi, __global CALreal * sumr, __global CALreal * logicalAndb,__global CALreal * logicalAndi,__global CALreal * logicalAndr, __global CALreal * logicalOrb,__global CALreal * logicalOri,__global CALreal * logicalOrr, __global CALreal * logicalXOrb,__global CALreal * logicalXOri, __global CALreal * logicalXOrr, __global CALreal * binaryAndb,__global CALreal * binaryAndi,__global CALreal * binaryAndr ,__global CALreal * binaryOrb, __global CALreal * binaryOri, __global CALreal * binaryOrr, __global CALreal * binaryXOrb,  __global CALreal * binaryXOri,  __global CALreal * binaryXOrr,  __global CALreal * prodb , __global CALreal * prodi , __global CALreal * prodr

/*! \brief Defines model parameters (This define is used in function calls) */
#define MODEL_2D CALCLrows,CALCLcolumns,CALCLbyteSubstatesNum,CALCLintSubstatesNum,CALCLrealSubstatesNum,CALCLcurrentByteSubstates,CALCLcurrentIntSubstates,CALCLcurrentRealSubstates,CALCLnextByteSubstates,CALCLnextIntSubstates,CALCLnextRealSubstates,CALCLactiveCells,CALCLactiveCellsNum,CALCLactiveCellsFlags,CALCLneighborhood,CALCLneighborhoodID,CALCLneighborhoodSize,CALCLboundaryCondition, CALCLstop, CALCLdiff, CALCLchunk, minimab, minimai, minimar, maximab, maximai, maximar, sumb, sumi, sumr, logicalAndb, logicalAndi, logicalAndr, logicalOrb, logicalOri, logicalOrb, logicalXOrb,logicalXOri, logicalXOrr, binaryAndb, binaryAndi, binaryAndr, binaryOrb, binaryOri, binaryOrr, binaryXOrb, binaryXOri, binaryXOrr, prodb, prodi, prodr

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
#define calclGetMax2Db(DEVICE_Q) maximab[DEVICE_Q]
#define calclGetMax2Di(DEVICE_Q) maximai[DEVICE_Q]
#define calclGetMax2Dr(DEVICE_Q) maximar[DEVICE_Q]
#define calclGetMin2Db(DEVICE_Q) minimab[DEVICE_Q]
#define calclGetMin2Di(DEVICE_Q) minimai[DEVICE_Q]
#define calclGetMin2Dr(DEVICE_Q) minimar[DEVICE_Q]
#define calclGetSum2Db(DEVICE_Q) sumb[DEVICE_Q]
#define calclGetSum2Di(DEVICE_Q) sumi[DEVICE_Q]
#define calclGetSum2Dr(DEVICE_Q) sumr[DEVICE_Q]
#define calclGetProd2Db(DEVICE_Q) prodb[DEVICE_Q]
#define calclGetProd2Di(DEVICE_Q) prodi[DEVICE_Q]
#define calclGetProd2Dr(DEVICE_Q) prodr[DEVICE_Q]
#define calclGetLogicalAnd2Db(DEVICE_Q) logicalAndb[DEVICE_Q]
#define calclGetLogicalAnd2Di(DEVICE_Q) logicalAndi[DEVICE_Q]
#define calclGetLogicalAnd2Dr(DEVICE_Q) logicalAndr[DEVICE_Q]
#define calclGetBinaryAnd2Db(DEVICE_Q) binaryAndb[DEVICE_Q]
#define calclGetBinaryAnd2Di(DEVICE_Q) binaryAndi[DEVICE_Q]
#define calclGetBinaryAnd2Dr(DEVICE_Q) binaryAndr[DEVICE_Q]
#define calclGetLogicalOr2Db(DEVICE_Q) logicalOrb[DEVICE_Q]
#define calclGetLogicalOr2Di(DEVICE_Q) logicalOri[DEVICE_Q]
#define calclGetLogicalOr2Dr(DEVICE_Q) logicalOrr[DEVICE_Q]
#define calclGetBinaryOr2Db(DEVICE_Q) binaryOrb[DEVICE_Q]
#define calclGetBinaryOr2Di(DEVICE_Q) binaryOri[DEVICE_Q]
#define calclGetBinaryOr2Dr(DEVICE_Q) binaryOrr[DEVICE_Q]
#define calclGetLogicalXor2Db(DEVICE_Q) logicalXOrb[DEVICE_Q]
#define calclGetLogicalXor2Di(DEVICE_Q) logicalXOri[DEVICE_Q]
#define calclGetLogicalXor2Dr(DEVICE_Q) logicalXOrr[DEVICE_Q]
#define calclGetBinaryXor2Db(DEVICE_Q) binaryXOrb[DEVICE_Q]
#define calclGetBinaryXor2Di(DEVICE_Q) binaryXOri[DEVICE_Q]
#define calclGetBinaryXor2Dr(DEVICE_Q) binaryXOrr[DEVICE_Q]

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
