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

#include <OpenCAL-CL/calcl2D.h>

__kernel void calclkernelUpdateSubstates2D(CALint columns,CALint rows,CALint byteSubstateNum,CALint intSubstateNum,CALint realSubstateNum,__global CALbyte * currentByteSubstate,__global CALint * currentIntSubstate, __global CALreal * currentRealSubstate,__global CALbyte * nextByteSubstate,__global CALint * nextIntSubstate,__global CALreal * nextRealSubstate,__global struct CALCell2D * activeCells,__global CALint * activeCellsNum ) {

	int k;

	int threadID = get_global_id (0);

	if(threadID>=*activeCellsNum)
		return;


	for (k = 0; k < byteSubstateNum; k++)
		calclCopyBufferActiveCells2Db(nextByteSubstate + rows * columns * k, currentByteSubstate + rows * columns * k, columns, activeCells, threadID);
	for (k = 0; k < intSubstateNum; k++)
		calclCopyBufferActiveCells2Di(nextIntSubstate + rows * columns * k, currentIntSubstate + rows * columns * k, columns, activeCells, threadID);
	for (k = 0; k < realSubstateNum; k++)
		calclCopyBufferActiveCells2Dr(nextRealSubstate + rows * columns * k, currentRealSubstate + rows * columns * k, columns, activeCells, threadID);
}
