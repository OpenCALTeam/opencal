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

#include <OpenCAL-CL/calcl3D.h>

__kernel void calclkernelUpdateSubstates3D(	CALint columns,	CALint rows,CALint slices,	CALint byteSubstateNum,	CALint intSubstateNum,	CALint realSubstateNum,	__global CALbyte * currentByteSubstate,	__global CALint * currentIntSubstate,	__global CALreal * currentRealSubstate,	__global CALbyte * nextByteSubstate, __global CALint * nextIntSubstate, __global CALreal * nextRealSubstate, __global struct CALCell3D * activeCells, __global CALint * activeCellsNum, CALint borderSize)
{

	int k;

	int threadID = get_global_id (0);

	if (threadID >= *activeCellsNum)
		return;

	for (k = 0; k < byteSubstateNum; k++)
		calclCopyBufferActiveCells3Db(nextByteSubstate + rows * columns * slices * k, currentByteSubstate + rows * columns * slices * k, columns,rows, activeCells, threadID,borderSize);
	for (k = 0; k < intSubstateNum; k++)
		calclCopyBufferActiveCells3Di(nextIntSubstate + rows * columns * slices * k, currentIntSubstate + rows * columns * slices * k, columns,rows, activeCells, threadID,borderSize);
	for (k = 0; k < realSubstateNum; k++)
		calclCopyBufferActiveCells3Dr(nextRealSubstate + rows * columns * slices * k, currentRealSubstate + rows * columns * slices * k, columns,rows, activeCells, threadID, borderSize);

}
