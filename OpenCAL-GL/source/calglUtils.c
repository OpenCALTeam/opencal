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

#include <OpenCAL-GL/calglUtils.h>

// Calculate the cross product of two vectors
void calglVectorCrossProduct(CALGLVector3 vU, CALGLVector3 vV, CALGLVector3 vResult)
{
	vResult[0] = vU[1] * vV[2] - vV[1] * vU[2];
	vResult[1] = -vU[0] * vV[2] + vV[0] * vU[2];
	vResult[2] = vU[0] * vV[1] - vV[0] * vU[1];
}

// Subtract one vector from another
void calglSubtractVectors(CALGLVector3 vFirst, CALGLVector3 vSecond, CALGLVector3 vResult)
{
	vResult[0] = vFirst[0] - vSecond[0];
	vResult[1] = vFirst[1] - vSecond[1];
	vResult[2] = vFirst[2] - vSecond[2];
}

// Given three points on a plane in counter clockwise order, calculate the unit normal
void calglGetNormalVector(CALGLVector3 vP1, CALGLVector3 vP2, CALGLVector3 vP3, CALGLVector3 vNormal)
{
	CALGLVector3 vV1, vV2;

	calglSubtractVectors(vP2, vP1, vV1);
	calglSubtractVectors(vP3, vP1, vV2);

	calglVectorCrossProduct(vV1, vV2, vNormal);
}

void calglAverageNormals(CALGLVector3 N24, CALGLVector3 N46, CALGLVector3 N68, CALGLVector3 N82, CALGLVector3 N)
{
	int i;
	for (i = 0; i < 3; i++)
		N[i] = N24[i] + N46[i] + N68[i] + N82[i];
}

