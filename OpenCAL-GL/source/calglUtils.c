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

#include <OpenCAL-GL/calglUtils.h>

// Calculate the cross product of two vectors
void calglVectorCrossProduct(CALGLVector3 vU, CALGLVector3 vV, CALGLVector3 vResult) {
	vResult[0] = vU[1]*vV[2]-vV[1]*vU[2];
	vResult[1] = -vU[0]*vV[2]+vV[0]*vU[2];
	vResult[2] = vU[0]*vV[1]-vV[0]*vU[1];
}

// Subtract one vector from another
void calglSubtractVectors(CALGLVector3 vFirst, CALGLVector3 vSecond, CALGLVector3 vResult) {
	vResult[0] = vFirst[0]-vSecond[0];
	vResult[1] = vFirst[1]-vSecond[1];
	vResult[2] = vFirst[2]-vSecond[2];
}

// Given three points on a plane in counter clockwise order, calculate the unit normal
void calglGetNormalVector(CALGLVector3 vP1, CALGLVector3 vP2, CALGLVector3 vP3, CALGLVector3 vNormal) {
	CALGLVector3 vV1, vV2;
	float lenght = 1.0f;

	calglSubtractVectors(vP2, vP1, vV1);
	calglSubtractVectors(vP3, vP1, vV2);

	calglVectorCrossProduct(vV1, vV2, vNormal);

	lenght = vNormal[0]*vNormal[0]+
		vNormal[1]*vNormal[1]+
		vNormal[2]*vNormal[2];
	lenght = sqrt(lenght);

	lenght = lenght>=0 ? lenght : 1.0f;
	vNormal[0] /= lenght;
	vNormal[1] /= lenght;
	vNormal[2] /= lenght;
}

void calglAverageNormals(CALGLVector3 N24, CALGLVector3 N46, CALGLVector3 N68, CALGLVector3 N82, CALGLVector3 N) {
	int i;
	for(i = 0; i<3; i++)
		N[i] = N24[i]+N46[i]+N68[i]+N82[i];
}
