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

#ifndef calglUtils_h
#define calglUtils_h
#include <OpenCAL-GL/calglCommon.h>
#include <math.h>
typedef float CALGLVector3[3];      // Three component floating point vector

/*! \brief Function that given three points calculate their normal vector
*/
DllExport
void calglGetNormalVector(
	CALGLVector3 vP1,		//!< First point
	CALGLVector3 vP2, 		//!< Second point
	CALGLVector3 vP3, 		//!< Third point
	CALGLVector3 vNormal	//!< Normal vector
	);

#endif
