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

#ifndef calglUtils_h
#define calglUtils_h
#include <math.h>
typedef float CALGLVector3[3];      // Three component floating point vector

/*! \brief Function that given three points calculate their normal vector
*/
void calglGetNormalVector(
	CALGLVector3 vP1,		//!< First point
	CALGLVector3 vP2, 		//!< Second point
	CALGLVector3 vP3, 		//!< Third point
	CALGLVector3 vNormal	//!< Normal vector
	);

#endif
