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

#ifndef calSteeringCommon_h
#define calSteeringCommon_h

enum STEERING_OPERATION {
	STEERING_NONE = 0,
	STEERING_MAX,
	STEERING_MIN,
	STEERING_SUM,
	STEERING_PROD,
	STEERING_LOGICAL_AND,
	STEERING_BINARY_AND,
	STEERING_LOGICAL_OR,
	STEERING_BINARY_OR,
	STEERING_LOGICAL_XOR,
	STEERING_BINARY_XOR
};

#endif
