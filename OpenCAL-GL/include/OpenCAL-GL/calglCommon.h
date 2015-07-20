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

#ifndef calglCommon_h
#define calglCommon_h

/*! Specify how the substate must be drawn
*/
enum CALGL_TYPE_INFO {
	CALGL_TYPE_INFO_NO_DATA = 0,
	CALGL_TYPE_INFO_VERTEX_DATA,
	CALGL_TYPE_INFO_COLOR_DATA,
	CALGL_TYPE_INFO_NORMAL_DATA,
	CALGL_TYPE_INFO_TEXTURE_DATA,
	CALGL_TYPE_INFO_STRING_DATA
};

/*! Specify how the type substate must be use
*/
enum CALGL_TYPE_INFO_USE {
	CALGL_TYPE_INFO_USE_DEFAULT = 0,
	CALGL_TYPE_INFO_USE_CONST_VALUE,
	CALGL_TYPE_INFO_USE_GRAY_SCALE,
	CALGL_TYPE_INFO_USE_RED_SCALE,
	CALGL_TYPE_INFO_USE_GREEN_SCALE,
	CALGL_TYPE_INFO_USE_BLUE_SCALE,
	CALGL_TYPE_INFO_USE_ALL_COLOR
};


/*! Specify how the CALModel must be drawn
*/
enum CALGL_DRAW_MODE {
	CALGL_DRAW_MODE_NO_DRAW = 0,
	CALGL_DRAW_MODE_FLAT,
	CALGL_DRAW_MODE_SURFACE
};

/*! Specify data's type
*/
enum CALGL_DATA_TYPE {
	CALGL_DATA_TYPE_UNKNOW = 0,
	CALGL_DATA_TYPE_STATIC,
	CALGL_DATA_TYPE_DYNAMIC
};

/*! Specify light's on/off
*/
enum CALGL_LIGHT {
	CALGL_LIGHT_OFF = 0,
	CALGL_LIGHT_ON
};

/*! Specify information bar orientation
*/
enum CALGL_INFO_BAR_ORIENTATION{
	CALGL_INFO_BAR_VERTICAL = 0,
	CALGL_INFO_BAR_HORIZONTAL
};

#endif