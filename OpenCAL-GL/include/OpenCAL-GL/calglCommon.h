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

/*! \brief Enumeration that specify how the substate must be drawn.
*/
enum CALGL_TYPE_INFO {
	CALGL_TYPE_INFO_NO_DATA = 0,	//!< Enumerator used for specify no action on a CALNodeData.
	CALGL_TYPE_INFO_VERTEX_DATA,	//!< Enumerator used for specify that a node contains vertex data.
	CALGL_TYPE_INFO_COLOR_DATA,		//!< Enumerator used for specify that a node contains color data.
	CALGL_TYPE_INFO_NORMAL_DATA,	//!< Enumerator used for specify that a node contains normal data.
	CALGL_TYPE_INFO_TEXTURE_DATA,	//!< Enumerator used for specify that a node contains texture data.
	CALGL_TYPE_INFO_STRING_DATA		//!< Enumerator used for specify that a node contains string data.
};

/*! \brief Enumeration that specify how the type substate must be use.
*/
enum CALGL_TYPE_INFO_USE {
	CALGL_TYPE_INFO_USE_DEFAULT = 0,	//!< Enumerator used for specify that a no color must be used on the node.
	CALGL_TYPE_INFO_USE_CONST_VALUE,	//!< Enumerator used for specify a const color to use on the node.
	CALGL_TYPE_INFO_USE_GRAY_SCALE,		//!< Enumerator used for specify a gray gradient to use on the node.
	CALGL_TYPE_INFO_USE_RED_SCALE,		//!< Enumerator used for specify a red gradient to use on the node.
	CALGL_TYPE_INFO_USE_GREEN_SCALE,	//!< Enumerator used for specify a green gradient to use on the node.
	CALGL_TYPE_INFO_USE_BLUE_SCALE,		//!< Enumerator used for specify a blu gradient to use on the node.
	CALGL_TYPE_INFO_USE_ALL_COLOR		//!< Enumerator used for specify a special color combination to use on the node.
};


/*! \brief Enumeration that specify how the CALModel must be drawn.
*/
enum CALGL_DRAW_MODE {
	CALGL_DRAW_MODE_NO_DRAW = 0,	//!< Enumerator used for specify that no drawing must be executed.
	CALGL_DRAW_MODE_FLAT,			//!< Enumerator used for specify that a flat mode must be used.
	CALGL_DRAW_MODE_SURFACE			//!< Enumerator used for specify that a surface mode must be used.
};

/*! \brief Enumeration that specify data's type.
*/
enum CALGL_DATA_TYPE {
	CALGL_DATA_TYPE_UNKNOW = 0,	//!< Enumerator used for specify an unknow type of data.
	CALGL_DATA_TYPE_STATIC,		//!< Enumerator used for specify that the data in the node will change during the execution.
	CALGL_DATA_TYPE_DYNAMIC		//!< Enumerator used for specify that the data in the node will not change during the execution.
};

/*! \brief Enumeration that specify light's on/off.
*/
enum CALGL_LIGHT {
	CALGL_LIGHT_OFF = 0,	//!< Enumerator used for set le light off.
	CALGL_LIGHT_ON			//!< Enumerator used for set le light on.
};

/*! \brief Enumeration that specify if an information bar must have fixed dimensions or relative to the window.
*/
enum CALGL_INFO_BAR_DIMENSION{
	CALGL_INFO_BAR_DIMENSION_ABSOLUTE = 0,	//!< Enumerator used for specify that the information bar must be fixed.
	CALGL_INFO_BAR_DIMENSION_RELATIVE		//!< Enumerator used for specify that the information bar must don't be fixed.
};

/*! \brief Enumeration that specify information bar orientation.
*/
enum CALGL_INFO_BAR_ORIENTATION{
	CALGL_INFO_BAR_ORIENTATION_UNKNOW = 0,	//!< Enumerator used for an unknow orientation of the information bar.
	CALGL_INFO_BAR_ORIENTATION_VERTICAL,	//!< Enumerator used for specify a vertical infomation bar.
	CALGL_INFO_BAR_ORIENTATION_HORIZONTAL	//!< Enumerator used for specify a horizontal infomation bar.
};

#endif
