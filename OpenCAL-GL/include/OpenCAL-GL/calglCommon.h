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

#ifndef calglCommon_h
#define calglCommon_h

#ifdef _MSC_VER
#define DllExport   __declspec( dllexport )
#else
#define DllExport
#endif


/*! \brief Enumeration that specify how the substate must be drawn.
*/
enum CALGL_TYPE_INFO {
	CALGL_TYPE_INFO_NO_DATA = 0,	//!< Enumerator used for specify no action on a CALNodeData.
	CALGL_TYPE_INFO_VERTEX_DATA,	//!< Enumerator used for specify that a node contains vertex data.
	CALGL_TYPE_INFO_COLOR_DATA,		//!< Enumerator used for specify that a node contains color data.
	CALGL_TYPE_INFO_NORMAL_DATA,	//!< Enumerator used for specify that a node contains normal data.
};

/*! \brief Enumeration that specify how the type substate must be use.
*/
enum CALGL_TYPE_INFO_USE {
	CALGL_TYPE_INFO_USE_NO_COLOR = 0,	//!< Enumerator used for specify that a no color must be used on the node.
	CALGL_TYPE_INFO_USE_CURRENT_COLOR,	//!< Enumerator used for specify a const color to use on the node.
	CALGL_TYPE_INFO_USE_GRAY_SCALE,		//!< Enumerator used for specify a gray gradient to use on the node.
	CALGL_TYPE_INFO_USE_RED_SCALE,		//!< Enumerator used for specify a red gradient to use on the node.
	CALGL_TYPE_INFO_USE_GREEN_SCALE,	//!< Enumerator used for specify a green gradient to use on the node.
	CALGL_TYPE_INFO_USE_BLUE_SCALE,		//!< Enumerator used for specify a blu gradient to use on the node.
	CALGL_TYPE_INFO_USE_RED_YELLOW_SCALE,	//!< Enumerator used for specify a Red and Yellow gradient to use on the node.
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

/*! \brief Enumeration that specify layout orientation when there are two models to draw.
*/
enum CALGL_LAYOUT_ORIENTATION {
	CALGL_LAYOUT_ORIENTATION_UNKNOW = 0,	//!< Enumerator used for an unknow layout orientation.
	CALGL_LAYOUT_ORIENTATION_VERTICAL,		//!< Enumerator used for specify a vertical layout orientation.
	CALGL_LAYOUT_ORIENTATION_HORIZONTAL		//!< Enumerator used for specify a horizontal layout orientation.
};

#endif
