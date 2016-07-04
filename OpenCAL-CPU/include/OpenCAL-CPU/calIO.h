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

#ifndef cal_iO_h
#define cal_iO_h

#include <OpenCAL-CPU/calModel.h>
#include <OpenCAL-CPU/calCommon.h>
#include <stdio.h>




/*! \brief Loads a byte substate from file.
*/
void calfLoadSubstate_b(struct CALModel* calModel, struct CALSubstate_b* Q, FILE* f);

/*! \brief Loads an int substate from file.
*/
void calfLoadSubstate_i(struct CALModel* calModel, struct CALSubstate_i* Q, FILE* f);

/*! \brief Loads a real (floating point) substate from file.
*/
void calfLoadSubstate_r(struct CALModel* calModel, struct CALSubstate_r* Q, FILE* f);



/*! \brief Loads a byte substate from file.
*/
CALbyte calLoadSubstate_b(struct CALModel* calModel, struct CALSubstate_b* Q, char* path);

/*! \brief Loads an int substate from file.
*/
CALbyte calLoadSubstate_i(struct CALModel* calModel, struct CALSubstate_i* Q, char* path);

/*! \brief Loads a real (floating point) substate from file.
*/
CALbyte calLoadSubstate_r(struct CALModel* calModel, struct CALSubstate_r* Q, char* path);



/*! \brief Saves a byte substate to file.
*/
void calfSaveSubstate_b(struct CALModel* calModel, struct CALSubstate_b* Q, FILE* f);

/*! \brief Saves an int substate to file.
*/
void calfSaveSubstate_i(struct CALModel* calModel, struct CALSubstate_i* Q, FILE* f);

/*! \brief Saves a real (floating point) substate to file.
*/
void calfSaveSubstate_r(struct CALModel* calModel, struct CALSubstate_r* Q, FILE* f);



/*! \brief Saves a byte substate to file.
*/
CALbyte calSaveSubstate_b(struct CALModel* calModel, struct CALSubstate_b* Q, char* path);

/*! \brief Saves a int substate to file.
*/
CALbyte calSaveSubstate_i(struct CALModel* calModel, struct CALSubstate_i* Q, char* path);

/*! \brief Saves a real (floating point) substate to file.
*/
CALbyte calSaveSubstate_r(struct CALModel* calModel, struct CALSubstate_r* Q, char* path);


#endif

