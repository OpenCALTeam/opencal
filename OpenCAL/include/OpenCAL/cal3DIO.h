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

#ifndef cal3DIO_h
#define cal3DIO_h

#include <OpenCAL/cal3D.h>
#include <OpenCAL/calCommon.h>
#include <stdio.h>


/*! \brief Loads a byte substate from file. 
*/
DllExport
void calfLoadSubstate3Db(struct CALModel3D* ca3D, struct CALSubstate3Db* Q, FILE* f);

/*! \brief Loads an int substate from file. 
*/
DllExport
void calfLoadSubstate3Di(struct CALModel3D* ca3D, struct CALSubstate3Di* Q, FILE* f);

/*! \brief Loads a real (floating point) substate from file. 
*/
DllExport
void calfLoadSubstate3Dr(struct CALModel3D* ca3D, struct CALSubstate3Dr* Q, FILE* f);



/*! \brief Loads a byte substate from file. 
*/
DllExport
CALbyte calLoadSubstate3Db(struct CALModel3D* ca3D, struct CALSubstate3Db* Q, char* path);

/*! \brief Loads an int substate from file.
*/
DllExport
CALbyte calLoadSubstate3Di(struct CALModel3D* ca3D, struct CALSubstate3Di* Q, char* path);

/*! \brief Loads a real (floating point) substate from file. 
*/
DllExport
CALbyte calLoadSubstate3Dr(struct CALModel3D* ca3D, struct CALSubstate3Dr* Q, char* path);



/*! \brief Saves a byte substate to file. 
*/
DllExport
void calfSaveSubstate3Db(struct CALModel3D* ca3D, struct CALSubstate3Db* Q, FILE* f);

/*! \brief Saves an int substate to file. 
*/
DllExport
void calfSaveSubstate3Di(struct CALModel3D* ca3D, struct CALSubstate3Di* Q, FILE* f);

/*! \brief Saves a real (floating point) substate to file. 
*/
DllExport
void calfSaveSubstate3Dr(struct CALModel3D* ca3D, struct CALSubstate3Dr* Q, FILE* f);



/*! \brief Saves a byte substate to file. 
*/
DllExport
CALbyte calSaveSubstate3Db(struct CALModel3D* ca3D, struct CALSubstate3Db* Q, char* path);

/*! \brief Saves a int substate to file. 
*/
DllExport
CALbyte calSaveSubstate3Di(struct CALModel3D* ca3D, struct CALSubstate3Di* Q, char* path);

/*! \brief Saves a real (floating point) substate to file. 
*/
DllExport
CALbyte calSaveSubstate3Dr(struct CALModel3D* ca3D, struct CALSubstate3Dr* Q, char* path);


#endif
