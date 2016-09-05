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

#ifndef cal2DIO_h
#define cal2DIO_h

#include <OpenCAL-OMP/cal2D.h>
#include <OpenCAL-OMP/calCommon.h>
#include <stdio.h>




/*! \brief Loads a byte substate from file. 
*/
DllExport
void calfLoadSubstate2Db(struct CALModel2D* ca2D, struct CALSubstate2Db* Q, FILE* f);

/*! \brief Loads an int substate from file. 
*/
DllExport
void calfLoadSubstate2Di(struct CALModel2D* ca2D, struct CALSubstate2Di* Q, FILE* f);

/*! \brief Loads a real (floating point) substate from file. 
*/
DllExport
void calfLoadSubstate2Dr(struct CALModel2D* ca2D, struct CALSubstate2Dr* Q, FILE* f);



/*! \brief Loads a byte substate from file. 
*/
DllExport
CALbyte calLoadSubstate2Db(struct CALModel2D* ca2D, struct CALSubstate2Db* Q, char* path);

/*! \brief Loads an int substate from file. 
*/
DllExport
CALbyte calLoadSubstate2Di(struct CALModel2D* ca2D, struct CALSubstate2Di* Q, char* path);

/*! \brief Loads a real (floating point) substate from file. 
*/
DllExport
CALbyte calLoadSubstate2Dr(struct CALModel2D* ca2D, struct CALSubstate2Dr* Q, char* path);



/*! \brief Saves a byte substate to file. 
*/
DllExport
void calfSaveSubstate2Db(struct CALModel2D* ca2D, struct CALSubstate2Db* Q, FILE* f);

/*! \brief Saves an int substate to file. 
*/
DllExport
void calfSaveSubstate2Di(struct CALModel2D* ca2D, struct CALSubstate2Di* Q, FILE* f);

/*! \brief Saves a real (floating point) substate to file. 
*/
DllExport
void calfSaveSubstate2Dr(struct CALModel2D* ca2D, struct CALSubstate2Dr* Q, FILE* f);



/*! \brief Saves a byte substate to file. 
*/
DllExport
CALbyte calSaveSubstate2Db(struct CALModel2D* ca2D, struct CALSubstate2Db* Q, char* path);

/*! \brief Saves a int substate to file. 
*/
DllExport
CALbyte calSaveSubstate2Di(struct CALModel2D* ca2D, struct CALSubstate2Di* Q, char* path);

/*! \brief Saves a real (floating point) substate to file. 
*/
DllExport
CALbyte calSaveSubstate2Dr(struct CALModel2D* ca2D, struct CALSubstate2Dr* Q, char* path);


#endif
