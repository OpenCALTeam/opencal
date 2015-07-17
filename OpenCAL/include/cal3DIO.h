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

#ifndef cal3DIO_h
#define cal3DIO_h

#include <calCommon.h>
#include <cal3D.h>
#include <stdio.h>


/*! \brief Loads a byte substate from file. 
*/
void calfLoadSubstate3Db(struct CALModel3D* ca3D, struct CALSubstate3Db* Q, FILE* f);

/*! \brief Loads an int substate from file. 
*/
void calfLoadSubstate3Di(struct CALModel3D* ca3D, struct CALSubstate3Di* Q, FILE* f);

/*! \brief Loads a real (floating point) substate from file. 
*/
void calfLoadSubstate3Dr(struct CALModel3D* ca3D, struct CALSubstate3Dr* Q, FILE* f);



/*! \brief Loads a byte substate from file. 
*/
CALbyte calLoadSubstate3Db(struct CALModel3D* ca3D, struct CALSubstate3Db* Q, char* path);

/*! \brief Loads an int substate from file.
*/
CALbyte calLoadSubstate3Di(struct CALModel3D* ca3D, struct CALSubstate3Di* Q, char* path);

/*! \brief Loads a real (floating point) substate from file. 
*/
CALbyte calLoadSubstate3Dr(struct CALModel3D* ca3D, struct CALSubstate3Dr* Q, char* path);



/*! \brief Saves a byte substate to file. 
*/
void calfSaveSubstate3Db(struct CALModel3D* ca3D, struct CALSubstate3Db* Q, FILE* f);

/*! \brief Saves an int substate to file. 
*/
void calfSaveSubstate3Di(struct CALModel3D* ca3D, struct CALSubstate3Di* Q, FILE* f);

/*! \brief Saves a real (floating point) substate to file. 
*/
void calfSaveSubstate3Dr(struct CALModel3D* ca3D, struct CALSubstate3Dr* Q, FILE* f);



/*! \brief Saves a byte substate to file. 
*/
CALbyte calSaveSubstate3Db(struct CALModel3D* ca3D, struct CALSubstate3Db* Q, char* path);

/*! \brief Saves a int substate to file. 
*/
CALbyte calSaveSubstate3Di(struct CALModel3D* ca3D, struct CALSubstate3Di* Q, char* path);

/*! \brief Saves a real (floating point) substate to file. 
*/
CALbyte calSaveSubstate3Dr(struct CALModel3D* ca3D, struct CALSubstate3Dr* Q, char* path);


#endif
