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

#ifndef cal2DIO_h
#define cal2DIO_h

#include <calCommon.h>
#include <cal2D.h>
#include <stdio.h>




/*! \brief Loads a byte substate from file. 
*/
void calfLoadSubstate2Db(struct CALModel2D* ca2D, struct CALSubstate2Db* Q, FILE* f);

/*! \brief Loads an int substate from file. 
*/
void calfLoadSubstate2Di(struct CALModel2D* ca2D, struct CALSubstate2Di* Q, FILE* f);

/*! \brief Loads a real (floating point) substate from file. 
*/
void calfLoadSubstate2Dr(struct CALModel2D* ca2D, struct CALSubstate2Dr* Q, FILE* f);



/*! \brief Loads a byte substate from file. 
*/
CALbyte calLoadSubstate2Db(struct CALModel2D* ca2D, struct CALSubstate2Db* Q, char* path);

/*! \brief Loads an int substate from file. 
*/
CALbyte calLoadSubstate2Di(struct CALModel2D* ca2D, struct CALSubstate2Di* Q, char* path);

/*! \brief Loads a real (floating point) substate from file. 
*/
CALbyte calLoadSubstate2Dr(struct CALModel2D* ca2D, struct CALSubstate2Dr* Q, char* path);



/*! \brief Saves a byte substate to file. 
*/
void calfSaveSubstate2Db(struct CALModel2D* ca2D, struct CALSubstate2Db* Q, FILE* f);

/*! \brief Saves an int substate to file. 
*/
void calfSaveSubstate2Di(struct CALModel2D* ca2D, struct CALSubstate2Di* Q, FILE* f);

/*! \brief Saves a real (floating point) substate to file. 
*/
void calfSaveSubstate2Dr(struct CALModel2D* ca2D, struct CALSubstate2Dr* Q, FILE* f);



/*! \brief Saves a byte substate to file. 
*/
CALbyte calSaveSubstate2Db(struct CALModel2D* ca2D, struct CALSubstate2Db* Q, char* path);

/*! \brief Saves a int substate to file. 
*/
CALbyte calSaveSubstate2Di(struct CALModel2D* ca2D, struct CALSubstate2Di* Q, char* path);

/*! \brief Saves a real (floating point) substate to file. 
*/
CALbyte calSaveSubstate2Dr(struct CALModel2D* ca2D, struct CALSubstate2Dr* Q, char* path);


#endif
