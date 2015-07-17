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

#include "calCommon.cuh"
#include <stdio.h>

/*! \brief Loads a byte substate from file for CUDA version. 
*/
CALbyte calCudaLoadSubstate2Db(struct CudaCALModel2D* ca2D, char* path, int i_substate);

/*! \brief Loads a byte substate from file for CUDA version. 
*/
CALbyte calCudaLoadSubstate2Di(struct CudaCALModel2D* ca2D, char* path, int i_substate);

/*! \brief Loads a real (floating point) substate from file directly in CudaCALModel2D struct for CUDA version. 
*/
CALbyte calCudaLoadSubstate2Dr(struct CudaCALModel2D* ca2D, char* path, int i_substate);

/*! \brief Saves a byte substate to file. 
*/
CALbyte calCudaSaveSubstate2Db(struct CudaCALModel2D* ca2D, char* path, CALint index_substate);

/*! \brief Saves a int substate to file. 
*/
CALbyte calCudaSaveSubstate2Di(struct CudaCALModel2D* ca2D, char* path, CALint index_substate);

/*! \brief Saves a real substate to file. 
*/
CALbyte calCudaSaveSubstate2Dr(struct CudaCALModel2D* ca2D, char* path, CALint index_substate);


#endif
