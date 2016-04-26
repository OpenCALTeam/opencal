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
