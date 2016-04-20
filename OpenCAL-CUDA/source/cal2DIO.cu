/*
 * Copyright (c) 2016 OpenCALTeam (https://github.com/OpenCALTeam),
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

#include ".\..\include\cal2D.cuh"
#include ".\..\include\cal2DBuffer.cuh"
#include ".\..\include\cal2DBufferIO.cuh"

CALbyte calCudaLoadSubstate2Db(struct CudaCALModel2D* ca2D, char* path, int i_substate) {
	CALbyte return_state = calCudaLoadMatrix2Db(ca2D->pQb_array_current, ca2D->rows, ca2D->columns, path, i_substate);
	if (ca2D->pQb_array_next){
		calCudaCopyBuffer2Db(ca2D->pQb_array_current, ca2D->pQb_array_next, ca2D->rows, ca2D->columns, i_substate+1);
	}
	return return_state;
}

CALbyte calCudaLoadSubstate2Di(struct CudaCALModel2D* ca2D, char* path, int i_substate) {
	CALbyte return_state = calCudaLoadMatrix2Di(ca2D->pQi_array_current, ca2D->rows, ca2D->columns, path, i_substate);
	if (ca2D->pQi_array_next){
		calCudaCopyBuffer2Di(ca2D->pQi_array_current, ca2D->pQi_array_next, ca2D->rows, ca2D->columns, i_substate+1);
	}
	return return_state;
}

CALbyte calCudaLoadSubstate2Dr(struct CudaCALModel2D* ca2D, char* path, int i_substate) {
	CALbyte return_state = calCudaLoadMatrix2Dr(ca2D->pQr_array_current, ca2D->rows, ca2D->columns, path, i_substate);
	if (ca2D->pQr_array_next)
		calCudaCopyBuffer2Dr(ca2D->pQr_array_current, ca2D->pQr_array_next, ca2D->rows, ca2D->columns, i_substate+1);
	return return_state;
}

CALbyte calCudaSaveSubstate2Db(struct CudaCALModel2D* ca2D, char* path, CALint index_substate) {
	CALbyte return_state = calCudaSaveMatrix2Db(ca2D->pQb_array_current, ca2D->rows, ca2D->columns, path, index_substate);
	return return_state;
}

CALbyte calCudaSaveSubstate2Di(struct CudaCALModel2D* ca2D, char* path, CALint index_substate) {
	CALbyte return_state = calCudaSaveMatrix2Di(ca2D->pQi_array_current, ca2D->rows, ca2D->columns, path, index_substate);
	return return_state;
}

CALbyte calCudaSaveSubstate2Dr(struct CudaCALModel2D* ca2D, char* path, CALint index_substate) {
	CALbyte return_state = calCudaSaveMatrix2Dr(ca2D->pQr_array_current, ca2D->rows, ca2D->columns, path, index_substate);
	return return_state;
}
