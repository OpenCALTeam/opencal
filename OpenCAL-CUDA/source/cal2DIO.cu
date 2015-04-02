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