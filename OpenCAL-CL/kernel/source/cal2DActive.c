#include "cal2DActive.h"

void calAddActiveCell2D(MODEL_DEFINITION2D, int i, int j) {

	if (CAL_FALSE == calGetBufferElement2D(get_active_cells_flags(), get_columns(), i, j)) {
		calSetBufferElement2D(get_active_cells_flags(), get_columns(), i, j, CAL_TRUE);
		CALint chunkNum = (i * get_columns() + j) / CALCLchunk;
		CALCLdiff[chunkNum] = CAL_TRUE;
	}

}
void calAddActiveCellX2D(MODEL_DEFINITION2D, int i, int j, int n) {

	if ((get_neighborhood_id() == CAL_HEXAGONAL_NEIGHBORHOOD_2D && j % 2 == 1) || (get_neighborhood_id() == CAL_HEXAGONAL_NEIGHBORHOOD_ALT_2D && i % 2 == 1))
		n += CAL_HEXAGONAL_SHIFT;

	CALint ix;
	CALint jx;

	if (get_boundary_condition() == CAL_SPACE_FLAT) {
		ix = i + get_neighborhood()[n].i;
		jx = j + get_neighborhood()[n].j;
	} else {
		ix = calGetToroidalX(i + get_neighborhood()[n].i, get_rows());
		jx = calGetToroidalX(j + get_neighborhood()[n].j, get_columns());
	}

	if (CAL_FALSE == calGetBufferElement2D(get_active_cells_flags(), get_columns(), ix, jx)) {
		calSetBufferElement2D(get_active_cells_flags(), get_columns(), ix, jx, CAL_TRUE);
		CALint chunkNum = (ix * get_columns() + jx) / CALCLchunk;
		CALCLdiff[chunkNum] = CAL_TRUE;
	}
}

void calRemoveActiveCell2D(MODEL_DEFINITION2D, int i, int j) {

	if (CAL_TRUE == calGetBufferElement2D(get_active_cells_flags(), get_columns(), i, j)) {
		calSetBufferElement2D(get_active_cells_flags(), get_columns(), i, j, CAL_FALSE);
		CALint chunkNum = (i * get_columns() + j) / CALCLchunk;
		CALCLdiff[chunkNum] = CAL_TRUE;
	}

}

void calInitSubstateActiveCell2Db(MODEL_DEFINITION2D, CALbyte value, int n, int substateNum) {
	__global CALbyte * current = calGetCurrentSubstate2Db(MODEL2D, substateNum);
	__global CALbyte * next = calGetNextSubstate2Db(MODEL2D, substateNum);
	calSetBufferActiveCells2Db(current, get_columns(), value, get_active_cells(), n);
	calSetBufferActiveCells2Db(next, get_columns(), value, get_active_cells(), n);
}
void calInitSubstateActiveCell2Di(MODEL_DEFINITION2D, CALint value, int n, int substateNum) {
	__global CALint * current = calGetCurrentSubstate2Di(MODEL2D, substateNum);
	__global CALint * next = calGetNextSubstate2Di(MODEL2D, substateNum);
	calSetBufferActiveCells2Di(current, get_columns(), value, get_active_cells(), n);
	calSetBufferActiveCells2Di(next, get_columns(), value, get_active_cells(), n);
}
void calInitSubstateActiveCell2Dr(MODEL_DEFINITION2D, CALreal value, int n, int substateNum) {
	__global CALreal * current = calGetCurrentSubstate2Dr(MODEL2D, substateNum);
	__global CALreal * next = calGetNextSubstate2Dr(MODEL2D, substateNum);
	calSetBufferActiveCells2Dr(current, get_columns(), value, get_active_cells(), n);
	calSetBufferActiveCells2Dr(next, get_columns(), value, get_active_cells(), n);
}

