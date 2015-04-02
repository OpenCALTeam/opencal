#include <cal3D.h>

void calInitSubstate3Db(MODEL_DEFINITION3D, CALbyte value, int i, int j, int k, CALint substateNum) {
	__global CALbyte * current = calGetCurrentSubstate3Db(MODEL3D, substateNum);
	__global CALbyte * next = calGetNextSubstate3Db(MODEL3D, substateNum);

	calSetBufferElement3D(current, get_rows(), get_columns(), i, j, k, value);
	calSetBufferElement3D(next, get_rows(), get_columns(), i, j, k, value);
}
void calInitSubstate3Di(MODEL_DEFINITION3D, CALint value, int i, int j, int k, CALint substateNum) {
	__global CALint * current = calGetCurrentSubstate3Di(MODEL3D, substateNum);
	__global CALint * next = calGetNextSubstate3Di(MODEL3D, substateNum);

	calSetBufferElement3D(current, get_rows(), get_columns(), i, j, k, value);
	calSetBufferElement3D(next, get_rows(), get_columns(), i, j, k, value);
}
void calInitSubstate3Dr(MODEL_DEFINITION3D, CALreal value, int i, int j, int k, CALint substateNum) {
	__global CALreal * current = calGetCurrentSubstate3Dr(MODEL3D, substateNum);
	__global CALreal * next = calGetNextSubstate3Dr(MODEL3D, substateNum);

	calSetBufferElement3D(current, get_rows(), get_columns(), i, j, k, value);
	calSetBufferElement3D(next, get_rows(), get_columns(), i, j, k, value);
}

__global CALbyte * calGetCurrentSubstate3Db(MODEL_DEFINITION3D, CALint substateNum) {
	return (get_current_byte_substates() + get_rows() * get_columns() * get_layers() * substateNum);
}
__global CALreal * calGetCurrentSubstate3Dr(MODEL_DEFINITION3D, CALint substateNum) {
	return (get_current_real_substates() + get_rows() * get_columns() * get_layers() * substateNum);
}
__global CALint * calGetCurrentSubstate3Di(MODEL_DEFINITION3D, CALint substateNum) {
	return (get_current_int_substates() + get_rows() * get_columns() * get_layers() * substateNum);
}

__global CALbyte * calGetNextSubstate3Db(MODEL_DEFINITION3D, CALint substateNum) {
	return (get_next_byte_substates() + get_rows() * get_columns() * get_layers() * substateNum);
}
__global CALreal * calGetNextSubstate3Dr(MODEL_DEFINITION3D, CALint substateNum) {
	return (get_next_real_substates() + get_rows() * get_columns() * get_layers() * substateNum);
}
__global CALint * calGetNextSubstate3Di(MODEL_DEFINITION3D, CALint substateNum) {
	return (get_next_int_substates() + get_rows() * get_columns() * get_layers() * substateNum);
}

CALbyte calGet3Db(MODEL_DEFINITION3D, int i, int j, int k, CALint substateNum) {
	__global CALbyte * current = calGetCurrentSubstate3Db(MODEL3D, substateNum);
	return calGetBufferElement3D(current, get_rows(), get_columns(), i, j, k);
}
CALint calGet3Di(MODEL_DEFINITION3D, int i, int j, int k, CALint substateNum) {
	__global CALint * current = calGetCurrentSubstate3Di(MODEL3D, substateNum);
	return calGetBufferElement3D(current, get_rows(), get_columns(), i, j, k);
}
CALreal calGet3Dr(MODEL_DEFINITION3D, int i, int j, int k, CALint substateNum) {
	__global CALreal * current = calGetCurrentSubstate3Dr(MODEL3D, substateNum);
	return calGetBufferElement3D(current, get_rows(), get_columns(), i, j, k);
}

void calSet3Db(MODEL_DEFINITION3D, CALbyte value, int i, int j, int k, CALint substateNum) {
	__global CALbyte * next = calGetNextSubstate3Db(MODEL3D, substateNum);
	calSetBufferElement3D(next, get_rows(), get_columns(), i, j, k, value);
}
void calSet3Di(MODEL_DEFINITION3D, CALint value, int i, int j, int k, CALint substateNum) {
	__global CALint * next = calGetNextSubstate3Di(MODEL3D, substateNum);
	calSetBufferElement3D(next, get_rows(), get_columns(), i, j, k, value);
}
void calSet3Dr(MODEL_DEFINITION3D, CALreal value, int i, int j, int k, CALint substateNum) {
	__global CALreal * next = calGetNextSubstate3Dr(MODEL3D, substateNum);
	calSetBufferElement3D(next, get_rows(), get_columns(), i, j, k, value);
}

CALbyte calGetX3Db(MODEL_DEFINITION3D, int i, int j, int k, int n, CALint substateNum) {
	__global CALbyte * current = calGetCurrentSubstate3Db(MODEL3D, substateNum);
	if (get_boundary_condition() == CAL_SPACE_FLAT)
		return calGetBufferElement3D(current, get_rows(), get_columns(), i + get_neighborhood()[n].i, j + get_neighborhood()[n].j, k + get_neighborhood()[n].k);
	else
		return calGetBufferElement3D(current, get_rows(), get_columns(), calGetToroidalX(i + get_neighborhood()[n].i, get_rows()), calGetToroidalX(j + get_neighborhood()[n].j, get_columns()), calGetToroidalX(k + get_neighborhood()[n].k, get_layers()));
}
CALint calGetX3Di(MODEL_DEFINITION3D, int i, int j, int k, int n, CALint substateNum) {
	__global CALint * current = calGetCurrentSubstate3Di(MODEL3D, substateNum);
	if (get_boundary_condition() == CAL_SPACE_FLAT)
		return calGetBufferElement3D(current, get_rows(), get_columns(), i + get_neighborhood()[n].i, j + get_neighborhood()[n].j, k + get_neighborhood()[n].k);
	else
		return calGetBufferElement3D(current, get_rows(), get_columns(), calGetToroidalX(i + get_neighborhood()[n].i, get_rows()), calGetToroidalX(j + get_neighborhood()[n].j, get_columns()), calGetToroidalX(k + get_neighborhood()[n].k, get_layers()));
}
CALreal calGetX3Dr(MODEL_DEFINITION3D, int i, int j, int k, int n, CALint substateNum) {
	__global CALreal * current = calGetCurrentSubstate3Dr(MODEL3D, substateNum);
	if (get_boundary_condition() == CAL_SPACE_FLAT)
		return calGetBufferElement3D(current, get_rows(), get_columns(), i + get_neighborhood()[n].i, j + get_neighborhood()[n].j, k + get_neighborhood()[n].k);
	else
		return calGetBufferElement3D(current, get_rows(), get_columns(), calGetToroidalX(i + get_neighborhood()[n].i, get_rows()), calGetToroidalX(j + get_neighborhood()[n].j, get_columns()), calGetToroidalX(k + get_neighborhood()[n].k, get_layers()));
}









