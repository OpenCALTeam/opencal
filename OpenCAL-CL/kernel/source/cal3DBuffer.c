#include "../../../OpenCAL-CL/kernel/include/cal3DBuffer.h"

void calCopyBufferActiveCells3Db(__global CALbyte* M_src, __global CALbyte* M_dest, int columns, int rows, __global struct CALCell3D* active_cells, int n) {
	int c = active_cells[n].k * columns * rows + active_cells[n].i * columns + active_cells[n].j;
	if (M_dest[c] != M_src[c])
		M_dest[c] = M_src[c];
}

void calCopyBufferActiveCells3Di(__global CALint* M_src, __global CALint* M_dest, int columns, int rows, __global struct CALCell3D* active_cells, int n) {
	int c = active_cells[n].k * columns * rows + active_cells[n].i * columns + active_cells[n].j;
	if (M_dest[c] != M_src[c])
		M_dest[c] = M_src[c];
}

void calCopyBufferActiveCells3Dr(__global CALreal* M_src, __global CALreal* M_dest, int columns, int rows, __global struct CALCell3D* active_cells, int n) {
	int c = active_cells[n].k * columns * rows + active_cells[n].i * columns + active_cells[n].j;
	if (M_dest[c] != M_src[c])
		M_dest[c] = M_src[c];
}

void calCopyBuffer3Db(__global CALbyte* M_src, __global CALbyte* M_dest, int columns, int rows, int i, int j, int k) {
	CALbyte value = calGetBufferElement3D(M_src, rows, columns, i, j, k);
	calSetBufferElement3D(M_dest, rows, columns, i, j, k, value);
}
void calCopyBuffer3Di(__global CALint* M_src, __global CALint* M_dest, int columns, int rows, int i, int j, int k) {
	CALint value = calGetBufferElement3D(M_src, rows, columns, i, j, k);
	calSetBufferElement3D(M_dest, rows, columns, i, j, k, value);
}
void calCopyBuffer3Dr(__global CALreal* M_src, __global CALreal* M_dest, int columns, int rows, int i, int j, int k) {
	CALreal value = calGetBufferElement3D(M_src, rows, columns, i, j, k);
	calSetBufferElement3D(M_dest, rows, columns, i, j, k, value);
}

void calAddMatrices3Db(__global CALbyte* M_op1, __global CALbyte* M_op3, __global CALbyte* M_dest, int i, int j, int k, int columns, int rows) {
	CALbyte sum = calGetBufferElement3D(M_op1, rows, columns, i, j, k);
	sum += calGetBufferElement3D(M_op3, rows, columns, i, j, k);
	calSetBufferElement3D(M_dest, rows, columns, i, j, k, sum);
}
void calAddMatrices3Di(__global CALint* M_op1, __global CALint* M_op3, __global CALint* M_dest, int i, int j, int k, int columns, int rows) {
	CALint sum = calGetBufferElement3D(M_op1, rows, columns, i, j, k);
	sum += calGetBufferElement3D(M_op3, rows, columns, i, j, k);
	calSetBufferElement3D(M_dest, rows, columns, i, j, k, sum);
}
void calAddMatrices3Dr(__global CALreal* M_op1, __global CALreal* M_op3, __global CALreal* M_dest, int i, int j, int k, int columns, int rows) {
	CALreal sum = calGetBufferElement3D(M_op1, rows, columns, i, j, k);
	sum += calGetBufferElement3D(M_op3, rows, columns, i, j, k);
	calSetBufferElement3D(M_dest, rows, columns, i, j, k, sum);
}

void calSubtractMatrices3Db(__global CALbyte* M_op1, __global CALbyte* M_op3, __global CALbyte* M_dest, int i, int j, int k, int columns, int rows) {
	CALbyte sub = calGetBufferElement3D(M_op1, rows, columns, i, j, k);
	sub -= calGetBufferElement3D(M_op3, rows, columns, i, j, k);
	calSetBufferElement3D(M_dest, rows, columns, i, j, k, sub);
}
void calSubtractMatrices3Di(__global CALint* M_op1, __global CALint* M_op3, __global CALint* M_dest, int i, int j, int k, int columns, int rows) {
	CALint sub = calGetBufferElement3D(M_op1, rows, columns, i, j, k);
	sub -= calGetBufferElement3D(M_op3, rows, columns, i, j, k);
	calSetBufferElement3D(M_dest, rows, columns, i, j, k, sub);
}
void calSubtractMatrices3Dr(__global CALreal* M_op1, __global CALreal* M_op3, __global CALreal* M_dest, int i, int j, int k, int columns, int rows) {
	CALreal sub = calGetBufferElement3D(M_op1, rows, columns, i, j, k);
	sub -= calGetBufferElement3D(M_op3, rows, columns, i, j, k);
	calSetBufferElement3D(M_dest, rows, columns, i, j, k, sub);
}

void calSetBufferActiveCells3Db(__global CALbyte* M, int columns, int rows, CALbyte value, __global struct CALCell3D* active_cells, int n) {
	calSetBufferElement3D(M, columns, rows, active_cells[n].i, active_cells[n].j, active_cells[n].k, value);
}
void calSetBufferActiveCells3Di(__global CALint* M, int columns, int rows, CALint value, __global struct CALCell3D* active_cells, int n) {
	calSetBufferElement3D(M, columns, rows, active_cells[n].i, active_cells[n].j, active_cells[n].k, value);
}
void calSetBufferActiveCells3Dr(__global CALreal* M, int columns, int rows, CALreal value, __global struct CALCell3D* active_cells, int n) {
	calSetBufferElement3D(M, columns, rows, active_cells[n].i, active_cells[n].j, active_cells[n].k, value);
}

