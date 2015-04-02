#include <cal3D.h>
#include <cal3DBuffer.h>
#include <cal3DBufferIO.h>


void calfLoadSubstate3Db(struct CALModel3D* ca3D, struct CALSubstate3Db* Q, FILE* f) {
	calfLoadBuffer3Db(Q->current, ca3D->rows, ca3D->columns, ca3D->layers, f);
	if (Q->next)
		calCopyBuffer3Db(Q->current, Q->next, ca3D->rows, ca3D->columns, ca3D->layers);
}
void calfLoadSubstate3Di(struct CALModel3D* ca3D, struct CALSubstate3Di* Q, FILE* f) {
	calfLoadBuffer3Di(Q->current, ca3D->rows, ca3D->columns, ca3D->layers, f);
	if (Q->next)
		calCopyBuffer3Di(Q->current, Q->next, ca3D->rows, ca3D->columns, ca3D->layers);
}
void calfLoadSubstate3Dr(struct CALModel3D* ca3D, struct CALSubstate3Dr* Q, FILE* f) {
	calfLoadBuffer3Dr(Q->current, ca3D->rows, ca3D->columns, ca3D->layers, f);
	if (Q->next)
		calCopyBuffer3Dr(Q->current, Q->next, ca3D->rows, ca3D->columns, ca3D->layers);
}


CALbyte calLoadSubstate3Db(struct CALModel3D* ca3D, struct CALSubstate3Db* Q, char* path) {
	CALbyte return_state = calLoadBuffer3Db(Q->current, ca3D->rows, ca3D->columns, ca3D->layers, path);
	if (Q->next)
		calCopyBuffer3Db(Q->current, Q->next, ca3D->rows, ca3D->columns, ca3D->layers);
	return return_state;
}
CALbyte calLoadSubstate3Di(struct CALModel3D* ca3D, struct CALSubstate3Di* Q, char* path) {
	CALbyte return_state = calLoadBuffer3Di(Q->current, ca3D->rows, ca3D->columns, ca3D->layers, path);
	if (Q->next)
		calCopyBuffer3Di(Q->current, Q->next, ca3D->rows, ca3D->columns, ca3D->layers);
	return return_state;
}
CALbyte calLoadSubstate3Dr(struct CALModel3D* ca3D, struct CALSubstate3Dr* Q, char* path) {
	CALbyte return_state = calLoadBuffer3Dr(Q->current, ca3D->rows, ca3D->columns, ca3D->layers, path);
	if (Q->next)
		calCopyBuffer3Dr(Q->current, Q->next, ca3D->rows, ca3D->columns, ca3D->layers);
	return return_state;
}


void calfSaveSubstate3Db(struct CALModel3D* ca3D, struct CALSubstate3Db* Q, FILE* f) {
	calfSaveBuffer3Db(Q->current, ca3D->rows, ca3D->columns, ca3D->layers, f);
}
void calfSaveSubstate3Di(struct CALModel3D* ca3D, struct CALSubstate3Di* Q, FILE* f) {
	calfSaveBuffer3Di(Q->current, ca3D->rows, ca3D->columns, ca3D->layers, f);
}
void calfSaveSubstate3Dr(struct CALModel3D* ca3D, struct CALSubstate3Dr* Q, FILE* f) {
	calfSaveBuffer3Dr(Q->current, ca3D->rows, ca3D->columns, ca3D->layers, f);
}


CALbyte calSaveSubstate3Db(struct CALModel3D* ca3D, struct CALSubstate3Db* Q, char* path) {
	CALbyte return_state = calSaveBuffer3Db(Q->current, ca3D->rows, ca3D->columns, ca3D->layers, path);
	return return_state;
}
CALbyte calSaveSubstate3Di(struct CALModel3D* ca3D, struct CALSubstate3Di* Q, char* path) {
	CALbyte return_state = calSaveBuffer3Di(Q->current, ca3D->rows, ca3D->columns, ca3D->layers, path);
	return return_state;
}
CALbyte calSaveSubstate3Dr(struct CALModel3D* ca3D, struct CALSubstate3Dr* Q, char* path) {
	CALbyte return_state = calSaveBuffer3Dr(Q->current, ca3D->rows, ca3D->columns, ca3D->layers, path);
	return return_state;
}
