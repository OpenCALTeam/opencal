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

#include <cal3D.h>
#include <cal3DBuffer.h>
#include <cal3DBufferIO.h>


void calfLoadSubstate3Db(struct CALModel3D* ca3D, struct CALSubstate3Db* Q, FILE* f) {
	calfLoadBuffer3Db(Q->current, ca3D->rows, ca3D->columns, ca3D->slices, f);
	if (Q->next)
		calCopyBuffer3Db(Q->current, Q->next, ca3D->rows, ca3D->columns, ca3D->slices);
}
void calfLoadSubstate3Di(struct CALModel3D* ca3D, struct CALSubstate3Di* Q, FILE* f) {
	calfLoadBuffer3Di(Q->current, ca3D->rows, ca3D->columns, ca3D->slices, f);
	if (Q->next)
		calCopyBuffer3Di(Q->current, Q->next, ca3D->rows, ca3D->columns, ca3D->slices);
}
void calfLoadSubstate3Dr(struct CALModel3D* ca3D, struct CALSubstate3Dr* Q, FILE* f) {
	calfLoadBuffer3Dr(Q->current, ca3D->rows, ca3D->columns, ca3D->slices, f);
	if (Q->next)
		calCopyBuffer3Dr(Q->current, Q->next, ca3D->rows, ca3D->columns, ca3D->slices);
}


CALbyte calLoadSubstate3Db(struct CALModel3D* ca3D, struct CALSubstate3Db* Q, char* path) {
	CALbyte return_state = calLoadBuffer3Db(Q->current, ca3D->rows, ca3D->columns, ca3D->slices, path);
	if (Q->next)
		calCopyBuffer3Db(Q->current, Q->next, ca3D->rows, ca3D->columns, ca3D->slices);
	return return_state;
}
CALbyte calLoadSubstate3Di(struct CALModel3D* ca3D, struct CALSubstate3Di* Q, char* path) {
	CALbyte return_state = calLoadBuffer3Di(Q->current, ca3D->rows, ca3D->columns, ca3D->slices, path);
	if (Q->next)
		calCopyBuffer3Di(Q->current, Q->next, ca3D->rows, ca3D->columns, ca3D->slices);
	return return_state;
}
CALbyte calLoadSubstate3Dr(struct CALModel3D* ca3D, struct CALSubstate3Dr* Q, char* path) {
	CALbyte return_state = calLoadBuffer3Dr(Q->current, ca3D->rows, ca3D->columns, ca3D->slices, path);
	if (Q->next)
		calCopyBuffer3Dr(Q->current, Q->next, ca3D->rows, ca3D->columns, ca3D->slices);
	return return_state;
}


void calfSaveSubstate3Db(struct CALModel3D* ca3D, struct CALSubstate3Db* Q, FILE* f) {
	calfSaveBuffer3Db(Q->current, ca3D->rows, ca3D->columns, ca3D->slices, f);
}
void calfSaveSubstate3Di(struct CALModel3D* ca3D, struct CALSubstate3Di* Q, FILE* f) {
	calfSaveBuffer3Di(Q->current, ca3D->rows, ca3D->columns, ca3D->slices, f);
}
void calfSaveSubstate3Dr(struct CALModel3D* ca3D, struct CALSubstate3Dr* Q, FILE* f) {
	calfSaveBuffer3Dr(Q->current, ca3D->rows, ca3D->columns, ca3D->slices, f);
}


CALbyte calSaveSubstate3Db(struct CALModel3D* ca3D, struct CALSubstate3Db* Q, char* path) {
	CALbyte return_state = calSaveBuffer3Db(Q->current, ca3D->rows, ca3D->columns, ca3D->slices, path);
	return return_state;
}
CALbyte calSaveSubstate3Di(struct CALModel3D* ca3D, struct CALSubstate3Di* Q, char* path) {
	CALbyte return_state = calSaveBuffer3Di(Q->current, ca3D->rows, ca3D->columns, ca3D->slices, path);
	return return_state;
}
CALbyte calSaveSubstate3Dr(struct CALModel3D* ca3D, struct CALSubstate3Dr* Q, char* path) {
	CALbyte return_state = calSaveBuffer3Dr(Q->current, ca3D->rows, ca3D->columns, ca3D->slices, path);
	return return_state;
}
