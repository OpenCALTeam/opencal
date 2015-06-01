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

#include <cal2D.h>
#include <cal2DBuffer.h>
#include <cal2DBufferIO.h>


void calfLoadSubstate2Db(struct CALModel2D* ca2D, struct CALSubstate2Db* Q, FILE* f) {
	calfLoadMatrix2Db(Q->current, ca2D->rows, ca2D->columns, f);
	if (Q->next)
		calCopyBuffer2Db(Q->current, Q->next, ca2D->rows, ca2D->columns);
}
void calfLoadSubstate2Di(struct CALModel2D* ca2D, struct CALSubstate2Di* Q, FILE* f) {
	calfLoadMatrix2Di(Q->current, ca2D->rows, ca2D->columns, f);
	if (Q->next)
		calCopyBuffer2Di(Q->current, Q->next, ca2D->rows, ca2D->columns);
}
void calfLoadSubstate2Dr(struct CALModel2D* ca2D, struct CALSubstate2Dr* Q, FILE* f) {
	calfLoadMatrix2Dr(Q->current, ca2D->rows, ca2D->columns, f);
	if (Q->next)
		calCopyBuffer2Dr(Q->current, Q->next, ca2D->rows, ca2D->columns);
}


CALbyte calLoadSubstate2Db(struct CALModel2D* ca2D, struct CALSubstate2Db* Q, char* path) {
	CALbyte return_state = calLoadMatrix2Db(Q->current, ca2D->rows, ca2D->columns, path);
	if (Q->next)
		calCopyBuffer2Db(Q->current, Q->next, ca2D->rows, ca2D->columns);
	return return_state;
}
CALbyte calLoadSubstate2Di(struct CALModel2D* ca2D, struct CALSubstate2Di* Q, char* path) {
	CALbyte return_state = calLoadMatrix2Di(Q->current, ca2D->rows, ca2D->columns, path);
	if (Q->next)
		calCopyBuffer2Di(Q->current, Q->next, ca2D->rows, ca2D->columns);
	return return_state;
}
CALbyte calLoadSubstate2Dr(struct CALModel2D* ca2D, struct CALSubstate2Dr* Q, char* path) {
	CALbyte return_state = calLoadMatrix2Dr(Q->current, ca2D->rows, ca2D->columns, path);
	if (Q->next)
		calCopyBuffer2Dr(Q->current, Q->next, ca2D->rows, ca2D->columns);
	return return_state;
}


void calfSaveSubstate2Db(struct CALModel2D* ca2D, struct CALSubstate2Db* Q, FILE* f) {
	calfSaveMatrix2Db(Q->current, ca2D->rows, ca2D->columns, f);
}
void calfSaveSubstate2Di(struct CALModel2D* ca2D, struct CALSubstate2Di* Q, FILE* f) {
	calfSaveMatrix2Di(Q->current, ca2D->rows, ca2D->columns, f);
}
void calfSaveSubstate2Dr(struct CALModel2D* ca2D, struct CALSubstate2Dr* Q, FILE* f) {
	calfSaveMatrix2Dr(Q->current, ca2D->rows, ca2D->columns, f);
}


CALbyte calSaveSubstate2Db(struct CALModel2D* ca2D, struct CALSubstate2Db* Q, char* path) {
	CALbyte return_state = calSaveMatrix2Db(Q->current, ca2D->rows, ca2D->columns, path);
	return return_state;
}
CALbyte calSaveSubstate2Di(struct CALModel2D* ca2D, struct CALSubstate2Di* Q, char* path) {
	CALbyte return_state = calSaveMatrix2Di(Q->current, ca2D->rows, ca2D->columns, path);
	return return_state;
}
CALbyte calSaveSubstate2Dr(struct CALModel2D* ca2D, struct CALSubstate2Dr* Q, char* path) {
	CALbyte return_state = calSaveMatrix2Dr(Q->current, ca2D->rows, ca2D->columns, path);
	return return_state;
}
