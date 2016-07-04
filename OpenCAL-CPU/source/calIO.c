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

#include <OpenCAL-CPU/calModel.h>
#include <OpenCAL-CPU/calBuffer.h>
#include <OpenCAL-CPU/calBufferIO.h>


void calfLoadSubstate_b(struct CALModel* calModel, struct CALSubstate_b* Q, FILE* f) {
    calfLoadMatrix_b(Q->current, calModel->rows, calModel->columns, f);
    if (Q->next)
        calCopyBuffer_b(Q->current, Q->next, calModel->cellularSpaceDimension);
}
void calfLoadSubstate_i(struct CALModel* calModel, struct CALSubstate_i* Q, FILE* f) {
    calfLoadMatrix_i(Q->current, calModel->rows, calModel->columns, f);
    if (Q->next)
        calCopyBuffer_i(Q->current, Q->next, calModel->cellularSpaceDimension);
}
void calfLoadSubstate_r(struct CALModel* calModel, struct CALSubstate_r* Q, FILE* f) {
    calfLoadMatrix_r(Q->current, calModel->rows, calModel->columns, f);
    if (Q->next)
        calCopyBuffer_r(Q->current, Q->next, calModel->cellularSpaceDimension);
}


CALbyte calLoadSubstate_b(struct CALModel* calModel, struct CALSubstate_b* Q, char* path) {
    CALbyte return_state = calLoadMatrix_b(Q->current, calModel->rows, calModel->columns, path);
    if (Q->next)
        calCopyBuffer_b(Q->current, Q->next, calModel->cellularSpaceDimension);
    return return_state;
}
CALbyte calLoadSubstate_i(struct CALModel* calModel, struct CALSubstate_i* Q, char* path) {
    CALbyte return_state = calLoadMatrix_i(Q->current, calModel->rows, calModel->columns, path);
    if (Q->next)
        calCopyBuffer_i(Q->current, Q->next, calModel->cellularSpaceDimension);
    return return_state;
}
CALbyte calLoadSubstate_r(struct CALModel* calModel, struct CALSubstate_r* Q, char* path) {
    CALbyte return_state = calLoadMatrix_r(Q->current, calModel->rows, calModel->columns, path);
    if (Q->next)
        calCopyBuffer_r(Q->current, Q->next, calModel->cellularSpaceDimension);
    return return_state;
}


void calfSaveSubstate_b(struct CALModel* calModel, struct CALSubstate_b* Q, FILE* f) {
    calfSaveMatrix_b(Q->current, calModel->rows, calModel->columns, f);
}
void calfSaveSubstate_i(struct CALModel* calModel, struct CALSubstate_i* Q, FILE* f) {
    calfSaveMatrix_i(Q->current, calModel->rows, calModel->columns, f);
}
void calfSaveSubstate_r(struct CALModel* calModel, struct CALSubstate_r* Q, FILE* f) {
    calfSaveMatrix_r(Q->current, calModel->rows, calModel->columns, f);
}


CALbyte calSaveSubstate_b(struct CALModel* calModel, struct CALSubstate_b* Q, char* path) {
    CALbyte return_state = calSaveMatrix_b(Q->current, calModel->rows, calModel->columns, path);
    return return_state;
}
CALbyte calSaveSubstate_i(struct CALModel* calModel, struct CALSubstate_i* Q, char* path) {
    CALbyte return_state = calSaveMatrix_i(Q->current, calModel->rows, calModel->columns, path);
    return return_state;
}
CALbyte calSaveSubstate_r(struct CALModel* calModel, struct CALSubstate_r* Q, char* path) {
    CALbyte return_state = calSaveMatrix_r(Q->current, calModel->rows, calModel->columns, path);
    return return_state;
}

