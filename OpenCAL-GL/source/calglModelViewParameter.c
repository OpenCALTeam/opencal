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

#include <OpenCAL-GL/calglModelViewParameter.h>

struct CALGLModelViewParameter* calglCreateModelViewParameter(GLfloat xT, GLfloat yT, GLfloat zT, 
	GLfloat xR, GLfloat yR, GLfloat zR, 
	GLfloat xS, GLfloat yS, GLfloat zS){
		struct CALGLModelViewParameter* modelView;

		modelView = (struct CALGLModelViewParameter*) malloc(sizeof(struct CALGLModelViewParameter));

		modelView->xTranslate = xT;
		modelView->yTranslate = yT;
		modelView->zTranslate = zT;

		modelView->xRotation = xR;
		modelView->yRotation = yR;
		modelView->zRotation = zR;

		modelView->xScale = xS;
		modelView->yScale = yS;
		modelView->zScale = zS;

		return modelView;
}

struct CALGLModelViewParameter* calglAutoCreateModelViewParameterFlat2D(struct CALDrawModel2D* calDrawModel){
	struct CALGLModelViewParameter* modelView = (struct CALGLModelViewParameter*) malloc(sizeof(struct CALGLModelViewParameter));

	modelView = calglCreateModelViewParameter(-calDrawModel->calModel->columns/2.0f, -calDrawModel->calModel->rows/2.0f, 0,
		90, 0, 0,
		1/(calDrawModel->calModel->columns/10.0f), 1/(calDrawModel->calModel->rows/10.0f), 1);

	return modelView;
}
struct CALGLModelViewParameter* calglAutoCreateModelViewParameterFlat3D(struct CALDrawModel3D* calDrawModel){
	struct CALGLModelViewParameter* modelView = (struct CALGLModelViewParameter*) malloc(sizeof(struct CALGLModelViewParameter));

	modelView = calglCreateModelViewParameter(-calDrawModel->calModel->rows*(1.0f/48.0f), -calDrawModel->calModel->columns*(1.0f/48.0f), 0,
		0,0,0,
		1/(calDrawModel->calModel->columns/10.0f), -1/(calDrawModel->calModel->rows/10.0f), 1/(calDrawModel->calModel->slices/10.0f));

	return modelView;
}
struct CALGLModelViewParameter* calglAutoCreateModelViewParameterSurface2D(struct CALDrawModel2D* calDrawModel){
	struct CALGLModelViewParameter* modelView = (struct CALGLModelViewParameter*) malloc(sizeof(struct CALGLModelViewParameter));
	GLfloat max, min;
	GLfloat yTranslate = 0.0f;
	GLfloat heightDiff = 0.0f;

	if(calDrawModel->byteModel){
		calglComputeExtremesToAll2Db(calDrawModel, calDrawModel->byteModel);
		min = (GLfloat) calDrawModel->byteModel->min;
		max = (GLfloat) calDrawModel->byteModel->nodes[1]->max;
		heightDiff = max - min;
	}
	if(calDrawModel->intModel){
		calglComputeExtremesToAll2Di(calDrawModel, calDrawModel->intModel);
		min = (GLfloat) calDrawModel->intModel->min;
		max = (GLfloat) calDrawModel->intModel->nodes[1]->max;
		heightDiff = max - min;
	}
	if(calDrawModel->realModel){
		calglComputeExtremesToAll2Dr(calDrawModel, calDrawModel->realModel);
		min = (GLfloat) calDrawModel->realModel->min;
		max = (GLfloat) calDrawModel->realModel->nodes[1]->max;
		heightDiff = max - min;
	}


	if(min >= 0){
		yTranslate = -min-(heightDiff/2);
	} else if(max <= 0){
		yTranslate = max+(heightDiff/2);
	} else {
		if((-min) > max){
			yTranslate = ((-min)-max)/2.0f;
		} else if((-min) < max){
			yTranslate = -(max-(-min))/2.0f;
		}
	}

	modelView = calglCreateModelViewParameter(
		(-calDrawModel->calModel->columns/2.0f) * calglGetGlobalSettings()->cellSize, yTranslate, (-calDrawModel->calModel->rows/2.0f)* calglGetGlobalSettings()->cellSize,
		0,0,0,
		1/(calDrawModel->calModel->columns/10.0f), 1/(heightDiff/10.0f), 1/(calDrawModel->calModel->rows/10.0f));

	return modelView;
}
struct CALGLModelViewParameter* calglAutoCreateModelViewParameterSurface3D(struct CALDrawModel3D* calDrawModel){
	struct CALGLModelViewParameter* modelView = (struct CALGLModelViewParameter*) malloc(sizeof(struct CALGLModelViewParameter));

	return modelView;
}

void calglDestroyModelViewParameter(struct CALGLModelViewParameter* calModelVieParameter){
	if(calModelVieParameter != NULL){
		free(calModelVieParameter);
	}
}

void calglApplyModelViewParameter(struct CALGLModelViewParameter* calModelVieParameter){
	glRotatef(calModelVieParameter->xRotation, 1.0f, 0.0f, 0.0f);
	glRotatef(calModelVieParameter->yRotation, 0.0f, 1.0f, 0.0f);
	glRotatef(calModelVieParameter->zRotation, 0.0f, 0.0f, 1.0f);

	glScalef(calModelVieParameter->xScale, calModelVieParameter->yScale, calModelVieParameter->zScale);

	glTranslatef(calModelVieParameter->xTranslate, calModelVieParameter->yTranslate, calModelVieParameter->zTranslate);
}
