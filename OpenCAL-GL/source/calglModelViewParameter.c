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

struct CALGLModelViewParameter* calglAutoCreateModelViewParameterFlat2D(struct CALGLDrawModel2D* calDrawModel){
	struct CALGLModelViewParameter* modelView = (struct CALGLModelViewParameter*) malloc(sizeof(struct CALGLModelViewParameter));

	modelView = calglCreateModelViewParameter(-calDrawModel->calModel->columns/2.0f, -calDrawModel->calModel->rows/2.0f, 0,
		90, 0, 0,
		1/(calDrawModel->calModel->columns/10.0f), 1/(calDrawModel->calModel->rows/10.0f), 1);

	return modelView;
}
struct CALGLModelViewParameter* calglAutoCreateModelViewParameterFlat3D(struct CALGLDrawModel3D* calDrawModel){
	struct CALGLModelViewParameter* modelView = (struct CALGLModelViewParameter*) malloc(sizeof(struct CALGLModelViewParameter));
	int rows = calDrawModel->calModel->rows;
	int columns = calDrawModel->calModel->columns;
	int slices = calDrawModel->calModel->slices;

	modelView = calglCreateModelViewParameter (
		0.0f, 0.0f, 0.0f,
		0, 0, 0,
		1/(columns/5.0f), -1/(rows/5.0f), 1/(slices/5.0f));


	return modelView;
}
struct CALGLModelViewParameter* calglAutoCreateModelViewParameterSurface2D(struct CALGLDrawModel2D* calDrawModel){
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

	//modelView = calglCreateModelViewParameter(
	//	(-calDrawModel->calModel->columns/2.0f) * calglGetGlobalSettings()->cellSize, yTranslate, (-calDrawModel->calModel->rows/2.0f)* calglGetGlobalSettings()->cellSize,
	//	0,0,0,
	//	1/(calDrawModel->calModel->columns/10.0f), 1/(heightDiff/10.0f), 1/(calDrawModel->calModel->rows/10.0f));

	modelView = calglCreateModelViewParameter(
		(-calDrawModel->calModel->columns/2.0f) * calglGetGlobalSettings()->cellSize, yTranslate, (-calDrawModel->calModel->rows/2.0f)* calglGetGlobalSettings()->cellSize,
		0, 0, 0,
		1/(calDrawModel->calModel->columns/1.0f), 1/(heightDiff/1.0f), 1/(calDrawModel->calModel->rows/1.0f));


	return modelView;
}
struct CALGLModelViewParameter* calglAutoCreateModelViewParameterSurface3D(struct CALGLDrawModel3D* calDrawModel){
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
