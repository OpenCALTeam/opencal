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

#include <stdio.h>
#include <stdlib.h>

#include <OpenCAL-GL/calgl2D.h>
#include <OpenCAL-GL/calgl2DWindow.h>
#include <OpenCAL-GL/calglUtils.h>

static GLuint componentColor[3] = {0};
static GLint currentIndex = 0;

struct CALDrawModel2D* calglDefDrawModel2D(enum CALGL_DRAW_MODE mode, const char* name, struct CALModel2D* calModel, struct CALRun2D* calRun){
	struct CALDrawModel2D* drawModel = (struct CALDrawModel2D*) malloc(sizeof(struct CALDrawModel2D));

	drawModel->drawMode = mode;
	drawModel->name = name;

	drawModel->calModel = calModel;

	drawModel->byteModel = NULL;
	drawModel->intModel = NULL;
	drawModel->realModel = NULL;
	drawModel->modelView = NULL;
	drawModel->modelLight = NULL;

	drawModel->redComponent = 1.0f;
	drawModel->greenComponent = 1.0f;
	drawModel->blueComponent = 1.0f;
	drawModel->alphaComponent = 1.0f;

	drawModel->drawICells = (GLshort*)malloc(sizeof(GLshort)*calModel->rows);
	drawModel->drawJCells = (GLshort*)malloc(sizeof(GLshort)*calModel->columns);

	calglDisplayDrawIBound2D(drawModel, 0, calModel->rows);
	calglDisplayDrawJBound2D(drawModel, 0, calModel->columns);
	
	drawModel->calUpdater = calglCreateUpdater2D(calRun);	
	drawModel->infoBar = NULL;

	calglShowModel2D(drawModel);

	return drawModel;
}

void calglDestoyDrawModel2D(struct CALDrawModel2D* drawModel){
	if(drawModel){
		if(drawModel->byteModel)
			calglDestroyNode2Db(drawModel->byteModel);
		if(drawModel->intModel)
			calglDestroyNode2Di(drawModel->intModel);
		if(drawModel->realModel)
			calglDestroyNode2Dr(drawModel->realModel);
		if (drawModel->drawICells)
			free(drawModel->drawICells);
		if (drawModel->drawJCells)
			free(drawModel->drawJCells);
		calglDestroyModelViewParameter(drawModel->modelView);
		calglDestroyLightParameter(drawModel->modelLight);
		calglDestroyUpdater2D(drawModel->calUpdater);
		free(drawModel);
	}	
}

#pragma region AddData
void calglAddToDrawModel2Db(struct CALDrawModel2D* drawModel, struct CALSubstate2Db* substateFather, struct CALSubstate2Db** substateToAdd, enum CALGL_TYPE_INFO typeInfo, enum CALGL_TYPE_INFO_USE typeInfoUseSubstate, enum CALGL_DATA_TYPE dataType){
	struct CALNode2Db* toReturn = NULL;
	struct CALNode2Db* nodeFather = NULL;

	// Add in treeModel for drawing
	if(substateFather == NULL){ // First Add
		toReturn = calglCreateNode2Db(NULL);
		toReturn->substate = *substateToAdd;
		toReturn->typeInfoSubstate = typeInfo;
		toReturn->typeInfoUseSubstate = typeInfoUseSubstate;
		drawModel->byteModel = toReturn;		
	} else {
		calglSearchSubstateDrawModel2Db(drawModel->byteModel, substateFather, &nodeFather);
		toReturn = calglAddDataNode2Db(nodeFather, *substateToAdd, typeInfo, typeInfoUseSubstate, dataType);
	}

	if(typeInfoUseSubstate == CALGL_TYPE_INFO_USE_CONST_VALUE){
		toReturn->redComponent = drawModel->redComponent;
		toReturn->greenComponent = drawModel->blueComponent;
		toReturn->blueComponent = drawModel->greenComponent;
		toReturn->alphaComponent = drawModel->alphaComponent;
	}
}
void calglAddToDrawModel2Di(struct CALDrawModel2D* drawModel, struct CALSubstate2Di* substateFather, struct CALSubstate2Di** substateToAdd, enum CALGL_TYPE_INFO typeInfo, enum CALGL_TYPE_INFO_USE typeInfoUseSubstate, enum CALGL_DATA_TYPE dataType){
	struct CALNode2Di* toReturn = NULL;
	struct CALNode2Di* nodeFather = NULL;

	// Add in treeModel for drawing
	if(substateFather == NULL){ // First Add
		toReturn = calglCreateNode2Di(NULL);
		toReturn->substate = *substateToAdd;
		toReturn->typeInfoSubstate = typeInfo;
		toReturn->typeInfoUseSubstate = typeInfoUseSubstate;
		drawModel->intModel = toReturn;		
	} else {
		calglSearchSubstateDrawModel2Di(drawModel->intModel, substateFather, &nodeFather);
		toReturn = calglAddDataNode2Di(nodeFather, *substateToAdd, typeInfo, typeInfoUseSubstate, dataType);
	}

	if(typeInfoUseSubstate == CALGL_TYPE_INFO_USE_CONST_VALUE){
		toReturn->redComponent = drawModel->redComponent;
		toReturn->greenComponent = drawModel->blueComponent;
		toReturn->blueComponent = drawModel->greenComponent;
		toReturn->alphaComponent = drawModel->alphaComponent;
	}
}
void calglAddToDrawModel2Dr(struct CALDrawModel2D* drawModel, struct CALSubstate2Dr* substateFather, struct CALSubstate2Dr** substateToAdd, enum CALGL_TYPE_INFO typeInfo, enum CALGL_TYPE_INFO_USE typeInfoUseSubstate, enum CALGL_DATA_TYPE dataType){
	struct CALNode2Dr* toReturn = NULL;
	struct CALNode2Dr* nodeFather = NULL;
	int i=0;

	// Add in treeModel for drawing
	if(substateFather == NULL){ // First Add
		toReturn = calglCreateNode2Dr(NULL);
		toReturn->substate = *substateToAdd;
		toReturn->typeInfoSubstate = typeInfo;
		toReturn->typeInfoUseSubstate = typeInfoUseSubstate;
		drawModel->realModel = toReturn;		
	} else {
		calglSearchSubstateDrawModel2Dr(drawModel->realModel, substateFather, &nodeFather);
		toReturn = calglAddDataNode2Dr(nodeFather, *substateToAdd, typeInfo, typeInfoUseSubstate, dataType);
	}

	if(typeInfoUseSubstate == CALGL_TYPE_INFO_USE_CONST_VALUE){
		toReturn->redComponent = drawModel->redComponent;
		toReturn->greenComponent = drawModel->blueComponent;
		toReturn->blueComponent = drawModel->greenComponent;
		toReturn->alphaComponent = drawModel->alphaComponent;
	}
}
#pragma endregion

#pragma region SearchSubstate
void calglSearchSubstateDrawModel2Db(struct CALNode2Db* currentNode, struct CALSubstate2Db* substateToSearch, struct CALNode2Db** nodeSearched){
	int i;

	if(currentNode == NULL){
		return;
	}

	if(currentNode->substate == substateToSearch){
		*nodeSearched = currentNode;
		return;
	} 

	for(i=1; i<currentNode->insertedNode && *nodeSearched==NULL; i++){
		if(currentNode->nodes[i] != NULL)
			calglSearchSubstateDrawModel2Db(currentNode->nodes[i], substateToSearch, nodeSearched);
	}
}
void calglSearchSubstateDrawModel2Di(struct CALNode2Di* currentNode, struct CALSubstate2Di* substateToSearch, struct CALNode2Di** nodeSearched){
	int i;

	if(currentNode == NULL){
		return;
	}

	if(currentNode->substate == substateToSearch){
		*nodeSearched = currentNode;
		return;
	} 

	for(i=1; i<currentNode->insertedNode && *nodeSearched==NULL; i++){
		if(currentNode->nodes[i] != NULL)
			calglSearchSubstateDrawModel2Di(currentNode->nodes[i], substateToSearch, nodeSearched);
	}
}
void calglSearchSubstateDrawModel2Dr(struct CALNode2Dr* currentNode, struct CALSubstate2Dr* substateToSearch, struct CALNode2Dr** nodeSearched){
	int i;

	if(currentNode == NULL){
		return;
	}

	if(currentNode->substate == substateToSearch){
		*nodeSearched = currentNode;
		return;
	} 

	for(i=1; i<currentNode->insertedNode && *nodeSearched==NULL; i++){
		if(currentNode->nodes[i] != NULL)
			calglSearchSubstateDrawModel2Dr(currentNode->nodes[i], substateToSearch, nodeSearched);
	}
}
#pragma endregion

void calglDisplayModel2D(struct CALDrawModel2D* calDrawModel){
	switch(calDrawModel->drawMode){
	case CALGL_DRAW_MODE_NO_DRAW: 
		break;

	case CALGL_DRAW_MODE_FLAT: 
		calglDrawDiscreetModel2D(calDrawModel);
		break;

	case CALGL_DRAW_MODE_SURFACE: 
		calglDrawRealModel2D(calDrawModel);
		break;
	}
}

#pragma region DrawDiscreetModel2D
void calglDrawDiscreetModel2D(struct CALDrawModel2D* calDrawModel){
	glPushMatrix();{
		glPushAttrib(GL_LIGHTING_BIT); {
			// Apply Light
			if(calDrawModel->modelLight){
				calglApplyLightParameter(calDrawModel->modelLight);
			} else if(calglAreLightsEnable()){
				calDrawModel->modelLight = calglCreateLightParameter(calglGetPositionLight(), calglGetAmbientLight(), calglGetDiffuseLight(), calglGetSpecularLight(), 1, NULL, 0.0f);
			}
			// Apply model view transformation
			if(calDrawModel->modelView){
				calglApplyModelViewParameter(calDrawModel->modelView);
			} else {
				calglSetModelViewParameter2D(calDrawModel, calglAutoCreateModelViewParameterFlat2D(calDrawModel));
				calglApplyModelViewParameter(calDrawModel->modelView);
			}
			// Draw model
			if(calDrawModel->byteModel){
				calglComputeExtremesToAll2Db(calDrawModel, calDrawModel->byteModel);
				calglDrawDiscreetModelDisplayNode2Db(calDrawModel, calDrawModel->byteModel);
			}
			if(calDrawModel->intModel){
				calglComputeExtremesToAll2Di(calDrawModel, calDrawModel->intModel);
				calglDrawDiscreetModelDisplayNode2Di(calDrawModel, calDrawModel->intModel);
			}
			if(calDrawModel->realModel){
				calglComputeExtremesToAll2Dr(calDrawModel, calDrawModel->realModel);
				calglDrawDiscreetModelDisplayNode2Dr(calDrawModel, calDrawModel->realModel);
			}
		}	glPopAttrib();
		// Draw BoundingSquare
		calglDrawBoundingSquare2D(calDrawModel);
	}	glPopMatrix();
}

void calglDrawDiscreetModelDisplayNode2Db(struct CALDrawModel2D* calDrawModel, struct CALNode2Db* calNode){
	int i;
	for(i=1; i < calNode->insertedNode; i++){
		calglDrawDiscreetModelDisplayNode2Db(calDrawModel, calNode->nodes[i]);
	}
	calglDrawDiscreetModelDisplayCurrentNode2Db(calDrawModel, calNode);
}
void calglDrawDiscreetModelDisplayNode2Di(struct CALDrawModel2D* calDrawModel, struct CALNode2Di* calNode){
	int i;
	for(i=1; i < calNode->insertedNode; i++){
		calglDrawDiscreetModelDisplayNode2Di(calDrawModel, calNode->nodes[i]);
	}
	calglDrawDiscreetModelDisplayCurrentNode2Di(calDrawModel, calNode);
}
void calglDrawDiscreetModelDisplayNode2Dr(struct CALDrawModel2D* calDrawModel, struct CALNode2Dr* calNode){
	int i;
	for(i=1; i < calNode->insertedNode; i++){
		calglDrawDiscreetModelDisplayNode2Dr(calDrawModel, calNode->nodes[i]);
	}
	calglDrawDiscreetModelDisplayCurrentNode2Dr(calDrawModel, calNode);
}

void calglDrawDiscreetModelDisplayCurrentNode2Db(struct CALDrawModel2D* calDrawModel, struct CALNode2Db* calNode){
	int i, j;
	int rows = calDrawModel->calModel->rows;
	int columns = calDrawModel->calModel->columns;

	// If no vertex data jump to next node
	if(calNode->typeInfoSubstate != CALGL_TYPE_INFO_VERTEX_DATA)
		return;

	// Check for static data -> use display list here
	if(calNode->dataType == CALGL_DATA_TYPE_STATIC){
		if(calNode->callList){
			glCallList(*calNode->callList);
		} else {
			*calNode->callList = glGenLists(1);
			glNewList(*calNode->callList, GL_COMPILE);{
				for(i=0; i < rows; i++){		
					for(j=0; j < columns; j++){
						glBegin(GL_QUADS); {
							if(calglSetColorData2Db(calDrawModel, calNode, i, j) == CAL_TRUE){
								calglSetNormalData2Db(calDrawModel, calNode, i, j);

								glVertex2i(j, i);
								glVertex2i(j, i+1);
								glVertex2i(j+1, i+1);
								glVertex2i(j+1, i);
							}
						} glEnd();
					}		
				}
			} glEndList();
		}
	} else {
		for(i=0; i < rows; i++){		
			for(j=0; j < columns; j++){
				glBegin(GL_QUADS); {
					if(calglSetColorData2Db(calDrawModel, calNode, i, j) == CAL_TRUE){
						calglSetNormalData2Db(calDrawModel, calNode, i, j);

						glVertex2i(j, i);
						glVertex2i(j, i+1);
						glVertex2i(j+1, i+1);
						glVertex2i(j+1, i);
					}
				} glEnd();
			}		
		}
	}
}
void calglDrawDiscreetModelDisplayCurrentNode2Di(struct CALDrawModel2D* calDrawModel, struct CALNode2Di* calNode){
	int i, j;
	int rows = calDrawModel->calModel->rows;
	int columns = calDrawModel->calModel->columns;

	// If no vertex data jump to next node
	if(calNode->typeInfoSubstate != CALGL_TYPE_INFO_VERTEX_DATA)
		return;

	// Check for static data -> use display list here
	if(calNode->dataType == CALGL_DATA_TYPE_STATIC){
		glCallList(*calNode->callList);
		if(calNode->callList){
			glCallList(*calNode->callList);
		} else {
			*calNode->callList = glGenLists(1);
			glNewList(*calNode->callList, GL_COMPILE);{
				for(i=0; i < rows; i++){		
					for(j=0; j < columns; j++){
						glBegin(GL_QUADS); {
							if(calglSetColorData2Di(calDrawModel, calNode, i, j) == CAL_TRUE){
								calglSetNormalData2Di(calDrawModel, calNode, i, j);

								glVertex2i(j, i);
								glVertex2i(j, i+1);
								glVertex2i(j+1, i+1);
								glVertex2i(j+1, i);		
							}
						} glEnd();
					}		
				}
			} glEndList();
		}
	} else {
		for(i=0; i < rows; i++){		
			for(j=0; j < columns; j++){
				glBegin(GL_QUADS); {
					if(calglSetColorData2Di(calDrawModel, calNode, i, j) == CAL_TRUE){
						calglSetNormalData2Di(calDrawModel, calNode, i, j);

						glVertex2i(j, i);
						glVertex2i(j, i+1);
						glVertex2i(j+1, i+1);
						glVertex2i(j+1, i);								
					}
				} glEnd();
			}		
		}
	}
}
void calglDrawDiscreetModelDisplayCurrentNode2Dr(struct CALDrawModel2D* calDrawModel, struct CALNode2Dr* calNode){
	int i, j;
	int rows = calDrawModel->calModel->rows;
	int columns = calDrawModel->calModel->columns;

	// If no vertex data jump to next node
	if(calNode->typeInfoSubstate != CALGL_TYPE_INFO_VERTEX_DATA)
		return;

	// Check for static data -> use display list here
	if(calNode->dataType == CALGL_DATA_TYPE_STATIC){
		glCallList(*calNode->callList);
		if(calNode->callList){
			glCallList(*calNode->callList);
		} else {
			*calNode->callList = glGenLists(1);
			glNewList(*calNode->callList, GL_COMPILE);{
				for(i=0; i < rows; i++){		
					for(j=0; j < columns; j++){
						glBegin(GL_QUADS); {
							if(calglSetColorData2Dr(calDrawModel, calNode, i, j) == CAL_TRUE){
								calglSetNormalData2Dr(calDrawModel, calNode, i, j);

								glVertex2i(j, i);
								glVertex2i(j, i+1);
								glVertex2i(j+1, i+1);
								glVertex2i(j+1, i);							
							}
						} glEnd();
					}		
				}
			} glEndList();
		}
	} else {
		for(i=0; i < rows; i++){		
			for(j=0; j < columns; j++){
				glBegin(GL_QUADS); {
					if(calglSetColorData2Dr(calDrawModel, calNode, i, j) == CAL_TRUE){
						calglSetNormalData2Dr(calDrawModel, calNode, i, j);

						glVertex2i(j, i);
						glVertex2i(j, i+1);
						glVertex2i(j+1, i+1);
						glVertex2i(j+1, i);
					}
				} glEnd();
			}		
		}
	}
}
#pragma endregion

#pragma region DrawRealModel2D
void calglDrawRealModel2D(struct CALDrawModel2D* calDrawModel){
	GLfloat max, min;

	glPushMatrix();{
		glPushAttrib(GL_LIGHTING_BIT); {
			// Apply Light
			if(calDrawModel->modelLight){
				calglApplyLightParameter(calDrawModel->modelLight);
			} else if(calglAreLightsEnable()){
				calDrawModel->modelLight = calglCreateLightParameter(calglGetPositionLight(), calglGetAmbientLight(), calglGetDiffuseLight(), calglGetSpecularLight(), 1, NULL, 0.0f);
			}
			// Apply model view transformation
			if(calDrawModel->modelView){
				calglApplyModelViewParameter(calDrawModel->modelView);
			} else {
				calglSetModelViewParameter2D(calDrawModel, calglAutoCreateModelViewParameterSurface2D(calDrawModel));
				calglApplyModelViewParameter(calDrawModel->modelView);
			}
			// Draw model
			if(calDrawModel->byteModel){
				calglComputeExtremesToAll2Db(calDrawModel, calDrawModel->byteModel);
				max = (GLfloat) calDrawModel->byteModel->max;
				min = (GLfloat) calDrawModel->byteModel->min;
				calglDrawRealModelDisplayNode2Db(calDrawModel, calDrawModel->byteModel);
			}
			if(calDrawModel->intModel){
				calglComputeExtremesToAll2Di(calDrawModel, calDrawModel->intModel);
				max = (GLfloat) calDrawModel->intModel->max;
				min = (GLfloat) calDrawModel->intModel->min;
				calglDrawRealModelDisplayNode2Di(calDrawModel, calDrawModel->intModel);
			}
			if(calDrawModel->realModel){
				calglComputeExtremesToAll2Dr(calDrawModel, calDrawModel->realModel);
				max = (GLfloat) calDrawModel->realModel->max;
				min = (GLfloat) calDrawModel->realModel->min;
				calglDrawRealModelDisplayNode2Dr(calDrawModel, calDrawModel->realModel);
			}
			// Draw BoundingBox
			calglDrawBoundingBox2D(calDrawModel, max, min);
		}	glPopAttrib();
	}	glPopMatrix();
}

void calglDrawRealModelDisplayNode2Db(struct CALDrawModel2D* calDrawModel, struct CALNode2Db* calNode){
	int i;
	calglDrawRealModelDisplayCurrentNode2Db(calDrawModel, calNode);
	for(i=1; i < calNode->insertedNode; i++){
		calglDrawRealModelDisplayNode2Db(calDrawModel, calNode->nodes[i]);
	}
}
void calglDrawRealModelDisplayNode2Di(struct CALDrawModel2D* calDrawModel, struct CALNode2Di* calNode){
	int i;
	calglDrawRealModelDisplayCurrentNode2Di(calDrawModel, calNode);
	for(i=1; i < calNode->insertedNode; i++){
		calglDrawRealModelDisplayNode2Di(calDrawModel, calNode->nodes[i]);
	}
}
void calglDrawRealModelDisplayNode2Dr(struct CALDrawModel2D* calDrawModel, struct CALNode2Dr* calNode){
	int i;
	calglDrawRealModelDisplayCurrentNode2Dr(calDrawModel, calNode);
	for(i=1; i < calNode->insertedNode; i++){
		calglDrawRealModelDisplayNode2Dr(calDrawModel, calNode->nodes[i]);
	}
}

void calglDrawRealModelDisplayCurrentNode2Db(struct CALDrawModel2D* calDrawModel, struct CALNode2Db* calNode){
	int i, j;
	int rows = calDrawModel->calModel->rows;
	int columns = calDrawModel->calModel->columns;

	// If no vertex data jump to next node
	if(calNode->typeInfoSubstate != CALGL_TYPE_INFO_VERTEX_DATA)
		return;

	// Check for static data -> use display list here
	if(calNode->dataType == CALGL_DATA_TYPE_STATIC){
		if(calNode->callList){
			glCallList(*calNode->callList);
		} else {
			*calNode->callList = glGenLists(1);
			glNewList(*calNode->callList, GL_COMPILE);{
				for (i = 0; i < rows - 1; i++){
					if (!calDrawModel->drawICells[i])
						continue;
					for (j = 0; j < columns - 1; j++){
						if (!calDrawModel->drawJCells[j])
							continue;
						glBegin(GL_TRIANGLES); {
							calglSetNormalData2Db(calDrawModel, calNode, i, j);
							calglSetColorData2Db(calDrawModel, calNode, i, j);
							calglDrawRealModelDisplayCurrentNodeSetVertexData2Db(calDrawModel, calNode, i, j);
							calglSetColorData2Db(calDrawModel, calNode, i+1, j);
							calglDrawRealModelDisplayCurrentNodeSetVertexData2Db(calDrawModel, calNode, i+1, j);
							calglSetColorData2Db(calDrawModel, calNode, i+1, j+1);
							calglDrawRealModelDisplayCurrentNodeSetVertexData2Db(calDrawModel, calNode, i+1, j+1);

							calglSetNormalData2Db(calDrawModel, calNode, i, j);
							calglSetColorData2Db(calDrawModel, calNode, i, j);
							calglDrawRealModelDisplayCurrentNodeSetVertexData2Db(calDrawModel, calNode, i, j);
							calglSetColorData2Db(calDrawModel, calNode, i+1, j+1);
							calglDrawRealModelDisplayCurrentNodeSetVertexData2Db(calDrawModel, calNode, i+1, j+1);
							calglSetColorData2Db(calDrawModel, calNode, i, j+1);
							calglDrawRealModelDisplayCurrentNodeSetVertexData2Db(calDrawModel, calNode, i, j+1);	
						} glEnd();
					}					
				}
			} glEndList();
		}
	} else {
		for (i = 0; i < rows - 1; i++){
			if (!calDrawModel->drawICells[i])
				continue;
			for (j = 0; j < columns - 1; j++){
				if (!calDrawModel->drawJCells[j])
					continue;
				glBegin(GL_TRIANGLES); {
					calglSetNormalData2Db(calDrawModel, calNode, i, j);
					calglSetColorData2Db(calDrawModel, calNode, i, j);
					calglDrawRealModelDisplayCurrentNodeSetVertexData2Db(calDrawModel, calNode, i, j);
					calglSetColorData2Db(calDrawModel, calNode, i+1, j);
					calglDrawRealModelDisplayCurrentNodeSetVertexData2Db(calDrawModel, calNode, i+1, j);
					calglSetColorData2Db(calDrawModel, calNode, i+1, j+1);
					calglDrawRealModelDisplayCurrentNodeSetVertexData2Db(calDrawModel, calNode, i+1, j+1);

					calglSetNormalData2Db(calDrawModel, calNode, i, j);
					calglSetColorData2Db(calDrawModel, calNode, i, j);
					calglDrawRealModelDisplayCurrentNodeSetVertexData2Db(calDrawModel, calNode, i, j);
					calglSetColorData2Db(calDrawModel, calNode, i+1, j+1);
					calglDrawRealModelDisplayCurrentNodeSetVertexData2Db(calDrawModel, calNode, i+1, j+1);
					calglSetColorData2Db(calDrawModel, calNode, i, j+1);
					calglDrawRealModelDisplayCurrentNodeSetVertexData2Db(calDrawModel, calNode, i, j+1);	
				} glEnd();
			}					
		}
	}
}
void calglDrawRealModelDisplayCurrentNode2Di(struct CALDrawModel2D* calDrawModel, struct CALNode2Di* calNode){
	int i, j;
	int rows = calDrawModel->calModel->rows;
	int columns = calDrawModel->calModel->columns;

	// If no vertex data jump to next node
	if(calNode->typeInfoSubstate != CALGL_TYPE_INFO_VERTEX_DATA)
		return;

	// Check for static data -> use display list here
	if(calNode->dataType == CALGL_DATA_TYPE_STATIC){
		if(calNode->callList){
			glCallList(*calNode->callList);
		} else {
			*calNode->callList = glGenLists(1);
			glNewList(*calNode->callList, GL_COMPILE);{
				for (i = 0; i < rows - 1; i++){
					if (!calDrawModel->drawICells[i])
						continue;
					for (j = 0; j < columns - 1; j++){
						if (!calDrawModel->drawJCells[j])
							continue;
						glBegin(GL_TRIANGLES); {
							calglSetNormalData2Di(calDrawModel, calNode, i, j);
							calglSetColorData2Di(calDrawModel, calNode, i, j);
							calglDrawRealModelDisplayCurrentNodeSetVertexData2Di(calDrawModel, calNode, i, j);
							calglSetColorData2Di(calDrawModel, calNode, i+1, j);
							calglDrawRealModelDisplayCurrentNodeSetVertexData2Di(calDrawModel, calNode, i+1, j);
							calglSetColorData2Di(calDrawModel, calNode, i+1, j+1);
							calglDrawRealModelDisplayCurrentNodeSetVertexData2Di(calDrawModel, calNode, i+1, j+1);

							calglSetNormalData2Di(calDrawModel, calNode, i, j);
							calglSetColorData2Di(calDrawModel, calNode, i, j);
							calglDrawRealModelDisplayCurrentNodeSetVertexData2Di(calDrawModel, calNode, i, j);
							calglSetColorData2Di(calDrawModel, calNode, i+1, j+1);
							calglDrawRealModelDisplayCurrentNodeSetVertexData2Di(calDrawModel, calNode, i+1, j+1);
							calglSetColorData2Di(calDrawModel, calNode, i, j+1);
							calglDrawRealModelDisplayCurrentNodeSetVertexData2Di(calDrawModel, calNode, i, j+1);	
						} glEnd();
					}					
				}
			} glEndList();
		}
	} else {
		for (i = 0; i < rows - 1; i++){
			if (!calDrawModel->drawICells[i])
				continue;
			for (j = 0; j < columns - 1; j++){
				if (!calDrawModel->drawJCells[j])
					continue;
				glBegin(GL_TRIANGLES); {
					calglSetNormalData2Di(calDrawModel, calNode, i, j);
					calglSetColorData2Di(calDrawModel, calNode, i, j);
					calglDrawRealModelDisplayCurrentNodeSetVertexData2Di(calDrawModel, calNode, i, j);
					calglSetColorData2Di(calDrawModel, calNode, i+1, j);
					calglDrawRealModelDisplayCurrentNodeSetVertexData2Di(calDrawModel, calNode, i+1, j);
					calglSetColorData2Di(calDrawModel, calNode, i+1, j+1);
					calglDrawRealModelDisplayCurrentNodeSetVertexData2Di(calDrawModel, calNode, i+1, j+1);

					calglSetNormalData2Di(calDrawModel, calNode, i, j);
					calglSetColorData2Di(calDrawModel, calNode, i, j);
					calglDrawRealModelDisplayCurrentNodeSetVertexData2Di(calDrawModel, calNode, i, j);
					calglSetColorData2Di(calDrawModel, calNode, i+1, j+1);
					calglDrawRealModelDisplayCurrentNodeSetVertexData2Di(calDrawModel, calNode, i+1, j+1);
					calglSetColorData2Di(calDrawModel, calNode, i, j+1);
					calglDrawRealModelDisplayCurrentNodeSetVertexData2Di(calDrawModel, calNode, i, j+1);	
				} glEnd();
			}					
		}
	}
}
void calglDrawRealModelDisplayCurrentNode2Dr(struct CALDrawModel2D* calDrawModel, struct CALNode2Dr* calNode){
	int i, j;
	int rows = calDrawModel->calModel->rows;
	int columns = calDrawModel->calModel->columns;

	// If no vertex data jump to next node
	if(calNode->typeInfoSubstate != CALGL_TYPE_INFO_VERTEX_DATA)
		return;

	// Check for static data -> use display list here
	if(calNode->dataType == CALGL_DATA_TYPE_STATIC){
		if(calNode->callList){
			glCallList(*calNode->callList);
		} else {
			*calNode->callList = glGenLists(1);
			glNewList(*calNode->callList, GL_COMPILE);{
				for (i = 0; i < rows - 1; i++){
					if (!calDrawModel->drawICells[i])
						continue;
					for (j = 0; j < columns - 1; j++){
						if (!calDrawModel->drawJCells[j])
							continue;
						glBegin(GL_TRIANGLES); {
							calglSetNormalData2Dr(calDrawModel, calNode, i, j);
							calglSetColorData2Dr(calDrawModel, calNode, i, j);
							calglDrawRealModelDisplayCurrentNodeSetVertexData2Dr(calDrawModel, calNode, i, j);
							calglSetColorData2Dr(calDrawModel, calNode, i+1, j);
							calglDrawRealModelDisplayCurrentNodeSetVertexData2Dr(calDrawModel, calNode, i+1, j);
							calglSetColorData2Dr(calDrawModel, calNode, i+1, j+1);
							calglDrawRealModelDisplayCurrentNodeSetVertexData2Dr(calDrawModel, calNode, i+1, j+1);

							calglSetNormalData2Dr(calDrawModel, calNode, i, j);
							calglSetColorData2Dr(calDrawModel, calNode, i, j);
							calglDrawRealModelDisplayCurrentNodeSetVertexData2Dr(calDrawModel, calNode, i, j);
							calglSetColorData2Dr(calDrawModel, calNode, i+1, j+1);
							calglDrawRealModelDisplayCurrentNodeSetVertexData2Dr(calDrawModel, calNode, i+1, j+1);
							calglSetColorData2Dr(calDrawModel, calNode, i, j+1);
							calglDrawRealModelDisplayCurrentNodeSetVertexData2Dr(calDrawModel, calNode, i, j+1);	
						} glEnd();
					}					
				}
			} glEndList();
		}
	} else {
		for (i = 0; i < rows - 1; i++){
			if (!calDrawModel->drawICells[i])
				continue;
			for (j = 0; j < columns - 1; j++){
				if (!calDrawModel->drawJCells[j])
					continue;
				glBegin(GL_TRIANGLES); {
					calglSetNormalData2Dr(calDrawModel, calNode, i, j);
					calglSetColorData2Dr(calDrawModel, calNode, i, j);
					calglDrawRealModelDisplayCurrentNodeSetVertexData2Dr(calDrawModel, calNode, i, j);
					calglSetColorData2Dr(calDrawModel, calNode, i+1, j);
					calglDrawRealModelDisplayCurrentNodeSetVertexData2Dr(calDrawModel, calNode, i+1, j);
					calglSetColorData2Dr(calDrawModel, calNode, i+1, j+1);
					calglDrawRealModelDisplayCurrentNodeSetVertexData2Dr(calDrawModel, calNode, i+1, j+1);

					calglSetNormalData2Dr(calDrawModel, calNode, i, j);
					calglSetColorData2Dr(calDrawModel, calNode, i, j);
					calglDrawRealModelDisplayCurrentNodeSetVertexData2Dr(calDrawModel, calNode, i, j);
					calglSetColorData2Dr(calDrawModel, calNode, i+1, j+1);
					calglDrawRealModelDisplayCurrentNodeSetVertexData2Dr(calDrawModel, calNode, i+1, j+1);
					calglSetColorData2Dr(calDrawModel, calNode, i, j+1);
					calglDrawRealModelDisplayCurrentNodeSetVertexData2Dr(calDrawModel, calNode, i, j+1);	
				} glEnd();
			}					
		}
	}
}

void calglDrawRealModelDisplayCurrentNodeSetVertexData2Db(struct CALDrawModel2D* calDrawModel, struct CALNode2Db*calNode, GLint i, GLint j){
	struct CALNode2Db* father = NULL;
	GLint intVertex[3] = {0};

	intVertex[0] = (int) (i * calglGetGlobalSettings()->cellSize);
	intVertex[1] = 0;
	father = calNode;

	if(calGet2Db(calDrawModel->calModel, father->substate, i, j) == father->noData){
		return;
	}

	while(father!=NULL && father->typeInfoSubstate == CALGL_TYPE_INFO_VERTEX_DATA){
		intVertex[1] += calGet2Db(calDrawModel->calModel, father->substate, i, j);
		father = calglGetFatherNode2Db(father);
	};
	intVertex[2] = (int) (j * calglGetGlobalSettings()->cellSize);

	glVertex3iv(intVertex);
}
void calglDrawRealModelDisplayCurrentNodeSetVertexData2Di(struct CALDrawModel2D* calDrawModel, struct CALNode2Di*calNode, GLint i, GLint j){
	struct CALNode2Di* father = NULL;
	GLint intVertex[3] = {0};

	intVertex[0] = (int) (i * calglGetGlobalSettings()->cellSize);
	intVertex[1] = 0;
	father = calNode;

	if(calGet2Di(calDrawModel->calModel, father->substate, i, j) == father->noData){
		return;
	}

	while(father!=NULL && father->typeInfoSubstate == CALGL_TYPE_INFO_VERTEX_DATA){
		intVertex[1] += calGet2Di(calDrawModel->calModel, father->substate, i, j);
		father = calglGetFatherNode2Di(father);
	};
	intVertex[2] = (int) (j * calglGetGlobalSettings()->cellSize);

	glVertex3iv(intVertex);
}
void calglDrawRealModelDisplayCurrentNodeSetVertexData2Dr(struct CALDrawModel2D* calDrawModel, struct CALNode2Dr*calNode, GLint i, GLint j){
	struct CALNode2Dr* father = NULL;
	GLdouble doubleVertex[3] = {0};

	doubleVertex[0] = j * calglGetGlobalSettings()->cellSize;
	doubleVertex[1] = 0;
	father = calNode;

	if(calGet2Dr(calDrawModel->calModel, father->substate, i, j) == father->noData){
		return;
	}

	while(father!=NULL && father->typeInfoSubstate == CALGL_TYPE_INFO_VERTEX_DATA){ 
		doubleVertex[1] += calGet2Dr(calDrawModel->calModel, father->substate, i, j);
		father = calglGetFatherNode2Dr(father);
	};
	doubleVertex[2] = i * calglGetGlobalSettings()->cellSize;

	glVertex3dv(doubleVertex);
}
#pragma endregion

#pragma region ComputeExtremes
void calglComputeExtremesDrawModel2Db(struct CALDrawModel2D* calDrawModel, struct CALNode2Db* calNode, GLdouble* m, GLdouble* M){
	GLint i, j;
	GLdouble tmp;

	//computing min and max z
	for (i=0; i<calDrawModel->calModel->rows; i++){
		for (j=0; j<calDrawModel->calModel->columns; j++){
			if (calGet2Db(calDrawModel->calModel, calNode->substate, i, j) > 0){
				*m = calGet2Db(calDrawModel->calModel, calNode->substate, i, j);
				*M = calGet2Db(calDrawModel->calModel, calNode->substate, i, j);
			}
		}
	}

	for (i=0; i<( (struct CALModel2D*) calDrawModel->calModel)->rows; i++){
		for (j=0; j<calDrawModel->calModel->columns; j++){
			tmp = calGet2Db(calDrawModel->calModel, calNode->substate, i, j);
			if (tmp > 0 && *M < tmp){
				*M = tmp;
			}
			if (tmp > 0 && *m > tmp){
				*m = tmp;
			}
		}
	}
}
void calglComputeExtremesDrawModel2Di(struct CALDrawModel2D* calDrawModel, struct CALNode2Di* calNode, GLdouble* m, GLdouble* M){
	GLint i, j, tmp;

	//computing min and max z
	for (i=0; i<calDrawModel->calModel->rows; i++){
		for (j=0; j<calDrawModel->calModel->columns; j++){
			if (calGet2Di(calDrawModel->calModel, calNode->substate, i, j) > 0){
				*m = calGet2Di(calDrawModel->calModel, calNode->substate, i, j);
				*M = calGet2Di(calDrawModel->calModel, calNode->substate, i, j);
			}
		}
	}

	for (i=0; i<calDrawModel->calModel->rows; i++){
		for (j=0; j<calDrawModel->calModel->columns; j++){
			tmp = calGet2Di(calDrawModel->calModel, calNode->substate, i, j);
			if (tmp > 0 && *M < tmp){
				*M = tmp;
			}
			if (tmp > 0 && *m > tmp){
				*m = tmp;
			}
		}
	}
}
void calglComputeExtremesDrawModel2Dr(struct CALDrawModel2D* calDrawModel, struct CALNode2Dr* calNode, GLdouble* m, GLdouble* M){
	GLint i = 0, j = 0;
	GLdouble tmp = 0;

	//computing min and max z
	for (i=0; i<calDrawModel->calModel->rows; i++){
		for (j=0; j<calDrawModel->calModel->columns; j++){
			if (calGet2Dr(calDrawModel->calModel, calNode->substate, i, j) > 0){
				*m = calGet2Dr(calDrawModel->calModel, calNode->substate, i, j);
				*M = calGet2Dr(calDrawModel->calModel, calNode->substate, i, j);
			}
		}
	}

	for (i=0; i<calDrawModel->calModel->rows; i++){
		for (j=0; j<calDrawModel->calModel->columns; j++){
			tmp = calGet2Dr(calDrawModel->calModel, calNode->substate, i, j);
			if (tmp > 0 && *M < tmp){
				*M = tmp;
			}
			if (tmp > 0 && *m > tmp){
				*m = tmp;
			}
		}
	}
}
#pragma endregion

#pragma region ComputeExtremesToAll
void calglComputeExtremesToAll2Db(struct CALDrawModel2D* calDrawModel, struct CALNode2Db* calNode){
	int i;
	if(calNode != NULL){
		calglComputeExtremesDrawModel2Db(calDrawModel, calNode, &calNode->min, &calNode->max);

		for(i=1; i<calNode->insertedNode; i++){
			calglComputeExtremesToAll2Db(calDrawModel, calNode->nodes[i]);
		}
	}
}
void calglComputeExtremesToAll2Di(struct CALDrawModel2D* calDrawModel, struct CALNode2Di* calNode){
	int i;
	if(calNode != NULL){
		calglComputeExtremesDrawModel2Di(calDrawModel, calNode, &calNode->min, &calNode->max);

		for(i=1; i<calNode->insertedNode; i++){
			calglComputeExtremesToAll2Di(calDrawModel, calNode->nodes[i]);
		}
	}
}
void calglComputeExtremesToAll2Dr(struct CALDrawModel2D* calDrawModel, struct CALNode2Dr* calNode){
	int i;
	if(calNode != NULL){
		calglComputeExtremesDrawModel2Dr(calDrawModel, calNode, &calNode->min, &calNode->max);

		for(i=1; i<calNode->insertedNode; i++){
			calglComputeExtremesToAll2Dr(calDrawModel, calNode->nodes[i]);
		}
	}
}
#pragma endregion

#pragma region SetNormalData
void calglSetNormalData2Db(struct CALDrawModel2D* calDrawModel, struct CALNode2Db* calNode, GLint i, GLint j){
	static GLint k;
	static CALGLVector3 vPoints[3];
	static CALGLVector3 vNormal;

	if(calGet2Db(calDrawModel->calModel, calNode->substate, i, j) == calNode->noData){
		return;
	}

	for(k=1; k<calNode->insertedNode; k++){
		if(calNode->nodes[k]->typeInfoSubstate == CALGL_TYPE_INFO_NORMAL_DATA){
			vPoints[0][0] = (GLfloat) j * calglGetGlobalSettings()->cellSize;	
			vPoints[0][1] = (GLfloat) calGet2Db(calDrawModel->calModel, calNode->nodes[k]->substate, i, j);	
			vPoints[0][2] = (GLfloat) i * calglGetGlobalSettings()->cellSize;

			vPoints[1][0] = (GLfloat) j * calglGetGlobalSettings()->cellSize;
			vPoints[1][1] = (GLfloat) calGet2Db(calDrawModel->calModel, calNode->nodes[k]->substate, i+1, j);	
			vPoints[1][2] = (GLfloat) (i+1) * calglGetGlobalSettings()->cellSize;

			vPoints[2][0] = (GLfloat) (j+1) * calglGetGlobalSettings()->cellSize;	
			vPoints[2][1] = (GLfloat) calGet2Db(calDrawModel->calModel, calNode->nodes[k]->substate, i, j+1);	
			vPoints[2][2] = (GLfloat) i * calglGetGlobalSettings()->cellSize;

			calglGetNormalVector(vPoints[0], vPoints[1], vPoints[2], vNormal);

			glNormal3f(vNormal[0], vNormal[1], vNormal[2]);
			break;
		}		
	}
}
void calglSetNormalData2Di(struct CALDrawModel2D* calDrawModel, struct CALNode2Di* calNode, GLint i, GLint j){
	static GLint k;
	static CALGLVector3 vPoints[3];
	static CALGLVector3 vNormal;

	if(calGet2Di(calDrawModel->calModel, calNode->substate, i, j) == calNode->noData){
		return;
	}

	for(k=1; k<calNode->insertedNode; k++){
		if(calNode->nodes[k]->typeInfoSubstate == CALGL_TYPE_INFO_NORMAL_DATA){
			vPoints[0][0] = (GLfloat) j * calglGetGlobalSettings()->cellSize;	
			vPoints[0][1] = (GLfloat) calGet2Di(calDrawModel->calModel, calNode->nodes[k]->substate, i, j);	
			vPoints[0][2] = (GLfloat) i * calglGetGlobalSettings()->cellSize;

			vPoints[1][0] = (GLfloat) j * calglGetGlobalSettings()->cellSize;
			vPoints[1][1] = (GLfloat) calGet2Di(calDrawModel->calModel, calNode->nodes[k]->substate, i+1, j);	
			vPoints[1][2] = (GLfloat) (i+1) * calglGetGlobalSettings()->cellSize;

			vPoints[2][0] = (GLfloat) (j+1) * calglGetGlobalSettings()->cellSize;	
			vPoints[2][1] = (GLfloat) calGet2Di(calDrawModel->calModel, calNode->nodes[k]->substate, i, j+1);	
			vPoints[2][2] = (GLfloat) i * calglGetGlobalSettings()->cellSize;

			calglGetNormalVector(vPoints[0], vPoints[1], vPoints[2], vNormal);

			glNormal3f(vNormal[0], vNormal[1], vNormal[2]);
			break;
		}		
	}
}
void calglSetNormalData2Dr(struct CALDrawModel2D* calDrawModel, struct CALNode2Dr* calNode, GLint i, GLint j){
	static GLint k;
	static CALGLVector3 vPoints[3];
	static CALGLVector3 vNormal;

	if(calGet2Dr(calDrawModel->calModel, calNode->substate, i, j) == calNode->noData){
		return;
	}

	for(k=1; k<calNode->insertedNode; k++){
		if(calNode->nodes[k]->typeInfoSubstate == CALGL_TYPE_INFO_NORMAL_DATA){
			vPoints[0][0] = (GLfloat) j * calglGetGlobalSettings()->cellSize;	
			vPoints[0][1] = (GLfloat) calGet2Dr(calDrawModel->calModel, calNode->nodes[k]->substate, i, j);	
			vPoints[0][2] = (GLfloat) i * calglGetGlobalSettings()->cellSize;

			vPoints[1][0] = (GLfloat) j * calglGetGlobalSettings()->cellSize;
			vPoints[1][1] = (GLfloat) calGet2Dr(calDrawModel->calModel, calNode->nodes[k]->substate, i+1, j);	
			vPoints[1][2] = (GLfloat) (i+1) * calglGetGlobalSettings()->cellSize;

			vPoints[2][0] = (GLfloat) (j+1) * calglGetGlobalSettings()->cellSize;	
			vPoints[2][1] = (GLfloat) calGet2Dr(calDrawModel->calModel, calNode->nodes[k]->substate, i, j+1);	
			vPoints[2][2] = (GLfloat) i * calglGetGlobalSettings()->cellSize;

			calglGetNormalVector(vPoints[0], vPoints[1], vPoints[2], vNormal);

			glNormal3f(vNormal[0], vNormal[1], vNormal[2]);
			break;
		}		
	}
}
#pragma endregion

#pragma region SetColorData
GLboolean calglSetColorData2Db(struct CALDrawModel2D* calDrawModel, struct CALNode2Db* calNode, GLint i, GLint j){
	GLint k = 0;
	GLboolean entered = CAL_FALSE;
	GLdouble tmp = 1.0;
	GLdouble doubleColor[4] = {1.0};

	for(k=1; k<calNode->insertedNode; k++){
		if(calNode->nodes[k]->typeInfoSubstate == CALGL_TYPE_INFO_COLOR_DATA){
			if(calGet2Db(calDrawModel->calModel, calNode->nodes[k]->substate, i, j) > 0){
				tmp = calGet2Db(calDrawModel->calModel, calNode->nodes[k]->substate, i, j);

				switch(calNode->nodes[k]->typeInfoUseSubstate){
				case CALGL_TYPE_INFO_USE_GRAY_SCALE: 
					entered = CAL_TRUE;
					doubleColor[0] = doubleColor[1] = doubleColor[2] = (tmp - calNode->nodes[k]->min) / (calNode->nodes[k]->max - calNode->nodes[k]->min);
					break;
				case CALGL_TYPE_INFO_USE_RED_SCALE:
					entered = CAL_TRUE;					
					doubleColor[1] = (tmp - calNode->nodes[k]->min) / (calNode->nodes[k]->max - calNode->nodes[k]->min);
					doubleColor[0] = 1.0; 
					doubleColor[2] = 0.0;
					break;
				case CALGL_TYPE_INFO_USE_GREEN_SCALE:
					entered = CAL_TRUE;
					doubleColor[1] = (tmp-calNode->nodes[k]->min)/(calNode->nodes[k]->max-calNode->nodes[k]->min);
					doubleColor[0] = doubleColor[2] = 0;
					break;
				case CALGL_TYPE_INFO_USE_BLUE_SCALE:
					entered = CAL_TRUE;
					doubleColor[2] = (tmp-calNode->nodes[k]->min)/(calNode->nodes[k]->max-calNode->nodes[k]->min);
					doubleColor[0] = doubleColor[1] = 0.0;
					break;
				case CALGL_TYPE_INFO_USE_CONST_VALUE:
					entered = CAL_TRUE;
					doubleColor[0] = calNode->nodes[k]->redComponent;
					doubleColor[1] = calNode->nodes[k]->greenComponent;
					doubleColor[2] = calNode->nodes[k]->blueComponent;
					doubleColor[3] = calNode->nodes[k]->alphaComponent;
					break;
				case CALGL_TYPE_INFO_USE_ALL_COLOR:
					entered = CAL_TRUE;					
					doubleColor[0] = componentColor[0]/255.0f;
					doubleColor[1] = componentColor[1]/255.0f;
					doubleColor[2] = componentColor[2]/255.0f;
					doubleColor[3] = 1.0f;
					componentColor[currentIndex]++;
					componentColor[currentIndex] = componentColor[currentIndex]%255;
					if(componentColor[currentIndex] == 0){
						currentIndex = (currentIndex+1)%3;
					}
					break;
				default:
					break;
				}
			}
		}
	}

	if(entered){
		glColor4d(doubleColor[0], doubleColor[1], doubleColor[2], doubleColor[3]);
	}

	return entered;
}
GLboolean calglSetColorData2Di(struct CALDrawModel2D* calDrawModel, struct CALNode2Di* calNode, GLint i, GLint j){
	GLint k = 0;
	GLboolean entered = CAL_FALSE;
	GLdouble tmp = 1.0;
	GLdouble doubleColor[4] = {1.0};

	for(k=1; k<calNode->insertedNode; k++){
		if(calNode->nodes[k]->typeInfoSubstate == CALGL_TYPE_INFO_COLOR_DATA){
			if(calGet2Di(calDrawModel->calModel, calNode->nodes[k]->substate, i, j) > 0){
				tmp = calGet2Di(calDrawModel->calModel, calNode->nodes[k]->substate, i, j);

				switch(calNode->nodes[k]->typeInfoUseSubstate){
				case CALGL_TYPE_INFO_USE_GRAY_SCALE: 
					entered = CAL_TRUE;
					doubleColor[0] = doubleColor[1] = doubleColor[2] = (tmp - calNode->nodes[k]->min) / (calNode->nodes[k]->max - calNode->nodes[k]->min);
					break;
				case CALGL_TYPE_INFO_USE_RED_SCALE:
					entered = CAL_TRUE;					
					doubleColor[1] = (tmp - calNode->nodes[k]->min) / (calNode->nodes[k]->max - calNode->nodes[k]->min);
					doubleColor[0] = 1.0; 
					doubleColor[2] = 0.0;
					break;
				case CALGL_TYPE_INFO_USE_GREEN_SCALE:
					entered = CAL_TRUE;
					doubleColor[1] = (tmp-calNode->nodes[k]->min)/(calNode->nodes[k]->max-calNode->nodes[k]->min);
					doubleColor[0] = doubleColor[2] = 0;
					break;
				case CALGL_TYPE_INFO_USE_BLUE_SCALE:
					entered = CAL_TRUE;
					doubleColor[2] = (tmp-calNode->nodes[k]->min)/(calNode->nodes[k]->max-calNode->nodes[k]->min);
					doubleColor[0] = doubleColor[1] = 0.0;
					break;
				case CALGL_TYPE_INFO_USE_CONST_VALUE:
					entered = CAL_TRUE;					
					doubleColor[0] = calNode->nodes[k]->redComponent;
					doubleColor[1] = calNode->nodes[k]->greenComponent;
					doubleColor[2] = calNode->nodes[k]->blueComponent;
					doubleColor[3] = calNode->nodes[k]->alphaComponent;
					break;
				case CALGL_TYPE_INFO_USE_ALL_COLOR:
					entered = CAL_TRUE;					
					doubleColor[0] = componentColor[0]/255.0f;
					doubleColor[1] = componentColor[1]/255.0f;
					doubleColor[2] = componentColor[2]/255.0f;
					doubleColor[3] = 1.0f;
					componentColor[currentIndex]++;
					componentColor[currentIndex] = componentColor[currentIndex]%255;
					if(componentColor[currentIndex] == 0){
						currentIndex = (currentIndex+1)%3;
					}
					break;
				default:
					break;
				}
			}
		}
	}

	if(entered){
		glColor4d(doubleColor[0], doubleColor[1], doubleColor[2], doubleColor[3]);
	}

	return entered;
}
GLboolean calglSetColorData2Dr(struct CALDrawModel2D* calDrawModel, struct CALNode2Dr* calNode, GLint i, GLint j){
	GLint k = 0;
	GLboolean entered = CAL_FALSE;
	GLdouble tmp = 1.0;
	GLdouble doubleColor[4] = {1.0};

	for(k=1; k<calNode->insertedNode; k++){
		if(calNode->nodes[k]->typeInfoSubstate == CALGL_TYPE_INFO_COLOR_DATA){
			if(calGet2Dr(calDrawModel->calModel, calNode->nodes[k]->substate, i, j) > 0){
				tmp = calGet2Dr(calDrawModel->calModel, calNode->nodes[k]->substate, i, j);

				switch(calNode->nodes[k]->typeInfoUseSubstate){
				case CALGL_TYPE_INFO_USE_GRAY_SCALE: 
					entered = CAL_TRUE;
					doubleColor[0] = doubleColor[1] = doubleColor[2] = (tmp - calNode->nodes[k]->min) / (calNode->nodes[k]->max - calNode->nodes[k]->min);
					break;
				case CALGL_TYPE_INFO_USE_RED_SCALE:
					entered = CAL_TRUE;					
					doubleColor[1] = (tmp - calNode->nodes[k]->min) / (calNode->nodes[k]->max - calNode->nodes[k]->min);
					doubleColor[0] = 1.0; 
					doubleColor[2] = 0.0;
					break;
				case CALGL_TYPE_INFO_USE_GREEN_SCALE:
					entered = CAL_TRUE;
					doubleColor[1] = (tmp-calNode->nodes[k]->min)/(calNode->nodes[k]->max-calNode->nodes[k]->min);
					doubleColor[0] = doubleColor[2] = 0;
					break;
				case CALGL_TYPE_INFO_USE_BLUE_SCALE:
					entered = CAL_TRUE;
					doubleColor[2] = (tmp-calNode->nodes[k]->min)/(calNode->nodes[k]->max-calNode->nodes[k]->min);
					doubleColor[0] = doubleColor[1] = 0.0;
					break;
				case CALGL_TYPE_INFO_USE_CONST_VALUE:
					entered = CAL_TRUE;					
					doubleColor[0] = calNode->nodes[k]->redComponent;
					doubleColor[1] = calNode->nodes[k]->greenComponent;
					doubleColor[2] = calNode->nodes[k]->blueComponent;
					doubleColor[3] = calNode->nodes[k]->alphaComponent;
					break;
				case CALGL_TYPE_INFO_USE_ALL_COLOR:
					entered = CAL_TRUE;					
					doubleColor[0] = componentColor[0]/255.0f;
					doubleColor[1] = componentColor[1]/255.0f;
					doubleColor[2] = componentColor[2]/255.0f;
					doubleColor[3] = 1.0f;
					componentColor[currentIndex]++;
					componentColor[currentIndex] = componentColor[currentIndex]%255;
					if(componentColor[currentIndex] == 0){
						currentIndex = (currentIndex+1)%3;
					}
					break;
				default:
					break;
				}
			}
		}
	}

	if(entered){
		glColor4d(doubleColor[0], doubleColor[1], doubleColor[2], doubleColor[3]);
	}

	return entered;
}
#pragma endregion

void calglColor2D(struct CALDrawModel2D* calDrawModel, GLfloat redComponent, GLfloat greenComponent, GLfloat blueComponent, GLfloat alphaComponent){
	calDrawModel->redComponent = redComponent;
	calDrawModel->greenComponent = greenComponent;
	calDrawModel->blueComponent = blueComponent;
	calDrawModel->alphaComponent = alphaComponent;
}

void calglSetModelViewParameter2D(struct CALDrawModel2D* calDrawModel, struct CALGLModelViewParameter* modelView){
	calglDestroyModelViewParameter(calDrawModel->modelView);
	calDrawModel->modelView = modelView;
}

void calglSetLightParameter2D(struct CALDrawModel2D* calDrawModel, struct CALGLLightParameter* modelLight){
	calglDestroyLightParameter(calDrawModel->modelLight);
	calDrawModel->modelLight = modelLight;
}

#pragma region BoundingBox
void calglDrawBoundingSquare2D(struct CALDrawModel2D* calDrawModel){
	int x;
	GLfloat xData[4];
	GLfloat yData[4];

	glColor3f(0.0, 1.0, 0.0);

	// Square coordinates
	xData[0] = 0.0f;	yData[0] = (GLfloat) calDrawModel->calModel->rows;
	xData[1] = 0.0f;	yData[1] = 0.0f;
	xData[2] = (GLfloat) calDrawModel->calModel->columns;	yData[2] = 0.0f;
	xData[3] = (GLfloat) calDrawModel->calModel->columns;	yData[3] = (GLfloat) calDrawModel->calModel->rows;

	glPushAttrib(GL_LIGHTING_BIT);{
		glDisable(GL_LIGHTING);
		glBegin(GL_LINE_LOOP);
		for(x = 0; x<4; x++){
			glVertex2f(xData[x], yData[x]);
		}
		glEnd();
	}	glPopAttrib();
}
void calglDrawBoundingBox2D(struct CALDrawModel2D* calDrawModel, GLfloat height, GLfloat low){
	int x;
	GLfloat xData[8];
	GLfloat yData[8];
	GLfloat zData[8];

	glColor3f(0.0, 1.0, 0.0);

	// Cube coordinates
	xData[0] = (GLfloat) calDrawModel->calModel->columns * calglGetGlobalSettings()->cellSize;	yData[0] = height;	zData[0] = 0.0f;
	xData[1] = (GLfloat) calDrawModel->calModel->columns * calglGetGlobalSettings()->cellSize;	yData[1] = height;	zData[1] = (GLfloat) calDrawModel->calModel->rows * calglGetGlobalSettings()->cellSize;
	xData[2] = 0.0f;	yData[2] = height;	zData[2] = 0.0f;
	xData[3] = 0.0f;	yData[3] = height;	zData[3] = (GLfloat) calDrawModel->calModel->rows * calglGetGlobalSettings()->cellSize;

	xData[6] = (GLfloat) calDrawModel->calModel->columns * calglGetGlobalSettings()->cellSize;	yData[6] = low;	zData[6] = 0.0f;
	xData[7] = (GLfloat) calDrawModel->calModel->columns * calglGetGlobalSettings()->cellSize;	yData[7] = low;	zData[7] = (GLfloat) calDrawModel->calModel->rows * calglGetGlobalSettings()->cellSize;
	xData[4] = 0.0f;	yData[4] = low;	zData[4] = 0.0f;
	xData[5] = 0.0f;	yData[5] = low;	zData[5] = (GLfloat) calDrawModel->calModel->rows * calglGetGlobalSettings()->cellSize;

	glPushAttrib(GL_LIGHTING_BIT);{
		glDisable(GL_LIGHTING);

		glBegin(GL_LINE_LOOP);
		for(x = 0; x<8; x+=2){
			glVertex3f(xData[x], yData[x], zData[x]);
		}
		glEnd();

		glBegin(GL_LINE_LOOP);
		for(x = 0; x<8; x+=2){
			glVertex3f(xData[x+1], yData[x+1], zData[x+1]);
		}
		glEnd();

		glBegin(GL_LINES);					
		glVertex3f(xData[0], yData[0], zData[0]);
		glVertex3f(xData[1], yData[1], zData[1]);
		glEnd();

		glBegin(GL_LINES);					
		glVertex3f(xData[6], yData[6], zData[6]);
		glVertex3f(xData[7], yData[7], zData[7]);
		glEnd();

		glBegin(GL_LINES);					
		glVertex3f(xData[4], yData[4], zData[4]);
		glVertex3f(xData[5], yData[5], zData[5]);
		glEnd();

		glBegin(GL_LINES);					
		glVertex3f(xData[2], yData[2], zData[2]);
		glVertex3f(xData[3], yData[3], zData[3]);
		glEnd();
	} glPopAttrib();
}
#pragma endregion

#pragma region InfoBar
void calglRelativeInfoBar2Db(struct CALDrawModel2D* calDrawModel, struct CALSubstate2Db* substate, const char* substateName, enum CALGL_TYPE_INFO_USE infoUse, enum CALGL_INFO_BAR_ORIENTATION orientation){
	calglDestroyInfoBar(calDrawModel->infoBar);
	calDrawModel->infoBar = calglCreateRelativeInfoBar2Db(substateName, infoUse, calDrawModel, substate, orientation); 
}
void calglRelativeInfoBar2Di(struct CALDrawModel2D* calDrawModel, struct CALSubstate2Di* substate, const char* substateName, enum CALGL_TYPE_INFO_USE infoUse, enum CALGL_INFO_BAR_ORIENTATION orientation){
	calglDestroyInfoBar(calDrawModel->infoBar);
	calDrawModel->infoBar = calglCreateRelativeInfoBar2Di(substateName, infoUse, calDrawModel, substate, orientation);
}
void calglRelativeInfoBar2Dr(struct CALDrawModel2D* calDrawModel, struct CALSubstate2Dr* substate, const char* substateName, enum CALGL_TYPE_INFO_USE infoUse, enum CALGL_INFO_BAR_ORIENTATION orientation){
	calglDestroyInfoBar(calDrawModel->infoBar);
	calDrawModel->infoBar = calglCreateRelativeInfoBar2Dr(substateName, infoUse, calDrawModel, substate, orientation);
}
void calglInfoBar2Db(struct CALDrawModel2D* calDrawModel, struct CALSubstate2Db* substate, const char* substateName, enum CALGL_TYPE_INFO_USE infoUse, GLfloat xPosition, GLfloat yPosition, GLint width, GLint height){
	calglDestroyInfoBar(calDrawModel->infoBar);
	calDrawModel->infoBar = calglCreateInfoBar2Db(substateName, infoUse, calDrawModel, substate, xPosition, yPosition, width, height);
}
void calglInfoBar2Di(struct CALDrawModel2D* calDrawModel, struct CALSubstate2Di* substate, const char* substateName, enum CALGL_TYPE_INFO_USE infoUse, GLfloat xPosition, GLfloat yPosition, GLint width, GLint height){
	calglDestroyInfoBar(calDrawModel->infoBar);
	calDrawModel->infoBar = calglCreateInfoBar2Di(substateName, infoUse, calDrawModel, substate, xPosition, yPosition, width, height);
}
void calglInfoBar2Dr(struct CALDrawModel2D* calDrawModel, struct CALSubstate2Dr* substate, const char* substateName, enum CALGL_TYPE_INFO_USE infoUse, GLfloat xPosition, GLfloat yPosition, GLint width, GLint height){
	calglDestroyInfoBar(calDrawModel->infoBar);
	calDrawModel->infoBar = calglCreateInfoBar2Dr(substateName, infoUse, calDrawModel, substate, xPosition, yPosition, width, height);
}
#pragma endregion

#pragma region DrawIntervals
void calglDisplayDrawIBound2D(struct CALDrawModel2D* calDrawModel, GLint min, GLint max){
	int i = 0;

	if (min < 0 || min > max || max > calDrawModel->calModel->rows)
		return;

	for (i = min; i < max; i++){
		calDrawModel->drawICells[i] = CAL_TRUE;
	}
}
void calglDisplayDrawJBound2D(struct CALDrawModel2D* calDrawModel, GLint min, GLint max){
	int i = 0;

	if (min < 0 || min > max || max > calDrawModel->calModel->columns)
		return;

	for (i = min; i < max; i++){
		calDrawModel->drawJCells[i] = CAL_TRUE;
	}
}
void calglHideDrawIBound2D(struct CALDrawModel2D* calDrawModel, GLint min, GLint max){
	int i = 0;

	if (min < 0 || min > max || max > calDrawModel->calModel->rows)
		return;

	for (i = min; i < max; i++){
		calDrawModel->drawICells[i] = CAL_FALSE;
	}
}
void calglHideDrawJBound2D(struct CALDrawModel2D* calDrawModel, GLint min, GLint max){
	int i = 0;

	if (min < 0 || min > max || max > calDrawModel->calModel->columns)
		return;

	for (i = min; i < max; i++){
		calDrawModel->drawJCells[i] = CAL_FALSE;
	}
}
#pragma endregion
