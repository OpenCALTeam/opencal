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

#include <OpenCAL-GL/calgl3D.h>
#include <OpenCAL-GL/calgl3DWindow.h>
#include <OpenCAL-GL/calglUtils.h>
#include <stdio.h>
#include <stdlib.h>

static GLuint componentColor[3] = {0};
static GLint currentIndex = 0;

struct CALDrawModel3D* calglDefDrawModel3D (enum CALGL_DRAW_MODE mode, const char* name, struct CALModel3D* calModel, struct CALRun3D* calRun) {
	struct CALDrawModel3D* drawModel = (struct CALDrawModel3D*) malloc (sizeof (struct CALDrawModel3D));

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

	drawModel->drawKCells = (GLshort*) malloc (sizeof (GLshort)*calModel->slices);
	drawModel->drawICells = (GLshort*) malloc (sizeof (GLshort)*calModel->rows);
	drawModel->drawJCells = (GLshort*) malloc (sizeof (GLshort)*calModel->columns);

	calglDisplayDrawKBound3D (drawModel, 0, calModel->slices);
	calglDisplayDrawIBound3D (drawModel, 0, calModel->rows);
	calglDisplayDrawJBound3D (drawModel, 0, calModel->columns);

	drawModel->calUpdater = calglCreateUpdater3D (calRun);
	drawModel->infoBar = NULL;

	calglShowModel3D (drawModel);

	return drawModel;
}

void calglDestoyDrawModel3D (struct CALDrawModel3D* drawModel) {
	if (drawModel) {
		if (drawModel->byteModel)
			calglDestroyNode3Db (drawModel->byteModel);
		if (drawModel->intModel)
			calglDestroyNode3Di (drawModel->intModel);
		if (drawModel->realModel)
			calglDestroyNode3Dr (drawModel->realModel);
		if (drawModel->drawKCells)
			free (drawModel->drawKCells);
		if (drawModel->drawICells)
			free (drawModel->drawICells);
		if (drawModel->drawJCells)
			free (drawModel->drawJCells);
		calglDestroyModelViewParameter (drawModel->modelView);
		calglDestroyLightParameter (drawModel->modelLight);
		calglDestroyUpdater3D (drawModel->calUpdater);
		free (drawModel);
	}
}

#pragma region AddData
void calglAddToDrawModel3Db (struct CALDrawModel3D* drawModel, struct CALSubstate3Db* substateFather, struct CALSubstate3Db** substateToAdd, enum CALGL_TYPE_INFO typeInfo, enum CALGL_TYPE_INFO_USE typeInfoUseSubstate, enum CALGL_DATA_TYPE dataType) {
	struct CALNode3Db* toReturn = NULL;
	struct CALNode3Db* nodeFather = NULL;

	// Add in treeModel for drawing
	if (substateFather==NULL) { // First Add
		toReturn = calglCreateNode3Db (NULL);
		toReturn->substate = *substateToAdd;
		toReturn->typeInfoSubstate = typeInfo;
		toReturn->typeInfoUseSubstate = typeInfoUseSubstate;
		drawModel->byteModel = toReturn;
	} else {
		calglSearchSubstateDrawModel3Db (drawModel->byteModel, substateFather, &nodeFather);
		toReturn = calglAddDataNode3Db (nodeFather, *substateToAdd, typeInfo, typeInfoUseSubstate, dataType);
	}

	if (typeInfoUseSubstate==CALGL_TYPE_INFO_USE_CONST_VALUE) {
		toReturn->redComponent = drawModel->redComponent;
		toReturn->greenComponent = drawModel->blueComponent;
		toReturn->blueComponent = drawModel->greenComponent;
		toReturn->alphaComponent = drawModel->alphaComponent;
	}
}
void calglAddToDrawModel3Di (struct CALDrawModel3D* drawModel, struct CALSubstate3Di* substateFather, struct CALSubstate3Di** substateToAdd, enum CALGL_TYPE_INFO typeInfo, enum CALGL_TYPE_INFO_USE typeInfoUseSubstate, enum CALGL_DATA_TYPE dataType) {
	struct CALNode3Di* toReturn = NULL;
	struct CALNode3Di* nodeFather = NULL;

	// Add in treeModel for drawing
	if (substateFather==NULL) { // First Add
		toReturn = calglCreateNode3Di (NULL);
		toReturn->substate = *substateToAdd;
		toReturn->typeInfoSubstate = typeInfo;
		toReturn->typeInfoUseSubstate = typeInfoUseSubstate;
		drawModel->intModel = toReturn;
	} else {
		calglSearchSubstateDrawModel3Di (drawModel->intModel, substateFather, &nodeFather);
		toReturn = calglAddDataNode3Di (nodeFather, *substateToAdd, typeInfo, typeInfoUseSubstate, dataType);
	}

	if (typeInfoUseSubstate==CALGL_TYPE_INFO_USE_CONST_VALUE) {
		toReturn->redComponent = drawModel->redComponent;
		toReturn->greenComponent = drawModel->blueComponent;
		toReturn->blueComponent = drawModel->greenComponent;
		toReturn->alphaComponent = drawModel->alphaComponent;
	}
}
void calglAddToDrawModel3Dr (struct CALDrawModel3D* drawModel, struct CALSubstate3Dr* substateFather, struct CALSubstate3Dr** substateToAdd, enum CALGL_TYPE_INFO typeInfo, enum CALGL_TYPE_INFO_USE typeInfoUseSubstate, enum CALGL_DATA_TYPE dataType) {
	struct CALNode3Dr* toReturn = NULL;
	struct CALNode3Dr* nodeFather = NULL;

	// Add in treeModel for drawing
	if (substateFather==NULL) { // First Add
		toReturn = calglCreateNode3Dr (NULL);
		toReturn->substate = *substateToAdd;
		toReturn->typeInfoSubstate = typeInfo;
		toReturn->typeInfoUseSubstate = typeInfoUseSubstate;
		drawModel->realModel = toReturn;
	} else {
		calglSearchSubstateDrawModel3Dr (drawModel->realModel, substateFather, &nodeFather);
		toReturn = calglAddDataNode3Dr (nodeFather, *substateToAdd, typeInfo, typeInfoUseSubstate, dataType);
	}

	if (typeInfoUseSubstate==CALGL_TYPE_INFO_USE_CONST_VALUE) {
		toReturn->redComponent = drawModel->redComponent;
		toReturn->greenComponent = drawModel->blueComponent;
		toReturn->blueComponent = drawModel->greenComponent;
		toReturn->alphaComponent = drawModel->alphaComponent;
	}
}
#pragma endregion

#pragma region SearchSubstate
void calglSearchSubstateDrawModel3Db (struct CALNode3Db* currentNode, struct CALSubstate3Db* substateToSearch, struct CALNode3Db** nodeSearched) {
	int i;

	if (currentNode==NULL) {
		return;
	}

	if (currentNode->substate==substateToSearch) {
		*nodeSearched = currentNode;
		return;
	}

	for (i = 1; i<currentNode->insertedNode && *nodeSearched==NULL; i++) {
		if (currentNode->nodes[i]!=NULL)
			calglSearchSubstateDrawModel3Db (currentNode->nodes[i], substateToSearch, nodeSearched);
	}
}
void calglSearchSubstateDrawModel3Di (struct CALNode3Di* currentNode, struct CALSubstate3Di* substateToSearch, struct CALNode3Di** nodeSearched) {
	int i;

	if (currentNode==NULL) {
		return;
	}

	if (currentNode->substate==substateToSearch) {
		*nodeSearched = currentNode;
		return;
	}

	for (i = 1; i<currentNode->insertedNode && *nodeSearched==NULL; i++) {
		if (currentNode->nodes[i]!=NULL)
			calglSearchSubstateDrawModel3Di (currentNode->nodes[i], substateToSearch, nodeSearched);
	}
}
void calglSearchSubstateDrawModel3Dr (struct CALNode3Dr* currentNode, struct CALSubstate3Dr* substateToSearch, struct CALNode3Dr** nodeSearched) {
	int i;

	if (currentNode==NULL) {
		return;
	}

	if (currentNode->substate==substateToSearch) {
		*nodeSearched = currentNode;
		return;
	}

	for (i = 1; i<currentNode->insertedNode && *nodeSearched==NULL; i++) {
		if (currentNode->nodes[i]!=NULL)
			calglSearchSubstateDrawModel3Dr (currentNode->nodes[i], substateToSearch, nodeSearched);
	}
}
#pragma endregion

void calglDisplayModel3D (struct CALDrawModel3D* calDrawModel) {
	switch (calDrawModel->drawMode) {
	case CALGL_DRAW_MODE_NO_DRAW:
		break;

	case CALGL_DRAW_MODE_FLAT:
		calglDrawDiscreetModel3D (calDrawModel);
		break;

	case CALGL_DRAW_MODE_SURFACE:
		calglDrawRealModel3D (calDrawModel);
		break;
	}
}

#pragma region DrawDiscreetModel3D
void calglDrawDiscreetModel3D (struct CALDrawModel3D* calDrawModel) {
	glPushMatrix (); {
		glPushAttrib (GL_LIGHTING_BIT); {
			// Apply Light
			if (calglAreLightsEnable ()) {
				if (calDrawModel->modelLight) {
					calglApplyLightParameter (calDrawModel->modelLight);
				} else {
					calDrawModel->modelLight = calglCreateLightParameter (
						calglGetPositionLight (),
						calglGetAmbientLight (),
						calglGetDiffuseLight (),
						calglGetSpecularLight (),
						1, NULL, 0.0f);
					calglApplyLightParameter (calDrawModel->modelLight);
				}
			}
			// Apply model view transformation
			if (calDrawModel->modelView) {
				calglApplyModelViewParameter (calDrawModel->modelView);
			} else {
				calglSetModelViewParameter3D (calDrawModel, calglAutoCreateModelViewParameterFlat3D (calDrawModel));
				calglApplyModelViewParameter (calDrawModel->modelView);
			}
			// Draw model
			if (calDrawModel->byteModel) {
				calglComputeExtremesToAll3Db (calDrawModel, calDrawModel->byteModel);
				calglDrawDiscreetModelDisplayNode3Db (calDrawModel, calDrawModel->byteModel);
			}
			if (calDrawModel->intModel) {
				calglComputeExtremesToAll3Di (calDrawModel, calDrawModel->intModel);
				calglDrawDiscreetModelDisplayNode3Di (calDrawModel, calDrawModel->intModel);
			}
			if (calDrawModel->realModel) {
				calglComputeExtremesToAll3Dr (calDrawModel, calDrawModel->realModel);
				calglDrawDiscreetModelDisplayNode3Dr (calDrawModel, calDrawModel->realModel);
			}
		}	glPopAttrib ();
		// Draw BoundingBox
		calglDrawBoundingBox3D (calDrawModel);
	}	glPopMatrix ();
}

void calglDrawDiscreetModelDisplayNode3Db (struct CALDrawModel3D* calDrawModel, struct CALNode3Db* calNode) {
	int i;
	calglDrawDiscreetModelDisplayCurrentNode3Db (calDrawModel, calNode);
	for (i = 1; i<calNode->insertedNode; i++) {
		calglDrawDiscreetModelDisplayNode3Db (calDrawModel, calNode->nodes[i]);
	}
}
void calglDrawDiscreetModelDisplayNode3Di (struct CALDrawModel3D* calDrawModel, struct CALNode3Di* calNode) {
	int i;
	calglDrawDiscreetModelDisplayCurrentNode3Di (calDrawModel, calNode);
	for (i = 1; i<calNode->insertedNode; i++) {
		calglDrawDiscreetModelDisplayNode3Di (calDrawModel, calNode->nodes[i]);
	}
}
void calglDrawDiscreetModelDisplayNode3Dr (struct CALDrawModel3D* calDrawModel, struct CALNode3Dr* calNode) {
	int i;
	calglDrawDiscreetModelDisplayCurrentNode3Dr (calDrawModel, calNode);
	for (i = 1; i<calNode->insertedNode; i++) {
		calglDrawDiscreetModelDisplayNode3Dr (calDrawModel, calNode->nodes[i]);
	}
}

void calglDrawDiscreetModelDisplayCurrentNode3Db (struct CALDrawModel3D* calDrawModel, struct CALNode3Db* calNode) {
	int i, j, k;
	int rows = calDrawModel->calModel->rows;
	int columns = calDrawModel->calModel->columns;
	int slices = calDrawModel->calModel->slices;

	// If no vertex data jump to next node
	if (calNode->typeInfoSubstate!=CALGL_TYPE_INFO_VERTEX_DATA)
		return;
	
	glTranslatef (-(rows/2.0f), -(columns/2.0f), -(slices/2.0f));

	glPushMatrix (); {
		// Check for static data -> use display list here
		if (calNode->dataType==CALGL_DATA_TYPE_STATIC) {
			if (calNode->callList) {
				glCallList (*calNode->callList);
			} else {
				*calNode->callList = glGenLists (1);
				glNewList (*calNode->callList, GL_COMPILE); {
					for (k = 0; k<slices; k++) {
						if (!calDrawModel->drawKCells[k])
							continue;
						for (i = 0; i<rows; i++) {
							if (!calDrawModel->drawICells[i])
								continue;
							for (j = 0; j<columns; j++) {
								if (!calDrawModel->drawJCells[j])
									continue;
								if (calglSetColorData3Db (calDrawModel, calNode, i, j, k)==CAL_TRUE) {
									glPushMatrix (); {
										glTranslatef ((GLfloat) i, (GLfloat) j, (GLfloat) k);
										glutSolidCube (1.0f);
									} glPopMatrix ();
								}
							}
						}
					}
				} glEndList ();
			}
		} else {
			for (k = 0; k<slices; k++) {
				if (!calDrawModel->drawKCells[k])
					continue;
				for (i = 0; i<rows; i++) {
					if (!calDrawModel->drawICells[i])
						continue;
					for (j = 0; j<columns; j++) {
						if (!calDrawModel->drawJCells[j])
							continue;
						if (calglSetColorData3Db (calDrawModel, calNode, i, j, k)==CAL_TRUE) {
							glPushMatrix (); {
								glTranslatef ((GLfloat) i, (GLfloat) j, (GLfloat) k);
								glutSolidCube (1.0f);
							} glPopMatrix ();
						}
					}
				}
			}
		}
	} glPopMatrix ();
}
void calglDrawDiscreetModelDisplayCurrentNode3Di (struct CALDrawModel3D* calDrawModel, struct CALNode3Di* calNode) {
	int i, j, k, x;
	int rows = calDrawModel->calModel->rows;
	int columns = calDrawModel->calModel->columns;
	int slices = calDrawModel->calModel->slices;

	GLfloat xData[8];
	GLfloat yData[8];
	GLfloat zData[8];

	// If no vertex data jump to next node
	if (calNode->typeInfoSubstate!=CALGL_TYPE_INFO_VERTEX_DATA)
		return;

	glTranslatef (-(rows/2.0f), -(slices/2.0f), -(columns/2.0f));

	// Check for static data -> use display list here
	if (calNode->dataType==CALGL_DATA_TYPE_STATIC) {
		if (calNode->callList) {
			glCallList (*calNode->callList);
		} else {
			*calNode->callList = glGenLists (1);
			glNewList (*calNode->callList, GL_COMPILE); {
				for (k = 0; k<slices; k++) {
					if (!calDrawModel->drawKCells[k])
						continue;
					for (i = 0; i<rows; i++) {
						if (!calDrawModel->drawICells[i])
							continue;
						for (j = 0; j<columns; j++) {
							if (!calDrawModel->drawJCells[j])
								continue;
							if (calglSetColorData3Di (calDrawModel, calNode, i, j, k)==CAL_TRUE) {
								// Normal ?

								// Cube coordinates
								xData[0] = j-0.5f;	yData[0] = k+0.5f;	zData[0] = i-0.5f;
								xData[1] = j-0.5f;	yData[1] = k+0.5f;	zData[1] = i+0.5f;
								xData[2] = j+0.5f;	yData[2] = k+0.5f;	zData[2] = i-0.5f;
								xData[3] = j+0.5f;	yData[3] = k+0.5f;	zData[3] = i+0.5f;
								xData[6] = j-0.5f;	yData[6] = k-0.5f;	zData[6] = i-0.5f;
								xData[7] = j-0.5f;	yData[7] = k-0.5f;	zData[7] = i+0.5f;
								xData[4] = j+0.5f;	yData[4] = k-0.5f;	zData[4] = i-0.5f;
								xData[5] = j+0.5f;	yData[5] = k-0.5f;	zData[5] = i+0.5f;

								glBegin (GL_QUAD_STRIP); {
									for (x = 0; x<8; x += 2) {
										glVertex3f (xData[x], yData[x], zData[x]);
										glVertex3f (xData[x+1], yData[x+1], zData[x+1]);
									}
									glVertex3f (xData[7], yData[7], zData[7]);
									glVertex3f (xData[0], yData[0], zData[0]);
								}glEnd ();

								glBegin (GL_QUADS); {
									glVertex3f (xData[0], yData[0], zData[0]);
									glVertex3f (xData[6], yData[6], zData[6]);
									glVertex3f (xData[4], yData[4], zData[4]);
									glVertex3f (xData[2], yData[2], zData[2]);
								}glEnd ();

								glBegin (GL_QUADS); {
									glVertex3f (xData[1], yData[1], zData[1]);
									glVertex3f (xData[7], yData[7], zData[7]);
									glVertex3f (xData[5], yData[5], zData[5]);
									glVertex3f (xData[3], yData[3], zData[3]);
								}glEnd ();
							}
						}
					}
				}
			} glEndList ();
		}
	} else {
		for (k = 0; k<slices; k++) {
			if (!calDrawModel->drawKCells[k])
				continue;
			for (i = 0; i<rows; i++) {
				if (!calDrawModel->drawICells[i])
					continue;
				for (j = 0; j<columns; j++) {
					if (!calDrawModel->drawJCells[j])
						continue;
					if (calglSetColorData3Di (calDrawModel, calNode, i, j, k)==CAL_TRUE) {
						// Normal ?

						// Cube coordinates
						xData[0] = j-0.5f;	yData[0] = k+0.5f;	zData[0] = i-0.5f;
						xData[1] = j-0.5f;	yData[1] = k+0.5f;	zData[1] = i+0.5f;
						xData[2] = j+0.5f;	yData[2] = k+0.5f;	zData[2] = i-0.5f;
						xData[3] = j+0.5f;	yData[3] = k+0.5f;	zData[3] = i+0.5f;
						xData[6] = j-0.5f;	yData[6] = k-0.5f;	zData[6] = i-0.5f;
						xData[7] = j-0.5f;	yData[7] = k-0.5f;	zData[7] = i+0.5f;
						xData[4] = j+0.5f;	yData[4] = k-0.5f;	zData[4] = i-0.5f;
						xData[5] = j+0.5f;	yData[5] = k-0.5f;	zData[5] = i+0.5f;

						glBegin (GL_QUAD_STRIP); {
							for (x = 0; x<8; x += 2) {
								glVertex3f (xData[x], yData[x], zData[x]);
								glVertex3f (xData[x+1], yData[x+1], zData[x+1]);
							}
							glVertex3f (xData[7], yData[7], zData[7]);
							glVertex3f (xData[0], yData[0], zData[0]);
						}glEnd ();

						glBegin (GL_QUADS); {
							glVertex3f (xData[0], yData[0], zData[0]);
							glVertex3f (xData[6], yData[6], zData[6]);
							glVertex3f (xData[4], yData[4], zData[4]);
							glVertex3f (xData[2], yData[2], zData[2]);
						}glEnd ();

						glBegin (GL_QUADS); {
							glVertex3f (xData[1], yData[1], zData[1]);
							glVertex3f (xData[7], yData[7], zData[7]);
							glVertex3f (xData[5], yData[5], zData[5]);
							glVertex3f (xData[3], yData[3], zData[3]);
						}glEnd ();
					}
				}
			}
		}
	}
}
void calglDrawDiscreetModelDisplayCurrentNode3Dr (struct CALDrawModel3D* calDrawModel, struct CALNode3Dr* calNode) {
	int i, j, k, x;
	int rows = calDrawModel->calModel->rows;
	int columns = calDrawModel->calModel->columns;
	int slices = calDrawModel->calModel->slices;

	GLfloat xData[8];
	GLfloat yData[8];
	GLfloat zData[8];

	// If no vertex data jump to next node
	if (calNode->typeInfoSubstate!=CALGL_TYPE_INFO_VERTEX_DATA)
		return;

	glTranslatef (-(rows/2.0f), -(slices/2.0f), -(columns/2.0f));

	// Check for static data -> use display list here
	if (calNode->dataType==CALGL_DATA_TYPE_STATIC) {
		if (calNode->callList) {
			glCallList (*calNode->callList);
		} else {
			*calNode->callList = glGenLists (1);
			glNewList (*calNode->callList, GL_COMPILE); {
				for (k = 0; k<slices; k++) {
					if (!calDrawModel->drawKCells[k])
						continue;
					for (i = 0; i<rows; i++) {
						if (!calDrawModel->drawICells[i])
							continue;
						for (j = 0; j<columns; j++) {
							if (!calDrawModel->drawJCells[j])
								continue;
							if (calglSetColorData3Dr (calDrawModel, calNode, i, j, k)==CAL_TRUE) {
								// Normal ?

								// Cube coordinates
								xData[0] = j-0.5f;	yData[0] = k+0.5f;	zData[0] = i-0.5f;
								xData[1] = j-0.5f;	yData[1] = k+0.5f;	zData[1] = i+0.5f;
								xData[2] = j+0.5f;	yData[2] = k+0.5f;	zData[2] = i-0.5f;
								xData[3] = j+0.5f;	yData[3] = k+0.5f;	zData[3] = i+0.5f;
								xData[6] = j-0.5f;	yData[6] = k-0.5f;	zData[6] = i-0.5f;
								xData[7] = j-0.5f;	yData[7] = k-0.5f;	zData[7] = i+0.5f;
								xData[4] = j+0.5f;	yData[4] = k-0.5f;	zData[4] = i-0.5f;
								xData[5] = j+0.5f;	yData[5] = k-0.5f;	zData[5] = i+0.5f;

								glBegin (GL_QUAD_STRIP); {
									for (x = 0; x<8; x += 2) {
										glVertex3f (xData[x], yData[x], zData[x]);
										glVertex3f (xData[x+1], yData[x+1], zData[x+1]);
									}
									glVertex3f (xData[7], yData[7], zData[7]);
									glVertex3f (xData[0], yData[0], zData[0]);
								}glEnd ();

								glBegin (GL_QUADS); {
									glVertex3f (xData[0], yData[0], zData[0]);
									glVertex3f (xData[6], yData[6], zData[6]);
									glVertex3f (xData[4], yData[4], zData[4]);
									glVertex3f (xData[2], yData[2], zData[2]);
								}glEnd ();

								glBegin (GL_QUADS); {
									glVertex3f (xData[1], yData[1], zData[1]);
									glVertex3f (xData[7], yData[7], zData[7]);
									glVertex3f (xData[5], yData[5], zData[5]);
									glVertex3f (xData[3], yData[3], zData[3]);
								}glEnd ();
							}
						}
					}
				}
			} glEndList ();
		}
	} else {
		for (k = 0; k<slices; k++) {
			if (!calDrawModel->drawKCells[k])
				continue;
			for (i = 0; i<rows; i++) {
				if (!calDrawModel->drawICells[i])
					continue;
				for (j = 0; j<columns; j++) {
					if (!calDrawModel->drawJCells[j])
						continue;
					if (calglSetColorData3Dr (calDrawModel, calNode, i, j, k)==CAL_TRUE) {
						// Normal ?

						// Cube coordinates
						xData[0] = j-0.5f;	yData[0] = k+0.5f;	zData[0] = i-0.5f;
						xData[1] = j-0.5f;	yData[1] = k+0.5f;	zData[1] = i+0.5f;
						xData[2] = j+0.5f;	yData[2] = k+0.5f;	zData[2] = i-0.5f;
						xData[3] = j+0.5f;	yData[3] = k+0.5f;	zData[3] = i+0.5f;
						xData[6] = j-0.5f;	yData[6] = k-0.5f;	zData[6] = i-0.5f;
						xData[7] = j-0.5f;	yData[7] = k-0.5f;	zData[7] = i+0.5f;
						xData[4] = j+0.5f;	yData[4] = k-0.5f;	zData[4] = i-0.5f;
						xData[5] = j+0.5f;	yData[5] = k-0.5f;	zData[5] = i+0.5f;

						glBegin (GL_QUAD_STRIP); {
							for (x = 0; x<8; x += 2) {
								glVertex3f (xData[x], yData[x], zData[x]);
								glVertex3f (xData[x+1], yData[x+1], zData[x+1]);
							}
							glVertex3f (xData[7], yData[7], zData[7]);
							glVertex3f (xData[0], yData[0], zData[0]);
						}glEnd ();

						glBegin (GL_QUADS); {
							glVertex3f (xData[0], yData[0], zData[0]);
							glVertex3f (xData[6], yData[6], zData[6]);
							glVertex3f (xData[4], yData[4], zData[4]);
							glVertex3f (xData[2], yData[2], zData[2]);
						}glEnd ();

						glBegin (GL_QUADS); {
							glVertex3f (xData[1], yData[1], zData[1]);
							glVertex3f (xData[7], yData[7], zData[7]);
							glVertex3f (xData[5], yData[5], zData[5]);
							glVertex3f (xData[3], yData[3], zData[3]);
						}glEnd ();
					}
				}
			}
		}
	}
}
#pragma endregion

#pragma region DrawRealModel3D
void calglDrawRealModel3D (struct CALDrawModel3D* calDrawModel) {
	glPushMatrix (); {
		glPushAttrib (GL_LIGHTING_BIT); {
			// Apply Light
			if (calDrawModel->modelLight) {
				calglApplyLightParameter (calDrawModel->modelLight);
			} else if (calglAreLightsEnable ()) {
				calDrawModel->modelLight = calglCreateLightParameter (calglGetPositionLight (), calglGetAmbientLight (), calglGetDiffuseLight (), calglGetSpecularLight (), 1, NULL, 0.0f);
			}
			// Apply model view transformation
			if (calDrawModel->modelView) {
				calglApplyModelViewParameter (calDrawModel->modelView);
			} else {
				calglSetModelViewParameter3D (calDrawModel, calglAutoCreateModelViewParameterSurface3D (calDrawModel));
				calglApplyModelViewParameter (calDrawModel->modelView);
			}
		} glPopAttrib ();
	}	glPopMatrix ();
}
#pragma endregion

#pragma region ComputeExtremes
void calglComputeExtremesDrawModel3Db (struct CALDrawModel3D* calDrawModel, struct CALNode3Db* calNode, GLdouble* m, GLdouble* M) {
	GLint i = 0, j = 0, k = 0;
	GLdouble tmp = 0;

	//computing min and max z

	for (i = 0; i<calDrawModel->calModel->rows; i++) {
		for (j = 0; j<calDrawModel->calModel->columns; j++) {
			for (k = 0; k<calDrawModel->calModel->slices; k++) {
				if (calGet3Db (calDrawModel->calModel, calNode->substate, i, j, k)>0) {
					*m = calGet3Db (calDrawModel->calModel, calNode->substate, i, j, k);
					*M = calGet3Db (calDrawModel->calModel, calNode->substate, i, j, k);
				}
			}
		}
	}

	for (i = 0; i<calDrawModel->calModel->rows; i++) {
		for (j = 0; j<calDrawModel->calModel->columns; j++) {
			for (k = 0; k<calDrawModel->calModel->slices; k++) {
				tmp = calGet3Db (calDrawModel->calModel, calNode->substate, i, j, k);
				if (tmp > 0&&*M<tmp) {
					*M = tmp;
				}
				if (tmp > 0&&*m>tmp) {
					*m = tmp;
				}
			}
		}
	}
}
void calglComputeExtremesDrawModel3Di (struct CALDrawModel3D* calDrawModel, struct CALNode3Di* calNode, GLdouble* m, GLdouble* M) {
	GLint i = 0, j = 0, k = 0;
	GLdouble tmp = 0;

	//computing min and max z

	for (i = 0; i<calDrawModel->calModel->rows; i++) {
		for (j = 0; j<calDrawModel->calModel->columns; j++) {
			for (k = 0; k<calDrawModel->calModel->slices; k++) {
				if (calGet3Di (calDrawModel->calModel, calNode->substate, i, j, k)>0) {
					*m = calGet3Di (calDrawModel->calModel, calNode->substate, i, j, k);
					*M = calGet3Di (calDrawModel->calModel, calNode->substate, i, j, k);
				}
			}
		}
	}

	for (i = 0; i<calDrawModel->calModel->rows; i++) {
		for (j = 0; j<calDrawModel->calModel->columns; j++) {
			for (k = 0; k<calDrawModel->calModel->slices; k++) {
				tmp = calGet3Di (calDrawModel->calModel, calNode->substate, i, j, k);
				if (tmp > 0&&*M<tmp) {
					*M = tmp;
				}
				if (tmp > 0&&*m>tmp) {
					*m = tmp;
				}
			}
		}
	}
}
void calglComputeExtremesDrawModel3Dr (struct CALDrawModel3D* calDrawModel, struct CALNode3Dr* calNode, GLdouble* m, GLdouble* M) {
	GLint i = 0, j = 0, k = 0;
	GLdouble tmp = 0;

	//computing min and max z

	for (i = 0; i<calDrawModel->calModel->rows; i++) {
		for (j = 0; j<calDrawModel->calModel->columns; j++) {
			for (k = 0; k<calDrawModel->calModel->slices; k++) {
				if (calGet3Dr (calDrawModel->calModel, calNode->substate, i, j, k)>0) {
					*m = calGet3Dr (calDrawModel->calModel, calNode->substate, i, j, k);
					*M = calGet3Dr (calDrawModel->calModel, calNode->substate, i, j, k);
				}
			}
		}
	}

	for (i = 0; i<calDrawModel->calModel->rows; i++) {
		for (j = 0; j<calDrawModel->calModel->columns; j++) {
			for (k = 0; k<calDrawModel->calModel->slices; k++) {
				tmp = calGet3Dr (calDrawModel->calModel, calNode->substate, i, j, k);
				if (tmp > 0&&*M<tmp) {
					*M = tmp;
				}
				if (tmp > 0&&*m>tmp) {
					*m = tmp;
				}
			}
		}
	}
}
#pragma endregion

#pragma region ComputeExtremesToAll
void calglComputeExtremesToAll3Db (struct CALDrawModel3D* calDrawModel, struct CALNode3Db* calNode) {
	int i;
	if (calNode!=NULL) {
		calglComputeExtremesDrawModel3Db (calDrawModel, calNode, &calNode->min, &calNode->max);

		for (i = 1; i<calNode->insertedNode; i++) {
			calglComputeExtremesToAll3Db (calDrawModel, calNode->nodes[i]);
		}
	}
}
void calglComputeExtremesToAll3Di (struct CALDrawModel3D* calDrawModel, struct CALNode3Di* calNode) {
	int i;
	if (calNode!=NULL) {
		calglComputeExtremesDrawModel3Di (calDrawModel, calNode, &calNode->min, &calNode->max);

		for (i = 1; i<calNode->insertedNode; i++) {
			calglComputeExtremesToAll3Di (calDrawModel, calNode->nodes[i]);
		}
	}
}
void calglComputeExtremesToAll3Dr (struct CALDrawModel3D* calDrawModel, struct CALNode3Dr* calNode) {
	int i;
	if (calNode!=NULL) {
		calglComputeExtremesDrawModel3Dr (calDrawModel, calNode, &calNode->min, &calNode->max);

		for (i = 1; i<calNode->insertedNode; i++) {
			calglComputeExtremesToAll3Dr (calDrawModel, calNode->nodes[i]);
		}
	}
}
#pragma endregion

#pragma region SetNormalData
void calglSetNormalData3Db (struct CALDrawModel3D* calDrawModel, struct CALNode3Db* calNode, GLint i, GLint j, GLint k) {
	GLint t;
	CALGLVector3 vPoints[3];
	CALGLVector3 vNormal;

	for (t = 1; t<calNode->insertedNode; t++) {
		if (calNode->nodes[t]->typeInfoSubstate==CALGL_TYPE_INFO_NORMAL_DATA) {
			vPoints[0][0] = (GLfloat) i * calglGetGlobalSettings ()->cellSize;
			vPoints[0][1] = (GLfloat) k * calglGetGlobalSettings ()->cellSize;
			vPoints[0][2] = (GLfloat) j * calglGetGlobalSettings ()->cellSize;
			vPoints[1][0] = (GLfloat) (i+1) * calglGetGlobalSettings ()->cellSize;
			vPoints[1][1] = (GLfloat) k * calglGetGlobalSettings ()->cellSize;
			vPoints[1][2] = (GLfloat) j * calglGetGlobalSettings ()->cellSize;
			vPoints[2][0] = (GLfloat) i * calglGetGlobalSettings ()->cellSize;
			vPoints[2][1] = (GLfloat) k * calglGetGlobalSettings ()->cellSize;
			vPoints[2][2] = (GLfloat) (j+1) * calglGetGlobalSettings ()->cellSize;

			calglGetNormalVector (vPoints[0], vPoints[1], vPoints[2], vNormal);
			glNormal3fv (vNormal);
		}
	}
}
void calglSetNormalData3Di (struct CALDrawModel3D* calDrawModel, struct CALNode3Di* calNode, GLint i, GLint j, GLint k) {
	GLint t;
	CALGLVector3 vPoints[3];
	CALGLVector3 vNormal;

	for (t = 1; t<calNode->insertedNode; t++) {
		if (calNode->nodes[t]->typeInfoSubstate==CALGL_TYPE_INFO_NORMAL_DATA) {
			vPoints[0][0] = (GLfloat) i * calglGetGlobalSettings ()->cellSize;
			vPoints[0][1] = (GLfloat) k * calglGetGlobalSettings ()->cellSize;
			vPoints[0][2] = (GLfloat) j * calglGetGlobalSettings ()->cellSize;
			vPoints[1][0] = (GLfloat) (i+1) * calglGetGlobalSettings ()->cellSize;
			vPoints[1][1] = (GLfloat) k * calglGetGlobalSettings ()->cellSize;
			vPoints[1][2] = (GLfloat) j * calglGetGlobalSettings ()->cellSize;
			vPoints[2][0] = (GLfloat) i * calglGetGlobalSettings ()->cellSize;
			vPoints[2][1] = (GLfloat) k * calglGetGlobalSettings ()->cellSize;
			vPoints[2][2] = (GLfloat) (j+1) * calglGetGlobalSettings ()->cellSize;

			calglGetNormalVector (vPoints[0], vPoints[1], vPoints[2], vNormal);
			glNormal3fv (vNormal);
		}
	}
}
void calglSetNormalData3Dr (struct CALDrawModel3D* calDrawModel, struct CALNode3Dr* calNode, GLint i, GLint j, GLint k) {
	GLint t;
	CALGLVector3 vPoints[3];
	CALGLVector3 vNormal;

	for (t = 1; t<calNode->insertedNode; t++) {
		if (calNode->nodes[t]->typeInfoSubstate==CALGL_TYPE_INFO_NORMAL_DATA) {
			vPoints[0][0] = (GLfloat) i * calglGetGlobalSettings ()->cellSize;
			vPoints[0][1] = (GLfloat) k * calglGetGlobalSettings ()->cellSize;
			vPoints[0][2] = (GLfloat) j * calglGetGlobalSettings ()->cellSize;
			vPoints[1][0] = (GLfloat) (i+1) * calglGetGlobalSettings ()->cellSize;
			vPoints[1][1] = (GLfloat) k * calglGetGlobalSettings ()->cellSize;
			vPoints[1][2] = (GLfloat) j * calglGetGlobalSettings ()->cellSize;
			vPoints[2][0] = (GLfloat) i * calglGetGlobalSettings ()->cellSize;
			vPoints[2][1] = (GLfloat) k * calglGetGlobalSettings ()->cellSize;
			vPoints[2][2] = (GLfloat) (j+1) * calglGetGlobalSettings ()->cellSize;

			calglGetNormalVector (vPoints[0], vPoints[1], vPoints[2], vNormal);
			glNormal3fv (vNormal);
		}
	}
}
#pragma endregion

#pragma region SetColorData
GLboolean calglSetColorData3Db (struct CALDrawModel3D* calDrawModel, struct CALNode3Db* calNode, GLint i, GLint j, GLint k) {
	GLint t = 0;
	GLboolean entered = CAL_FALSE;
	GLdouble tmp = 1.0;
	GLdouble doubleColor[4] = {1.0};

	for (t = 1; t<calNode->insertedNode; t++) {
		if (calNode->nodes[t]->typeInfoSubstate==CALGL_TYPE_INFO_COLOR_DATA) {
			tmp = calGet3Db (calDrawModel->calModel, calNode->nodes[t]->substate, i, j, k);
			if (tmp>0) {
				switch (calNode->nodes[t]->typeInfoUseSubstate) {
				case CALGL_TYPE_INFO_USE_GRAY_SCALE:
					entered = CAL_TRUE;
					doubleColor[0] = doubleColor[1] = doubleColor[2] = ((tmp-calNode->nodes[t]->min)/(calNode->nodes[t]->max-calNode->nodes[t]->min));
					break;
				case CALGL_TYPE_INFO_USE_RED_SCALE:
					entered = CAL_TRUE;
					doubleColor[1] = ((tmp-calNode->nodes[t]->min)/(calNode->nodes[t]->max-calNode->nodes[t]->min));
					doubleColor[0] = 1;
					doubleColor[2] = 0;
					break;
				case CALGL_TYPE_INFO_USE_GREEN_SCALE:
					entered = CAL_TRUE;
					doubleColor[1] = ((tmp-calNode->nodes[t]->min)/(calNode->nodes[t]->max-calNode->nodes[t]->min));
					doubleColor[0] = doubleColor[2] = 0;
					break;
				case CALGL_TYPE_INFO_USE_BLUE_SCALE:
					entered = CAL_TRUE;
					doubleColor[2] = ((tmp-calNode->nodes[t]->min)/(calNode->nodes[t]->max-calNode->nodes[t]->min));
					doubleColor[0] = doubleColor[1] = 0;
					break;
				case CALGL_TYPE_INFO_USE_CONST_VALUE:
					entered = CAL_TRUE;
					doubleColor[0] = calNode->nodes[t]->redComponent;
					doubleColor[1] = calNode->nodes[t]->greenComponent;
					doubleColor[2] = calNode->nodes[t]->blueComponent;
					doubleColor[3] = calNode->nodes[t]->alphaComponent;
					break;
				case CALGL_TYPE_INFO_USE_ALL_COLOR:
					entered = CAL_TRUE;
					doubleColor[0] = componentColor[0]/255.0f;
					doubleColor[1] = componentColor[1]/255.0f;
					doubleColor[2] = componentColor[2]/255.0f;
					doubleColor[3] = 1.0f;
					componentColor[currentIndex]++;
					componentColor[currentIndex] = componentColor[currentIndex]%255;
					if (componentColor[currentIndex]==0) {
						currentIndex = (currentIndex+1)%3;
					}
					break;
				default:
					break;
				}
			}
		}
	}

	if (entered) {
		glColor4dv (doubleColor);
	}

	return entered;
}
GLboolean calglSetColorData3Di (struct CALDrawModel3D* calDrawModel, struct CALNode3Di* calNode, GLint i, GLint j, GLint k) {
	GLint t = 0;
	GLboolean entered = CAL_FALSE;
	GLdouble tmp = 1.0;
	GLdouble doubleColor[4] = {1.0};

	for (t = 1; t<calNode->insertedNode; t++) {
		if (calNode->nodes[t]->typeInfoSubstate==CALGL_TYPE_INFO_COLOR_DATA) {
			if (calGet3Di (calDrawModel->calModel, calNode->nodes[t]->substate, i, j, k)>0) {
				tmp = calGet3Di (calDrawModel->calModel, calNode->nodes[t]->substate, i, j, k);

				switch (calNode->nodes[t]->typeInfoUseSubstate) {
				case CALGL_TYPE_INFO_USE_GRAY_SCALE:
					entered = CAL_TRUE;
					doubleColor[0] = doubleColor[1] = doubleColor[2] = (tmp-calNode->nodes[t]->min)/(calNode->nodes[t]->max-calNode->nodes[t]->min);
					break;
				case CALGL_TYPE_INFO_USE_RED_SCALE:
					entered = CAL_TRUE;
					doubleColor[1] = (tmp-calNode->nodes[t]->min)/(calNode->nodes[t]->max-calNode->nodes[t]->min);
					doubleColor[0] = 1.0;
					doubleColor[2] = 0.0;
					break;
				case CALGL_TYPE_INFO_USE_GREEN_SCALE:
					entered = CAL_TRUE;
					doubleColor[1] = (tmp-calNode->nodes[t]->min)/(calNode->nodes[t]->max-calNode->nodes[t]->min);
					doubleColor[0] = doubleColor[2] = 0;
					break;
				case CALGL_TYPE_INFO_USE_BLUE_SCALE:
					entered = CAL_TRUE;
					doubleColor[2] = (tmp-calNode->nodes[t]->min)/(calNode->nodes[t]->max-calNode->nodes[t]->min);
					doubleColor[0] = doubleColor[1] = 0.0;
					break;
				case CALGL_TYPE_INFO_USE_CONST_VALUE:
					entered = CAL_TRUE;
					doubleColor[0] = calNode->nodes[t]->redComponent;
					doubleColor[1] = calNode->nodes[t]->greenComponent;
					doubleColor[2] = calNode->nodes[t]->blueComponent;
					doubleColor[3] = calNode->nodes[t]->alphaComponent;
					break;
				case CALGL_TYPE_INFO_USE_ALL_COLOR:
					entered = CAL_TRUE;
					doubleColor[0] = componentColor[0]/255.0f;
					doubleColor[1] = componentColor[1]/255.0f;
					doubleColor[2] = componentColor[2]/255.0f;
					doubleColor[3] = 1.0f;
					componentColor[currentIndex]++;
					componentColor[currentIndex] = componentColor[currentIndex]%255;
					if (componentColor[currentIndex]==0) {
						currentIndex = (currentIndex+1)%3;
					}
					break;
				default:
					break;
				}
			}
		}
	}

	if (entered) {
		glColor4d (doubleColor[0], doubleColor[1], doubleColor[2], doubleColor[3]);
	}

	return entered;
}
GLboolean calglSetColorData3Dr (struct CALDrawModel3D* calDrawModel, struct CALNode3Dr* calNode, GLint i, GLint j, GLint k) {
	GLint t = 0;
	GLboolean entered = CAL_FALSE;
	GLdouble tmp = 1.0;
	GLdouble doubleColor[4] = {1.0};

	for (t = 1; t<calNode->insertedNode; t++) {
		if (calNode->nodes[t]->typeInfoSubstate==CALGL_TYPE_INFO_COLOR_DATA) {
			if (calGet3Dr (calDrawModel->calModel, calNode->nodes[t]->substate, i, j, k)>0) {
				tmp = calGet3Dr (calDrawModel->calModel, calNode->nodes[t]->substate, i, j, k);

				switch (calNode->nodes[t]->typeInfoUseSubstate) {
				case CALGL_TYPE_INFO_USE_GRAY_SCALE:
					entered = CAL_TRUE;
					doubleColor[0] = doubleColor[1] = doubleColor[2] = (tmp-calNode->nodes[t]->min)/(calNode->nodes[t]->max-calNode->nodes[t]->min);
					break;
				case CALGL_TYPE_INFO_USE_RED_SCALE:
					entered = CAL_TRUE;
					doubleColor[1] = (tmp-calNode->nodes[t]->min)/(calNode->nodes[t]->max-calNode->nodes[t]->min);
					doubleColor[0] = 1.0;
					doubleColor[2] = 0.0;
					break;
				case CALGL_TYPE_INFO_USE_GREEN_SCALE:
					entered = CAL_TRUE;
					doubleColor[1] = (tmp-calNode->nodes[t]->min)/(calNode->nodes[t]->max-calNode->nodes[t]->min);
					doubleColor[0] = doubleColor[2] = 0;
					break;
				case CALGL_TYPE_INFO_USE_BLUE_SCALE:
					entered = CAL_TRUE;
					doubleColor[2] = (tmp-calNode->nodes[t]->min)/(calNode->nodes[t]->max-calNode->nodes[t]->min);
					doubleColor[0] = doubleColor[1] = 0.0;
					break;
				case CALGL_TYPE_INFO_USE_CONST_VALUE:
					entered = CAL_TRUE;
					doubleColor[0] = calNode->nodes[t]->redComponent;
					doubleColor[1] = calNode->nodes[t]->greenComponent;
					doubleColor[2] = calNode->nodes[t]->blueComponent;
					doubleColor[3] = calNode->nodes[t]->alphaComponent;
					break;
				case CALGL_TYPE_INFO_USE_ALL_COLOR:
					entered = CAL_TRUE;
					doubleColor[0] = componentColor[0]/255.0f;
					doubleColor[1] = componentColor[1]/255.0f;
					doubleColor[2] = componentColor[2]/255.0f;
					doubleColor[3] = 1.0f;
					componentColor[currentIndex]++;
					componentColor[currentIndex] = componentColor[currentIndex]%255;
					if (componentColor[currentIndex]==0) {
						currentIndex = (currentIndex+1)%3;
					}
					break;
				default:
					break;
				}
			}
		}
	}

	if (entered) {
		glColor4d (doubleColor[0], doubleColor[1], doubleColor[2], doubleColor[3]);
	}

	return entered;
}
#pragma endregion

void calglColor3D (struct CALDrawModel3D* calDrawModel, GLfloat redComponent, GLfloat greenComponent, GLfloat blueComponent, GLfloat alphaComponent) {
	calDrawModel->redComponent = redComponent;
	calDrawModel->greenComponent = greenComponent;
	calDrawModel->blueComponent = blueComponent;
	calDrawModel->alphaComponent = alphaComponent;
}

void calglSetModelViewParameter3D (struct CALDrawModel3D* calDrawModel, struct CALGLModelViewParameter* modelView) {
	calglDestroyModelViewParameter (calDrawModel->modelView);
	calDrawModel->modelView = modelView;
}

void calglSetLightParameter3D (struct CALDrawModel3D* calDrawModel, struct CALGLLightParameter* modelLight) {
	calglDestroyLightParameter (calDrawModel->modelLight);
	calDrawModel->modelLight = modelLight;
}

#pragma region BoundingBox
void calglDrawBoundingBox3D (struct CALDrawModel3D* calDrawModel) {
	int x;
	GLfloat xData[8];
	GLfloat yData[8];
	GLfloat zData[8];

	glPushMatrix (); {
		glTranslatef (-0.5f, -0.5f, 0.0f);

		glColor3f (0.0, 1.0, 0.0);

		// Cube coordinates
		xData[0] = (GLfloat) calDrawModel->calModel->columns;	yData[0] = (GLfloat) calDrawModel->calModel->slices;	zData[0] = 0.0f;
		xData[1] = (GLfloat) calDrawModel->calModel->columns;	yData[1] = (GLfloat) calDrawModel->calModel->slices;	zData[1] = (GLfloat) calDrawModel->calModel->rows;
		xData[2] = 0.0f;	yData[2] = (GLfloat) calDrawModel->calModel->slices;	zData[2] = 0.0f;
		xData[3] = 0.0f;	yData[3] = (GLfloat) calDrawModel->calModel->slices;	zData[3] = (GLfloat) calDrawModel->calModel->rows;

		xData[6] = (GLfloat) calDrawModel->calModel->columns;	yData[6] = 0.0f;	zData[6] = 0.0f;
		xData[7] = (GLfloat) calDrawModel->calModel->columns;	yData[7] = 0.0f;	zData[7] = (GLfloat) calDrawModel->calModel->rows;
		xData[4] = 0.0f;	yData[4] = 0.0f;	zData[4] = 0.0f;
		xData[5] = 0.0f;	yData[5] = 0.0f;	zData[5] = (GLfloat) calDrawModel->calModel->rows;

		glPushAttrib (GL_LIGHTING_BIT); {
			glDisable (GL_LIGHTING);

			glBegin (GL_LINE_LOOP);
			for (x = 0; x<8; x += 2) {
				glVertex3f (xData[x], yData[x], zData[x]);
			}
			glEnd ();

			glBegin (GL_LINE_LOOP);
			for (x = 0; x<8; x += 2) {
				glVertex3f (xData[x+1], yData[x+1], zData[x+1]);
			}
			glEnd ();

			glBegin (GL_LINES);
			glVertex3f (xData[0], yData[0], zData[0]);
			glVertex3f (xData[1], yData[1], zData[1]);
			glEnd ();

			glBegin (GL_LINES);
			glVertex3f (xData[6], yData[6], zData[6]);
			glVertex3f (xData[7], yData[7], zData[7]);
			glEnd ();

			glBegin (GL_LINES);
			glVertex3f (xData[4], yData[4], zData[4]);
			glVertex3f (xData[5], yData[5], zData[5]);
			glEnd ();

			glBegin (GL_LINES);
			glVertex3f (xData[2], yData[2], zData[2]);
			glVertex3f (xData[3], yData[3], zData[3]);
			glEnd ();
		} glPopAttrib ();
	} glPopMatrix ();
}
#pragma endregion

#pragma region InfoBar
void calglRelativeInfoBar3Db (struct CALDrawModel3D* calDrawModel, struct CALSubstate3Db* substate, const char* substateName, enum CALGL_TYPE_INFO_USE infoUse, enum CALGL_INFO_BAR_ORIENTATION orientation) {
	calglDestroyInfoBar (calDrawModel->infoBar);
	calDrawModel->infoBar = calglCreateRelativeInfoBar3Db (substateName, CALGL_TYPE_INFO_USE_RED_SCALE, calDrawModel, substate, orientation);
}
void calglRelativeInfoBar3Di (struct CALDrawModel3D* calDrawModel, struct CALSubstate3Di* substate, const char* substateName, enum CALGL_TYPE_INFO_USE infoUse, enum CALGL_INFO_BAR_ORIENTATION orientation) {
	calglDestroyInfoBar (calDrawModel->infoBar);
	calDrawModel->infoBar = calglCreateRelativeInfoBar3Di (substateName, CALGL_TYPE_INFO_USE_RED_SCALE, calDrawModel, substate, orientation);
}
void calglRelativeInfoBar3Dr (struct CALDrawModel3D* calDrawModel, struct CALSubstate3Dr* substate, const char* substateName, enum CALGL_TYPE_INFO_USE infoUse, enum CALGL_INFO_BAR_ORIENTATION orientation) {
	calglDestroyInfoBar (calDrawModel->infoBar);
	calDrawModel->infoBar = calglCreateRelativeInfoBar3Dr (substateName, CALGL_TYPE_INFO_USE_RED_SCALE, calDrawModel, substate, orientation);
}
void calglAbsoluteInfoBar3Db (struct CALDrawModel3D* calDrawModel, struct CALSubstate3Db* substate, const char* substateName, enum CALGL_TYPE_INFO_USE infoUse, GLfloat xPosition, GLfloat yPosition, GLint width, GLint height) {
	calglDestroyInfoBar (calDrawModel->infoBar);
	calDrawModel->infoBar = calglCreateAbsoluteInfoBar3Db (substateName, infoUse, calDrawModel, substate, xPosition, yPosition, width, height);
}
void calglAbsoluteInfoBar3Di (struct CALDrawModel3D* calDrawModel, struct CALSubstate3Di* substate, const char* substateName, enum CALGL_TYPE_INFO_USE infoUse, GLfloat xPosition, GLfloat yPosition, GLint width, GLint height) {
	calglDestroyInfoBar (calDrawModel->infoBar);
	calDrawModel->infoBar = calglCreateAbsoluteInfoBar3Di (substateName, infoUse, calDrawModel, substate, xPosition, yPosition, width, height);
}
void calglAbsoluteInfoBar3Dr (struct CALDrawModel3D* calDrawModel, struct CALSubstate3Dr* substate, const char* substateName, enum CALGL_TYPE_INFO_USE infoUse, GLfloat xPosition, GLfloat yPosition, GLint width, GLint height) {
	calglDestroyInfoBar (calDrawModel->infoBar);
	calDrawModel->infoBar = calglCreateAbsoluteInfoBar3Dr (substateName, infoUse, calDrawModel, substate, xPosition, yPosition, width, height);
}
#pragma endregion

#pragma region DrawIntervals
void calglDisplayDrawKBound3D (struct CALDrawModel3D* calDrawModel, GLint min, GLint max) {
	int i = 0;

	if (min < 0||min > max||max>calDrawModel->calModel->slices)
		return;

	for (i = min; i<max; i++) {
		calDrawModel->drawKCells[i] = CAL_TRUE;
	}
}
void calglDisplayDrawIBound3D (struct CALDrawModel3D* calDrawModel, GLint min, GLint max) {
	int i = 0;

	if (min < 0||min > max||max>calDrawModel->calModel->rows)
		return;

	for (i = min; i<max; i++) {
		calDrawModel->drawICells[i] = CAL_TRUE;
	}
}
void calglDisplayDrawJBound3D (struct CALDrawModel3D* calDrawModel, GLint min, GLint max) {
	int i = 0;

	if (min < 0||min > max||max>calDrawModel->calModel->columns)
		return;

	for (i = min; i<max; i++) {
		calDrawModel->drawJCells[i] = CAL_TRUE;
	}
}
void calglHideDrawKBound3D (struct CALDrawModel3D* calDrawModel, GLint min, GLint max) {
	int i = 0;

	if (min < 0||min > max||max>calDrawModel->calModel->slices)
		return;

	for (i = min; i<max; i++) {
		calDrawModel->drawKCells[i] = CAL_FALSE;
	}
}
void calglHideDrawIBound3D (struct CALDrawModel3D* calDrawModel, GLint min, GLint max) {
	int i = 0;

	if (min < 0||min > max||max>calDrawModel->calModel->rows)
		return;

	for (i = min; i<max; i++) {
		calDrawModel->drawICells[i] = CAL_FALSE;
	}
}
void calglHideDrawJBound3D (struct CALDrawModel3D* calDrawModel, GLint min, GLint max) {
	int i = 0;

	if (min < 0||min > max||max>calDrawModel->calModel->columns)
		return;

	for (i = min; i<max; i++) {
		calDrawModel->drawJCells[i] = CAL_FALSE;
	}
}
#pragma endregion
