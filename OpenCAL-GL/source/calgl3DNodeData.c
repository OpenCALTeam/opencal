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

#include <OpenCAL-GL/calgl3DNodeData.h>
#include <stdio.h>
#include <stdlib.h>

#pragma region Create
struct CALNode3Db* calglCreateNode3Db(struct CALNode3Db* father){
	struct CALNode3Db* node = (struct CALNode3Db*)malloc(sizeof(struct CALNode3Db));

	node->dataType = CALGL_DATA_TYPE_UNKNOW;
	node->callList = NULL;
	node->typeInfoSubstate = CALGL_TYPE_INFO_NO_DATA;
	node->capacityNode = 2;
	node->insertedNode = 1;
	node->substate = NULL;
	node->nodes = (struct CALNode3Db**) malloc(sizeof(struct CALNode3Db)*node->capacityNode);
	node->nodes[0] = father;
	node->nodes[1] = NULL;

	node->redComponent = 0.0f;
	node->greenComponent = 0.0f;
	node->blueComponent = 0.0f;
	node->alphaComponent = 1.0f;
	node->noData = 0;

	return node;
}
struct CALNode3Di* calglCreateNode3Di(struct CALNode3Di* father){
	struct CALNode3Di* node = (struct CALNode3Di*)malloc(sizeof(struct CALNode3Di));

	node->dataType = CALGL_DATA_TYPE_UNKNOW;
	node->callList = NULL;
	node->typeInfoSubstate = CALGL_TYPE_INFO_NO_DATA;
	node->capacityNode = 2;
	node->insertedNode = 1;
	node->substate = NULL;
	node->nodes = (struct CALNode3Di**) malloc(sizeof(struct CALNode3Di)*node->capacityNode);
	node->nodes[0] = father;
	node->nodes[1] = NULL;

	node->redComponent = 0.0f;
	node->greenComponent = 0.0f;
	node->blueComponent = 0.0f;
	node->alphaComponent = 1.0f;
	node->noData = 0;

	return node;
}
struct CALNode3Dr* calglCreateNode3Dr(struct CALNode3Dr* father){
	struct CALNode3Dr* node = (struct CALNode3Dr*)malloc(sizeof(struct CALNode3Dr));

	node->dataType = CALGL_DATA_TYPE_UNKNOW;
	node->callList = NULL;
	node->typeInfoSubstate = CALGL_TYPE_INFO_NO_DATA;
	node->capacityNode = 2;
	node->insertedNode = 1;
	node->substate = NULL;
	node->nodes = (struct CALNode3Dr**) malloc(sizeof(struct CALNode3Dr)*node->capacityNode);
	node->nodes[0] = father;
	node->nodes[1] = NULL;

	node->redComponent = 0.0f;
	node->greenComponent = 0.0f;
	node->blueComponent = 0.0f;
	node->alphaComponent = 1.0f;
	node->noData = 0;

	return node;
}
#pragma endregion

#pragma region Destroy
void calglDestroyNode3Db(struct CALNode3Db* node){
	int i = 0;

	if(node){
		if(node->insertedNode>1){
			for(i=1; i<node->insertedNode; i++){
				calglDestroyNode3Db(node->nodes[i]);
			}
		}

		if(node->nodes){
			free(node->nodes);
		}
		free(node);
	}
}
void calglDestroyNode3Di(struct CALNode3Di* node){
	int i = 0;

	if(node){
		if(node->insertedNode>1){
			for(i=1; i<node->insertedNode; i++){
				calglDestroyNode3Di(node->nodes[i]);
			}
		}

		if(node->nodes){
			free(node->nodes);
		}
		free(node);
	}
}
void calglDestroyNode3Dr(struct CALNode3Dr* node){
	int i = 0;

	if(node){
		if(node->insertedNode>1){
			for(i=1; i<node->insertedNode; i++){
				calglDestroyNode3Dr(node->nodes[i]);
			}
		}

		if(node->nodes){
			free(node->nodes);
		}
		free(node);
	}
}
#pragma endregion

#pragma region IncreaseData
void calglIncreaseDataNode3Db(struct CALNode3Db* node){
	int i = 0;
	struct CALNode3Db** nodes = NULL;

	node->capacityNode += 3;
	nodes = (struct CALNode3Db**) malloc(sizeof(struct CALNode3Db) * (node->capacityNode));

	for(i=0; i<node->insertedNode; i++){
		nodes[i] = node->nodes[i];
	}

	free(node->nodes);
	node->nodes = nodes;
}
void calglIncreaseDataNode3Di(struct CALNode3Di* node){
	int i = 0;
	struct CALNode3Di** nodes = NULL;

	node->capacityNode += 3;
	nodes = (struct CALNode3Di**) malloc(sizeof(struct CALNode3Di) * (node->capacityNode));

	for(i=0; i<node->insertedNode; i++){
		nodes[i] = node->nodes[i];
	}

	free(node->nodes);
	node->nodes = nodes;
}
void calglIncreaseDataNode3Dr(struct CALNode3Dr* node){
	int i = 0;
	struct CALNode3Dr** nodes = NULL;

	node->capacityNode += 3;
	nodes = (struct CALNode3Dr**) malloc(sizeof(struct CALNode3Dr) * (node->capacityNode));

	for(i=0; i<node->insertedNode; i++){
		nodes[i] = node->nodes[i];
	}

	free(node->nodes);
	node->nodes = nodes;
}
#pragma endregion

#pragma region DecreaseData
void calglDecreaseDataNode3Db(struct CALNode3Db* node){
	int i = 0;
	struct CALNode3Db** nodes = NULL;

	if(node->capacityNode-node->insertedNode > 3){
		node->capacityNode -= 3;
		nodes = (struct CALNode3Db**) malloc(sizeof(struct CALNode3Db) * (node->capacityNode));

		for(i=0; i<node->insertedNode; i++){
			nodes[i] = node->nodes[i];
		}

		if(node->nodes){
			free(node->nodes);
		}
		node->nodes = nodes;
	}
}
void calglDecreaseDataNode3Di(struct CALNode3Di* node){
	int i = 0;
	struct CALNode3Di** nodes = NULL;

	if(node->capacityNode-node->insertedNode > 3){
		node->capacityNode -= 3;
		nodes = (struct CALNode3Di**) malloc(sizeof(struct CALNode3Di) * (node->capacityNode));

		for(i=0; i<node->insertedNode; i++){
			nodes[i] = node->nodes[i];
		}

		if(node->nodes){
			free(node->nodes);
		}
		node->nodes = nodes;
	}
}
void calglDecreaseDataNode3Dr(struct CALNode3Dr* node){
	int i = 0;
	struct CALNode3Dr** nodes = NULL;

	if(node->capacityNode-node->insertedNode > 3){
		node->capacityNode -= 3;
		nodes = (struct CALNode3Dr**) malloc(sizeof(struct CALNode3Dr) * (node->capacityNode));

		for(i=0; i<node->insertedNode; i++){
			nodes[i] = node->nodes[i];
		}

		if(node->nodes){
			free(node->nodes);
		}
		node->nodes = nodes;
	}
}
#pragma endregion

#pragma region AddData
struct CALNode3Db* calglAddDataNode3Db(struct CALNode3Db* node, struct CALSubstate3Db* substate, enum CALGL_TYPE_INFO typeInfoSubstate, enum CALGL_TYPE_INFO_USE typeInfoUseSubstate, enum CALGL_DATA_TYPE dataType){
	struct CALNode3Db* nodeToAdd = calglCreateNode3Db(node);
	nodeToAdd->typeInfoSubstate = typeInfoSubstate;
	nodeToAdd->typeInfoUseSubstate = typeInfoUseSubstate;
	nodeToAdd->substate = substate;
	nodeToAdd->dataType = dataType;

	if(node->insertedNode >= node->capacityNode){
		calglIncreaseDataNode3Db(node);
	}

	node->nodes[node->insertedNode++] = nodeToAdd;

	return nodeToAdd;
}
struct CALNode3Di* calglAddDataNode3Di(struct CALNode3Di* node, struct CALSubstate3Di* substate, enum CALGL_TYPE_INFO typeInfoSubstate, enum CALGL_TYPE_INFO_USE typeInfoUseSubstate, enum CALGL_DATA_TYPE dataType){
	struct CALNode3Di* nodeToAdd = calglCreateNode3Di(node);
	nodeToAdd->typeInfoSubstate = typeInfoSubstate;
	nodeToAdd->typeInfoUseSubstate = typeInfoUseSubstate;
	nodeToAdd->substate = substate;
	nodeToAdd->dataType = dataType;

	if(node->insertedNode >= node->capacityNode){
		calglIncreaseDataNode3Di(node);
	}

	node->nodes[node->insertedNode++] = nodeToAdd;

	return nodeToAdd;
}
struct CALNode3Dr* calglAddDataNode3Dr(struct CALNode3Dr* node, struct CALSubstate3Dr* substate, enum CALGL_TYPE_INFO typeInfoSubstate, enum CALGL_TYPE_INFO_USE typeInfoUseSubstate, enum CALGL_DATA_TYPE dataType){
	struct CALNode3Dr* nodeToAdd = calglCreateNode3Dr(node);
	nodeToAdd->typeInfoSubstate = typeInfoSubstate;
	nodeToAdd->typeInfoUseSubstate = typeInfoUseSubstate;
	nodeToAdd->substate = substate;
	nodeToAdd->dataType = dataType;

	if(node->insertedNode >= node->capacityNode){
		calglIncreaseDataNode3Dr(node);
	}

	node->nodes[node->insertedNode++] = nodeToAdd;

	return nodeToAdd;
}
#pragma endregion

#pragma region RemoveData
void calglRemoveDataNode3Db(struct CALNode3Db* node, struct CALSubstate3Db* substate){
	int i = 0;
	int indexToRemove = -1;
	struct CALNode3Db* nodeToRemove = NULL;

	for(i = 1; i<node->insertedNode; i++){
		if(node->nodes[i]->substate == substate){
			indexToRemove = i;
		}
	}

	if(indexToRemove != -1){
		nodeToRemove = node->nodes[indexToRemove];
		node->nodes[indexToRemove] = NULL;
		calglDestroyNode3Db(nodeToRemove);
		calglShiftLeftFromIndexNode3Db(node, indexToRemove);
		node->insertedNode--;

		if(node->capacityNode-node->insertedNode > 3){
			calglDecreaseDataNode3Db(node);
		}
	}
}
void calglRemoveDataNode3Di(struct CALNode3Di* node, struct CALSubstate3Di* substate){
	int i = 0;
	int indexToRemove = -1;
	struct CALNode3Di* nodeToRemove = NULL;

	for(i = 1; i<node->insertedNode; i++){
		if(node->nodes[i]->substate == substate){
			indexToRemove = i;
		}
	}

	if(indexToRemove != -1){
		nodeToRemove = node->nodes[indexToRemove];
		node->nodes[indexToRemove] = NULL;
		calglDestroyNode3Di(nodeToRemove);
		calglShiftLeftFromIndexNode3Di(node, indexToRemove);
		node->insertedNode--;

		if(node->capacityNode-node->insertedNode > 3){
			calglDecreaseDataNode3Di(node);
		}
	}
}
void calglRemoveDataNode3Dr(struct CALNode3Dr* node, struct CALSubstate3Dr* substate){
	int i = 0;
	int indexToRemove = -1;
	struct CALNode3Dr* nodeToRemove = NULL;

	for(i = 1; i<node->insertedNode; i++){
		if(node->nodes[i]->substate == substate){
			indexToRemove = i;
		}
	}

	if(indexToRemove != -1){
		nodeToRemove = node->nodes[indexToRemove];
		node->nodes[indexToRemove] = NULL;
		calglDestroyNode3Dr(nodeToRemove);
		calglShiftLeftFromIndexNode3Dr(node, indexToRemove);
		node->insertedNode--;

		if(node->capacityNode-node->insertedNode > 3){
			calglDecreaseDataNode3Dr(node);
		}
	}
}
#pragma endregion

#pragma region ShiftLeftFromIndex
void calglShiftLeftFromIndexNode3Db(struct CALNode3Db* node, int index){
	int i=0;

	for(i = index; i<node->insertedNode-1; i++){
		node->nodes[i] = node->nodes[i+1];
	}
	node->nodes[node->insertedNode] = NULL;
}
void calglShiftLeftFromIndexNode3Di(struct CALNode3Di* node, int index){
	int i=0;

	for(i = index; i<node->insertedNode-1; i++){
		node->nodes[i] = node->nodes[i+1];
	}
	node->nodes[node->insertedNode] = NULL;
}
void calglShiftLeftFromIndexNode3Dr(struct CALNode3Dr* node, int index){
	int i=0;

	for(i = index; i<node->insertedNode-1; i++){
		node->nodes[i] = node->nodes[i+1];
	}
	node->nodes[node->insertedNode] = NULL;
}
#pragma endregion

#pragma region GetFather
struct CALNode3Db* calglGetFatherNode3Db(struct CALNode3Db* node){
	return node->insertedNode>0 ? node->nodes[0] : NULL;
}
struct CALNode3Di* calglGetFatherNode3Di(struct CALNode3Di* node){
	return node->insertedNode>0 ? node->nodes[0] : NULL;
}
struct CALNode3Dr* calglGetFatherNode3Dr(struct CALNode3Dr* node){
	return node->insertedNode>0 ? node->nodes[0] : NULL;
}
#pragma endregion

#pragma region SetNoData
void calglSetNoDataToNode3Db(struct CALNode3Db* node, CALbyte noData){
	node->noData = noData;
}
void calglSetNoDataToNode3Di(struct CALNode3Di* node, CALint noData){
	node->noData = noData;
}
void calglSetNoDataToNode3Dr(struct CALNode3Dr* node, CALreal noData){
	node->noData = noData;
}
#pragma endregion
