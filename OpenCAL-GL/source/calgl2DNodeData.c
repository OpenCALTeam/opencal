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

#include <calgl2DNodeData.h>
#include <stdio.h>
#include <stdlib.h>

#pragma region Create
struct CALNode2Db* calglCreateNode2Db(struct CALNode2Db* father){
	struct CALNode2Db* node = (struct CALNode2Db*)malloc(sizeof(struct CALNode2Db));

	node->dataType = CALGL_DATA_TYPE_UNKNOW;
	node->callList = NULL;
	node->typeInfoSubstate = CALGL_TYPE_INFO_NO_DATA;
	node->capacityNode = 2;
	node->insertedNode = 1;
	node->substate = NULL;
	node->nodes = (struct CALNode2Db**) malloc(sizeof(struct CALNode2Db)*node->capacityNode);
	node->nodes[0] = father;
	node->nodes[1] = NULL;

	node->redComponent = 0.0f;
	node->greenComponent = 0.0f;
	node->blueComponent = 0.0f;
	node->alphaComponent = 1.0f;
	node->noData = 0;

	return node;
}
struct CALNode2Di* calglCreateNode2Di(struct CALNode2Di* father){
	struct CALNode2Di* node = (struct CALNode2Di*)malloc(sizeof(struct CALNode2Di));

	node->dataType = CALGL_DATA_TYPE_UNKNOW;
	node->callList = NULL;
	node->typeInfoSubstate = CALGL_TYPE_INFO_NO_DATA;
	node->capacityNode = 2;
	node->insertedNode = 1;
	node->substate = NULL;
	node->nodes = (struct CALNode2Di**) malloc(sizeof(struct CALNode2Di)*node->capacityNode);
	node->nodes[0] = father;
	node->nodes[1] = NULL;

	node->redComponent = 0.0f;
	node->greenComponent = 0.0f;
	node->blueComponent = 0.0f;
	node->alphaComponent = 1.0f;
	node->noData = 0;

	return node;
}
struct CALNode2Dr* calglCreateNode2Dr(struct CALNode2Dr* father){
	struct CALNode2Dr* node = (struct CALNode2Dr*)malloc(sizeof(struct CALNode2Dr));

	node->dataType = CALGL_DATA_TYPE_UNKNOW;
	node->callList = NULL;
	node->typeInfoSubstate = CALGL_TYPE_INFO_NO_DATA;
	node->capacityNode = 2;
	node->insertedNode = 1;
	node->substate = NULL;
	node->nodes = (struct CALNode2Dr**) malloc(sizeof(struct CALNode2Dr)*node->capacityNode);
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
void calglDestroyNode2Db(struct CALNode2Db* node){
	int i = 0;

	if(node){
		if(node->insertedNode>1){
			for(i=1; i<node->insertedNode; i++){
				calglDestroyNode2Db(node->nodes[i]);
			}
		}

		if(node->nodes){
			free(node->nodes);
		}
		free(node);
	}
}
void calglDestroyNode2Di(struct CALNode2Di* node){
	int i = 0;

	if(node){
		if(node->insertedNode>1){
			for(i=1; i<node->insertedNode; i++){
				calglDestroyNode2Di(node->nodes[i]);
			}
		}

		if(node->nodes){
			free(node->nodes);
		}
		free(node);
	}
}
void calglDestroyNode2Dr(struct CALNode2Dr* node){
	int i = 0;

	if(node){
		if(node->insertedNode>1){
			for(i=1; i<node->insertedNode; i++){
				calglDestroyNode2Dr(node->nodes[i]);
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
void calglIncreaseDataNode2Db(struct CALNode2Db* node){
	int i = 0;
	struct CALNode2Db** nodes = NULL;

	node->capacityNode += 3;
	nodes = (struct CALNode2Db**) malloc(sizeof(struct CALNode2Db) * (node->capacityNode));

	for(i=0; i<node->insertedNode; i++){
		nodes[i] = node->nodes[i];
	}

	free(node->nodes);
	node->nodes = nodes;
}
void calglIncreaseDataNode2Di(struct CALNode2Di* node){
	int i = 0;
	struct CALNode2Di** nodes = NULL;

	node->capacityNode += 3;
	nodes = (struct CALNode2Di**) malloc(sizeof(struct CALNode2Di) * (node->capacityNode));

	for(i=0; i<node->insertedNode; i++){
		nodes[i] = node->nodes[i];
	}

	free(node->nodes);
	node->nodes = nodes;
}
void calglIncreaseDataNode2Dr(struct CALNode2Dr* node){
	int i = 0;
	struct CALNode2Dr** nodes = NULL;

	node->capacityNode += 3;
	nodes = (struct CALNode2Dr**) malloc(sizeof(struct CALNode2Dr) * (node->capacityNode));

	for(i=0; i<node->insertedNode; i++){
		nodes[i] = node->nodes[i];
	}

	free(node->nodes);
	node->nodes = nodes;
}
#pragma endregion

#pragma region DecreaseData
void calglDecreaseDataNode2Db(struct CALNode2Db* node){
	int i = 0;
	struct CALNode2Db** nodes = NULL;

	if(node->capacityNode-node->insertedNode > 3){
		node->capacityNode -= 3;
		nodes = (struct CALNode2Db**) malloc(sizeof(struct CALNode2Db) * (node->capacityNode));

		for(i=0; i<node->insertedNode; i++){
			nodes[i] = node->nodes[i];
		}

		if(node->nodes){
			free(node->nodes);
		}
		node->nodes = nodes;
	}
}
void calglDecreaseDataNode2Di(struct CALNode2Di* node){
	int i = 0;
	struct CALNode2Di** nodes = NULL;

	if(node->capacityNode-node->insertedNode > 3){
		node->capacityNode -= 3;
		nodes = (struct CALNode2Di**) malloc(sizeof(struct CALNode2Di) * (node->capacityNode));

		for(i=0; i<node->insertedNode; i++){
			nodes[i] = node->nodes[i];
		}

		if(node->nodes){
			free(node->nodes);
		}
		node->nodes = nodes;
	}
}
void calglDecreaseDataNode2Dr(struct CALNode2Dr* node){
	int i = 0;
	struct CALNode2Dr** nodes = NULL;

	if(node->capacityNode-node->insertedNode > 3){
		node->capacityNode -= 3;
		nodes = (struct CALNode2Dr**) malloc(sizeof(struct CALNode2Dr) * (node->capacityNode));

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
struct CALNode2Db* calglAddDataNode2Db(struct CALNode2Db* node, struct CALSubstate2Db* substate, enum CALGL_TYPE_INFO typeInfoSubstate, enum CALGL_TYPE_INFO_USE typeInfoUseSubstate, enum CALGL_DATA_TYPE dataType){
	struct CALNode2Db* nodeToAdd = calglCreateNode2Db(node);
	nodeToAdd->typeInfoSubstate = typeInfoSubstate;
	nodeToAdd->typeInfoUseSubstate = typeInfoUseSubstate;
	nodeToAdd->substate = substate;
	nodeToAdd->dataType = dataType;

	if(node->insertedNode >= node->capacityNode){
		calglIncreaseDataNode2Db(node);
	}

	node->nodes[node->insertedNode++] = nodeToAdd;

	return nodeToAdd;
}
struct CALNode2Di* calglAddDataNode2Di(struct CALNode2Di* node, struct CALSubstate2Di* substate, enum CALGL_TYPE_INFO typeInfoSubstate, enum CALGL_TYPE_INFO_USE typeInfoUseSubstate, enum CALGL_DATA_TYPE dataType){
	struct CALNode2Di* nodeToAdd = calglCreateNode2Di(node);
	nodeToAdd->typeInfoSubstate = typeInfoSubstate;
	nodeToAdd->typeInfoUseSubstate = typeInfoUseSubstate;
	nodeToAdd->substate = substate;
	nodeToAdd->dataType = dataType;

	if(node->insertedNode >= node->capacityNode){
		calglIncreaseDataNode2Di(node);
	}

	node->nodes[node->insertedNode++] = nodeToAdd;

	return nodeToAdd;
}
struct CALNode2Dr* calglAddDataNode2Dr(struct CALNode2Dr* node, struct CALSubstate2Dr* substate, enum CALGL_TYPE_INFO typeInfoSubstate, enum CALGL_TYPE_INFO_USE typeInfoUseSubstate, enum CALGL_DATA_TYPE dataType){
	struct CALNode2Dr* nodeToAdd = calglCreateNode2Dr(node);
	nodeToAdd->typeInfoSubstate = typeInfoSubstate;
	nodeToAdd->typeInfoUseSubstate = typeInfoUseSubstate;
	nodeToAdd->substate = substate;
	nodeToAdd->dataType = dataType;

	if(node->insertedNode >= node->capacityNode){
		calglIncreaseDataNode2Dr(node);
	}

	node->nodes[node->insertedNode] = nodeToAdd;
	node->insertedNode++;

	return nodeToAdd;
}
#pragma endregion

#pragma region RemoveData
void calglRemoveDataNode2Db(struct CALNode2Db* node, struct CALSubstate2Db* substate){
	int i = 0;
	int indexToRemove = -1;
	struct CALNode2Db* nodeToRemove = NULL;

	for(i = 1; i<node->insertedNode; i++){
		if(node->nodes[i]->substate == substate){
			indexToRemove = i;
		}
	}

	if(indexToRemove != -1){
		nodeToRemove = node->nodes[indexToRemove];
		node->nodes[indexToRemove] = NULL;
		calglDestroyNode2Db(nodeToRemove);
		calglShiftLeftFromIndexNode2Db(node, indexToRemove);
		node->insertedNode--;

		if(node->capacityNode-node->insertedNode > 3){
			calglDecreaseDataNode2Db(node);
		}
	}
}
void calglRemoveDataNode2Di(struct CALNode2Di* node, struct CALSubstate2Di* substate){
	int i = 0;
	int indexToRemove = -1;
	struct CALNode2Di* nodeToRemove = NULL;

	for(i = 1; i<node->insertedNode; i++){
		if(node->nodes[i]->substate == substate){
			indexToRemove = i;
		}
	}

	if(indexToRemove != -1){
		nodeToRemove = node->nodes[indexToRemove];
		node->nodes[indexToRemove] = NULL;
		calglDestroyNode2Di(nodeToRemove);
		calglShiftLeftFromIndexNode2Di(node, indexToRemove);
		node->insertedNode--;

		if(node->capacityNode-node->insertedNode > 3){
			calglDecreaseDataNode2Di(node);
		}
	}
}
void calglRemoveDataNode2Dr(struct CALNode2Dr* node, struct CALSubstate2Dr* substate){
	int i = 0;
	int indexToRemove = -1;
	struct CALNode2Dr* nodeToRemove = NULL;

	for(i = 1; i<node->insertedNode; i++){
		if(node->nodes[i]->substate == substate){
			indexToRemove = i;
		}
	}

	if(indexToRemove != -1){
		nodeToRemove = node->nodes[indexToRemove];
		node->nodes[indexToRemove] = NULL;
		calglDestroyNode2Dr(nodeToRemove);
		calglShiftLeftFromIndexNode2Dr(node, indexToRemove);
		node->insertedNode--;

		if(node->capacityNode-node->insertedNode > 3){
			calglDecreaseDataNode2Dr(node);
		}
	}
}
#pragma endregion

#pragma region ShiftLeftFromIndex
void calglShiftLeftFromIndexNode2Db(struct CALNode2Db* node, int index){
	int i=0;

	for(i = index; i<node->insertedNode-1; i++){
		node->nodes[i] = node->nodes[i+1];
	}
	node->nodes[node->insertedNode] = NULL;
}
void calglShiftLeftFromIndexNode2Di(struct CALNode2Di* node, int index){
	int i=0;

	for(i = index; i<node->insertedNode-1; i++){
		node->nodes[i] = node->nodes[i+1];
	}
	node->nodes[node->insertedNode] = NULL;
}
void calglShiftLeftFromIndexNode2Dr(struct CALNode2Dr* node, int index){
	int i=0;

	for(i = index; i<node->insertedNode-1; i++){
		node->nodes[i] = node->nodes[i+1];
	}
	node->nodes[node->insertedNode] = NULL;
}
#pragma endregion

#pragma region GetFather
struct CALNode2Db* calglGetFatherNode2Db(struct CALNode2Db* node){
	return node->insertedNode>0 ? node->nodes[0] : NULL; 
}
struct CALNode2Di* calglGetFatherNode2Di(struct CALNode2Di* node){
	return node->insertedNode>0 ? node->nodes[0] : NULL; 
}
struct CALNode2Dr* calglGetFatherNode2Dr(struct CALNode2Dr* node){
	return node->insertedNode>0 ? node->nodes[0] : NULL; 
}
#pragma endregion

#pragma region SetNoData
void calglSetNoDataToNode2Db(struct CALNode2Db* node, CALbyte noData){
	node->noData = noData;
}
void calglSetNoDataToNode2Di(struct CALNode2Di* node, CALint noData){
	node->noData = noData;
}
void calglSetNoDataToNode2Dr(struct CALNode2Dr* node, CALreal noData){
	node->noData = noData;
}
#pragma endregion



