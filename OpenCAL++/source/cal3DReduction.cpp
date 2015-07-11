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

#include <cal3DReduction.h>
#include <omp.h>
#include <stdlib.h>



CALbyte calReductionOperation3Db(struct CALModel3D* model, struct CALSubstate3Db* substate, enum REDUCTION_OPERATION operation){
  CALint i;
  CALbyte valueComputed = 0, tmp = 0;
  CALint numThreads = 1;

  valueComputed = getValue3DbAtIndex(substate, 0);

  CALint start, end, threadId;
  threadId = omp_get_thread_num();
  start = threadId * (model->rows*model->columns*model->slices) / numThreads;
  end = (threadId + 1) * (model->rows*model->columns*model->slices) / numThreads;
  if (threadId == numThreads - 1){
    end = model->rows;
  }

  for (i = start; i<end; i++){
    if (i == 0){
      continue;
    }
    switch (operation){
    case REDUCTION_MAX:
      tmp = getValue3DbAtIndex(substate, i);
      if (valueComputed < tmp){
	valueComputed = tmp;
      }
      break;
    case REDUCTION_MIN:
      tmp = getValue3DbAtIndex(substate, i);
      if (valueComputed > tmp){
	valueComputed = tmp;
      }
      break;
    case REDUCTION_SUM:
      valueComputed += getValue3DbAtIndex(substate, i);
      break;
    case REDUCTION_PROD:
      valueComputed *= getValue3DbAtIndex(substate, i);
      break;
    case REDUCTION_LOGICAL_AND:
      valueComputed = (valueComputed && getValue3DbAtIndex(substate, i));
      break;
    case REDUCTION_BINARY_AND:
      valueComputed = (valueComputed & getValue3DbAtIndex(substate, i));
      break;
    case REDUCTION_LOGICAL_OR:
      valueComputed = (valueComputed || getValue3DbAtIndex(substate, i));
      break;
    case REDUCTION_BINARY_OR:
      valueComputed = (valueComputed | getValue3DbAtIndex(substate, i));
      break;
    case REDUCTION_LOGICAL_XOR:
      valueComputed = (!!valueComputed != !!getValue3DbAtIndex(substate, i));
      break;
    case REDUCTION_BINARY_XOR:
      valueComputed = (valueComputed ^ getValue3DbAtIndex(substate, i));
      break;
    default:
      break;
    }
  }

  return valueComputed;
}
CALint calReductionOperation3Di(struct CALModel3D* model, struct CALSubstate3Di* substate, enum REDUCTION_OPERATION operation){
  CALint i, valueComputed = 0, tmp = 0;
  CALint numThreads = 1;

  valueComputed = getValue3DiAtIndex(substate, 0);

  CALint start, end, threadId;
  threadId = omp_get_thread_num();
  start = threadId * (model->rows*model->columns*model->slices) / numThreads;
  end = (threadId + 1) * (model->rows*model->columns*model->slices) / numThreads;
  if (threadId == numThreads - 1){
    end = model->rows;
  }

  for (i = start; i<end; i++){
    if (i == 0){
      continue;
    }
    switch (operation){
    case REDUCTION_MAX:
      tmp = getValue3DiAtIndex(substate, i);
      if (valueComputed < tmp){
	valueComputed = tmp;
      }
      break;
    case REDUCTION_MIN:
      tmp = getValue3DiAtIndex(substate, i);
      if (valueComputed > tmp){
	valueComputed = tmp;
      }
      break;
    case REDUCTION_SUM:
      valueComputed += getValue3DiAtIndex(substate, i);
      break;
    case REDUCTION_PROD:
      valueComputed *= getValue3DiAtIndex(substate, i);
      break;
    case REDUCTION_LOGICAL_AND:
      valueComputed = (valueComputed && getValue3DiAtIndex(substate, i));
      break;
    case REDUCTION_BINARY_AND:
      valueComputed = (valueComputed & getValue3DiAtIndex(substate, i));
      break;
    case REDUCTION_LOGICAL_OR:
      valueComputed = (valueComputed || getValue3DiAtIndex(substate, i));
      break;
    case REDUCTION_BINARY_OR:
      valueComputed = (valueComputed | getValue3DiAtIndex(substate, i));
      break;
    case REDUCTION_LOGICAL_XOR:
      valueComputed = (!!valueComputed != !!getValue3DiAtIndex(substate, i));
      break;
    case REDUCTION_BINARY_XOR:
      valueComputed = (valueComputed ^ getValue3DiAtIndex(substate, i));
      break;
    default:
      break;
    }
  }

  return valueComputed;
}
CALreal calReductionOperation3Dr(struct CALModel3D* model, struct CALSubstate3Dr* substate, enum REDUCTION_OPERATION operation){
  CALint i;
  CALreal valueComputed = 0, tmp = 0;
  CALint numThreads = 1;

  valueComputed = getValue3DrAtIndex(substate, 0);

  CALint start, end, threadId;
  threadId = omp_get_thread_num();
  start = threadId * (model->rows*model->columns*model->slices) / numThreads;
  end = (threadId + 1) * (model->rows*model->columns*model->slices) / numThreads;
  if (threadId == numThreads - 1){
    end = model->rows;
  }

  for (i = start; i<end; i++){
    if (i == 0){
      continue;
    }
    switch (operation){
    case REDUCTION_MAX:
      tmp = getValue3DrAtIndex(substate, i);
      if (valueComputed < tmp){
	valueComputed = tmp;
      }
      break;
    case REDUCTION_MIN:
      tmp = getValue3DrAtIndex(substate, i);
      if (valueComputed > tmp){
	valueComputed = tmp;
      }
      break;
    case REDUCTION_SUM:
      valueComputed += getValue3DrAtIndex(substate, i);
      break;
    case REDUCTION_PROD:
      valueComputed *= getValue3DrAtIndex(substate, i);
      break;
    case REDUCTION_LOGICAL_AND:
      valueComputed = (valueComputed && getValue3DrAtIndex(substate, i));
      break;
    case REDUCTION_BINARY_AND:
      valueComputed = (CALreal)((CALint)valueComputed & (CALint)getValue3DrAtIndex(substate, i));
      break;
    case REDUCTION_LOGICAL_OR:
      valueComputed = (valueComputed || getValue3DrAtIndex(substate, i));
      break;
    case REDUCTION_BINARY_OR:
      valueComputed = (CALreal)((CALint)valueComputed | (CALint)getValue3DrAtIndex(substate, i));
      break;
    case REDUCTION_LOGICAL_XOR:
      valueComputed = (!!valueComputed != !!getValue3DrAtIndex(substate, i));
      break;
    case REDUCTION_BINARY_XOR:
      valueComputed = (CALreal)((CALint)valueComputed ^ (CALint)getValue3DrAtIndex(substate, i));
      break;
    default:
      break;
    }
  }
		
  return valueComputed;
}



CALbyte calReductionComputeMax3Db(struct CALModel3D* model, struct CALSubstate3Db* substate){
  return calReductionOperation3Db(model, substate, REDUCTION_MAX);
}
CALint calReductionComputeMax3Di(struct CALModel3D* model, struct CALSubstate3Di* substate){
  return calReductionOperation3Di(model, substate, REDUCTION_MAX);
}
CALreal calReductionComputeMax3Dr(struct CALModel3D* model, struct CALSubstate3Dr* substate){
  return calReductionOperation3Dr(model, substate, REDUCTION_MAX);
}

CALbyte calReductionComputeMin3Db(struct CALModel3D* model, struct CALSubstate3Db* substate){
  return calReductionOperation3Db(model, substate, REDUCTION_MIN);
}
CALint calReductionComputeMin3Di(struct CALModel3D* model, struct CALSubstate3Di* substate){
  return calReductionOperation3Di(model, substate, REDUCTION_MIN);
}
CALreal calReductionComputeMin3Dr(struct CALModel3D* model, struct CALSubstate3Dr* substate){
  return calReductionOperation3Dr(model, substate, REDUCTION_MIN);
}

CALbyte calReductionComputeSum3Db(struct CALModel3D* model, struct CALSubstate3Db* substate){
  return calReductionOperation3Db(model, substate, REDUCTION_SUM);
}
CALint calReductionComputeSum3Di(struct CALModel3D* model, struct CALSubstate3Di* substate){
  return calReductionOperation3Di(model, substate, REDUCTION_SUM);
}
CALreal calReductionComputeSum3Dr(struct CALModel3D* model, struct CALSubstate3Dr* substate){
  return calReductionOperation3Dr(model, substate, REDUCTION_SUM);
}

CALbyte calReductionComputeProd3Db(struct CALModel3D* model, struct CALSubstate3Db* substate){	
  return calReductionOperation3Db(model, substate, REDUCTION_PROD);
}
CALint calReductionComputeProd3Di(struct CALModel3D* model, struct CALSubstate3Di* substate){
  return calReductionOperation3Di(model, substate, REDUCTION_SUM);
}
CALreal calReductionComputeProd3Dr(struct CALModel3D* model, struct CALSubstate3Dr* substate){
  return calReductionOperation3Dr(model, substate, REDUCTION_SUM);
}

CALbyte calReductionComputeLogicalAnd3Db(struct CALModel3D* model, struct CALSubstate3Db* substate){ 
  return calReductionOperation3Db(model, substate, REDUCTION_LOGICAL_AND);
}
CALint calReductionComputeLogicalAnd3Di(struct CALModel3D* model, struct CALSubstate3Di* substate){
  return calReductionOperation3Di(model, substate, REDUCTION_LOGICAL_AND);
}
CALreal calReductionComputeLogicalAnd3Dr(struct CALModel3D* model, struct CALSubstate3Dr* substate){
  return calReductionOperation3Dr(model, substate, REDUCTION_LOGICAL_AND);
}

CALbyte calReductionComputeBinaryAnd3Db(struct CALModel3D* model, struct CALSubstate3Db* substate){
  return calReductionOperation3Db(model, substate, REDUCTION_BINARY_AND);
}
CALint calReductionComputeBinaryAnd3Di(struct CALModel3D* model, struct CALSubstate3Di* substate){
  return calReductionOperation3Di(model, substate, REDUCTION_BINARY_AND);
}
CALreal calReductionComputeBinaryAnd3Dr(struct CALModel3D* model, struct CALSubstate3Dr* substate){ 
  return calReductionOperation3Dr(model, substate, REDUCTION_BINARY_AND);
}

CALbyte calReductionComputeLogicalOr3Db(struct CALModel3D* model, struct CALSubstate3Db* substate){
  return calReductionOperation3Db(model, substate, REDUCTION_LOGICAL_OR);
}
CALint calReductionComputeLogicalOr3Di(struct CALModel3D* model, struct CALSubstate3Di* substate){
  return calReductionOperation3Di(model, substate, REDUCTION_LOGICAL_OR);
}
CALreal calReductionComputeLogicalOr3Dr(struct CALModel3D* model, struct CALSubstate3Dr* substate){
  return calReductionOperation3Dr(model, substate, REDUCTION_LOGICAL_OR);
}

CALbyte calReductionComputeBinaryOr3Db(struct CALModel3D* model, struct CALSubstate3Db* substate){
  return calReductionOperation3Db(model, substate, REDUCTION_BINARY_OR);
}
CALint calReductionComputeBinaryOr3Di(struct CALModel3D* model, struct CALSubstate3Di* substate){ 
  return calReductionOperation3Di(model, substate, REDUCTION_BINARY_OR);
}
CALreal calReductionComputeBinaryOr3Dr(struct CALModel3D* model, struct CALSubstate3Dr* substate){
  return calReductionOperation3Dr(model, substate, REDUCTION_BINARY_OR);
}

CALbyte calReductionComputeLogicalXor3Db(struct CALModel3D* model, struct CALSubstate3Db* substate){
  return calReductionOperation3Db(model, substate, REDUCTION_LOGICAL_XOR);
}
CALint calReductionComputeLogicalXor3Di(struct CALModel3D* model, struct CALSubstate3Di* substate){ 
  return calReductionOperation3Di(model, substate, REDUCTION_LOGICAL_XOR);
}
CALreal calReductionComputeLogicalXor3Dr(struct CALModel3D* model, struct CALSubstate3Dr* substate){
  return calReductionOperation3Dr(model, substate, REDUCTION_LOGICAL_XOR);
}

CALbyte calReductionComputeBinaryXor3Db(struct CALModel3D* model, struct CALSubstate3Db* substate){
  return calReductionOperation3Db(model, substate, REDUCTION_BINARY_XOR);
}
CALint calReductionComputeBinaryXor3Di(struct CALModel3D* model, struct CALSubstate3Di* substate){ 
  return calReductionOperation3Di(model, substate, REDUCTION_BINARY_XOR);
}
CALreal calReductionComputeBinaryXor3Dr(struct CALModel3D* model, struct CALSubstate3Dr* substate){ 
  return calReductionOperation3Dr(model, substate, REDUCTION_BINARY_XOR);
}

CALbyte getValue3DbAtIndex(struct CALSubstate3Db* substate, CALint index){
  return substate->current[index];
}
CALint getValue3DiAtIndex(struct CALSubstate3Di* substate, CALint index){
  return substate->current[index];
}
CALreal getValue3DrAtIndex(struct CALSubstate3Dr* substate, CALint index){
  return substate->current[index];
}
