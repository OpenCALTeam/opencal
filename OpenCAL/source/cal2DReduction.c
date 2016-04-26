/*
 * Copyright (c) 2016 OpenCALTeam (https://github.com/OpenCALTeam),
 * Telesio Research Group,
 * Department of Mathematics and Computer Science,
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

#include <OpenCAL/cal2DReduction.h>
#include <stdlib.h>


CALbyte calReductionOperation2Db(struct CALModel2D* model, struct CALSubstate2Db* substate, enum REDUCTION_OPERATION operation){
  CALint i;
  CALint start, end;
  CALbyte valueComputed = 0, tmp = 0;

  start = 1;
  end = model->rows*model->columns;

  valueComputed = getValue2DbAtIndex(substate, 0);

  for (i = start; i<end; i++){
    switch (operation){
    case REDUCTION_MAX:
      tmp = getValue2DbAtIndex(substate, i);
      if (valueComputed < tmp){
	valueComputed = tmp;
      }
      break;
    case REDUCTION_MIN:
      tmp = getValue2DbAtIndex(substate, i);
      if (valueComputed > tmp){
	valueComputed = tmp;
      }
      break;
    case REDUCTION_SUM:
      valueComputed += getValue2DbAtIndex(substate, i);
      break;
    case REDUCTION_PROD:
      valueComputed *= getValue2DbAtIndex(substate, i);
      break;
    case REDUCTION_LOGICAL_AND:
      valueComputed = (valueComputed && getValue2DbAtIndex(substate, i));
      break;
    case REDUCTION_BINARY_AND:
      valueComputed = (valueComputed & getValue2DbAtIndex(substate, i));
      break;
    case REDUCTION_LOGICAL_OR:
      valueComputed = (valueComputed || getValue2DbAtIndex(substate, i));
      break;
    case REDUCTION_BINARY_OR:
      valueComputed = (valueComputed | getValue2DbAtIndex(substate, i));
      break;
    case REDUCTION_LOGICAL_XOR:
      valueComputed = (!!valueComputed != !!getValue2DbAtIndex(substate, i));
      break;
    case REDUCTION_BINARY_XOR:
      valueComputed = (valueComputed ^ getValue2DbAtIndex(substate, i));
      break;
    default:
      break;
    }
  }

  return valueComputed;
}
CALint calReductionOperation2Di(struct CALModel2D* model, struct CALSubstate2Di* substate, enum REDUCTION_OPERATION operation){
  CALint i;
  CALint start, end;
  CALint valueComputed = 0, tmp = 0;

  start = 1;
  end = model->rows*model->columns;

  valueComputed = getValue2DiAtIndex(substate, 0);

  for (i = start; i<end; i++){
    switch (operation){
    case REDUCTION_MAX:
      tmp = getValue2DiAtIndex(substate, i);
      if (valueComputed < tmp){
	valueComputed = tmp;
      }
      break;
    case REDUCTION_MIN:
      tmp = getValue2DiAtIndex(substate, i);
      if (valueComputed > tmp){
	valueComputed = tmp;
      }
      break;
    case REDUCTION_SUM:
      valueComputed += getValue2DiAtIndex(substate, i);
      break;
    case REDUCTION_PROD:
      valueComputed *= getValue2DiAtIndex(substate, i);
      break;
    case REDUCTION_LOGICAL_AND:
      valueComputed = (valueComputed && getValue2DiAtIndex(substate, i));
      break;
    case REDUCTION_BINARY_AND:
      valueComputed = (valueComputed & getValue2DiAtIndex(substate, i));
      break;
    case REDUCTION_LOGICAL_OR:
      valueComputed = (valueComputed || getValue2DiAtIndex(substate, i));
      break;
    case REDUCTION_BINARY_OR:
      valueComputed = (valueComputed | getValue2DiAtIndex(substate, i));
      break;
    case REDUCTION_LOGICAL_XOR:
      valueComputed = (!!valueComputed != !!getValue2DiAtIndex(substate, i));
      break;
    case REDUCTION_BINARY_XOR:
      valueComputed = (valueComputed ^ getValue2DiAtIndex(substate, i));
      break;
    default:
      break;
    }
  }

  return valueComputed;
}
CALreal calReductionOperation2Dr(struct CALModel2D* model, struct CALSubstate2Dr* substate, enum REDUCTION_OPERATION operation){
  CALint i;
  CALint start, end;
  CALreal valueComputed = 0, tmp = 0;

  start = 1;
  end = model->rows*model->columns;

  valueComputed = getValue2DrAtIndex(substate, 0);

  for (i = start; i<end; i++){
    switch (operation){
    case REDUCTION_MAX:
      tmp = getValue2DrAtIndex(substate, i);
      if (valueComputed < tmp){
	valueComputed = tmp;
      }
      break;
    case REDUCTION_MIN:
      tmp = getValue2DrAtIndex(substate, i);
      if (valueComputed > tmp){
	valueComputed = tmp;
      }
      break;
    case REDUCTION_SUM:
      valueComputed += getValue2DrAtIndex(substate, i);
      break;
    case REDUCTION_PROD:
      valueComputed *= getValue2DrAtIndex(substate, i);
      break;
    case REDUCTION_LOGICAL_AND:
      valueComputed = (valueComputed && getValue2DrAtIndex(substate, i));
      break;
    case REDUCTION_BINARY_AND:
      valueComputed = (CALreal)((CALint)valueComputed & (CALint)getValue2DrAtIndex(substate, i));
      break;
    case REDUCTION_LOGICAL_OR:
      valueComputed = (valueComputed || getValue2DrAtIndex(substate, i));
      break;
    case REDUCTION_BINARY_OR:
      valueComputed = (CALreal)((CALint)valueComputed | (CALint)getValue2DrAtIndex(substate, i));
      break;
    case REDUCTION_LOGICAL_XOR:
      valueComputed = (!!valueComputed != !!getValue2DrAtIndex(substate, i));
      break;
    case REDUCTION_BINARY_XOR:
      valueComputed = (CALreal)((CALint)valueComputed ^ (CALint)getValue2DrAtIndex(substate, i));
      break;
    default:
      break;
    }
  }

  return valueComputed;
}



CALbyte calReductionComputeMax2Db(struct CALModel2D* model, struct CALSubstate2Db* substate){
  return calReductionOperation2Db(model, substate, REDUCTION_MAX);
}
CALint calReductionComputeMax2Di(struct CALModel2D* model, struct CALSubstate2Di* substate){
  return calReductionOperation2Di(model, substate, REDUCTION_MAX);
}
CALreal calReductionComputeMax2Dr(struct CALModel2D* model, struct CALSubstate2Dr* substate){
  return calReductionOperation2Dr(model, substate, REDUCTION_MAX);
}

CALbyte calReductionComputeMin2Db(struct CALModel2D* model, struct CALSubstate2Db* substate){
  return calReductionOperation2Db(model, substate, REDUCTION_MIN);
}
CALint calReductionComputeMin2Di(struct CALModel2D* model, struct CALSubstate2Di* substate){
  return calReductionOperation2Di(model, substate, REDUCTION_MIN);
}
CALreal calReductionComputeMin2Dr(struct CALModel2D* model, struct CALSubstate2Dr* substate){
  return calReductionOperation2Dr(model, substate, REDUCTION_MIN);
}

CALbyte calReductionComputeSum2Db(struct CALModel2D* model, struct CALSubstate2Db* substate){
  return calReductionOperation2Db(model, substate, REDUCTION_SUM);
}
CALint calReductionComputeSum2Di(struct CALModel2D* model, struct CALSubstate2Di* substate){
  return calReductionOperation2Di(model, substate, REDUCTION_SUM);
}
CALreal calReductionComputeSum2Dr(struct CALModel2D* model, struct CALSubstate2Dr* substate){
  return calReductionOperation2Dr(model, substate, REDUCTION_SUM);
}

CALbyte calReductionComputeProd2Db(struct CALModel2D* model, struct CALSubstate2Db* substate){
  return calReductionOperation2Db(model, substate, REDUCTION_PROD);
}
CALint calReductionComputeProd2Di(struct CALModel2D* model, struct CALSubstate2Di* substate){
  return calReductionOperation2Di(model, substate, REDUCTION_SUM);
}
CALreal calReductionComputeProd2Dr(struct CALModel2D* model, struct CALSubstate2Dr* substate){
  return calReductionOperation2Dr(model, substate, REDUCTION_SUM);
}

CALbyte calReductionComputeLogicalAnd2Db(struct CALModel2D* model, struct CALSubstate2Db* substate){
  return calReductionOperation2Db(model, substate, REDUCTION_LOGICAL_AND);
}
CALint calReductionComputeLogicalAnd2Di(struct CALModel2D* model, struct CALSubstate2Di* substate){
  return calReductionOperation2Di(model, substate, REDUCTION_LOGICAL_AND);
}
CALreal calReductionComputeLogicalAnd2Dr(struct CALModel2D* model, struct CALSubstate2Dr* substate){
  return calReductionOperation2Dr(model, substate, REDUCTION_LOGICAL_AND);
}

CALbyte calReductionComputeBinaryAnd2Db(struct CALModel2D* model, struct CALSubstate2Db* substate){
  return calReductionOperation2Db(model, substate, REDUCTION_BINARY_AND);
}
CALint calReductionComputeBinaryAnd2Di(struct CALModel2D* model, struct CALSubstate2Di* substate){
  return calReductionOperation2Di(model, substate, REDUCTION_BINARY_AND);
}
CALreal calReductionComputeBinaryAnd2Dr(struct CALModel2D* model, struct CALSubstate2Dr* substate){
  return calReductionOperation2Dr(model, substate, REDUCTION_BINARY_AND);
}

CALbyte calReductionComputeLogicalOr2Db(struct CALModel2D* model, struct CALSubstate2Db* substate){
  return calReductionOperation2Db(model, substate, REDUCTION_LOGICAL_OR);
}
CALint calReductionComputeLogicalOr2Di(struct CALModel2D* model, struct CALSubstate2Di* substate){
  return calReductionOperation2Di(model, substate, REDUCTION_LOGICAL_OR);
}
CALreal calReductionComputeLogicalOr2Dr(struct CALModel2D* model, struct CALSubstate2Dr* substate){
  return calReductionOperation2Dr(model, substate, REDUCTION_LOGICAL_OR);
}

CALbyte calReductionComputeBinaryOr2Db(struct CALModel2D* model, struct CALSubstate2Db* substate){
  return calReductionOperation2Db(model, substate, REDUCTION_BINARY_OR);
}
CALint calReductionComputeBinaryOr2Di(struct CALModel2D* model, struct CALSubstate2Di* substate){
  return calReductionOperation2Di(model, substate, REDUCTION_BINARY_OR);
}
CALreal calReductionComputeBinaryOr2Dr(struct CALModel2D* model, struct CALSubstate2Dr* substate){
  return calReductionOperation2Dr(model, substate, REDUCTION_BINARY_OR);
}

CALbyte calReductionComputeLogicalXor2Db(struct CALModel2D* model, struct CALSubstate2Db* substate){
  return calReductionOperation2Db(model, substate, REDUCTION_LOGICAL_XOR);
}
CALint calReductionComputeLogicalXor2Di(struct CALModel2D* model, struct CALSubstate2Di* substate){
  return calReductionOperation2Di(model, substate, REDUCTION_LOGICAL_XOR);
}
CALreal calReductionComputeLogicalXor2Dr(struct CALModel2D* model, struct CALSubstate2Dr* substate){
  return calReductionOperation2Dr(model, substate, REDUCTION_LOGICAL_XOR);
}

CALbyte calReductionComputeBinaryXor2Db(struct CALModel2D* model, struct CALSubstate2Db* substate){
  return calReductionOperation2Db(model, substate, REDUCTION_BINARY_XOR);
}
CALint calReductionComputeBinaryXor2Di(struct CALModel2D* model, struct CALSubstate2Di* substate){
  return calReductionOperation2Di(model, substate, REDUCTION_BINARY_XOR);
}
CALreal calReductionComputeBinaryXor2Dr(struct CALModel2D* model, struct CALSubstate2Dr* substate){
  return calReductionOperation2Dr(model, substate, REDUCTION_BINARY_XOR);
}

CALbyte getValue2DbAtIndex(struct CALSubstate2Db* substate, CALint index){
  return substate->current[index];
}
CALint getValue2DiAtIndex(struct CALSubstate2Di* substate, CALint index){
  return substate->current[index];
}
CALreal getValue2DrAtIndex(struct CALSubstate2Dr* substate, CALint index){
  return substate->current[index];
}
