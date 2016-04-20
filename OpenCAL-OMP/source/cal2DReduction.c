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

#include <OpenCAL-OMP/cal2DReduction.h>
#include <omp.h>
#include <stdlib.h>



CALbyte calReductionOperation2Db(struct CALModel2D* model, struct CALSubstate2Db* substate, enum REDUCTION_OPERATION operation){
	CALint i;
	CALbyte valueComputed = 0, tmp = 0;
	CALint numThreads = 1;
#ifdef _OPENMP
	CALbyte *values;

	numThreads = omp_get_num_procs();
	CAL_SET_NUM_THREADS(numThreads);
	values = (CALbyte*)malloc(sizeof(CALbyte) * numThreads);
#endif

	valueComputed = getValue2DbAtIndex(substate, 0);
#ifdef _OPENMP
	for (i = 0; i<numThreads; i++){
		values[i] = valueComputed;
	}
#endif

#pragma omp parallel private(tmp, i) shared(numThreads)
	{
		CALint start, end, threadId;
		threadId = CAL_GET_THREAD_NUM();
		start = threadId * (model->rows*model->columns) / numThreads;
		end = (threadId + 1) * (model->rows*model->columns) / numThreads;
		if (threadId == numThreads - 1){
			end = model->rows;
		}

		for (i = start; i<end; i++){
			if (i == 0){
				continue;
			}
			switch (operation){
			case REDUCTION_MAX:
				tmp = getValue2DbAtIndex(substate, i);
#ifdef _OPENMP
				if (values[threadId] < tmp){
					values[threadId] = tmp;
				}
#else
				if (valueComputed < tmp){
					valueComputed = tmp;
				}
#endif
				break;
			case REDUCTION_MIN:
				tmp = getValue2DbAtIndex(substate, i);
#ifdef _OPENMP
				if (values[threadId] > tmp){
					values[threadId] = tmp;
				}
#else
				if (valueComputed > tmp){
					valueComputed = tmp;
				}
#endif
				break;
			case REDUCTION_SUM:
#ifdef _OPENMP
				values[threadId] += getValue2DbAtIndex(substate, i);
#else
				valueComputed += getValue2DbAtIndex(substate, i);
#endif
				break;
			case REDUCTION_PROD:
#ifdef _OPENMP
				values[threadId] *= getValue2DbAtIndex(substate, i);
#else
				valueComputed *= getValue2DbAtIndex(substate, i);
#endif
				break;
			case REDUCTION_LOGICAL_AND:
#ifdef _OPENMP
				values[threadId] = (values[threadId] && getValue2DbAtIndex(substate, i));
#else
				valueComputed = (valueComputed && getValue2DbAtIndex(substate, i));
#endif
				break;
			case REDUCTION_BINARY_AND:
#ifdef _OPENMP
				values[threadId] = (values[threadId] & getValue2DbAtIndex(substate, i));
#else
				valueComputed = (valueComputed & getValue2DbAtIndex(substate, i));
#endif
				break;
			case REDUCTION_LOGICAL_OR:
#ifdef _OPENMP
				values[threadId] = (values[threadId] || getValue2DbAtIndex(substate, i));
#else
				valueComputed = (valueComputed || getValue2DbAtIndex(substate, i));
#endif
				break;
			case REDUCTION_BINARY_OR:
#ifdef _OPENMP
				values[threadId] = (values[threadId] | getValue2DbAtIndex(substate, i));
#else
				valueComputed = (valueComputed | getValue2DbAtIndex(substate, i));
#endif
				break;
			case REDUCTION_LOGICAL_XOR:
#ifdef _OPENMP
				values[threadId] = (!!values[threadId] != !!getValue2DbAtIndex(substate, i));
#else
				valueComputed = (!!valueComputed != !!getValue2DbAtIndex(substate, i));
#endif
				break;
			case REDUCTION_BINARY_XOR:
#ifdef _OPENMP
				values[threadId] = (values[threadId] ^ getValue2DbAtIndex(substate, i));
#else
				valueComputed = (valueComputed ^ getValue2DbAtIndex(substate, i));
#endif
				break;
			default:
				break;
			}
		}
	}

#ifdef _OPENMP
	valueComputed = values[0];

	for (i = 1; i<numThreads; i++){
		switch (operation){
		case REDUCTION_MAX:
			if (valueComputed < values[i]){
				valueComputed = values[i];
			}
			break;
		case REDUCTION_MIN:
			if (valueComputed > values[i]){
				valueComputed = values[i];
			}
			break;
		case REDUCTION_SUM:
			valueComputed += values[i];
			break;
		case REDUCTION_PROD:
			valueComputed *= values[i];
			break;
		case REDUCTION_LOGICAL_AND:
			valueComputed = (valueComputed && values[i]);
			break;
		case REDUCTION_BINARY_AND:
			valueComputed = (valueComputed & values[i]);
			break;
		case REDUCTION_LOGICAL_OR:
			valueComputed = (valueComputed || values[i]);
			break;
		case REDUCTION_BINARY_OR:
			valueComputed = (valueComputed | values[i]);
			break;
		case REDUCTION_LOGICAL_XOR:
			valueComputed = (!!valueComputed != !!values[i]);
			break;
		case REDUCTION_BINARY_XOR:
			valueComputed = (valueComputed ^ values[i]);
			break;
		default:
			break;
		}
	}

	if (values){
		free(values);
	}
#endif

	return valueComputed;
}
CALint calReductionOperation2Di(struct CALModel2D* model, struct CALSubstate2Di* substate, enum REDUCTION_OPERATION operation){
	CALint i, valueComputed = 0, tmp = 0;
	CALint numThreads = 1;
#ifdef _OPENMP
	CALint *values;

	numThreads = CAL_GET_NUM_PROCS();
	CAL_SET_NUM_THREADS(numThreads);
	values = (CALint*)malloc(sizeof(CALint) * numThreads);
#endif

	valueComputed = getValue2DiAtIndex(substate, 0);
#ifdef _OPENMP
	for (i = 0; i<numThreads; i++){
		values[i] = valueComputed;
	}
#endif

#pragma omp parallel private(tmp, i) shared(numThreads)
	{
		CALint start, end, threadId;
		threadId = CAL_GET_THREAD_NUM();
		start = threadId * (model->rows*model->columns) / numThreads;
		end = (threadId + 1) * (model->rows*model->columns) / numThreads;
		if (threadId == numThreads - 1){
			end = model->rows;
		}

		for (i = start; i<end; i++){
			if (i == 0){
				continue;
			}
			switch (operation){
			case REDUCTION_MAX:
				tmp = getValue2DiAtIndex(substate, i);
#ifdef _OPENMP
				if (values[threadId] < tmp){
					values[threadId] = tmp;
				}
#else
				if (valueComputed < tmp){
					valueComputed = tmp;
				}
#endif
				break;
			case REDUCTION_MIN:
				tmp = getValue2DiAtIndex(substate, i);
#ifdef _OPENMP
				if (values[threadId] > tmp){
					values[threadId] = tmp;
				}
#else
				if (valueComputed > tmp){
					valueComputed = tmp;
				}
#endif
				break;
			case REDUCTION_SUM:
#ifdef _OPENMP
				values[threadId] += getValue2DiAtIndex(substate, i);
#else
				valueComputed += getValue2DiAtIndex(substate, i);
#endif
				break;
			case REDUCTION_PROD:
#ifdef _OPENMP
				values[threadId] *= getValue2DiAtIndex(substate, i);
#else
				valueComputed *= getValue2DiAtIndex(substate, i);
#endif
				break;
			case REDUCTION_LOGICAL_AND:
#ifdef _OPENMP
				values[threadId] = (values[threadId] && getValue2DiAtIndex(substate, i));
#else
				valueComputed = (valueComputed && getValue2DiAtIndex(substate, i));
#endif
				break;
			case REDUCTION_BINARY_AND:
#ifdef _OPENMP
				values[threadId] = (values[threadId] & getValue2DiAtIndex(substate, i));
#else
				valueComputed = (valueComputed & getValue2DiAtIndex(substate, i));
#endif
				break;
			case REDUCTION_LOGICAL_OR:
#ifdef _OPENMP
				values[threadId] = (values[threadId] || getValue2DiAtIndex(substate, i));
#else
				valueComputed = (valueComputed || getValue2DiAtIndex(substate, i));
#endif
				break;
			case REDUCTION_BINARY_OR:
#ifdef _OPENMP
				values[threadId] = (values[threadId] | getValue2DiAtIndex(substate, i));
#else
				valueComputed = (valueComputed | getValue2DiAtIndex(substate, i));
#endif
				break;
			case REDUCTION_LOGICAL_XOR:
#ifdef _OPENMP
				values[threadId] = (!!values[threadId] != !!getValue2DiAtIndex(substate, i));
#else
				valueComputed = (!!valueComputed != !!getValue2DiAtIndex(substate, i));
#endif
				break;
			case REDUCTION_BINARY_XOR:
#ifdef _OPENMP
				values[threadId] = (values[threadId] ^ getValue2DiAtIndex(substate, i));
#else
				valueComputed = (valueComputed ^ getValue2DiAtIndex(substate, i));
#endif
				break;
			default:
				break;
			}
		}
	}

#ifdef _OPENMP
	valueComputed = values[0];

	for (i = 1; i<numThreads; i++){
		switch (operation){
		case REDUCTION_MAX:
			if (valueComputed < values[i]){
				valueComputed = values[i];
			}
			break;
		case REDUCTION_MIN:
			if (valueComputed > values[i]){
				valueComputed = values[i];
			}
			break;
		case REDUCTION_SUM:
			valueComputed += values[i];
			break;
		case REDUCTION_PROD:
			valueComputed *= values[i];
			break;
		case REDUCTION_LOGICAL_AND:
			valueComputed = (valueComputed && values[i]);
			break;
		case REDUCTION_BINARY_AND:
			valueComputed = (valueComputed & values[i]);
			break;
		case REDUCTION_LOGICAL_OR:
			valueComputed = (valueComputed || values[i]);
			break;
		case REDUCTION_BINARY_OR:
			valueComputed = (valueComputed | values[i]);
			break;
		case REDUCTION_LOGICAL_XOR:
			valueComputed = (!!valueComputed != !!values[i]);
			break;
		case REDUCTION_BINARY_XOR:
			valueComputed = (valueComputed ^ values[i]);
			break;
		default:
			break;
		}
	}

	if (values){
		free(values);
	}
#endif

	return valueComputed;
}
CALreal calReductionOperation2Dr(struct CALModel2D* model, struct CALSubstate2Dr* substate, enum REDUCTION_OPERATION operation){
	CALint i;
	CALreal valueComputed = 0, tmp = 0;
	CALint numThreads = 1;
#ifdef _OPENMP
	CALreal *values;

	numThreads = CAL_GET_NUM_PROCS();
	CAL_SET_NUM_THREADS(numThreads);
	values = (CALreal*)malloc(sizeof(CALreal) * numThreads);
#endif

	valueComputed = getValue2DrAtIndex(substate, 0);
#ifdef _OPENMP
	for (i = 0; i<numThreads; i++){
		values[i] = valueComputed;
	}
#endif

#pragma omp parallel private(tmp, i) shared(numThreads)
	{
		CALint start, end, threadId;
		threadId = CAL_GET_THREAD_NUM();
		start = threadId * (model->rows*model->columns) / numThreads;
		end = (threadId + 1) * (model->rows*model->columns) / numThreads;
		if (threadId == numThreads - 1){
			end = model->rows;
		}

		for (i = start; i<end; i++){
			if (i == 0){
				continue;
			}
			switch (operation){
			case REDUCTION_MAX:
				tmp = getValue2DrAtIndex(substate, i);
#ifdef _OPENMP
				if (values[threadId] < tmp){
					values[threadId] = tmp;
				}
#else
				if (valueComputed < tmp){
					valueComputed = tmp;
				}
#endif
				break;
			case REDUCTION_MIN:
				tmp = getValue2DrAtIndex(substate, i);
#ifdef _OPENMP
				if (values[threadId] > tmp){
					values[threadId] = tmp;
				}
#else
				if (valueComputed > tmp){
					valueComputed = tmp;
				}
#endif
				break;
			case REDUCTION_SUM:
#ifdef _OPENMP
				values[threadId] += getValue2DrAtIndex(substate, i);
#else
				valueComputed += getValue2DrAtIndex(substate, i);
#endif
				break;
			case REDUCTION_PROD:
#ifdef _OPENMP
				values[threadId] *= getValue2DrAtIndex(substate, i);
#else
				valueComputed *= getValue2DrAtIndex(substate, i);
#endif
				break;
			case REDUCTION_LOGICAL_AND:
#ifdef _OPENMP
				values[threadId] = (values[threadId] && getValue2DrAtIndex(substate, i));
#else
				valueComputed = (valueComputed && getValue2DrAtIndex(substate, i));
#endif
				break;
			case REDUCTION_BINARY_AND:
#ifdef _OPENMP
				values[threadId] = (CALreal)((CALint)values[threadId] & (CALint)getValue2DrAtIndex(substate, i));
#else
				valueComputed = (CALreal)((CALint)valueComputed & (CALint)getValue2DrAtIndex(substate, i));
#endif
				break;
			case REDUCTION_LOGICAL_OR:
#ifdef _OPENMP
				values[threadId] = (values[threadId] || getValue2DrAtIndex(substate, i));
#else
				valueComputed = (valueComputed || getValue2DrAtIndex(substate, i));
#endif
				break;
			case REDUCTION_BINARY_OR:
#ifdef _OPENMP
				values[threadId] = (CALreal)((CALint)values[threadId] | (CALint)getValue2DrAtIndex(substate, i));
#else
				valueComputed = (CALreal)((CALint)valueComputed | (CALint)getValue2DrAtIndex(substate, i));
#endif
				break;
			case REDUCTION_LOGICAL_XOR:
#ifdef _OPENMP
				values[threadId] = (!!values[threadId] != !!getValue2DrAtIndex(substate, i));
#else
				valueComputed = (!!valueComputed != !!getValue2DrAtIndex(substate, i));
#endif
				break;
			case REDUCTION_BINARY_XOR:
#ifdef _OPENMP
				values[threadId] = (CALreal)((CALint)values[threadId] ^ (CALint)getValue2DrAtIndex(substate, i));
#else
				valueComputed = (CALreal)((CALint)valueComputed ^ (CALint)getValue2DrAtIndex(substate, i));
#endif
				break;
			default:
				break;
			}
		}
	}

#ifdef _OPENMP
	valueComputed = values[0];

	for (i = 1; i<numThreads; i++){
		switch (operation){
		case REDUCTION_MAX:
			if (valueComputed < values[i]){
				valueComputed = values[i];
			}
			break;
		case REDUCTION_MIN:
			if (valueComputed > values[i]){
				valueComputed = values[i];
			}
			break;
		case REDUCTION_SUM:
			valueComputed += values[i];
			break;
		case REDUCTION_PROD:
			valueComputed *= values[i];
			break;
		case REDUCTION_LOGICAL_AND:
			valueComputed = (valueComputed && values[i]);
			break;
		case REDUCTION_BINARY_AND:
			valueComputed = (CALreal)((CALint)valueComputed & (CALint)values[i]);
			break;
		case REDUCTION_LOGICAL_OR:
			valueComputed = (valueComputed || values[i]);
			break;
		case REDUCTION_BINARY_OR:
			valueComputed = (CALreal)((CALint)valueComputed | (CALint)values[i]);
			break;
		case REDUCTION_LOGICAL_XOR:
			valueComputed = (!!valueComputed != !!values[i]);
			break;
		case REDUCTION_BINARY_XOR:
			valueComputed = (CALreal)((CALint)valueComputed ^ (CALint)values[i]);
			break;
		default:
			break;
		}
	}

	if (values){
		free(values);
	}
#endif

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
