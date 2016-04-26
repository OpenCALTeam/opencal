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

#include <OpenCAL-OMP/cal3DReduction.h>
#include <omp.h>
#include <stdlib.h>



CALbyte calReductionOperation3Db(struct CALModel3D* model, struct CALSubstate3Db* substate, enum REDUCTION_OPERATION operation){
	CALint i;
	CALbyte valueComputed = 0, tmp = 0;
	CALint numThreads = 1;
#ifdef _OPENMP
	CALbyte *values;

	numThreads = CAL_GET_NUM_PROCS();
	CAL_SET_NUM_THREADS(numThreads);
	values = (CALbyte*)malloc(sizeof(CALbyte) * numThreads);
#endif

	valueComputed = getValue3DbAtIndex(substate, 0);
#ifdef _OPENMP
	for (i = 0; i<numThreads; i++){
		values[i] = valueComputed;
	}
#endif

#pragma omp parallel private(tmp, i) shared(numThreads)
	{
		CALint start, end, threadId;
		threadId = CAL_GET_THREAD_NUM();
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
				tmp = getValue3DbAtIndex(substate, i);
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
				values[threadId] += getValue3DbAtIndex(substate, i);
#else
				valueComputed += getValue3DbAtIndex(substate, i);
#endif
				break;
			case REDUCTION_PROD:
#ifdef _OPENMP
				values[threadId] *= getValue3DbAtIndex(substate, i);
#else
				valueComputed *= getValue3DbAtIndex(substate, i);
#endif
				break;
			case REDUCTION_LOGICAL_AND:
#ifdef _OPENMP
				values[threadId] = (values[threadId] && getValue3DbAtIndex(substate, i));
#else
				valueComputed = (valueComputed && getValue3DbAtIndex(substate, i));
#endif
				break;
			case REDUCTION_BINARY_AND:
#ifdef _OPENMP
				values[threadId] = (values[threadId] & getValue3DbAtIndex(substate, i));
#else
				valueComputed = (valueComputed & getValue3DbAtIndex(substate, i));
#endif
				break;
			case REDUCTION_LOGICAL_OR:
#ifdef _OPENMP
				values[threadId] = (values[threadId] || getValue3DbAtIndex(substate, i));
#else
				valueComputed = (valueComputed || getValue3DbAtIndex(substate, i));
#endif
				break;
			case REDUCTION_BINARY_OR:
#ifdef _OPENMP
				values[threadId] = (values[threadId] | getValue3DbAtIndex(substate, i));
#else
				valueComputed = (valueComputed | getValue3DbAtIndex(substate, i));
#endif
				break;
			case REDUCTION_LOGICAL_XOR:
#ifdef _OPENMP
				values[threadId] = (!!values[threadId] != !!getValue3DbAtIndex(substate, i));
#else
				valueComputed = (!!valueComputed != !!getValue3DbAtIndex(substate, i));
#endif
				break;
			case REDUCTION_BINARY_XOR:
#ifdef _OPENMP
				values[threadId] = (values[threadId] ^ getValue3DbAtIndex(substate, i));
#else
				valueComputed = (valueComputed ^ getValue3DbAtIndex(substate, i));
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
CALint calReductionOperation3Di(struct CALModel3D* model, struct CALSubstate3Di* substate, enum REDUCTION_OPERATION operation){
	CALint i, valueComputed = 0, tmp = 0;
	CALint numThreads = 1;
#ifdef _OPENMP
	CALint *values;

	numThreads = CAL_GET_NUM_PROCS();
	CAL_SET_NUM_THREADS(numThreads);
	values = (CALint*)malloc(sizeof(CALint) * numThreads);
#endif

	valueComputed = getValue3DiAtIndex(substate, 0);
#ifdef _OPENMP
	for (i = 0; i<numThreads; i++){
		values[i] = valueComputed;
	}
#endif

#pragma omp parallel private(tmp, i) shared(numThreads)
	{
		CALint start, end, threadId;
		threadId = CAL_GET_THREAD_NUM();
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
				tmp = getValue3DiAtIndex(substate, i);
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
				values[threadId] += getValue3DiAtIndex(substate, i);
#else
				valueComputed += getValue3DiAtIndex(substate, i);
#endif
				break;
			case REDUCTION_PROD:
#ifdef _OPENMP
				values[threadId] *= getValue3DiAtIndex(substate, i);
#else
				valueComputed *= getValue3DiAtIndex(substate, i);
#endif
				break;
			case REDUCTION_LOGICAL_AND:
#ifdef _OPENMP
				values[threadId] = (values[threadId] && getValue3DiAtIndex(substate, i));
#else
				valueComputed = (valueComputed && getValue3DiAtIndex(substate, i));
#endif
				break;
			case REDUCTION_BINARY_AND:
#ifdef _OPENMP
				values[threadId] = (values[threadId] & getValue3DiAtIndex(substate, i));
#else
				valueComputed = (valueComputed & getValue3DiAtIndex(substate, i));
#endif
				break;
			case REDUCTION_LOGICAL_OR:
#ifdef _OPENMP
				values[threadId] = (values[threadId] || getValue3DiAtIndex(substate, i));
#else
				valueComputed = (valueComputed || getValue3DiAtIndex(substate, i));
#endif
				break;
			case REDUCTION_BINARY_OR:
#ifdef _OPENMP
				values[threadId] = (values[threadId] | getValue3DiAtIndex(substate, i));
#else
				valueComputed = (valueComputed | getValue3DiAtIndex(substate, i));
#endif
				break;
			case REDUCTION_LOGICAL_XOR:
#ifdef _OPENMP
				values[threadId] = (!!values[threadId] != !!getValue3DiAtIndex(substate, i));
#else
				valueComputed = (!!valueComputed != !!getValue3DiAtIndex(substate, i));
#endif
				break;
			case REDUCTION_BINARY_XOR:
#ifdef _OPENMP
				values[threadId] = (values[threadId] ^ getValue3DiAtIndex(substate, i));
#else
				valueComputed = (valueComputed ^ getValue3DiAtIndex(substate, i));
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
CALreal calReductionOperation3Dr(struct CALModel3D* model, struct CALSubstate3Dr* substate, enum REDUCTION_OPERATION operation){
	CALint i;
	CALreal valueComputed = 0, tmp = 0;
	CALint numThreads = 1;
#ifdef _OPENMP
	CALreal *values;

	numThreads = CAL_GET_NUM_PROCS();
	CAL_SET_NUM_THREADS(numThreads);
	values = (CALreal*)malloc(sizeof(CALreal) * numThreads);
#endif

	valueComputed = getValue3DrAtIndex(substate, 0);
#ifdef _OPENMP
	for (i = 0; i<numThreads; i++){
		values[i] = valueComputed;
	}
#endif

#pragma omp parallel private(tmp, i) shared(numThreads)
	{
		CALint start, end, threadId;
		threadId = CAL_GET_THREAD_NUM();
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
				tmp = getValue3DrAtIndex(substate, i);
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
				values[threadId] += getValue3DrAtIndex(substate, i);
#else
				valueComputed += getValue3DrAtIndex(substate, i);
#endif
				break;
			case REDUCTION_PROD:
#ifdef _OPENMP
				values[threadId] *= getValue3DrAtIndex(substate, i);
#else
				valueComputed *= getValue3DrAtIndex(substate, i);
#endif
				break;
			case REDUCTION_LOGICAL_AND:
#ifdef _OPENMP
				values[threadId] = (values[threadId] && getValue3DrAtIndex(substate, i));
#else
				valueComputed = (valueComputed && getValue3DrAtIndex(substate, i));
#endif
				break;
			case REDUCTION_BINARY_AND:
#ifdef _OPENMP
				values[threadId] = (CALreal)((CALint)values[threadId] & (CALint)getValue3DrAtIndex(substate, i));
#else
				valueComputed = (CALreal)((CALint)valueComputed & (CALint)getValue3DrAtIndex(substate, i));
#endif
				break;
			case REDUCTION_LOGICAL_OR:
#ifdef _OPENMP
				values[threadId] = (values[threadId] || getValue3DrAtIndex(substate, i));
#else
				valueComputed = (valueComputed || getValue3DrAtIndex(substate, i));
#endif
				break;
			case REDUCTION_BINARY_OR:
#ifdef _OPENMP
				values[threadId] = (CALreal)((CALint)values[threadId] | (CALint)getValue3DrAtIndex(substate, i));
#else
				valueComputed = (CALreal)((CALint)valueComputed | (CALint)getValue3DrAtIndex(substate, i));
#endif
				break;
			case REDUCTION_LOGICAL_XOR:
#ifdef _OPENMP
				values[threadId] = (!!values[threadId] != !!getValue3DrAtIndex(substate, i));
#else
				valueComputed = (!!valueComputed != !!getValue3DrAtIndex(substate, i));
#endif
				break;
			case REDUCTION_BINARY_XOR:
#ifdef _OPENMP
				values[threadId] = (CALreal)((CALint)values[threadId] ^ (CALint)getValue3DrAtIndex(substate, i));
#else
				valueComputed = (CALreal)((CALint)valueComputed ^ (CALint)getValue3DrAtIndex(substate, i));
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
