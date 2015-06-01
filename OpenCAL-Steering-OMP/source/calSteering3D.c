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

#include <calSteering3D.h>
#include <omp.h>
#include <stdlib.h>

CALbyte calSteeringComputeMax3Db(struct CALModel3D* model, struct CALSubstate3Db* substate){
	return calSteeringOperation3Db(model, substate, STEERING_MAX);
}
CALint calSteeringComputeMax3Di(struct CALModel3D* model, struct CALSubstate3Di* substate){
	return calSteeringOperation3Di(model, substate, STEERING_MAX);
}
CALreal calSteeringComputeMax3Dr(struct CALModel3D* model, struct CALSubstate3Dr* substate){
	return calSteeringOperation3Dr(model, substate, STEERING_MAX);
}

CALbyte calSteeringComputeMin3Db(struct CALModel3D* model, struct CALSubstate3Db* substate){
	return calSteeringOperation3Db(model, substate, STEERING_MIN);
}
CALint calSteeringComputeMin3Di(struct CALModel3D* model, struct CALSubstate3Di* substate){
	return calSteeringOperation3Di(model, substate, STEERING_MIN);
}
CALreal calSteeringComputeMin3Dr(struct CALModel3D* model, struct CALSubstate3Dr* substate){
	return calSteeringOperation3Dr(model, substate, STEERING_MIN);
}

CALbyte calSteeringComputeSum3Db(struct CALModel3D* model, struct CALSubstate3Db* substate){
	return calSteeringOperation3Db(model, substate, STEERING_SUM);
}
CALint calSteeringComputeSum3Di(struct CALModel3D* model, struct CALSubstate3Di* substate){
	return calSteeringOperation3Di(model, substate, STEERING_SUM);
}
CALreal calSteeringComputeSum3Dr(struct CALModel3D* model, struct CALSubstate3Dr* substate){
	return calSteeringOperation3Dr(model, substate, STEERING_SUM);
}

CALbyte calSteeringComputeProd3Db(struct CALModel3D* model, struct CALSubstate3Db* substate){	
	return calSteeringOperation3Db(model, substate, STEERING_PROD);
}
CALint calSteeringComputeProd3Di(struct CALModel3D* model, struct CALSubstate3Di* substate){
	return calSteeringOperation3Di(model, substate, STEERING_SUM);
}
CALreal calSteeringComputeProd3Dr(struct CALModel3D* model, struct CALSubstate3Dr* substate){
	return calSteeringOperation3Dr(model, substate, STEERING_SUM);
}

CALbyte calSteeringComputeLogicalAnd3Db(struct CALModel3D* model, struct CALSubstate3Db* substate){ 
	return calSteeringOperation3Db(model, substate, STEERING_LOGICAL_AND);
}
CALint calSteeringComputeLogicalAnd3Di(struct CALModel3D* model, struct CALSubstate3Di* substate){
	return calSteeringOperation3Di(model, substate, STEERING_LOGICAL_AND);
}
CALreal calSteeringComputeLogicalAnd3Dr(struct CALModel3D* model, struct CALSubstate3Dr* substate){
	return calSteeringOperation3Dr(model, substate, STEERING_LOGICAL_AND);
}

CALbyte calSteeringComputeBinaryAnd3Db(struct CALModel3D* model, struct CALSubstate3Db* substate){
	return calSteeringOperation3Db(model, substate, STEERING_BINARY_AND);
}
CALint calSteeringComputeBinaryAnd3Di(struct CALModel3D* model, struct CALSubstate3Di* substate){
	return calSteeringOperation3Di(model, substate, STEERING_BINARY_AND);
}
CALreal calSteeringComputeBinaryAnd3Dr(struct CALModel3D* model, struct CALSubstate3Dr* substate){ 
	return calSteeringOperation3Dr(model, substate, STEERING_BINARY_AND);
}

CALbyte calSteeringComputeLogicalOr3Db(struct CALModel3D* model, struct CALSubstate3Db* substate){
	return calSteeringOperation3Db(model, substate, STEERING_LOGICAL_OR);
}
CALint calSteeringComputeLogicalOr3Di(struct CALModel3D* model, struct CALSubstate3Di* substate){
	return calSteeringOperation3Di(model, substate, STEERING_LOGICAL_OR);
}
CALreal calSteeringComputeLogicalOr3Dr(struct CALModel3D* model, struct CALSubstate3Dr* substate){
	return calSteeringOperation3Dr(model, substate, STEERING_LOGICAL_OR);
}

CALbyte calSteeringComputeBinaryOr3Db(struct CALModel3D* model, struct CALSubstate3Db* substate){
	return calSteeringOperation3Db(model, substate, STEERING_BINARY_OR);
}
CALint calSteeringComputeBinaryOr3Di(struct CALModel3D* model, struct CALSubstate3Di* substate){ 
	return calSteeringOperation3Di(model, substate, STEERING_BINARY_OR);
}
CALreal calSteeringComputeBinaryOr3Dr(struct CALModel3D* model, struct CALSubstate3Dr* substate){
	return calSteeringOperation3Dr(model, substate, STEERING_BINARY_OR);
}

CALbyte calSteeringComputeLogicalXor3Db(struct CALModel3D* model, struct CALSubstate3Db* substate){
	return calSteeringOperation3Db(model, substate, STEERING_LOGICAL_XOR);
}
CALint calSteeringComputeLogicalXor3Di(struct CALModel3D* model, struct CALSubstate3Di* substate){ 
	return calSteeringOperation3Di(model, substate, STEERING_LOGICAL_XOR);
}
CALreal calSteeringComputeLogicalXor3Dr(struct CALModel3D* model, struct CALSubstate3Dr* substate){
	return calSteeringOperation3Dr(model, substate, STEERING_LOGICAL_XOR);
}

CALbyte calSteeringComputeBinaryXor3Db(struct CALModel3D* model, struct CALSubstate3Db* substate){
	return calSteeringOperation3Db(model, substate, STEERING_BINARY_XOR);
}
CALint calSteeringComputeBinaryXor3Di(struct CALModel3D* model, struct CALSubstate3Di* substate){ 
	return calSteeringOperation3Di(model, substate, STEERING_BINARY_XOR);
}
CALreal calSteeringComputeBinaryXor3Dr(struct CALModel3D* model, struct CALSubstate3Dr* substate){ 
	return calSteeringOperation3Dr(model, substate, STEERING_BINARY_XOR);
}

CALbyte calSteeringOperation3Db(struct CALModel3D* model, struct CALSubstate3Db* substate, enum STEERING_OPERATION operation){
	CALint i;
	CALbyte valueComputed = 0, tmp = 0;
	CALint numThreads = 1;
#ifdef _OPENMP	
	CALbyte *values;

	numThreads = omp_get_num_procs();
	omp_set_num_threads(numThreads);
	values = (CALbyte*) malloc(sizeof(CALbyte) * numThreads);
#endif

	valueComputed = getValue3DbAtIndex(substate, 0);
#ifdef _OPENMP
	for (i=0; i<numThreads; i++){
		values[i] = valueComputed;
	} 
#endif

#pragma omp parallel private(tmp, i) shared(numThreads)
	{
		CALint start, end, threadId;
		threadId = omp_get_thread_num();
		start = threadId * (model->rows*model->columns*model->layers)/numThreads;
		end = (threadId+1) * (model->rows*model->columns*model->layers)/numThreads;
		if(threadId == numThreads-1){
			end = model->rows;
		}

		for (i=start; i<end; i++){
			if(i==0){
				continue;
			}
			switch (operation){
			case STEERING_MAX: 
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
			case STEERING_MIN: 
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
			case STEERING_SUM: 
#ifdef _OPENMP
				values[threadId] += getValue3DbAtIndex(substate, i);	
#else
				valueComputed += getValue3DbAtIndex(substate, i);
#endif				
				break;
			case STEERING_PROD: 
#ifdef _OPENMP
				values[threadId] *= getValue3DbAtIndex(substate, i);	
#else
				valueComputed *= getValue3DbAtIndex(substate, i);
#endif			
				break;
			case STEERING_LOGICAL_AND:  
#ifdef _OPENMP
				values[threadId] = (values[threadId] && getValue3DbAtIndex(substate, i));
#else
				valueComputed = (valueComputed && getValue3DbAtIndex(substate, i));
#endif			 
				break;
			case STEERING_BINARY_AND:   
#ifdef _OPENMP
				values[threadId] = (values[threadId] & getValue3DbAtIndex(substate, i));
#else
				valueComputed = (valueComputed & getValue3DbAtIndex(substate, i));
#endif			  
				break;
			case STEERING_LOGICAL_OR:   
#ifdef _OPENMP
				values[threadId] = (values[threadId] || getValue3DbAtIndex(substate, i));
#else
				valueComputed = (valueComputed || getValue3DbAtIndex(substate, i));
#endif			 
				break;
			case STEERING_BINARY_OR:    
#ifdef _OPENMP
				values[threadId] = (values[threadId] | getValue3DbAtIndex(substate, i));
#else
				valueComputed = (valueComputed | getValue3DbAtIndex(substate, i)); 
#endif			  
				break;
			case STEERING_LOGICAL_XOR:   
#ifdef _OPENMP
				values[threadId] = (!!values[threadId] != !!getValue3DbAtIndex(substate, i));
#else
				valueComputed = (!!valueComputed != !!getValue3DbAtIndex(substate, i));
#endif			  
				break;
			case STEERING_BINARY_XOR:    
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

	for (i=1; i<numThreads; i++){
		switch (operation){
		case STEERING_MAX:
			if (valueComputed < values[i]){
				valueComputed = values[i];
			}
			break;
		case STEERING_MIN: 
			if (valueComputed > values[i]){
				valueComputed = values[i];
			}
			break;
		case STEERING_SUM: 
			valueComputed += values[i];
			break;
		case STEERING_PROD: 
			valueComputed *= values[i];
			break;
		case STEERING_LOGICAL_AND: 
			valueComputed = (valueComputed && values[i]); 
			break;
		case STEERING_BINARY_AND: 
			valueComputed = (valueComputed & values[i]);
			break;
		case STEERING_LOGICAL_OR: 
			valueComputed = (valueComputed || values[i]); 
			break;
		case STEERING_BINARY_OR: 
			valueComputed = (valueComputed | values[i]);
			break;
		case STEERING_LOGICAL_XOR: 
			valueComputed = (!!valueComputed != !!values[i]); 
			break;
		case STEERING_BINARY_XOR:  
			valueComputed = (valueComputed ^ values[i]);
			break;
		default:
			break;
		}
	} 

	if(values){
		free(values);
	} 
#endif

	return valueComputed;
}
CALint calSteeringOperation3Di(struct CALModel3D* model, struct CALSubstate3Di* substate, enum STEERING_OPERATION operation){
	CALint i, valueComputed = 0, tmp = 0;
	CALint numThreads = 1;
#ifdef _OPENMP	
	CALint *values;

	numThreads = omp_get_num_procs();
	omp_set_num_threads(numThreads);
	values = (CALint*) malloc(sizeof(CALint) * numThreads);
#endif

	valueComputed = getValue3DiAtIndex(substate, 0);
#ifdef _OPENMP
	for (i=0; i<numThreads; i++){
		values[i] = valueComputed;
	} 
#endif

#pragma omp parallel private(tmp, i) shared(numThreads)
	{
		CALint start, end, threadId;
		threadId = omp_get_thread_num();
		start = threadId * (model->rows*model->columns*model->layers)/numThreads;
		end = (threadId+1) * (model->rows*model->columns*model->layers)/numThreads;
		if(threadId == numThreads-1){
			end = model->rows;
		}

		for (i=start; i<end; i++){
			if(i==0){
				continue;
			}
			switch (operation){
			case STEERING_MAX: 
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
			case STEERING_MIN: 
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
			case STEERING_SUM: 
#ifdef _OPENMP
				values[threadId] += getValue3DiAtIndex(substate, i);	
#else
				valueComputed += getValue3DiAtIndex(substate, i);
#endif				
				break;
			case STEERING_PROD: 
#ifdef _OPENMP
				values[threadId] *= getValue3DiAtIndex(substate, i);	
#else
				valueComputed *= getValue3DiAtIndex(substate, i);
#endif			
				break;
			case STEERING_LOGICAL_AND:  
#ifdef _OPENMP
				values[threadId] = (values[threadId] && getValue3DiAtIndex(substate, i));
#else
				valueComputed = (valueComputed && getValue3DiAtIndex(substate, i));
#endif			 
				break;
			case STEERING_BINARY_AND:   
#ifdef _OPENMP
				values[threadId] = (values[threadId] & getValue3DiAtIndex(substate, i));
#else
				valueComputed = (valueComputed & getValue3DiAtIndex(substate, i));
#endif			  
				break;
			case STEERING_LOGICAL_OR:   
#ifdef _OPENMP
				values[threadId] = (values[threadId] || getValue3DiAtIndex(substate, i));
#else
				valueComputed = (valueComputed || getValue3DiAtIndex(substate, i));
#endif			 
				break;
			case STEERING_BINARY_OR:    
#ifdef _OPENMP
				values[threadId] = (values[threadId] | getValue3DiAtIndex(substate, i));
#else
				valueComputed = (valueComputed | getValue3DiAtIndex(substate, i)); 
#endif			  
				break;
			case STEERING_LOGICAL_XOR:   
#ifdef _OPENMP
				values[threadId] = (!!values[threadId] != !!getValue3DiAtIndex(substate, i));
#else
				valueComputed = (!!valueComputed != !!getValue3DiAtIndex(substate, i));
#endif			  
				break;
			case STEERING_BINARY_XOR:    
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

	for (i=1; i<numThreads; i++){
		switch (operation){
		case STEERING_MAX:
			if (valueComputed < values[i]){
				valueComputed = values[i];
			}
			break;
		case STEERING_MIN: 
			if (valueComputed > values[i]){
				valueComputed = values[i];
			}
			break;
		case STEERING_SUM: 
			valueComputed += values[i];
			break;
		case STEERING_PROD: 
			valueComputed *= values[i];
			break;
		case STEERING_LOGICAL_AND: 
			valueComputed = (valueComputed && values[i]); 
			break;
		case STEERING_BINARY_AND: 
			valueComputed = (valueComputed & values[i]);
			break;
		case STEERING_LOGICAL_OR: 
			valueComputed = (valueComputed || values[i]); 
			break;
		case STEERING_BINARY_OR: 
			valueComputed = (valueComputed | values[i]);
			break;
		case STEERING_LOGICAL_XOR: 
			valueComputed = (!!valueComputed != !!values[i]); 
			break;
		case STEERING_BINARY_XOR:  
			valueComputed = (valueComputed ^ values[i]);
			break;
		default:
			break;
		}
	} 

	if(values){
		free(values);
	} 
#endif

	return valueComputed;
}
CALreal calSteeringOperation3Dr(struct CALModel3D* model, struct CALSubstate3Dr* substate, enum STEERING_OPERATION operation){
	CALint i;
	CALreal valueComputed = 0, tmp = 0;
	CALint numThreads = 1;
#ifdef _OPENMP	
	CALreal *values;

	numThreads = omp_get_num_procs();
	omp_set_num_threads(numThreads);
	values = (CALreal*) malloc(sizeof(CALreal) * numThreads);
#endif

	valueComputed = getValue3DrAtIndex(substate, 0);
#ifdef _OPENMP
	for (i=0; i<numThreads; i++){
		values[i] = valueComputed;
	} 
#endif

#pragma omp parallel private(tmp, i) shared(numThreads)
	{
		CALint start, end, threadId;
		threadId = omp_get_thread_num();
		start = threadId * (model->rows*model->columns*model->layers)/numThreads;
		end = (threadId+1) * (model->rows*model->columns*model->layers)/numThreads;
		if(threadId == numThreads-1){
			end = model->rows;
		}

		for (i=start; i<end; i++){
			if(i==0){
				continue;
			}
			switch (operation){
			case STEERING_MAX: 
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
			case STEERING_MIN: 
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
			case STEERING_SUM: 
#ifdef _OPENMP
				values[threadId] += getValue3DrAtIndex(substate, i);	
#else
				valueComputed += getValue3DrAtIndex(substate, i);
#endif				
				break;
			case STEERING_PROD: 
#ifdef _OPENMP
				values[threadId] *= getValue3DrAtIndex(substate, i);	
#else
				valueComputed *= getValue3DrAtIndex(substate, i);
#endif			
				break;
			case STEERING_LOGICAL_AND:  
#ifdef _OPENMP
				values[threadId] = (values[threadId] && getValue3DrAtIndex(substate, i));
#else
				valueComputed = (valueComputed && getValue3DrAtIndex(substate, i));
#endif			 
				break;
			case STEERING_BINARY_AND:   
#ifdef _OPENMP
				values[threadId] = (CALreal) ((CALint)values[threadId] & (CALint)getValue3DrAtIndex(substate, i));
#else
				valueComputed = (CALreal) ((CALint)valueComputed & (CALint)getValue3DrAtIndex(substate, i));
#endif			  
				break;
			case STEERING_LOGICAL_OR:   
#ifdef _OPENMP
				values[threadId] = (values[threadId] || getValue3DrAtIndex(substate, i));
#else
				valueComputed = (valueComputed || getValue3DrAtIndex(substate, i));
#endif			 
				break;
			case STEERING_BINARY_OR:    
#ifdef _OPENMP
				values[threadId] = (CALreal) ((CALint)values[threadId] | (CALint)getValue3DrAtIndex(substate, i));
#else
				valueComputed = (CALreal) ((CALint)valueComputed | (CALint)getValue3DrAtIndex(substate, i)); 
#endif			  
				break;
			case STEERING_LOGICAL_XOR:   
#ifdef _OPENMP
				values[threadId] = (!!values[threadId] != !!getValue3DrAtIndex(substate, i));
#else
				valueComputed = (!!valueComputed != !!getValue3DrAtIndex(substate, i));
#endif			  
				break;
			case STEERING_BINARY_XOR:    
#ifdef _OPENMP
				values[threadId] = (CALreal) ((CALint)values[threadId] ^ (CALint)getValue3DrAtIndex(substate, i));
#else
				valueComputed = (CALreal) ((CALint)valueComputed ^ (CALint)getValue3DrAtIndex(substate, i)); 
#endif			    
				break;
			default:
				break;
			}
		}
	}

#ifdef _OPENMP
	valueComputed = values[0];

	for (i=1; i<numThreads; i++){
		switch (operation){
		case STEERING_MAX:
			if (valueComputed < values[i]){
				valueComputed = values[i];
			}
			break;
		case STEERING_MIN: 
			if (valueComputed > values[i]){
				valueComputed = values[i];
			}
			break;
		case STEERING_SUM: 
			valueComputed += values[i];
			break;
		case STEERING_PROD: 
			valueComputed *= values[i];
			break;
		case STEERING_LOGICAL_AND: 
			valueComputed = (valueComputed && values[i]); 
			break;
		case STEERING_BINARY_AND: 
			valueComputed = (CALreal) ((CALint)valueComputed & (CALint)values[i]);
			break;
		case STEERING_LOGICAL_OR: 
			valueComputed = (valueComputed || values[i]); 
			break;
		case STEERING_BINARY_OR: 
			valueComputed = (CALreal) ((CALint)valueComputed | (CALint)values[i]);
			break;
		case STEERING_LOGICAL_XOR: 
			valueComputed = (!!valueComputed != !!values[i]); 
			break;
		case STEERING_BINARY_XOR:  
			valueComputed = (CALreal) ((CALint)valueComputed ^ (CALint)values[i]);
			break;
		default:
			break;
		}
	} 

	if(values){
		free(values);
	} 
#endif

	return valueComputed;
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

