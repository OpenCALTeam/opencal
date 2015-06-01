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

#include <calSteering2D.h>
#include <omp.h>
#include <stdlib.h>

CALbyte calSteeringComputeMax2Db(struct CALModel2D* model, struct CALSubstate2Db* substate){
	return calSteeringOperation2Db(model, substate, STEERING_MAX);
}
CALint calSteeringComputeMax2Di(struct CALModel2D* model, struct CALSubstate2Di* substate){
	return calSteeringOperation2Di(model, substate, STEERING_MAX);
}
CALreal calSteeringComputeMax2Dr(struct CALModel2D* model, struct CALSubstate2Dr* substate){
	return calSteeringOperation2Dr(model, substate, STEERING_MAX);
}

CALbyte calSteeringComputeMin2Db(struct CALModel2D* model, struct CALSubstate2Db* substate){
	return calSteeringOperation2Db(model, substate, STEERING_MIN);
}
CALint calSteeringComputeMin2Di(struct CALModel2D* model, struct CALSubstate2Di* substate){
	return calSteeringOperation2Di(model, substate, STEERING_MIN);
}
CALreal calSteeringComputeMin2Dr(struct CALModel2D* model, struct CALSubstate2Dr* substate){
	return calSteeringOperation2Dr(model, substate, STEERING_MIN);
}

CALbyte calSteeringComputeSum2Db(struct CALModel2D* model, struct CALSubstate2Db* substate){
	return calSteeringOperation2Db(model, substate, STEERING_SUM);
}
CALint calSteeringComputeSum2Di(struct CALModel2D* model, struct CALSubstate2Di* substate){
	return calSteeringOperation2Di(model, substate, STEERING_SUM);
}
CALreal calSteeringComputeSum2Dr(struct CALModel2D* model, struct CALSubstate2Dr* substate){
	return calSteeringOperation2Dr(model, substate, STEERING_SUM);
}

CALbyte calSteeringComputeProd2Db(struct CALModel2D* model, struct CALSubstate2Db* substate){	
	return calSteeringOperation2Db(model, substate, STEERING_PROD);
}
CALint calSteeringComputeProd2Di(struct CALModel2D* model, struct CALSubstate2Di* substate){
	return calSteeringOperation2Di(model, substate, STEERING_SUM);
}
CALreal calSteeringComputeProd2Dr(struct CALModel2D* model, struct CALSubstate2Dr* substate){
	return calSteeringOperation2Dr(model, substate, STEERING_SUM);
}

CALbyte calSteeringComputeLogicalAnd2Db(struct CALModel2D* model, struct CALSubstate2Db* substate){ 
	return calSteeringOperation2Db(model, substate, STEERING_LOGICAL_AND);
}
CALint calSteeringComputeLogicalAnd2Di(struct CALModel2D* model, struct CALSubstate2Di* substate){
	return calSteeringOperation2Di(model, substate, STEERING_LOGICAL_AND);
}
CALreal calSteeringComputeLogicalAnd2Dr(struct CALModel2D* model, struct CALSubstate2Dr* substate){
	return calSteeringOperation2Dr(model, substate, STEERING_LOGICAL_AND);
}

CALbyte calSteeringComputeBinaryAnd2Db(struct CALModel2D* model, struct CALSubstate2Db* substate){
	return calSteeringOperation2Db(model, substate, STEERING_BINARY_AND);
}
CALint calSteeringComputeBinaryAnd2Di(struct CALModel2D* model, struct CALSubstate2Di* substate){
	return calSteeringOperation2Di(model, substate, STEERING_BINARY_AND);
}
CALreal calSteeringComputeBinaryAnd2Dr(struct CALModel2D* model, struct CALSubstate2Dr* substate){ 
	return calSteeringOperation2Dr(model, substate, STEERING_BINARY_AND);
}

CALbyte calSteeringComputeLogicalOr2Db(struct CALModel2D* model, struct CALSubstate2Db* substate){
	return calSteeringOperation2Db(model, substate, STEERING_LOGICAL_OR);
}
CALint calSteeringComputeLogicalOr2Di(struct CALModel2D* model, struct CALSubstate2Di* substate){
	return calSteeringOperation2Di(model, substate, STEERING_LOGICAL_OR);
}
CALreal calSteeringComputeLogicalOr2Dr(struct CALModel2D* model, struct CALSubstate2Dr* substate){
	return calSteeringOperation2Dr(model, substate, STEERING_LOGICAL_OR);
}

CALbyte calSteeringComputeBinaryOr2Db(struct CALModel2D* model, struct CALSubstate2Db* substate){
	return calSteeringOperation2Db(model, substate, STEERING_BINARY_OR);
}
CALint calSteeringComputeBinaryOr2Di(struct CALModel2D* model, struct CALSubstate2Di* substate){ 
	return calSteeringOperation2Di(model, substate, STEERING_BINARY_OR);
}
CALreal calSteeringComputeBinaryOr2Dr(struct CALModel2D* model, struct CALSubstate2Dr* substate){
	return calSteeringOperation2Dr(model, substate, STEERING_BINARY_OR);
}

CALbyte calSteeringComputeLogicalXor2Db(struct CALModel2D* model, struct CALSubstate2Db* substate){
	return calSteeringOperation2Db(model, substate, STEERING_LOGICAL_XOR);
}
CALint calSteeringComputeLogicalXor2Di(struct CALModel2D* model, struct CALSubstate2Di* substate){ 
	return calSteeringOperation2Di(model, substate, STEERING_LOGICAL_XOR);
}
CALreal calSteeringComputeLogicalXor2Dr(struct CALModel2D* model, struct CALSubstate2Dr* substate){
	return calSteeringOperation2Dr(model, substate, STEERING_LOGICAL_XOR);
}

CALbyte calSteeringComputeBinaryXor2Db(struct CALModel2D* model, struct CALSubstate2Db* substate){
	return calSteeringOperation2Db(model, substate, STEERING_BINARY_XOR);
}
CALint calSteeringComputeBinaryXor2Di(struct CALModel2D* model, struct CALSubstate2Di* substate){ 
	return calSteeringOperation2Di(model, substate, STEERING_BINARY_XOR);
}
CALreal calSteeringComputeBinaryXor2Dr(struct CALModel2D* model, struct CALSubstate2Dr* substate){ 
	return calSteeringOperation2Dr(model, substate, STEERING_BINARY_XOR);
}

CALbyte calSteeringOperation2Db(struct CALModel2D* model, struct CALSubstate2Db* substate, enum STEERING_OPERATION operation){
	CALint i;
	CALbyte valueComputed = 0, tmp = 0;
	CALint numThreads = 1;
#ifdef _OPENMP	
	CALbyte *values;

	numThreads = omp_get_num_procs();
	omp_set_num_threads(numThreads);
	values = (CALbyte*) malloc(sizeof(CALbyte) * numThreads);
#endif

	valueComputed = getValue2DbAtIndex(substate, 0);
#ifdef _OPENMP
	for (i=0; i<numThreads; i++){
		values[i] = valueComputed;
	} 
#endif

#pragma omp parallel private(tmp, i) shared(numThreads)
	{
		CALint start, end, threadId;
		threadId = omp_get_thread_num();
		start = threadId * (model->rows*model->columns)/numThreads;
		end = (threadId+1) * (model->rows*model->columns)/numThreads;
		if(threadId == numThreads-1){
			end = model->rows;
		}

		for (i=start; i<end; i++){
			if(i==0){
				continue;
			}
			switch (operation){
			case STEERING_MAX: 
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
			case STEERING_MIN: 
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
			case STEERING_SUM: 
#ifdef _OPENMP
				values[threadId] += getValue2DbAtIndex(substate, i);	
#else
				valueComputed += getValue2DbAtIndex(substate, i);
#endif				
				break;
			case STEERING_PROD: 
#ifdef _OPENMP
				values[threadId] *= getValue2DbAtIndex(substate, i);	
#else
				valueComputed *= getValue2DbAtIndex(substate, i);
#endif			
				break;
			case STEERING_LOGICAL_AND:  
#ifdef _OPENMP
				values[threadId] = (values[threadId] && getValue2DbAtIndex(substate, i));
#else
				valueComputed = (valueComputed && getValue2DbAtIndex(substate, i));
#endif			 
				break;
			case STEERING_BINARY_AND:   
#ifdef _OPENMP
				values[threadId] = (values[threadId] & getValue2DbAtIndex(substate, i));
#else
				valueComputed = (valueComputed & getValue2DbAtIndex(substate, i));
#endif			  
				break;
			case STEERING_LOGICAL_OR:   
#ifdef _OPENMP
				values[threadId] = (values[threadId] || getValue2DbAtIndex(substate, i));
#else
				valueComputed = (valueComputed || getValue2DbAtIndex(substate, i));
#endif			 
				break;
			case STEERING_BINARY_OR:    
#ifdef _OPENMP
				values[threadId] = (values[threadId] | getValue2DbAtIndex(substate, i));
#else
				valueComputed = (valueComputed | getValue2DbAtIndex(substate, i)); 
#endif			  
				break;
			case STEERING_LOGICAL_XOR:   
#ifdef _OPENMP
				values[threadId] = (!!values[threadId] != !!getValue2DbAtIndex(substate, i));
#else
				valueComputed = (!!valueComputed != !!getValue2DbAtIndex(substate, i));
#endif			  
				break;
			case STEERING_BINARY_XOR:    
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
CALint calSteeringOperation2Di(struct CALModel2D* model, struct CALSubstate2Di* substate, enum STEERING_OPERATION operation){
	CALint i, valueComputed = 0, tmp = 0;
	CALint numThreads = 1;
#ifdef _OPENMP	
	CALint *values;

	numThreads = omp_get_num_procs();
	omp_set_num_threads(numThreads);
	values = (CALint*) malloc(sizeof(CALint) * numThreads);
#endif

	valueComputed = getValue2DiAtIndex(substate, 0);
#ifdef _OPENMP
	for (i=0; i<numThreads; i++){
		values[i] = valueComputed;
	} 
#endif

#pragma omp parallel private(tmp, i) shared(numThreads)
	{
		CALint start, end, threadId;
		threadId = omp_get_thread_num();
		start = threadId * (model->rows*model->columns)/numThreads;
		end = (threadId+1) * (model->rows*model->columns)/numThreads;
		if(threadId == numThreads-1){
			end = model->rows;
		}

		for (i=start; i<end; i++){
			if(i==0){
				continue;
			}
			switch (operation){
			case STEERING_MAX: 
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
			case STEERING_MIN: 
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
			case STEERING_SUM: 
#ifdef _OPENMP
				values[threadId] += getValue2DiAtIndex(substate, i);	
#else
				valueComputed += getValue2DiAtIndex(substate, i);
#endif				
				break;
			case STEERING_PROD: 
#ifdef _OPENMP
				values[threadId] *= getValue2DiAtIndex(substate, i);	
#else
				valueComputed *= getValue2DiAtIndex(substate, i);
#endif			
				break;
			case STEERING_LOGICAL_AND:  
#ifdef _OPENMP
				values[threadId] = (values[threadId] && getValue2DiAtIndex(substate, i));
#else
				valueComputed = (valueComputed && getValue2DiAtIndex(substate, i));
#endif			 
				break;
			case STEERING_BINARY_AND:   
#ifdef _OPENMP
				values[threadId] = (values[threadId] & getValue2DiAtIndex(substate, i));
#else
				valueComputed = (valueComputed & getValue2DiAtIndex(substate, i));
#endif			  
				break;
			case STEERING_LOGICAL_OR:   
#ifdef _OPENMP
				values[threadId] = (values[threadId] || getValue2DiAtIndex(substate, i));
#else
				valueComputed = (valueComputed || getValue2DiAtIndex(substate, i));
#endif			 
				break;
			case STEERING_BINARY_OR:    
#ifdef _OPENMP
				values[threadId] = (values[threadId] | getValue2DiAtIndex(substate, i));
#else
				valueComputed = (valueComputed | getValue2DiAtIndex(substate, i)); 
#endif			  
				break;
			case STEERING_LOGICAL_XOR:   
#ifdef _OPENMP
				values[threadId] = (!!values[threadId] != !!getValue2DiAtIndex(substate, i));
#else
				valueComputed = (!!valueComputed != !!getValue2DiAtIndex(substate, i));
#endif			  
				break;
			case STEERING_BINARY_XOR:    
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
CALreal calSteeringOperation2Dr(struct CALModel2D* model, struct CALSubstate2Dr* substate, enum STEERING_OPERATION operation){
	CALint i;
	CALreal valueComputed = 0, tmp = 0;
	CALint numThreads = 1;
#ifdef _OPENMP	
	CALreal *values;

	numThreads = omp_get_num_procs();
	omp_set_num_threads(numThreads);
	values = (CALreal*) malloc(sizeof(CALreal) * numThreads);
#endif

	valueComputed = getValue2DrAtIndex(substate, 0);
#ifdef _OPENMP
	for (i=0; i<numThreads; i++){
		values[i] = valueComputed;
	} 
#endif

#pragma omp parallel private(tmp, i) shared(numThreads)
	{
		CALint start, end, threadId;
		threadId = omp_get_thread_num();
		start = threadId * (model->rows*model->columns)/numThreads;
		end = (threadId+1) * (model->rows*model->columns)/numThreads;
		if(threadId == numThreads-1){
			end = model->rows;
		}

		for (i=start; i<end; i++){
			if(i==0){
				continue;
			}
			switch (operation){
			case STEERING_MAX: 
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
			case STEERING_MIN: 
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
			case STEERING_SUM: 
#ifdef _OPENMP
				values[threadId] += getValue2DrAtIndex(substate, i);	
#else
				valueComputed += getValue2DrAtIndex(substate, i);
#endif				
				break;
			case STEERING_PROD: 
#ifdef _OPENMP
				values[threadId] *= getValue2DrAtIndex(substate, i);	
#else
				valueComputed *= getValue2DrAtIndex(substate, i);
#endif			
				break;
			case STEERING_LOGICAL_AND:  
#ifdef _OPENMP
				values[threadId] = (values[threadId] && getValue2DrAtIndex(substate, i));
#else
				valueComputed = (valueComputed && getValue2DrAtIndex(substate, i));
#endif			 
				break;
			case STEERING_BINARY_AND:   
#ifdef _OPENMP
				values[threadId] = (CALreal) ((CALint)values[threadId] & (CALint)getValue2DrAtIndex(substate, i));
#else
				valueComputed = (CALreal) ((CALint)valueComputed & (CALint)getValue2DrAtIndex(substate, i));
#endif			  
				break;
			case STEERING_LOGICAL_OR:   
#ifdef _OPENMP
				values[threadId] = (values[threadId] || getValue2DrAtIndex(substate, i));
#else
				valueComputed = (valueComputed || getValue2DrAtIndex(substate, i));
#endif			 
				break;
			case STEERING_BINARY_OR:    
#ifdef _OPENMP
				values[threadId] = (CALreal) ((CALint)values[threadId] | (CALint)getValue2DrAtIndex(substate, i));
#else
				valueComputed = (CALreal) ((CALint)valueComputed | (CALint)getValue2DrAtIndex(substate, i)); 
#endif			  
				break;
			case STEERING_LOGICAL_XOR:   
#ifdef _OPENMP
				values[threadId] = (!!values[threadId] != !!getValue2DrAtIndex(substate, i));
#else
				valueComputed = (!!valueComputed != !!getValue2DrAtIndex(substate, i));
#endif			  
				break;
			case STEERING_BINARY_XOR:    
#ifdef _OPENMP
				values[threadId] = (CALreal) ((CALint)values[threadId] ^ (CALint)getValue2DrAtIndex(substate, i));
#else
				valueComputed = (CALreal) ((CALint)valueComputed ^ (CALint)getValue2DrAtIndex(substate, i)); 
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

CALbyte getValue2DbAtIndex(struct CALSubstate2Db* substate, CALint index){
	return substate->current[index];
}
CALint getValue2DiAtIndex(struct CALSubstate2Di* substate, CALint index){
	return substate->current[index];
}
CALreal getValue2DrAtIndex(struct CALSubstate2Dr* substate, CALint index){
	return substate->current[index];
}
