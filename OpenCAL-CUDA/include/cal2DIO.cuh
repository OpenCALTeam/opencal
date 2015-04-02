#ifndef cal2DIO_h
#define cal2DIO_h

#include "calCommon.cuh"
#include <stdio.h>

/*! \brief Loads a byte substate from file for CUDA version. 
*/
CALbyte calCudaLoadSubstate2Db(struct CudaCALModel2D* ca2D, char* path, int i_substate);

/*! \brief Loads a byte substate from file for CUDA version. 
*/
CALbyte calCudaLoadSubstate2Di(struct CudaCALModel2D* ca2D, char* path, int i_substate);

/*! \brief Loads a real (floating point) substate from file directly in CudaCALModel2D struct for CUDA version. 
*/
CALbyte calCudaLoadSubstate2Dr(struct CudaCALModel2D* ca2D, char* path, int i_substate);

/*! \brief Saves a byte substate to file. 
*/
CALbyte calCudaSaveSubstate2Db(struct CudaCALModel2D* ca2D, char* path, CALint index_substate);

/*! \brief Saves a int substate to file. 
*/
CALbyte calCudaSaveSubstate2Di(struct CudaCALModel2D* ca2D, char* path, CALint index_substate);

/*! \brief Saves a real substate to file. 
*/
CALbyte calCudaSaveSubstate2Dr(struct CudaCALModel2D* ca2D, char* path, CALint index_substate);


#endif
