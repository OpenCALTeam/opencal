#ifndef cal2DBufferIO_h
#define cal2DBufferIO_h

#include "calCommon.cuh"
#include <stdio.h>

/*! \brief Loads a byte matrix from file. 
*/
void calfCudaLoadMatrix2Db(CALbyte* M, int rows, int columns, FILE* f, int i_substate);

/*! \brief Loads a byte matrix from file. 
*/
void calfCudaLoadMatrix2Di(CALint* M, int rows, int columns, FILE* f, int i_substate);

/*! \brief Loads a real (floating point) matrix from file. 
*/
void calfCudaLoadMatrix2Dr(CALreal* M, int rows, int columns, FILE* f, int i_substate);

/*! \brief Loads a real (floating point) matrix from file. 
*/
CALbyte calCudaLoadMatrix2Db(CALbyte* M, int rows, int columns, char* path, int i_substate);

/*! \brief Loads a real (floating point) matrix from file. 
*/
CALbyte calCudaLoadMatrix2Di(CALint* M, int rows, int columns, char* path, int i_substate);

/*! \brief Loads a real (floating point) matrix from file. 
*/
CALbyte calCudaLoadMatrix2Dr(CALreal* M, int rows, int columns, char* path, int i_substate);


/*! \brief Saves a byte matrix to file. 
*/
void calCudafSaveMatrix2Db(CALbyte* M, int rows, int columns, FILE* f, CALint index_substate);

/*! \brief Saves an int matrix to file. 
*/
void calCudafSaveMatrix2Di(CALint* M, int rows, int columns, FILE* f, CALint index_substate);

/*! \brief Saves a real (floating point) matrix to file. 
*/
void calCudafSaveMatrix2Dr(CALreal* M, int rows, int columns, FILE* f, CALint index_substate);


/*! \brief Saves a byte matrix to file. 
*/
CALbyte calCudaSaveMatrix2Db(CALbyte* M, int rows, int columns, char* path, CALint index_substate);
/*! \brief Saves a int matrix to file. 
*/
CALbyte calCudaSaveMatrix2Di(CALint* M, int rows, int columns, char* path, CALint index_substate);

/*! \brief Saves a real (floating point) matrix to file.
*/
CALbyte calCudaSaveMatrix2Dr(CALreal* M, int rows, int columns, char* path, CALint index_substate);


#endif
