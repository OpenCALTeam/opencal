#ifndef cal2DBufferIO_h
#define cal2DBufferIO_h

#include <calCommon.h>
#include <stdio.h>



/*! \brief Loads a byte matrix from file.
*/
void calfLoadMatrix2Db(CALbyte* M, int rows, int columns, FILE* f);

/*! \brief Loads an int matrix from file. 
*/
void calfLoadMatrix2Di(CALint* M, int rows, int columns, FILE* f);

/*! \brief Loads a real (floating point) matrix from file. 
*/
void calfLoadMatrix2Dr(CALreal* M, int rows, int columns, FILE* f);



/*! \brief Loads a byte matrix from file. 
*/
CALbyte calLoadMatrix2Db(CALbyte* M, int rows, int columns, char* path);

/*! \brief Loads an int matrix from file. 
*/
CALbyte calLoadMatrix2Di(CALint* M, int rows, int columns, char* path);

/*! \brief Loads a real (floating point) matrix from file. 
*/
CALbyte calLoadMatrix2Dr(CALreal* M, int rows, int columns, char* path);



/*! \brief Saves a byte matrix to file. 
*/
void calfSaveMatrix2Db(CALbyte* M, int rows, int columns, FILE* f);
/*! \brief Saves an int matrix to file. 
*/
void calfSaveMatrix2Di(CALint* M, int rows, int columns, FILE* f);

/*! \brief Saves a real (floating point) matrix to file. 
*/
void calfSaveMatrix2Dr(CALreal* M, int rows, int columns, FILE* f);



/*! \brief Saves a byte matrix to file. 
*/
CALbyte calSaveMatrix2Db(CALbyte* M, int rows, int columns, char* path);
/*! \brief Saves a int matrix to file. 
*/
CALbyte calSaveMatrix2Di(CALint* M, int rows, int columns, char* path);
/*! \brief Saves a real (floating point) matrix to file.
*/
CALbyte calSaveMatrix2Dr(CALreal* M, int rows, int columns, char* path);



#endif
