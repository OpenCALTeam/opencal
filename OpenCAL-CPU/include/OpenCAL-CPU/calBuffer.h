#ifndef cal_buffer
#define cal_buffer

#include <OpenCAL-CPU/calCommon.h>



void calSetBufferOperations(enum CALExecutionType execution_type);

/*! \brief Allocates a byte linearized matrix.
*/
CALbyte* calAllocBuffer_b(CALIndexes dimensions, int num_of_dimensions);

/*! \brief Allocates an int linearized matrix.
*/
CALint* calAllocBuffer_i(CALIndexes dimensions, int num_of_dimensions);

/*! \brief Allocates a real (floating point) linearized matrix.
*/
CALreal* calAllocBuffer_r(CALIndexes dimensions, int num_of_dimensions);



/*! \brief Deletes the memory associated to a byte linearized matrix.
*/
void calDeleteBuffer_b(CALbyte* M);

/*! \brief Deletes the memory associated to an int linearized matrix.
*/
void calDeleteBuffer_i(CALint* M);

/*! \brief Deletes the memory associated to a real (floating point) linearized matrix.
*/
void calDeleteBuffer_r(CALreal* M);

/*! \brief Byte linearized matrix copy function.
 */
extern void (* calCopyBuffer_b)(CALbyte* M_src, CALbyte* M_dest, int buffer_dimension);
/*! \brief Int linearized matrix copy function.
*/
extern void (* calCopyBuffer_i)(CALint* M_src, CALint* M_dest, int buffer_dimension);
/*! \brief Real linearized matrix copy function.
*/
extern void (* calCopyBuffer_r)(CALreal* M_src, CALreal* M_dest, int buffer_dimension);

/*! \brief Byte linearized matrix add function.
*/extern void (* calAddBuffer_b)(CALbyte* M_op1, CALbyte* M_op2,  CALbyte* M_dest, int buffer_dimension);
/*! \brief Int linearized matrix add function.
*/
extern void (* calAddBuffer_i)(CALint* M_op1, CALint* M_op2,  CALint* M_dest, int buffer_dimension);
/*! \brief Real linearized matrix add function.
*/
extern void (* calAddBuffer_r)(CALreal* M_op1, CALreal* M_op2,  CALreal* M_dest, int buffer_dimension);

/*! \brief Byte linearized matrix subtract function.
*/
extern void (* calSubtractBuffer_b)(CALbyte* M_op1, CALbyte* M_op2,  CALbyte* M_dest, int buffer_dimension);
/*! \brief Int linearized matrix subtract function.
*/
extern void (* calSubtractBuffer_i)(CALint* M_op1, CALint* M_op2,  CALint* M_dest, int buffer_dimension);
/*! \brief Real linearized matrix subtract function.
*/
extern void (* calSubtractBuffer_r)(CALreal* M_op1, CALreal* M_op2,  CALreal* M_dest, int buffer_dimension);

/*! \brief Sets a byte matrix to a constant value.
*/
extern void (*calSetBuffer_b)(CALbyte* M, int buffer_dimension, CALbyte value);
/*! \brief Sets a int matrix to a constant value.
*/
extern void (*calSetBuffer_i)(CALint* M, int buffer_dimension, CALint value);
/*! \brief Sets a real matrix to a constant value.
*/
extern void (*calSetBuffer_r)(CALreal* M, int buffer_dimension, CALreal value);




#endif
