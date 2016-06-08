#include <OpenCAL-CPU/calBuffer.h>
#include <OpenCAL-CPU/calRunSerial.h>
#include <OpenCAL-CPU/calRunParallel.h>

void calSetBufferOperations(enum CALExecutionType execution_type)
{
    if(execution_type == SERIAL)
    {
        calCopyBuffer_b = calSerialCopyBuffer_b;
        calCopyBuffer_r = calSerialCopyBuffer_r;
        calCopyBuffer_i = calSerialCopyBuffer_i;
    }
    else
    {

    }
}

CALbyte*calAllocBuffer_b(CALIndexes dimensions, int num_of_dimensions)
{

}

CALint*calAllocBuffer_i(CALIndexes dimensions, int num_of_dimensions)
{

}

CALreal*calAllocBuffer_r(CALIndexes dimensions, int num_of_dimensions)
{

}

void calDeleteBuffer_b(CALbyte* M)
{

}

void calDeleteBuffer_i(CALint* M)
{

}

void calDeleteBuffer_r(CALreal* M)
{

}

