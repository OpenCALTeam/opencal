#include <OpenCAL-CPU/calBuffer.h>
#include <OpenCAL-CPU/calRunSerial.h>
#include <OpenCAL-CPU/calRunParallel.h>


CALbyte*calAllocBuffer_b(CALIndexes dimensions, int num_of_dimensions)
{
    int overall_dimension = 1;
    int i = 0;
    for( ; i < num_of_dimensions; i++ )
        overall_dimension *= dimensions[i];
    return (CALbyte*)malloc(sizeof(CALbyte) * overall_dimension );
}

CALint*calAllocBuffer_i(CALIndexes dimensions, int num_of_dimensions)
{
    int overall_dimension = 1;
    int i = 0;
    for( ; i < num_of_dimensions; i++ )
        overall_dimension *= dimensions[i];
    return (CALint*)malloc(sizeof(CALint) * overall_dimension );
}

CALreal*calAllocBuffer_r(CALIndexes dimensions, int num_of_dimensions)
{
    int overall_dimension = 1;
    int i = 0;
    for( ; i < num_of_dimensions; i++ )
        overall_dimension *= dimensions[i];
    return (CALreal*)malloc(sizeof(CALreal) * overall_dimension );

}

void calDeleteBuffer_b(CALbyte* M)
{
    free(M);
}

void calDeleteBuffer_i(CALint* M)
{
    free(M);
}

void calDeleteBuffer_r(CALreal* M)
{
    free(M);
}


void calCopyBuffer_b(CALbyte* M_src, CALbyte* M_dest, int buffer_dimension)
{
    int tn;
    int ttotal;
    size_t size;

    int start;
    int chunk;

    size = buffer_dimension;

#pragma omp parallel private (start, chunk, tn, ttotal)
    {
        ttotal = CAL_GET_NUM_THREADS();

        tn = CAL_GET_THREAD_NUM();
        chunk = size / ttotal;
        start = tn * chunk;

        if (tn == ttotal - 1)
            chunk = size - start;

        memcpy(M_dest + start, M_src + start,
               sizeof(CALbyte) * chunk);
    }
}

void calCopyBuffer_i(CALint* M_src, CALint* M_dest, int buffer_dimension)
{
    int tn;
    int ttotal;
    size_t size;

    int start;
    int chunk;

    size = buffer_dimension;

#pragma omp parallel private (start, chunk, tn, ttotal)
    {
        ttotal = CAL_GET_NUM_THREADS();

        tn = CAL_GET_THREAD_NUM();
        chunk = size / ttotal;
        start = tn * chunk;

        if (tn == ttotal - 1)
            chunk = size - start;

        memcpy(M_dest + start, M_src + start,
               sizeof(CALint) * chunk);
    }
}

void calCopyBuffer_r(CALreal* M_src, CALreal* M_dest, int buffer_dimension)
{
    int tn;
    int ttotal;
    size_t size;

    int start;
    int chunk;

    size = buffer_dimension;

#pragma omp parallel private (tn, start, chunk, ttotal)
    {
        ttotal = CAL_GET_NUM_THREADS();


        tn = CAL_GET_THREAD_NUM();
        chunk = size / ttotal;
        start = tn * chunk;

        if (tn == ttotal - 1)
            chunk = size - start;

        memcpy(M_dest + start, M_src + start,
               sizeof(CALreal) * chunk);
    }


}

void calAddBuffer_b(CALbyte* M_op1, CALbyte* M_op2, CALbyte* M_dest, int buffer_dimension)
{
    int i;
#pragma omp parallel for firstprivate(M_op1, M_op2, M_dest)
    for( i = 0; i < buffer_dimension; i++ )
    {
        M_dest[i] = M_op1[i] + M_op2[i];
    }
}

void calAddBuffer_i(CALint* M_op1, CALint* M_op2, CALint* M_dest, int buffer_dimension)
{
    int i;
#pragma omp parallel for firstprivate(M_op1, M_op2, M_dest)
    for( i = 0; i < buffer_dimension; i++ )
    {
        M_dest[i] = M_op1[i] + M_op2[i];
    }
}

void calAddBuffer_r(CALreal* M_op1, CALreal* M_op2, CALreal* M_dest, int buffer_dimension)
{
    int i;
#pragma omp parallel for firstprivate(M_op1, M_op2, M_dest)
    for( i = 0; i < buffer_dimension; i++ )
    {
        M_dest[i] = M_op1[i] + M_op2[i];
    }
}

void calSubtractBuffer_b(CALbyte* M_op1, CALbyte* M_op2, CALbyte* M_dest, int buffer_dimension)
{
    int i;
#pragma omp parallel for firstprivate(M_op1, M_op2, M_dest)
    for( i = 0; i < buffer_dimension; i++ )
    {
        M_dest[i] = M_op1[i] - M_op2[i];
    }
}

void calSubtractBuffer_i(CALint* M_op1, CALint* M_op2, CALint* M_dest, int buffer_dimension)
{
    int i;
#pragma omp parallel for firstprivate(M_op1, M_op2, M_dest)
    for( i = 0; i < buffer_dimension; i++ )
    {
        M_dest[i] = M_op1[i] - M_op2[i];
    }
}

void calSubtractBuffer_r(CALreal* M_op1, CALreal* M_op2, CALreal* M_dest, int buffer_dimension)
{
    int i;
#pragma omp parallel for firstprivate(M_op1, M_op2, M_dest)
    for( i = 0; i < buffer_dimension; i++ )
    {
        M_dest[i] = M_op1[i] - M_op2[i];
    }
}

void calSetBuffer_b(CALbyte* M, int buffer_dimension, CALbyte value)
{
    memset(M, value, sizeof(CALbyte)*buffer_dimension);
}

void calSetBuffer_i(CALint* M, int buffer_dimension, CALint value)
{
    memset(M, value, sizeof(CALint)*buffer_dimension);
}

void calSetBuffer_r(CALreal* M, int buffer_dimension, CALreal value)
{
    memset(M, value, sizeof(CALreal)*buffer_dimension);
}
