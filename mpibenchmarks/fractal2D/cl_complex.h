#ifndef calcl_complex_h
#define calcl_complex_h


typedef double2 cl_double_complex;
typedef float2 cl_float_complex;

#ifdef DOUBLE_PRECISION
typedef cl_double_complex cl_complex;
typedef double TYPE;
#else
typedef cl_float_complex cl_complex;
typedef float TYPE;
#endif

#define IMAGINARY_UNIT ((cl_complex)(0.0, 1.0))


/*
 * Returns Real  component of complex number:
 */
inline TYPE cl_complex_real_part(const cl_complex& n);

/*
 * Returns Imaginary component of complex number:
 */
inline TYPE cl_complex_imaginary_part(const cl_complex& n);

/*
 * Returns modulus of complex number (its length):
 */
inline TYPE cl_complex_modulus(const cl_complex n);

/* 
 *  Multiplies two complex numbers according to the following rule:
 *  *  a = (aReal + I*aImag)
 *  b = (bReal + I*bImag)
 *  a * b = (aReal + I*aImag) * (bReal + I*bImag)
 *        = aReal*bReal +I*aReal*bImag +I*aImag*bReal +I^2*aImag*bImag
 *        = (aReal*bReal - aImag*bImag) + I*(aReal*bImag + aImag*bReal)
 * Note: I = IMAGINARY_UNIT
 */
inline cl_complex cl_complex_multiply(cl_complex a, cl_complex b);



/*
 * Divide two complex numbers:
 *
 *  aReal + I*aImag     (aReal + I*aImag) * (bReal - I*bImag)
 * ----------------- = ---------------------------------------
 *  bReal + I*bImag     (bReal + I*bImag) * (bReal - I*bImag)
 * 
 *        aReal*bReal - I*aReal*bImag + I*aImag*bReal - I^2*aImag*bImag
 *     = ---------------------------------------------------------------
 *            bReal^2 - I*bReal*bImag + I*bImag*bReal  -I^2*bImag^2
 * 
 *        aReal*bReal + aImag*bImag         aImag*bReal - Real*bImag 
 *     = ---------------------------- + I* --------------------------
 *            bReal^2 + bImag^2                bReal^2 + bImag^2
 * 
 * Note: I = IMAGINARY_UNIT
 */
inline cl_complex cl_complex_divide(const cl_complex a&, const cl_complex& b);

/*
 *  Returns the Square root of complex number.
 *  Although a complex number has two square roots,
 *  only  one of them -the principal square root- is computed.
 *  see wikipedia:http://en.wikipedia.org/wiki/Square_root#Principal_square_root_of_a_complex_number
 */
inline TYPE cl_complex_sqrt(const cl_complex& a);

/*
 * Get the argument of a complex number (its angle):
 * http://en.wikipedia.org/wiki/Complex_number#Absolute_value_and_argument
 */
inline TYPE cl_complex_argument(const cl_complex& a);

#endif //calcl_complex_h