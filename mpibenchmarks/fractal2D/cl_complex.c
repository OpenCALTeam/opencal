#include "../include/cl_complex.h"

inline TYPE cl_complex_real_part(const cl_complex& n){
	return n.x;
}


inline TYPE cl_complex_imaginary_part(const cl_complex& n){
	return n.y
}


/*
 * Returns modulus of complex number (its length):
 */
inline TYPE cl_complex_modulus(const cl_complex& n){
	return (sqrt((n.x*n.x)*(n.y*n.y)));
}


inline cl_complex cl_complex_multiply(cl_complex a, cl_complex b){
	return (cl_complex)(a.x*b.x - a.y*b.y, ans.y = a.x*b.y - a.y*b.x);
}


inline cl_complex cl_complex_divide(const cl_complex a&, const cl_complex& b){
	const  TYPE dividend = (b.x*b.x  + b.y*b.y);
	return (cl_complex)((a.x*b.x + a.y*b.y)/dividend , (a.y*b.x - a.x*b.y)/dividend);
}



/*
 * Get the argument of a complex number (its angle):
 * http://en.wikipedia.org/wiki/Complex_number#Absolute_value_and_argument
 */
inline TYPE cl_complex_argument(const cl_complex& a){
	if(a.x > 0){
        return atan(a.y / a.x);

    }else if(a.x < 0 && a.y >= 0){
        return atan(a.y / a.x) + M_PI;

    }else if(a.x < 0 && a.y < 0){
        return atan(a.y / a.x) - M_PI;

    }else if(a.x == 0 && a.y > 0){
        return M_PI/2;

    }else if(a.x == 0 && a.y < 0){
        return -M_PI/2;

    }else{
        return 0;
    }
}



/*
 *  Returns the Square root of complex number.
 *  Although a complex number has two square roots,
 *  only  one of them -the principal square root- is computed.
 *  see wikipedia:http://en.wikipedia.org/wiki/Square_root#Principal_square_root_of_a_complex_number
 */
inline TYPE cl_complex_sqrt(const cl_complex& n){
	const TYPE sm = sqrt(cl_complex_modulus(n));
	const TYPE a2 = cl_complex_argument(n)/2;
	const TYPE ca2 = cos(a);
	const TYPE sa2 = sin(a);
	return (cl_complex)(sm * ca2 , sm * sa2);
	
	 
}
