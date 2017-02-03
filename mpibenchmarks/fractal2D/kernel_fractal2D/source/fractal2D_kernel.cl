// Conway's game of Life transition function kernel

#include <OpenCAL-CL/calcl2D.h>
#define DOUBLE_PRECISION (1)

typedef double2 cl_double_complex;
typedef float2 cl_float_complex;

#ifdef DOUBLE_PRECISION
typedef cl_double_complex cl_complex;
typedef double TYPE;
#else
typedef cl_float_complex cl_complex;
typedef float TYPE;
#endif


inline TYPE cl_complex_real_part(const cl_complex* n){
	return n->x;
}


inline TYPE cl_complex_imaginary_part(const cl_complex* n){
	return n->y;
}


/*
 * Returns modulus of complex number (its length):
 */
inline TYPE cl_complex_modulus(const cl_complex* n){
	return (sqrt( (n->x*n->x)+(n->y*n->y) ));
}

inline cl_complex cl_complex_add(const cl_complex* a, const cl_complex* b){
	return (cl_complex)( a->x + b->x, a->y + b->y );
}

inline cl_complex cl_complex_multiply(const cl_complex* a, const cl_complex* b){
	return (cl_complex)(a->x*b->x - a->y*b->y,  a->x*b->y + a->y*b->x);
}



inline cl_complex cl_complex_ipow(const cl_complex* base ,  int exp ){
	cl_complex res;
	res.x=res.y=1;
	while(exp){
		if(exp & 1)
			res=cl_complex_multiply(&res,base);
		exp>>=1;
		res = cl_complex_multiply(&res,&res);
		}
	
	return res;
}


inline cl_complex cl_complex_divide(const cl_complex* a, const cl_complex* b){
	const  TYPE dividend = (b->x*b->x  + b->y*b->y);
	return (cl_complex)((a->x*b->x + a->y*b->y)/dividend , (a->y*b->x - a->x*b->y)/dividend);
}



/*
 * Get the argument of a complex number (its angle):
 * http://en.wikipedia.org/wiki/Complex_number#Absolute_value_and_argument
 */
inline TYPE cl_complex_argument(const cl_complex* a){
	if(a->x > 0){
        return atan(a->y / a->x);

    }else if(a->x < 0 && a->y >= 0){
        return atan(a->y / a->x) + M_PI;

    }else if(a->x < 0 && a->y < 0){
        return atan(a->y / a->x) - M_PI;

    }else if(a->x == 0 && a->y > 0){
        return M_PI/2;

    }else if(a->x == 0 && a->y < 0){
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
inline cl_complex cl_complex_sqrt(const cl_complex* n){
	const TYPE sm = sqrt(cl_complex_modulus(n));
	const TYPE a2 = cl_complex_argument(n)/2;
	const TYPE ca2 = cos(a2);
	const TYPE sa2 = sin(a2);
	return (cl_complex)(sm * ca2 , sm * sa2);
	
	 
}

// MODEL KERNEL STARTS HERE -----------------------------------------

#define DEVICE_Q_fractal (0)



#define MAXITERATIONS (5000)
#define SIZE (16384)

#define moveX (0)
#define moveY (0)


cl_complex convertToComplex(const int x , const int y, const TYPE zoom, const int DIMX, const int DIMY){
	TYPE jx = 1.5 * (x - DIMX / 2.0) / (0.5 * zoom * DIMX) + moveX;	
	TYPE jy = (y - DIMY / 2.0) / (0.5 * zoom * DIMY) + moveY;
	return (cl_complex)(jx,jy);
}

cl_complex juliaFunctor(const cl_complex p, cl_complex c){
	//return cuCaddf(cuCmulf(p,p),c);
	const cl_complex c_ipow = cl_complex_multiply(&p,&p);//cl_complex_ipow(&p,2);
	return cl_complex_add( &c_ipow, &c);
	//cTYPE a = cMakecuComplex(-1.0,0);

	//return cuCaddf(cuCmulf(cuCexp(p),p),c);

}

int evolveComplexPoint(cl_complex p,cl_complex c){
	//printf("%f %f , %f \n", p.x,p.y,cl_complex_modulus(&p));
	int it =1;
	while(it <= MAXITERATIONS && cl_complex_modulus(&p) <=10){
		p=juliaFunctor(p,c);
		it++;
		
	}
	
	//printf("re=%f,im=%f, it=%i\n",p.y,p.y,it);
	return it;
}


__kernel void fractal2D_transitionFunction(__CALCL_MODEL_2D)
{

	calclThreadCheck2D();
    int i = calclGlobalRow()+borderSize;   
	int j = calclGlobalColumn();
	
	const TYPE zoom = 1.0;
	const cl_complex c;
	c.x = -0.391;
	c.y = -0.587;
	
		
	int global_i = i-borderSize + offset;

	cl_complex p = convertToComplex(global_i,j,zoom, SIZE, SIZE);	
	
	//printf("%d\n", calclGetRows());
	calclSet2Di(MODEL_2D, 
				DEVICE_Q_fractal, 
				i, 
				j, 
				evolveComplexPoint(p,c)
	);

	return;
	
	


}
