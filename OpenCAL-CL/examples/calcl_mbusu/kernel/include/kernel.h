
#ifndef kernel_h
#define kernel_h


#define MOORE_NEIGHBORS 9
#define VON_NEUMANN_NEIGHBORS 7

#include <OpenCAL-CL/calcl3DActive.h>

#define YOUT 29
#define YIN 0
#define XE 159
#define XW 0
#define ZSUP 129
#define ZFONDO 0

#define TETA 0
#define MOIST_CONT 1
#define PSI 2
#define K 3
#define H 4
#define DQDH 5
#define CONVERGENCE 6
#define MOIST_DIFF 7
//mbusu->Q->teta = calAddSubstate3Dr(mbusu->model);
//	mbusu->Q->moist_cont = calAddSubstate3Dr(mbusu->model);
//	mbusu->Q->psi = calAddSubstate3Dr(mbusu->model);
//	mbusu->Q->k = calAddSubstate3Dr(mbusu->model);
//	mbusu->Q->h = calAddSubstate3Dr(mbusu->model);
//	mbusu->Q->dqdh = calAddSubstate3Dr(mbusu->model);
//	mbusu->Q->convergence = calAddSubstate3Dr(mbusu->model);
//	mbusu->Q->moist_diff = calAddSubstate3Dr(mbusu->model);



typedef struct {
	CALint ascii_output_time_step;				//[s] in seconds
	CALreal lato ;
	CALreal delta_t ;
	CALreal delta_t_cum ;
	CALreal delta_t_cum_prec;
	CALreal tetas1 , tetas2 , tetas3 , tetas4;
	CALreal tetar1 , tetar2 , tetar3 , tetar4 ;
	CALreal alfa1, alfa2, alfa3, alfa4;
	CALreal n1 , n2, n3, n4;
	CALreal ks1 , ks2 , ks3 , ks4;
	CALreal rain;
} Parameters;

#endif
