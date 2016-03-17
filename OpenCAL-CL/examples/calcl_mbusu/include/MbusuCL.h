/*
 * mbusuCL.h
 *
 *  Created on: 16/apr/2015
 *      Author: alessio
 */
#ifndef CA_H_
#define CA_H_

extern "C"{
#include <OpenCAL/cal3D.h>
#include <OpenCAL/cal3DIO.h>
#include <OpenCAL/cal3DBuffer.h>
#include <OpenCAL/cal3DBufferIO.h>
#include <OpenCAL/cal3DRun.h>
}
#include <math.h>
#include <stdlib.h>

#define YOUT 29
#define YIN 0
#define XE 159
#define XW 0
#define ZSUP 129
#define ZFONDO 0

#define COLS YOUT+1
#define ROWS XE+1
#define LAYERS ZSUP+1

#define MOORE_NEIGHBORS 9
#define VON_NEUMANN_NEIGHBORS 5


#define SAVE_PATH "./results/ris10g.txt"


#define ROOT_DATA_DIR "."

#define KERNEL_SRC_AC ROOT_DATA_DIR"/kernelActive/source/"
#define KERNEL_INC_AC ROOT_DATA_DIR"/kernelActive/include/"
#define KERNEL_SRC ROOT_DATA_DIR"/CALCL_Mbusu/kernel/source/"
#define KERNEL_INC ROOT_DATA_DIR"/CALCL_Mbusu/kernel/include/"

typedef struct {
	struct CALSubstate3Dr *teta;
	struct CALSubstate3Dr *moist_cont;
	struct CALSubstate3Dr *psi;
	struct CALSubstate3Dr *k;
	struct CALSubstate3Dr *h;
	struct CALSubstate3Dr *dqdh;
	struct CALSubstate3Dr *convergence;
	struct CALSubstate3Dr *moist_diff;
} mbusuSubstates;

typedef struct {
	CALint ascii_output_time_step;				//[s] in seconds
	CALreal lato ;
	CALreal delta_t;
	CALreal delta_t_cum;
	CALreal delta_t_cum_prec;
	CALreal tetas1 , tetas2, tetas3, tetas4;
	CALreal tetar1 , tetar2, tetar3, tetar4;
	CALreal alfa1 , alfa2, alfa3, alfa4;
	CALreal n1 , n2, n3, n4;
	CALreal ks1 , ks2, ks3, ks4;
	CALreal rain;
} Parameters;

typedef struct {

	CALint rows;
	CALint cols;
	CALint layers;
	Parameters parameters;
	mbusuSubstates* Q;
	CALModel3D * host_CA;

} Mbusu;

extern Mbusu* mbusu;							//the cellular automaton

void initMbusu();
void exit();
void simulationInitialize();

#endif /* CA_H_ */
