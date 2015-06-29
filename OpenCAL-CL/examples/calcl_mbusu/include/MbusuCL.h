/*
 * mbusuCL.h
 *
 *  Created on: 16/apr/2015
 *      Author: alessio
 */
#ifndef CA_H_
#define CA_H_

extern "C"{
#include <cal3D.h>
#include <cal3DIO.h>
#include <cal3DBuffer.h>
#include <cal3DBufferIO.h>
#include <cal3DRun.h>
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
	CALint ascii_output_time_step = 864000;				//[s] in seconds
	CALreal lato = 5.0;
	CALreal delta_t = 10.0;
	CALreal delta_t_cum = 0.0;
	CALreal delta_t_cum_prec = 0.0;
	CALreal tetas1 = 0.368, tetas2 = 0.351, tetas3 = 0.325, tetas4 = 0.325;
	CALreal tetar1 = 0.102, tetar2 = 0.0985, tetar3 = 0.0859, tetar4 = 0.0859;
	CALreal alfa1 = 0.0334, alfa2 = 0.0363, alfa3 = 0.0345, alfa4 = 0.0345;
	CALreal n1 = 1.982, n2 = 1.632, n3 = 1.573, n4 = 1.573;
	CALreal ks1 = 0.009154, ks2 = 0.005439, ks3 = 0.004803, ks4 = 0.048032;
	CALreal rain = 0.000023148148;
} Parameters;

typedef struct {

	CALint rows;
	CALint cols;
	CALint layers;
	Parameters parameters;
	mbusuSubstates* Q;
	CALModel3D * model;

} Mbusu;

extern Mbusu* mbusu;							//the cellular automaton

void initMbusu();
void exit();
void simulationInitialize();

#endif /* CA_H_ */
