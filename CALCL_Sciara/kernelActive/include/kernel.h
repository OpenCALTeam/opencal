#ifndef kernel_h
#define kernel_h

#include <cal2DActive.h>
#define F(i) (i+3)
#define SZ 0
#define SLT 1
#define ST 2
#define NUMBER_OF_OUTFLOWS 8
#define MOORE_NEIGHBORS 9
#define VON_NEUMANN_NEIGHBORS 5

typedef struct{
	int x;
	int y;
} Vent;

typedef struct{

	// Parameters
	CALreal Pclock;	//AC clock [s]
	CALreal Pc;		//cell side
	CALreal Pac;		//area of the cell
	CALreal PTsol;	//temperature of solidification
	CALreal PTvent;	//temperature of lava at vent
	// new Paramenters
	CALreal Pr_Tsol;
	CALreal Pr_Tvent;
	CALreal a;		// parametro per calcolo Pr
	CALreal b;		// parametro per calcolo Pr
	CALreal Phc_Tsol;
	CALreal Phc_Tvent;
	CALreal c;		// parametro per calcolo hc
	CALreal d;		// parametro per calcolo hc
	CALreal Pcool;
	CALreal Prho;	//density
	CALreal Pepsilon;	//emissivity
	CALreal Psigma;	//Stephen-Boltzamnn constant
	CALreal Pcv;		//Specific heat
	CALreal rad2;
	unsigned int emission_time;
	CALreal effusion_duration;

} Parameters;


#endif



