#ifndef kernel_h
#define kernel_h

#include <OpenCAL-CL/calcl2DActive.h>
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
	double Pclock;	//AC clock [s]
	double Pc;		//cell side
	double Pac;		//area of the cell
	double PTsol;	//temperature of solidification
	double PTvent;	//temperature of lava at vent
	// new Paramenters
	double Pr_Tsol;
	double Pr_Tvent;
	double a;		// parametro per calcolo Pr
	double b;		// parametro per calcolo Pr
	double Phc_Tsol;
	double Phc_Tvent;
	double c;		// parametro per calcolo hc
	double d;		// parametro per calcolo hc
	double Pcool;
	double Prho;	//density
	double Pepsilon;	//emissivity
	double Psigma;	//Stephen-Boltzamnn constant
	double Pcv;		//Specific heat
	double rad2;
	unsigned int emission_time;
	double effusion_duration;

} Parameters;


#endif
