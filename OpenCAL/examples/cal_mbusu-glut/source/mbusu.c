#include <math.h>
#include <OpenCAL/cal3DReduction.h>
#include "mbusu.h"
#include <stdio.h>
#include <stdlib.h>

FILE *stream;

//-----------------------------------------------------------------------
//	   THE mbusu cellular automaton definition section
//-----------------------------------------------------------------------

//cadef and rundef
struct CALModel3D* mbusu;							//the cellular automaton
struct mbusuSubstates Q;							//the set of substates
struct CALRun3D* mbusuSimulation;					//the simulartion run

int ascii_output_time_step = 864000;				//[s] in seconds
float lato = 5.0;
float delta_t = 10.0;
float delta_t_cum = 0.0;
float delta_t_cum_prec = 0.0;
float tetas1 = 0.368, tetas2 = 0.351, tetas3 = 0.325, tetas4 = 0.325;
float tetar1 = 0.102, tetar2 = 0.0985, tetar3 = 0.0859, tetar4 = 0.0859;
float alfa1 = 0.0334, alfa2 = 0.0363, alfa3 = 0.0345, alfa4 = 0.0345;
float n1 = 1.982, n2 = 1.632, n3 = 1.573, n4 = 1.573;
float ks1 = 0.009154, ks2 = 0.005439, ks3 = 0.004803, ks4 = 0.048032;
float rain = 0.000023148148;
float prm_vis = 0.0;

//------------------------------------------------------------------------------
//					mbusu transition function
//------------------------------------------------------------------------------

//first elementary process
void mbusuTransitionFunction(struct CALModel3D* ca, int _i, int _j, int _k)
{
	int _k_inv;

	double quota, teta, satur, psi, h, k, uno_su_dqdh, teta_pioggia; //
	double alfa, n, tetar, tetas, ks, Delta_h, moist_cont;
	double denom_pow, denompow_uno, denompow_due, denompow_tre;
	double exp_c, exp_d, satur_expc, satur_expd;	// 	
	double volume, moist_print;
	int i, Ymid;
	double convergence;
	double teta_start, denom_pow_start, moist_diff;
	double temp_value;

	volume = lato*lato*lato;
	_k_inv = (mbusu->slices - 1) - _k;
	quota = lato*_k_inv;
	h = calGet3Dr(ca, Q.h, _i, _j, _k);

	//---- PARAMETRI SUOLO

	if ((_i > 19) && (_i < 60) && (_k_inv>79) && (_k_inv < 100))
	{
		tetas = tetas4;
		tetar = tetar4;
		alfa = alfa4;
		n = n4;
		ks = ks4;
	}
	else if ((_k_inv > 111) && (_k_inv < 124))
	{
		tetas = tetas2;
		tetar = tetar2;
		alfa = alfa2;
		n = n2;
		ks = ks2;
	}
	else if (_k_inv > 123)
	{
		tetas = tetas1;
		tetar = tetar1;
		alfa = alfa1;
		n = n1;
		ks = ks1;
	}
	else
	{
		tetas = tetas3;
		tetar = tetar3;
		alfa = alfa3;
		n = n3;
		ks = ks3;
	}

	//-------------------------------- AGGIORNAMENTO CELLE

	for (i = 1; i < mbusu->sizeof_X; i++)
	{
		Delta_h = h - calGetX3Dr(mbusu, Q.h, _i, _j, _k, i);
		if (_k_inv == ZSUP && i == 5)	//6
			Delta_h = 0;
		if (_k_inv == ZFONDO && i == 6)	//5
			Delta_h = 0;
		if (_i == XW && i == 1)	//2
			Delta_h = 0;
		if (_i == XE && i == 4)	//3
			Delta_h = 0;
		if (_j == YOUT && i == 3)	//1
			Delta_h = 0;
		if (_j == YIN && i == 2)	//4
			Delta_h = 0;

		temp_value = ((calGet3Dr(mbusu, Q.k, _i, _j, _k) + calGetX3Dr(mbusu, Q.k, _i, _j, _k, i)) / 2.0) * calGet3Dr(mbusu, Q.dqdh, _i, _j, _k);
		h = h - ((Delta_h / (lato*lato))*delta_t*temp_value);
	}

	//---- CONDIZIONE AL CONTORNO SUPERIORE

	Ymid = YOUT / 2;
	if (_k_inv == ZSUP && _i<45 && _j>(Ymid - 20) && _j < (Ymid + 20))
	{
		teta_pioggia = lato*rain*delta_t / volume;
		h = h + teta_pioggia*calGet3Dr(mbusu, Q.dqdh, _i, _j, _k);
	}

	//--------------------------------- PEDOFUNZIONI

	psi = h - quota;

	//---------------------------(dteta/dh)^(-1) SECONDO RICHARDS

	denompow_uno = pow(alfa*(-psi), (1 - n));
	denompow_due = pow(alfa*(-psi), n);
	denompow_tre = pow((1 / (1 + denompow_due)), (1 / n - 2));
	uno_su_dqdh = (denompow_uno / (alfa*(n - 1)*(tetas - tetar)))*denompow_tre;

	denom_pow = pow(alfa*(-psi), n);
	teta = tetar + ((tetas - tetar)*pow((1 / (1 + denom_pow)), (1 - 1 / n)));
	denom_pow_start = pow(alfa*(734), n);
	teta_start = tetar + ((tetas - tetar)*pow((1 / (1 + denom_pow_start)), (1 - 1 / n)));
	moist_cont = teta / tetas;
	moist_diff = moist_cont - teta_start / tetas;

	satur = (teta - tetar) / (tetas - tetar);
	exp_c = n / (n - 1);
	satur_expc = pow(satur, exp_c);
	exp_d = 1 - (1 / n);
	satur_expd = pow((1 - satur_expc), exp_d);
	k = ks*pow(satur, 0.5)*pow((1 - satur_expd), 2);

	//--------------------------------- PEDOFUNZIONI

	//----- Verifica convergenza Deltat variabile

	if ((k > 0) && (uno_su_dqdh > 0))
	{
		convergence = lato*lato / (6 * k*uno_su_dqdh);
	}
	else
	{
		convergence = 1.0;
	}

	//---Update

	calSet3Dr(mbusu, Q.dqdh, _i, _j, _k, uno_su_dqdh);
	calSet3Dr(mbusu, Q.psi, _i, _j, _k, psi);
	calSet3Dr(mbusu, Q.k, _i, _j, _k, k);
	calSet3Dr(mbusu, Q.h, _i, _j, _k, h);
	calSet3Dr(mbusu, Q.teta, _i, _j, _k, teta);
	calSet3Dr(mbusu, Q.moist_cont, _i, _j, _k, moist_cont);
	calSet3Dr(mbusu, Q.moist_diff, _i, _j, _k, moist_diff);
	calSet3Dr(mbusu, Q.convergence, _i, _j, _k, convergence);
}

//------------------------------------------------------------------------------
//					mbusu simulation functions
//------------------------------------------------------------------------------

void mbusuSimulationInit(struct CALModel3D* mbusu)
{
	int _i, _j, _k, _k_inv;

	double quota, teta, satur, psi, h, k, uno_su_dqdh, teta_pioggia;
	double alfa, n, tetar, tetas, ks, Delta_h, moist_cont;
	double denom_pow, denompow_uno, denompow_due, denompow_tre;
	double exp_c, exp_d, satur_expc, satur_expd;
	double volume, moist_print;
	double convergence;
	double teta_start, denom_pow_start, moist_diff;
	double temp_value;

	for (_k = 0; _k < mbusu->slices; _k++)
		for (_i = 0; _i < mbusu->rows; _i++)
			for (_j = 0; _j < mbusu->columns; _j++)
			{
				_k_inv = (mbusu->slices - 1) - _k;
				quota = lato * _k_inv;
				h = -734 + quota;

				//--------------------------------- PEDOFUNZIONI

				psi = h - quota;

				//--------------------------- (dteta/dh)^(-1) SECONDO RICHARDS

				//---- PARAMETRI SUOLO

				if (_i>19 && _i < 60 && _k_inv>79 && _k_inv < 100)
				{
					tetas = tetas4;
					tetar = tetar4;
					alfa = alfa4;
					n = n4;
					ks = ks4;
				}
				else if (_k_inv > 111 && _k_inv < 124)
				{
					tetas = tetas2;
					tetar = tetar2;
					alfa = alfa2;
					n = n2;
					ks = ks2;
				}
				else if (_k_inv > 123)
				{
					tetas = tetas1;
					tetar = tetar1;
					alfa = alfa1;
					n = n1;
					ks = ks1;
				}
				else
				{
					tetas = tetas3;
					tetar = tetar3;
					alfa = alfa3;
					n = n3;
					ks = ks3;
				}

				denompow_uno = pow(alfa*(-psi), (1 - n));
				denompow_due = pow(alfa*(-psi), n);
				denompow_tre = pow((1 / (1 + denompow_due)), (1 / n - 2));
				uno_su_dqdh = (denompow_uno / (alfa*(n - 1)*(tetas - tetar)))*denompow_tre;

				denom_pow = pow(alfa*(-psi), n);
				teta = tetar + ((tetas - tetar)*pow((1 / (1 + denom_pow)), (1 - 1 / n)));
				denom_pow_start = pow(alfa*(734), n);
				teta_start = tetar + ((tetas - tetar)*pow((1 / (1 + denom_pow_start)), (1 - 1 / n)));
				moist_cont = teta / tetas;
				moist_diff = moist_cont - teta_start / tetas;

				satur = (teta - tetar) / (tetas - tetar);
				exp_c = n / (n - 1);
				satur_expc = pow(satur, exp_c);
				exp_d = 1 - (1 / n);
				satur_expd = pow((1 - satur_expc), exp_d);
				k = ks* pow(satur, 0.5) *pow((1 - satur_expd), 2);

				calSet3Dr(mbusu, Q.dqdh, _i, _j, _k, uno_su_dqdh);
				calSet3Dr(mbusu, Q.psi, _i, _j, _k, psi);
				calSet3Dr(mbusu, Q.k, _i, _j, _k, k);
				calSet3Dr(mbusu, Q.h, _i, _j, _k, h);
				calSet3Dr(mbusu, Q.teta, _i, _j, _k, teta);
				calSet3Dr(mbusu, Q.moist_cont, _i, _j, _k, moist_cont);
				calSet3Dr(mbusu, Q.moist_diff, _i, _j, _k, moist_diff);
				//--------------------------------- PEDOFUNZIONI
			}

	//calUpdate3D(mbusu);
}

void mbusuSimulationSteering(struct CALModel3D* mbusu)
{
        double min;
        min = calReductionComputeMin3Dr(mbusu, Q.convergence);

	if (min > 105.0)
		min = 105.0;

	delta_t = 0.95*min;
	delta_t_cum_prec = delta_t_cum;
	delta_t_cum += delta_t;

	stream = fopen("deltat.txt", "a");
	fprintf(stream, "%f, %f\n", delta_t, delta_t_cum);
	fclose(stream);
}

CALbyte mbusuSimulationStopCondition(struct CALModel3D* mbusu)
{
	int i, j, k;
	CALreal moist_print;
	CALint k_inv;

	//Stop condition
	if (delta_t_cum >= ascii_output_time_step && delta_t_cum_prec <= ascii_output_time_step)
	{
		j = YOUT/2;
		for (k = 0; k < mbusu->slices; k++)
			for (i = 0; i < mbusu->rows; i++)
				//for (j = 0; j < mbusu->columns; j++)
				{
					k_inv = (mbusu->slices - 1) - k;
					moist_print = calGet3Dr(mbusu, Q.moist_cont, i, j, k);

					if (i == XW && k_inv == ZSUP)
					{
						stream = fopen("ris10g.txt", "a");
						fprintf(stream, "%f\t", moist_print);
					}
					else if (i == XE && k_inv == ZFONDO)
					{
						fprintf(stream, "%f\n", moist_print);
						fclose(stream);
					}
					else if (i == XE)
						fprintf(stream, "%f\n", moist_print);
					else
						fprintf(stream, "%f\t", moist_print);
				}

		return CAL_TRUE;
	}

	return CAL_FALSE;
}

//------------------------------------------------------------------------------
//					mbusu CADef and runDef
//------------------------------------------------------------------------------

void mbusuCADef()
{
	//cadef and rundef
	mbusu = calCADef3D(ROWS, COLS, LAYERS, CAL_VON_NEUMANN_NEIGHBORHOOD_3D, CAL_SPACE_TOROIDAL, CAL_NO_OPT);
	mbusuSimulation = calRunDef3D(mbusu, 1, CAL_RUN_LOOP, CAL_UPDATE_IMPLICIT);

	//add transition function's elementary processes
	calAddElementaryProcess3D(mbusu, mbusuTransitionFunction);

	//add substates
	Q.teta = calAddSubstate3Dr(mbusu);
	Q.moist_cont = calAddSubstate3Dr(mbusu);
	Q.psi = calAddSubstate3Dr(mbusu);
	Q.k = calAddSubstate3Dr(mbusu);
	Q.h = calAddSubstate3Dr(mbusu);
	Q.dqdh = calAddSubstate3Dr(mbusu);
	Q.convergence = calAddSubstate3Dr(mbusu);
	Q.moist_diff = calAddSubstate3Dr(mbusu);

	//simulation run setup
	calRunAddInitFunc3D(mbusuSimulation, mbusuSimulationInit); calRunInitSimulation3D(mbusuSimulation);
	calRunAddSteeringFunc3D(mbusuSimulation, mbusuSimulationSteering);
	calRunAddStopConditionFunc3D(mbusuSimulation, mbusuSimulationStopCondition);
}

//------------------------------------------------------------------------------
//					mbusu finalization function
//------------------------------------------------------------------------------

void mbusuExit()
{
	//finalizations
	calRunFinalize3D(mbusuSimulation);
	calFinalize3D(mbusu);
}
