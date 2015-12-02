/*
 * kernel.c
 *
 *  Created on: 16/apr/2015
 *      Author: alessio
 */

#ifndef __OPENCL_VERSION__
#define __kernel
#define __global
#define __local
#define get_global_id (int)
#endif

#include "kernel.h"

__kernel void mbusuTransitionFunction(MODEL_DEFINITION3D, __global Parameters* parameters) {

	initThreads3D();

	int _i = getRow();
	int _j = getCol();
	int _k = getSlice();
	int _k_inv;

	double quota, teta, satur, psi, h, k, uno_su_dqdh, teta_pioggia; //
	double alfa, n, tetar, tetas, ks, Delta_h, moist_cont;
	double denom_pow, denompow_uno, denompow_due, denompow_tre;
	double exp_c, exp_d, satur_expc, satur_expd;//
	double volume, moist_print;
	int i, Ymid;
	double convergence;
	double teta_start, denom_pow_start, moist_diff;
	double temp_value;

	volume = parameters->lato*parameters->lato*parameters->lato;
	_k_inv = (get_slices() - 1) - _k;
	quota = parameters->lato*_k_inv;
	h = calGet3Dr(MODEL3D,H, _i, _j, _k);

	//---- PARAMETRI SUOLO

	if ((_i > 19) && (_i < 60) && (_k_inv>79) && (_k_inv < 100))
	{
		tetas = parameters->tetas4;
		tetar = parameters->tetar4;
		alfa = parameters->alfa4;
		n = parameters->n4;
		ks = parameters->ks4;
	}
	else if ((_k_inv > 111) && (_k_inv < 124))
	{
		tetas = parameters->tetas2;
		tetar = parameters->tetar2;
		alfa = parameters->alfa2;
		n = parameters->n2;
		ks = parameters->ks2;
	}
	else if (_k_inv > 123)
	{
		tetas = parameters->tetas1;
		tetar = parameters->tetar1;
		alfa = parameters->alfa1;
		n = parameters->n1;
		ks = parameters->ks1;
	}
	else
	{
		tetas = parameters->tetas3;
		tetar = parameters->tetar3;
		alfa = parameters->alfa3;
		n = parameters->n3;
		ks = parameters->ks3;
	}

	//-------------------------------- AGGIORNAMENTO CELLE

	for (i = 1; i < VON_NEUMANN_NEIGHBORS; i++)
	{
		Delta_h = h - calGetX3Dr(MODEL3D,H, _i, _j, _k, i);

		if (_k_inv == ZSUP && i == 5)	//6
		Delta_h = 0;
		if (_k_inv == ZFONDO && i == 6)//5
		Delta_h = 0;
		if (_i == XW && i == 1)//2
		Delta_h = 0;
		if (_i == XE && i == 4)//3
		Delta_h = 0;
		if (_j == YOUT && i == 3)//1
		Delta_h = 0;
		if (_j == YIN && i == 2)//4
		Delta_h = 0;

		temp_value = ((calGet3Dr(MODEL3D,K, _i, _j, _k) + calGetX3Dr(MODEL3D,K, _i, _j, _k, i)) / 2.0) * calGet3Dr(MODEL3D,DQDH, _i, _j, _k);
		h = h - ((Delta_h / (parameters->lato*parameters->lato))*parameters->delta_t*temp_value);

	}

	//---- CONDIZIONE AL CONTORNO SUPERIORE

	Ymid = YOUT / 2;
	if (_k_inv == ZSUP && _i<45 && _j>(Ymid - 20) && _j < (Ymid + 20))
	{
		teta_pioggia = parameters->lato*parameters->rain*parameters->delta_t / volume;
		h = h + teta_pioggia*calGet3Dr(MODEL3D,DQDH, _i, _j, _k);

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
		convergence = parameters->lato* parameters->lato / (6 * k*uno_su_dqdh);
	}
	else
	{
		convergence = 1.0;
	}

	//---Update

	calSet3Dr(MODEL3D,DQDH, _i, _j, _k,uno_su_dqdh);
	calSet3Dr(MODEL3D,PSI, _i, _j, _k,psi);
	calSet3Dr(MODEL3D,K, _i, _j, _k,k);
	calSet3Dr(MODEL3D,H, _i, _j, _k,h);
	calSet3Dr(MODEL3D,TETA, _i, _j, _k,teta);
	calSet3Dr(MODEL3D,MOIST_CONT, _i, _j, _k,moist_cont);
	calSet3Dr(MODEL3D,MOIST_DIFF, _i, _j, _k,moist_diff);
	calSet3Dr(MODEL3D,CONVERGENCE,_i, _j, _k,convergence);

}

__kernel void steering(MODEL_DEFINITION3D, __global Parameters* parameters) {

	initThreads3D();

	int i = getRow();
	int j = getCol();
	int k = getSlice();

	if(i==0 && j==0 && k==0) {
		double min = calGet3Dr(MODEL3D,CONVERGENCE, 0, 0,0);

		for(int s =0; s < get_slices(); s++)
		for(int x =0; x < get_rows(); x++)
		for(int y =0; y < get_columns(); y++) {
			double tempMin = calGet3Dr(MODEL3D,CONVERGENCE, s, x, y);
			if (min > tempMin)
			min = tempMin;
		}

		if (min > 105.0)
		min = 105.0;
		parameters->delta_t = 0.95*min;
		parameters->delta_t_cum_prec = parameters->delta_t_cum;
		parameters->delta_t_cum += parameters->delta_t;
	}

}

__kernel void stopCondition(MODEL_DEFINITION3D, __global Parameters* parameters) {

	initThreads3D();

	int i = getRow();
	int j = getCol();
	int k = getSlice();

	//Stop condition
	if ((parameters->delta_t_cum >= parameters->ascii_output_time_step && parameters->delta_t_cum_prec <= parameters->ascii_output_time_step))
	{

		stopExecution();
	}

}

