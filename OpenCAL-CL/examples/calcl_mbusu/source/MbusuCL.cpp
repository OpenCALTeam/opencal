#include "MbusuCL.h"

void initMbusu() {
	mbusu= new Mbusu;
	mbusu->rows = ROWS;
	mbusu->cols = COLS;
	mbusu->layers = LAYERS;

	mbusu->host_CA = calCADef3D(mbusu->rows,mbusu->cols,mbusu->layers,CAL_VON_NEUMANN_NEIGHBORHOOD_3D, CAL_SPACE_TOROIDAL, CAL_NO_OPT);

	mbusu->Q = new mbusuSubstates();

	mbusu->Q->teta = calAddSubstate3Dr(mbusu->host_CA);
	mbusu->Q->moist_cont = calAddSubstate3Dr(mbusu->host_CA);
	mbusu->Q->psi = calAddSubstate3Dr(mbusu->host_CA);
	mbusu->Q->k = calAddSubstate3Dr(mbusu->host_CA);
	mbusu->Q->h = calAddSubstate3Dr(mbusu->host_CA);
	mbusu->Q->dqdh = calAddSubstate3Dr(mbusu->host_CA);
	mbusu->Q->convergence = calAddSubstate3Dr(mbusu->host_CA);
	mbusu->Q->moist_diff = calAddSubstate3Dr(mbusu->host_CA);


//	calInitSubstate2Dr(sciara->host_CA, sciara->substates->Sz, 0);
//	calInitSubstate2Dr(sciara->host_CA, sciara->substates->Slt, 0);
//	calInitSubstate2Dr(sciara->host_CA, sciara->substates->St, 0);
//
//	//TODO single layer initialization
//	for (int i = 0; i < sciara->rows * sciara->cols; ++i) {
//		sciara->substates->Mb->current[i] = CAL_FALSE;
//		sciara->substates->Mv->current[i] = 0;
//		sciara->substates->Msl->current[i] = 0;
//		sciara->substates->Sz_t0->current[i] = 0;
//	}
//
//	for (int i = 0; i < NUMBER_OF_OUTFLOWS; ++i) {
//		sciara->substates->f[i] = calAddSubstate2Dr(sciara->host_CA);
//		calInitSubstate2Dr(sciara->host_CA, sciara->substates->f[i], 0);
//	}

}

void simulationInitialize() {
	int _i, _j, _k, _k_inv;

	double quota, teta, satur, psi, h, k, uno_su_dqdh, teta_pioggia;
	double alfa, n, tetar, tetas, ks, Delta_h, moist_cont;
	double denom_pow, denompow_uno, denompow_due, denompow_tre;
	double exp_c, exp_d, satur_expc, satur_expd;
	double volume, moist_print;
	double convergence;
	double teta_start, denom_pow_start, moist_diff;
	double temp_value;

	for (_k = 0; _k < mbusu->layers; _k++)
		for (_i = 0; _i < mbusu->rows; _i++)
			for (_j = 0; _j < mbusu->cols; _j++)
			{
				_k_inv = (mbusu->layers - 1) - _k;
				quota = mbusu->parameters.lato * _k_inv;
				h = -734 + quota;

				//--------------------------------- PEDOFUNZIONI

				psi = h - quota;

				//--------------------------- (dteta/dh)^(-1) SECONDO RICHARDS

				//---- PARAMETRI SUOLO

				if (_i>19 && _i < 60 && _k_inv>79 && _k_inv < 100)
				{
					tetas = mbusu->parameters.tetas4;
					tetar = mbusu->parameters.tetar4;
					alfa = mbusu->parameters.alfa4;
					n = mbusu->parameters.n4;
					ks = mbusu->parameters.ks4;
				}
				else if (_k_inv > 111 && _k_inv < 124)
				{
					tetas = mbusu->parameters.tetas2;
					tetar = mbusu->parameters.tetar2;
					alfa = mbusu->parameters.alfa2;
					n = mbusu->parameters.n2;
					ks = mbusu->parameters.ks2;
				}
				else if (_k_inv > 123)
				{
					tetas = mbusu->parameters.tetas1;
					tetar = mbusu->parameters.tetar1;
					alfa = mbusu->parameters.alfa1;
					n = mbusu->parameters.n1;
					ks = mbusu->parameters.ks1;
				}
				else
				{
					tetas = mbusu->parameters.tetas3;
					tetar = mbusu->parameters.tetar3;
					alfa = mbusu->parameters.alfa3;
					n = mbusu->parameters.n3;
					ks = mbusu->parameters.ks3;
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

				calSet3Dr(mbusu->host_CA, mbusu->Q->dqdh, _i, _j, _k, uno_su_dqdh);
				calSet3Dr(mbusu->host_CA, mbusu->Q->psi, _i, _j, _k, psi);
				calSet3Dr(mbusu->host_CA, mbusu->Q->k, _i, _j, _k, k);
				calSet3Dr(mbusu->host_CA, mbusu->Q->h, _i, _j, _k, h);
				calSet3Dr(mbusu->host_CA, mbusu->Q->teta, _i, _j, _k, teta);
				calSet3Dr(mbusu->host_CA, mbusu->Q->moist_cont, _i, _j, _k, moist_cont);
				calSet3Dr(mbusu->host_CA, mbusu->Q->moist_diff, _i, _j, _k, moist_diff);
				//--------------------------------- PEDOFUNZIONI

			}
	calUpdate3D(mbusu->host_CA);
}

void exit() {
	calFinalize3D(mbusu->host_CA);
}
