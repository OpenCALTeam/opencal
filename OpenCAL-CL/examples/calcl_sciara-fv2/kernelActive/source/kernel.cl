#ifndef __OPENCL_VERSION__
#define __kernel
#define __global
#define __local
#define get_global_id (int)
#endif

#include "kernel.h"

CALreal thickness(CALreal sim_elapsed_time, CALreal Pt, unsigned int emission_time, CALreal Pac, __global CALreal * emission_rate, int sizeEmissionRate) {
	unsigned int i;

	i = (unsigned int) (sim_elapsed_time / emission_time);
	if (i >= sizeEmissionRate)
		return 0;
	else
		return emission_rate[i] / Pac * Pt;
}

__kernel void updateVentsEmission(__CALCL_MODEL_2D, __global Vent* vents, __global CALreal * emissionRates, __global CALreal * elapsed_time, int sizeVents, int sizeEmissionRate, Parameters parameters) {

	calclActiveThreadCheck2D();

	int threadID = calclGlobalRow();
	int i = calclActiveCellRow(threadID);
	int j = calclActiveCellColumns(threadID);

	CALreal emitted_lava = 0;


	for (unsigned int k = 0; k < sizeVents; k++) {
		int iVent = vents[k].y;
		int jVent = vents[k].x;
		if (i == iVent && j == jVent) {
			emitted_lava = thickness(*elapsed_time, parameters.Pclock, parameters.emission_time, parameters.Pac, emissionRates + k * sizeEmissionRate, sizeEmissionRate);
			if (emitted_lava > 0) {
				CALreal slt = calclGet2Dr(MODEL_2D, SLT, iVent, jVent) + emitted_lava;
				calclSet2Dr(MODEL_2D, SLT, iVent, jVent, slt);
				calclSet2Dr(MODEL_2D, ST, iVent, jVent, parameters.PTvent);
			}
		}
	}
}

CALreal powerLaw(CALreal k1, CALreal k2, CALreal T) {
	CALreal log_value = (k2*T)+k1;
	return powr(10, (double)log_value);
}

void outflowsMin(__CALCL_MODEL_2D, int i, int j, CALreal *f, Parameters parameters) {

	bool n_eliminated[MOORE_NEIGHBORS];
	CALreal z[MOORE_NEIGHBORS];
	CALreal h[MOORE_NEIGHBORS];
	CALreal H[MOORE_NEIGHBORS];
	CALreal theta[MOORE_NEIGHBORS];
	bool loop;
	int counter;
	CALreal avg, _w, _Pr, hc;

	CALreal t = calclGet2Dr(MODEL_2D, ST, i, j);

	_w = parameters.Pc;
	_Pr = powerLaw(parameters.a, parameters.b, t);
	hc = powerLaw(parameters.c, parameters.d, t);

	for (int k = 0; k < MOORE_NEIGHBORS; k++) {

		h[k] = calclGetX2Dr(MODEL_2D, SLT, i, j, k);
		H[k] = f[k] = theta[k] = 0;
		CALreal sz = calclGetX2Dr(MODEL_2D, SZ, i, j, k);
		CALreal sz0 = calclGet2Dr(MODEL_2D, SZ, i, j);
		if (k < VON_NEUMANN_NEIGHBORS)
			z[k] = sz;
		else
			z[k] = sz0 - (sz0 - sz) / parameters.rad2;
	}

	H[0] = z[0];
	n_eliminated[0] = true;

	for (int k = 1; k < MOORE_NEIGHBORS; k++)
		if (z[0] + h[0] > z[k] + h[k]) {
			H[k] = z[k] + h[k];
			theta[k] = atan((double)(((z[0] + h[0]) - (z[k] + h[k])) / _w));
			n_eliminated[k] = true;
		} else
			n_eliminated[k] = false;

	do {
		loop = false;
		avg = h[0];
		counter = 0;
		for (int k = 0; k < MOORE_NEIGHBORS; k++)
			if (n_eliminated[k]) {
				avg += H[k];
				counter++;
			}
		if (counter != 0)
			avg = avg / ((CALreal) counter);
		for (int k = 0; k < MOORE_NEIGHBORS; k++)
			if (n_eliminated[k] && avg <= H[k]) {
				n_eliminated[k] = false;
				loop = true;
			}
	} while (loop);

	for (int k = 1; k < MOORE_NEIGHBORS; k++) {
		if (n_eliminated[k] && h[0] > hc * cos((double)theta[k])) {
			f[k] = _Pr * (avg - H[k]);
		}
	}

}

__kernel void empiricalFlows(__CALCL_MODEL_2D, Parameters parameters) {

	calclActiveThreadCheck2D();

	int threadID = calclGlobalRow();
	int i = calclActiveCellRow(threadID);
	int j = calclActiveCellColumns(threadID);

	if (calclGet2Dr(MODEL_2D, SLT, i, j) > 0) {
		CALreal f[MOORE_NEIGHBORS];
		outflowsMin(MODEL_2D, i, j, f, parameters);

		for (int k = 1; k < MOORE_NEIGHBORS; k++)
			if (f[k] > 0) {
				calclSet2Dr(MODEL_2D, F(k-1), i, j, f[k]);
				calclAddActiveCellX2D(MODEL_2D, i, j, k);
			}
	}
}

__kernel void width_update(__CALCL_MODEL_2D) {

	calclActiveThreadCheck2D();

	int threadID = calclGlobalRow();
	int i = calclActiveCellRow(threadID);
	int j = calclActiveCellColumns(threadID);

	CALint outFlowsIndexes[NUMBER_OF_OUTFLOWS] = { 3, 2, 1, 0, 6, 7, 4, 5 };
	CALint n;
	CALreal initial_h = calclGet2Dr(MODEL_2D, SLT, i, j);
	CALreal initial_t = calclGet2Dr(MODEL_2D, ST, i, j);
	CALreal residualTemperature = initial_h * initial_t;
	CALreal residualLava = initial_h;
	CALreal h_next = initial_h;
	CALreal t_next;

	CALreal ht = 0;
	CALreal inSum = 0;
	CALreal outSum = 0;

	for (n = 1; n < calclGetNeighborhoodSize(); n++) {
		CALreal inFlow = calclGetX2Dr(MODEL_2D, F(outFlowsIndexes[n - 1]), i, j, n);
		CALreal outFlow = calclGet2Dr(MODEL_2D, F(n - 1), i, j);
		CALreal neigh_t = calclGetX2Dr(MODEL_2D, ST, i, j, n);
		ht =(inFlow*neigh_t)+ht;
		inSum += inFlow;
		outSum += outFlow;
	}
	h_next += inSum - outSum;
	calclSet2Dr(MODEL_2D, SLT, i, j, h_next);
	if (inSum > 0 || outSum > 0) {
		residualLava -= outSum;
		t_next = (residualLava*initial_t+ht) / (residualLava + inSum);
		calclSet2Dr(MODEL_2D, ST, i, j, t_next);
	}
}

__kernel void updateTemperature(__CALCL_MODEL_2D, __global CALbyte * Mb, __global CALreal * Msl, Parameters parameters) {

	calclActiveThreadCheck2D();

	int threadID = calclGlobalRow();
	int i = calclActiveCellRow(threadID);
	int j = calclActiveCellColumns(threadID);

	CALreal aus = 0;
	CALreal sh = calclGet2Dr(MODEL_2D, SLT, i, j);
	CALreal st = calclGet2Dr(MODEL_2D, ST, i, j);
	CALreal sz = calclGet2Dr(MODEL_2D, SZ, i, j);

	if (sh > 0 && calclGetBufferElement2D(Mb, calclGetColumns(), i, j) == CAL_FALSE) {
		aus = 1.0 + (3 * pow(st, 3.0) * parameters.Pepsilon * parameters.Psigma * parameters.Pclock * parameters.Pcool) / (parameters.Prho * parameters.Pcv * sh * parameters.Pac);
		st = st / pow(aus, (1.0 / 3.0));
		calclSet2Dr(MODEL_2D, ST, i, j, st);

		//solidification
		if (st <= parameters.PTsol) {
			calclSet2Dr(MODEL_2D, SZ, i, j, sz + sh);
			calclSetBufferElement2D(Msl, calclGetColumns(), i, j, calclGetBufferElement2D(Msl, calclGetColumns(), i, j) + sh);
			calclSet2Dr(MODEL_2D, SLT, i, j, 0);
			calclSet2Dr(MODEL_2D, ST, i, j, parameters.PTsol);
		}
	}
}

__kernel void removeActiveCells(__CALCL_MODEL_2D, __global CALbyte * Mb, Parameters parameters){

	calclActiveThreadCheck2D();

	int threadID = calclGlobalRow();
	int i = calclActiveCellRow(threadID);
	int j = calclActiveCellColumns(threadID);;

	CALreal st = calclGet2Dr(MODEL_2D, ST, i, j);
	if (st <= parameters.PTsol && calclGetBufferElement2D(Mb, calclGetColumns(), i, j) == CAL_FALSE)
		//calclSetBufferElement2D(calclGetActiveCellsFlags(),calclGetColumns(),i,j,CAL_FALSE);
		calclRemoveActiveCell2D(MODEL_2D,i,j);
}


__kernel void stopCondition(__CALCL_MODEL_2D, Parameters parameters, __global CALreal* elapsed_time) {

	calclActiveThreadCheck2D();

	int threadID = calclGlobalRow();
	int i = calclActiveCellRow(threadID);
	int j = calclActiveCellColumns(threadID);

	if (threadID == 0){
		if (*elapsed_time >= parameters.effusion_duration)
			calclRunStop();
	}
}

__kernel void steering(__CALCL_MODEL_2D, __global CALbyte * Mb, Parameters parameters, __global CALreal* elapsed_time) {

	calclActiveThreadCheck2D();

	int threadID = calclGlobalRow();
	int i = calclActiveCellRow(threadID);
	int j = calclActiveCellColumns(threadID);

	for (int k = 0; k < NUMBER_OF_OUTFLOWS; ++k)
		calclInitSubstate2Dr(MODEL_2D, F(k), i, j,0);

	if (calclGetBufferElement2D(Mb, calclGetColumns(), i, j) == CAL_TRUE) {
		calclSet2Dr(MODEL_2D, SLT, i, j, 0);
		calclSet2Dr(MODEL_2D, ST, i, j, 0);
	}
	if (threadID == 0)
		*elapsed_time += parameters.Pclock;

}
