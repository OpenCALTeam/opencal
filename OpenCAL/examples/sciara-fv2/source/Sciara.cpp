#include "Sciara.h"

void updateVentsEmission(struct CALModel2D * model, int i, int j) {
	double emitted_lava = 0;
	for (unsigned int k = 0; k < sciara->vent.size(); k++) {
		int xVent = sciara->vent[k].x();
		int yVent = sciara->vent[k].y();
		if (i == yVent && j == xVent) {
			emitted_lava = sciara->vent[k].thickness(sciara->elapsed_time, sciara->Pclock, sciara->emission_time, sciara->Pac);
			if (emitted_lava > 0) {
				calSet2Dr(model, sciara->substates->Slt, yVent, xVent, calGet2Dr(sciara->model, sciara->substates->Slt, yVent, xVent) + emitted_lava);
				calSet2Dr(model, sciara->substates->St, yVent, xVent, sciara->PTvent);
			}
		}
	}
}

double powerLaw(double k1, double k2, double T) {
	double log_value = k1 + k2 * T;
	return pow(10, log_value);
}

void outflowsMin(struct CALModel2D * model, int i, int j, CALreal *f) {

	bool n_eliminated[MOORE_NEIGHBORS];
	double z[MOORE_NEIGHBORS];
	double h[MOORE_NEIGHBORS];
	double H[MOORE_NEIGHBORS];
	double theta[MOORE_NEIGHBORS];
	double w[MOORE_NEIGHBORS];		//Distances between central and adjecent cells
	double Pr[MOORE_NEIGHBORS];		//Relaiation rate arraj
	bool loop;
	int counter;
	double avg, _w, _Pr, hc, sum, sumZ;

	CALreal t = calGet2Dr(model, sciara->substates->St, i, j);

	_w = sciara->Pc;
	_Pr = powerLaw(sciara->a, sciara->b, t);
	hc = powerLaw(sciara->c, sciara->d, t);
	for (int k = 0; k < MOORE_NEIGHBORS; k++) {

		h[k] = calGetX2Dr(model, sciara->substates->Slt, i, j, k);
		H[k] = f[k] = theta[k] = 0;
		w[k] = _w;
		Pr[k] = _Pr;
		CALreal sz = calGetX2Dr(model, sciara->substates->Sz, i, j, k);
		CALreal sz0 = calGet2Dr(model, sciara->substates->Sz, i, j);
		if (k < VON_NEUMANN_NEIGHBORS)
			z[k] = calGetX2Dr(model, sciara->substates->Sz, i, j, k);
		else
			z[k] = sz0 - (sz0 - sz) / sciara->rad2;
	}

	H[0] = z[0];
	n_eliminated[0] = true;

	for (int k = 1; k < MOORE_NEIGHBORS; k++)
		if (z[0] + h[0] > z[k] + h[k]) {
			H[k] = z[k] + h[k];
			theta[k] = atan(((z[0] + h[0]) - (z[k] + h[k])) / w[k]);
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
			avg = avg / double(counter);
		for (int k = 0; k < MOORE_NEIGHBORS; k++)
			if (n_eliminated[k] && avg <= H[k]) {
				n_eliminated[k] = false;
				loop = true;
			}
	} while (loop);

	for (int k = 1; k < MOORE_NEIGHBORS; k++) {
		if (n_eliminated[k] && h[0] > hc * cos(theta[k])) {
			f[k] = Pr[k] * (avg - H[k]);
		}
	}

}

void empiricalFlows(struct CALModel2D * model, int i, int j) {

	if (calGet2Dr(model, sciara->substates->Slt, i, j) > 0) {
		CALreal f[MOORE_NEIGHBORS];
		outflowsMin(model, i, j, f);

		for (int k = 1; k < MOORE_NEIGHBORS; k++)
			if (f[k] > 0) {
				calSet2Dr(model, sciara->substates->f[k - 1], i, j, f[k]);
				if (active)
					calAddActiveCellX2D(model, i, j, k);
			}
	}
}
void width_update(struct CALModel2D* model, int i, int j) {
	CALint outFlowsIndexes[NUMBER_OF_OUTFLOWS] = { 3, 2, 1, 0, 6, 7, 4, 5 };
	CALint n;
	CALreal initial_h = calGet2Dr(model, sciara->substates->Slt, i, j);
	CALreal initial_t = calGet2Dr(model, sciara->substates->St, i, j);
	CALreal residualTemperature = initial_h * initial_t;
	CALreal residualLava = initial_h;
	CALreal h_next = initial_h;
	CALreal t_next;

	CALreal ht = 0;
	CALreal inSum = 0;
	CALreal outSum = 0;

	for (n = 1; n < model->sizeof_X; n++) {
		CALreal inFlow = calGetX2Dr(model, sciara->substates->f[outFlowsIndexes[n - 1]], i, j, n);
		CALreal outFlow = calGet2Dr(model, sciara->substates->f[n - 1], i, j);
		CALreal neigh_t = calGetX2Dr(model, sciara->substates->St, i, j, n);
		ht += inFlow * neigh_t;
		inSum += inFlow;
		outSum += outFlow;
	}
	h_next += inSum - outSum;
	calSet2Dr(model, sciara->substates->Slt, i, j, h_next);
	if (inSum > 0 || outSum > 0) {
		residualLava -= outSum;
		t_next = (residualLava * initial_t + ht) / (residualLava + inSum);
		calSet2Dr(model, sciara->substates->St, i, j, t_next);
	}
}

void updateTemperature(struct CALModel2D* model, int i, int j) {
	CALreal nT, h, T, aus;
	CALreal sh = calGet2Dr(model, sciara->substates->Slt, i, j);
	CALreal st = calGet2Dr(model, sciara->substates->St, i, j);
	CALreal sz = calGet2Dr(model, sciara->substates->Sz, i, j);

	if (sh > 0 && !calGet2Db(model, sciara->substates->Mb, i, j)) {
		h = sh;
		T = st;
		if (h != 0) {
//			nT = T / h;
			nT = T;

			/*nT -= Pepsilon * Psigma * pow(nT, 4.0) * Pclock * Pcool/ (Prho * Pcv * h * Pac);
			 nSt[x][y]=nT;*/

			aus = 1.0 + (3 * pow(nT, 3.0) * sciara->Pepsilon * sciara->Psigma * sciara->Pclock * sciara->Pcool) / (sciara->Prho * sciara->Pcv * h * sciara->Pac);
			st = nT / pow(aus, 1.0 / 3.0);
			calSet2Dr(model, sciara->substates->St, i, j, st);

		}

		//solidification
		if (st <= sciara->PTsol && sh > 0) {
			calSet2Dr(model, sciara->substates->Sz, i, j, sz + sh);
			calSetCurrent2Dr(model, sciara->substates->Msl, i, j, calGet2Dr(model, sciara->substates->Msl, i, j) + sh);
			calSet2Dr(model, sciara->substates->Slt, i, j, 0);
			calSet2Dr(model, sciara->substates->St, i, j, sciara->PTsol);

		} else
			calSet2Dr(model, sciara->substates->Sz, i, j, sz);
	}
}

void removeActiveCells(struct CALModel2D* model, int i, int j) {
	CALreal st = calGet2Dr(model, sciara->substates->St, i, j);
	if (st <= sciara->PTsol && !calGet2Db(model, sciara->substates->Mb, i, j))
		calRemoveActiveCell2D(model, i, j);
}

CALbyte stopCondition(struct CALModel2D* model) {
	if (sciara->elapsed_time >= sciara->effusion_duration)
		return CAL_TRUE;

	//La simulazione pu� continuare
	return CAL_FALSE;
}

void steering(struct CALModel2D* model) {
	for (int i = 0; i < NUMBER_OF_OUTFLOWS; ++i)
		calInitSubstate2Dr(model, sciara->substates->f[i], 0);

	for (int i = 0; i < sciara->rows; i++)
		for (int j = 0; j < sciara->cols; j++)
			if (calGet2Db(model, sciara->substates->Mb, i, j) == CAL_TRUE) {
				calSet2Dr(model, sciara->substates->Slt, i, j, 0);
				calSet2Dr(model, sciara->substates->St, i, j, 0);
			}
	sciara->elapsed_time += sciara->Pclock;
//	updateVentsEmission(model);
}

void evaluatePowerLawParams(CALreal value_sol, CALreal value_vent, CALreal &k1, CALreal &k2) {
	k2 = (log10(value_vent) - log10(value_sol)) / (sciara->PTvent - sciara->PTsol);
	k1 = log10(value_sol) - k2 * (sciara->PTsol);
}

void MakeBorder() {
	int j, i;

	//prima riga
	i = 0;
	for (j = 0; j < sciara->cols; j++)
		if (calGet2Dr(sciara->model, sciara->substates->Sz, i, j) >= 0) {
			calSetCurrent2Db(sciara->model, sciara->substates->Mb, i, j, CAL_TRUE);
			if (active)
				calAddActiveCell2D(sciara->model, i, j);
		}
	//ultima riga
	i = sciara->rows - 1;
	for (j = 0; j < sciara->cols; j++)
		if (calGet2Dr(sciara->model, sciara->substates->Sz, i, j) >= 0) {
			calSetCurrent2Db(sciara->model, sciara->substates->Mb, i, j, CAL_TRUE);
			if (active)
				calAddActiveCell2D(sciara->model, i, j);
		}
	//prima colonna
	j = 0;
	for (i = 0; i < sciara->rows; i++)
		if (calGet2Dr(sciara->model, sciara->substates->Sz, i, j) >= 0) {
			calSetCurrent2Db(sciara->model, sciara->substates->Mb, i, j, CAL_TRUE);
			if (active)
				calAddActiveCell2D(sciara->model, i, j);
		}
	//ultima colonna
	j = sciara->cols - 1;
	for (i = 0; i < sciara->rows; i++)
		if (calGet2Dr(sciara->model, sciara->substates->Sz, i, j) >= 0) {
			calSetCurrent2Db(sciara->model, sciara->substates->Mb, i, j, CAL_TRUE);
			if (active)
				calAddActiveCell2D(sciara->model, i, j);
		}
	//il resto
	for (int i = 1; i < sciara->rows - 1; i++)
		for (int j = 1; j < sciara->cols - 1; j++)
			if (calGet2Dr(sciara->model, sciara->substates->Sz, i, j) >= 0) {
				for (int k = 1; k < sciara->model->sizeof_X; k++)
					if (calGetX2Dr(sciara->model, sciara->substates->Sz, i, j, k) < 0) {
						calSetCurrent2Db(sciara->model, sciara->substates->Mb, i, j, CAL_TRUE);
						if (active)
							calAddActiveCell2D(sciara->model, i, j);
						break;
					}
			}

}

void simulationInitialize(struct CALModel2D* model) {

	//dichiarazioni
	unsigned int maximum_number_of_emissions = 0;

	//azzeramento dello step dell'AC
	sciara->step = 0;
	sciara->elapsed_time = 0;

	//determinazione numero massimo di passi
	for (unsigned int i = 0; i < sciara->emission_rate.size(); i++)
		if (maximum_number_of_emissions < sciara->emission_rate[i].size())
			maximum_number_of_emissions = sciara->emission_rate[i].size();
	//maximum_steps_from_emissions = (int)(emission_time/Pclock*maximum_number_of_emissions);
	sciara->effusion_duration = sciara->emission_time * maximum_number_of_emissions;

	//definisce il bordo della morfologia
	MakeBorder();

	//calcolo a b (parametri viscosit�) c d (parametri resistenza al taglio)
	evaluatePowerLawParams(sciara->Pr_Tsol, sciara->Pr_Tvent, sciara->a, sciara->b);
	evaluatePowerLawParams(sciara->Phc_Tsol, sciara->Phc_Tvent, sciara->c, sciara->d);
//	updateVentsEmission(model);
	if (active)
		for (unsigned int k = 0; k < sciara->vent.size(); k++) {
			int xVent = sciara->vent[k].x();
			int yVent = sciara->vent[k].y();
			calAddActiveCell2D(model, yVent, xVent);
		}
	calUpdate2D(model);

}

void initSciara(char const* demPath, int steps) {

	sciara = new Sciara;
	FILE * demFile = fopen(demPath, "r");
	if (demFile == NULL) {
		perror("Cannot open dem file\n");
		exit(EXIT_FAILURE);
	}
	TGISInfo sciaraINFO;
	int err = LeggiGISInfo(sciaraINFO, demFile);
	if (err > 0) {
		perror("Error while reading GIS INFO\n");
		exit(EXIT_FAILURE);
	}

	sciara->rows = sciaraINFO.nrows;
	sciara->cols = sciaraINFO.ncols;
	sciara->elapsed_time = 0.0;
	sciara->rad2 = sqrt(2.0);
	sciara->effusion_duration = 0;

	if (active)
		sciara->model = calCADef2D(sciara->rows, sciara->cols, CAL_MOORE_NEIGHBORHOOD_2D, CAL_SPACE_FLAT, CAL_OPT_ACTIVE_CELLS);
	else
		sciara->model = calCADef2D(sciara->rows, sciara->cols, CAL_MOORE_NEIGHBORHOOD_2D, CAL_SPACE_FLAT, CAL_NO_OPT);

	sciara->substates = new SciaraSubstates();

	sciara->substates->Sz = calAddSubstate2Dr(sciara->model);
	sciara->substates->Slt = calAddSubstate2Dr(sciara->model);
	sciara->substates->St = calAddSubstate2Dr(sciara->model);

	sciara->substates->Mb = calAddSingleLayerSubstate2Db(sciara->model);
	sciara->substates->Mv = calAddSingleLayerSubstate2Di(sciara->model);
	sciara->substates->Msl = calAddSingleLayerSubstate2Dr(sciara->model);
	sciara->substates->Sz_t0 = calAddSingleLayerSubstate2Dr(sciara->model);

	calInitSubstate2Dr(sciara->model, sciara->substates->Sz, 0);
	calInitSubstate2Dr(sciara->model, sciara->substates->Slt, 0);
	calInitSubstate2Dr(sciara->model, sciara->substates->St, 0);

	//TODO single layer initialization
	for (int i = 0; i < sciara->rows * sciara->cols; ++i) {
		sciara->substates->Mb->current[i] = CAL_FALSE;
		sciara->substates->Mv->current[i] = 0;
		sciara->substates->Msl->current[i] = 0;
		sciara->substates->Sz_t0->current[i] = 0;
	}

	for (int i = 0; i < NUMBER_OF_OUTFLOWS; ++i) {
		sciara->substates->f[i] = calAddSubstate2Dr(sciara->model);
		calInitSubstate2Dr(sciara->model, sciara->substates->f[i], 0);
	}

	fclose(demFile);

	calAddElementaryProcess2D(sciara->model, updateVentsEmission);
	calAddElementaryProcess2D(sciara->model, empiricalFlows);
	calAddElementaryProcess2D(sciara->model, width_update);
	calAddElementaryProcess2D(sciara->model, updateTemperature);
	if (active)
		calAddElementaryProcess2D(sciara->model, removeActiveCells);

	sciara->run = calRunDef2D(sciara->model, 1, steps, CAL_UPDATE_IMPLICIT);

	calRunAddInitFunc2D(sciara->run, simulationInitialize);
	calRunAddSteeringFunc2D(sciara->run, steering);
	calRunAddStopConditionFunc2D(sciara->run, stopCondition);

}

void runSciara() {
	calRun2D(sciara->run);
}

