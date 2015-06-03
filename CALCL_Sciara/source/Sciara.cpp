#include "Sciara.h"


void evaluatePowerLawParams(CALreal value_sol, CALreal value_vent, CALreal &k1, CALreal &k2) {
	k2 = (log10(value_vent) - log10(value_sol)) / (sciara->parameters.PTvent - sciara->parameters.PTsol);
	k1 = log10(value_sol) - k2 * (sciara->parameters.PTsol);
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
	sciara->parameters.effusion_duration = sciara->parameters.emission_time * maximum_number_of_emissions;

	//definisce il bordo della morfologia
	MakeBorder();

	//calcolo a b (parametri viscosit�) c d (parametri resistenza al taglio)
	evaluatePowerLawParams(sciara->parameters.Pr_Tsol, sciara->parameters.Pr_Tvent, sciara->parameters.a, sciara->parameters.b);
	evaluatePowerLawParams(sciara->parameters.Phc_Tsol, sciara->parameters.Phc_Tvent, sciara->parameters.c, sciara->parameters.d);
//	updateVentsEmission(model);

	if (active)
		for (unsigned int k = 0; k < sciara->vent.size(); k++) {
			int xVent = sciara->vent[k].x();
			int yVent = sciara->vent[k].y();
			calAddActiveCell2D(model, yVent, xVent);
		}
	calUpdate2D(model);

}

void initSciara(char * demPath) {

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
	sciara->parameters.rad2 = sqrt(2.0);
	sciara->parameters.effusion_duration = 0;

	if (active)
		sciara->model = calCADef2D(sciara->rows, sciara->cols, CAL_MOORE_NEIGHBORHOOD_2D, CAL_SPACE_TOROIDAL, CAL_OPT_ACTIVE_CELLS);
	else
		sciara->model = calCADef2D(sciara->rows, sciara->cols, CAL_MOORE_NEIGHBORHOOD_2D, CAL_SPACE_TOROIDAL, CAL_NO_OPT);

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

}


void saveConfigSciara() {
	char save_path[1024] = SAVE_PATH;
	calSaveSubstate2Dr(sciara->model, sciara->substates->Slt, save_path);
}

void exitSciara() {
	calFinalize2D(sciara->model);
}

