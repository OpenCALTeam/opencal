//---------------------------------------------------------------------------
#include "io.h"
//---------------------------------------------------------------------------
#define FILE_ERROR	0
#define FILE_OK		1
//---------------------------------------------------------------------------
// Autosave state variables
bool storing = false;      //se � true avviene il salvataggio automatico
int storing_step = 0;          //Ogni storing_step passi salva la configurazione
char storing_path[1024] = "./config"; //percorso in cui viene salvata la configurazione
TGISInfo gis_info_Sz;
TGISInfo gis_info_generic;
TGISInfo gis_info_nodata0;


/****************************************************************************************
 * 										PRIVATE FUNCTIONS
 ****************************************************************************************/

void saveMatrixr(CALreal * M, char configuration_path[1024],Sciara * sciara){
	FILE* input_file = fopen(configuration_path,"w");
	SalvaGISInfo(gis_info_Sz,input_file);
	calfSaveMatrix2Dr(M,sciara->rows,sciara->cols,input_file);
	fclose(input_file);
}
void saveMatrixi(CALint * M, char configuration_path[1024],Sciara * sciara){
	FILE* input_file = fopen(configuration_path,"w");
	SalvaGISInfo(gis_info_Sz,input_file);
	calfSaveMatrix2Di(M,sciara->rows,sciara->cols,input_file);
	fclose(input_file);
}

int SaveConfigurationEmission(Sciara* sciara, char const *path, char const *name)
{
    char s[1024];
    if (ConfigurationFileSavingPath((char*)path, sciara->step, (char*)name, ".txt", s) == false)
        return FILE_ERROR;
    else
    {
        //Salvataggio su file
        FILE *s_file;
        if ( ( s_file = fopen(s,"w") ) == NULL)
        {
            char str[1024];
            strcpy(str, "Cannot save ");
            strcat(str, (char*)name);
            return FILE_ERROR;
        }
        saveEmissionRates(s_file, sciara->emission_time, sciara->emission_rate);
        fclose(s_file);
		return FILE_OK;
    }
}
/****************************************************************************************
 * 										PUBLIC FUNCTIONS
 ****************************************************************************************/


//---------------------------------------------------------------------------
int loadParameters(char const * path, Sciara* sciara) {
	char str[256];
	FILE *f;
	fpos_t position;

	if ((f = fopen(path, "r")) == NULL)
		return FILE_ERROR;

	fgetpos(f, &position);

	fscanf(f, "%s", str);
	if (strcmp(str, "maximum_steps_(0_for_loop)") != 0)
		return FILE_ERROR;
	fscanf(f, "%s", str);

	fscanf(f, "%s", str);
	if (strcmp(str, "stopping_threshold_(height)") != 0)
		return FILE_ERROR;
	fscanf(f, "%s", str);
	fscanf(f, "%s", str);
	if (strcmp(str, "refreshing_step") != 0)
		return FILE_ERROR;
	fscanf(f, "%s", str);
	fscanf(f, "%s", str);
	if (strcmp(str, "thickness_visual_threshold") != 0)
		return FILE_ERROR;
	fscanf(f, "%s", str);
	fscanf(f, "%s", str);
	if (strcmp(str, "Pclock") != 0)
		return FILE_ERROR;
	fscanf(f, "%s", str);
	fscanf(f, "%s", str);
	if (strcmp(str, "PTsol") != 0)
		return FILE_ERROR;
	fscanf(f, "%s", str);
	fscanf(f, "%s", str);
	if (strcmp(str, "PTvent") != 0)
		return FILE_ERROR;
	fscanf(f, "%s", str);
	fscanf(f, "%s", str);
	if (strcmp(str, "Pr(Tsol)") != 0)
		return FILE_ERROR;
	fscanf(f, "%s", str);
	fscanf(f, "%s", str);
	if (strcmp(str, "Pr(Tvent)") != 0)
		return FILE_ERROR;
	fscanf(f, "%s", str);
	fscanf(f, "%s", str);
	if (strcmp(str, "Phc(Tsol)") != 0)
		return FILE_ERROR;
	fscanf(f, "%s", str);
	fscanf(f, "%s", str);
	if (strcmp(str, "Phc(Tvent)") != 0)
		return FILE_ERROR;
	fscanf(f, "%s", str);
	fscanf(f, "%s", str);
	if (strcmp(str, "Pcool") != 0)
		return FILE_ERROR;
	fscanf(f, "%s", str);
	fscanf(f, "%s", str);
	if (strcmp(str, "Prho") != 0)
		return FILE_ERROR;
	fscanf(f, "%s", str);
	fscanf(f, "%s", str);
	if (strcmp(str, "Pepsilon") != 0)
		return FILE_ERROR;
	fscanf(f, "%s", str);
	fscanf(f, "%s", str);
	if (strcmp(str, "Psigma") != 0)
		return FILE_ERROR;
	fscanf(f, "%s", str);
	fscanf(f, "%s", str);
	if (strcmp(str, "Pcv") != 0)
		return FILE_ERROR;
	fscanf(f, "%s", str);
	fscanf(f, "%s", str);
	if (strcmp(str, "algorithm") != 0)
		return FILE_ERROR;
	fscanf(f, "%s", str);
	if (strcmp(str, "MIN") != 0 && strcmp(str, "PROP") != 0)
		return FILE_ERROR;

	fsetpos(f, &position);

	//fake readings
	fscanf(f, "%s", str);
	fscanf(f, "%s", str);
	fscanf(f, "%s", str);
	fscanf(f, "%s", str);
	fscanf(f, "%s", str);
	fscanf(f, "%s", str);
	fscanf(f, "%s", str);
	fscanf(f, "%s", str);

	fscanf(f, "%s", str);
	fscanf(f, "%s", str);
	sciara->Pclock = atof(str);
	fscanf(f, "%s", str);
	fscanf(f, "%s", str);
	sciara->PTsol = atof(str);
	fscanf(f, "%s", str);
	fscanf(f, "%s", str);
	sciara->PTvent = atof(str);
	fscanf(f, "%s", str);
	fscanf(f, "%s", str);
	sciara->Pr_Tsol = atof(str);
	fscanf(f, "%s", str);
	fscanf(f, "%s", str);
	sciara->Pr_Tvent = atof(str);
	fscanf(f, "%s", str);
	fscanf(f, "%s", str);
	sciara->Phc_Tsol = atof(str);
	fscanf(f, "%s", str);
	fscanf(f, "%s", str);
	sciara->Phc_Tvent = atof(str);
	fscanf(f, "%s", str);
	fscanf(f, "%s", str);
	sciara->Pcool = atof(str);
	fscanf(f, "%s", str);
	fscanf(f, "%s", str);
	sciara->Prho = atof(str);
	fscanf(f, "%s", str);
	fscanf(f, "%s", str);
	sciara->Pepsilon = atof(str);
	fscanf(f, "%s", str);
	fscanf(f, "%s", str);
	sciara->Psigma = atof(str);
	fscanf(f, "%s", str);
	fscanf(f, "%s", str);
	sciara->Pcv = atof(str);
	fscanf(f, "%s", str);
	fscanf(f, "%s", str);

	fclose(f);
	return FILE_OK;
}
//---------------------------------------------------------------------------
int saveParameters(char* path, Sciara* sciara) {
	FILE *f;
	if ((f = fopen(path, "w")) == NULL)
		return FILE_ERROR;

	fprintf(f, "Pclock				%f\n", sciara->Pclock);
	fprintf(f, "PTsol				%f\n", sciara->PTsol);
	fprintf(f, "PTvent				%f\n", sciara->PTvent);
	fprintf(f, "Pr(Tsol)			%f\n", sciara->Pr_Tsol);
	fprintf(f, "Pr(Tvent)			%f\n", sciara->Pr_Tvent);
	fprintf(f, "Phc(Tsol)			%f\n", sciara->Phc_Tsol);
	fprintf(f, "Phc(Tvent)			%f\n", sciara->Phc_Tvent);
	fprintf(f, "Pcool				%f\n", sciara->Pcool);
	fprintf(f, "Prho				%f\n", sciara->Prho);
	fprintf(f, "Pepsilon			%f\n", sciara->Pepsilon);
	fprintf(f, "Psigma				%e\n", sciara->Psigma);
	fprintf(f, "Pcv				%f\n", sciara->Pcv);

	fclose(f);
	return FILE_OK;
}
//---------------------------------------------------------------------------
void printParameters(Sciara* sciara) {
	printf("---------------------------------------------\n");
	printf("Paramater		Value\n");
	printf("---------------------------------------------\n");
	printf("Pclock			%f\n", sciara->Pclock);
	printf("PTsol			%f\n", sciara->PTsol);
	printf("PTvent			%f\n", sciara->PTvent);
	printf("Pr(Tsol)		%f\n", sciara->Pr_Tsol);
	printf("Pr(Tvent)		%f\n", sciara->Pr_Tvent);
	printf("Phc(Tsol)		%f\n", sciara->Phc_Tsol);
	printf("Phc(Tvent)		%f\n", sciara->Phc_Tvent);
	printf("Pcool			%f\n", sciara->Pcool);
	printf("Prho			%f\n", sciara->Prho);
	printf("Pepsilon		%f\n", sciara->Pepsilon);
	printf("Psigma			%e\n", sciara->Psigma);
	printf("Pcv			%f\n", sciara->Pcv);
}
//---------------------------------------------------------------------------
int loadMorphology(char* path, Sciara* sciara) {
	FILE *input_file;

	if ((input_file = fopen(path, "r")) == NULL)
		return FILE_ERROR;

	CALint gis_info_status = LeggiGISInfo(gis_info_Sz, input_file);
	if (gis_info_status != GIS_FILE_OK) {
		fclose(input_file);
		return FILE_ERROR;
	}
	initGISInfoNODATA0(gis_info_Sz, gis_info_nodata0);

//	sciara->Pa			= gis_info_Sz.cell_size;
//	sciara->Ple			= 2./sqrt(3.) * sciara->Pa;
//	sciara->Pae			= 3 * sciara->Ple * sciara->Pa;
	sciara->Pc = gis_info_Sz.cell_size;
	sciara->Pac = sciara->Pc * sciara->Pc;

	//legge il file contenente la morfologia
	calfLoadMatrix2Dr(sciara->substates->Sz->current, sciara->rows, sciara->cols, input_file);
	calCopyBuffer2Dr(sciara->substates->Sz->current, sciara->substates->Sz->next, sciara->rows, sciara->cols);
	calCopyBuffer2Dr(sciara->substates->Sz->current, sciara->substates->Sz_t0->current, sciara->rows, sciara->cols);

	fclose(input_file);

	return FILE_OK;
}
//---------------------------------------------------------------------------
int loadVents(char* path, Sciara* sciara) {
	FILE *input_file;
	if ((input_file = fopen(path, "r")) == NULL)
		return FILE_ERROR;

	CALint gis_info_status = LeggiGISInfo(gis_info_generic, input_file);
	CALint gis_info_verify = VerificaGISInfo(gis_info_generic, gis_info_Sz);
	if (gis_info_status != GIS_FILE_OK || gis_info_verify != GIS_FILE_OK) {
		fclose(input_file);
		return FILE_ERROR;
	}

	//Alloca e legge
	calfLoadMatrix2Di(sciara->substates->Mv->current, sciara->rows, sciara->cols, input_file);
	fclose(input_file);

	//verifica della consistenza della matrice
	initVents(sciara->substates->Mv->current, sciara->cols, sciara->rows, sciara->vent);

	calDeleteBuffer2Di(sciara->substates->Mv->current);



	return FILE_OK;
}
//---------------------------------------------------------------------------
int loadEmissionRate(char *path, Sciara* sciara) {
	FILE *input_file;
	if ((input_file = fopen(path, "r")) == NULL)
		return FILE_ERROR;

	CALint emission_rate_file_status = loadEmissionRates(input_file, sciara->emission_time, sciara->emission_rate, sciara->vent);
	fclose(input_file);

	//verifica della consistenza del file e definisce il vettore vent
	CALint error = defineVents(sciara->emission_rate, sciara->vent);
	if (error || emission_rate_file_status != EMISSION_RATE_FILE_OK)
		return FILE_ERROR;


	return 1;
}
//---------------------------------------------------------------------------
template<class Tipo> bool verifySubstate(Tipo **M, CALint lx, CALint ly, CALreal no_data) {
	Tipo sum = 0;
	for (int x = 0; x < lx; x++)
		for (int y = 0; y < ly; y++)
			if (M[x][y] > 0 && M[x][y] != no_data)
				sum += M[x][y];
	return (sum > 0);
}
//---------------------------------------------------------------------------
int loadAlreadyAllocatedMap(char *path, CALint* S, CALint* nS, CALint lx, CALint ly) {
	FILE *input_file;
	if ((input_file = fopen(path, "r")) == NULL)
		return FILE_ERROR;

	int gis_info_status = LeggiGISInfo(gis_info_generic, input_file);
	int gis_info_verify = VerificaGISInfo(gis_info_generic, gis_info_Sz);
	if (gis_info_status != GIS_FILE_OK || gis_info_verify != GIS_FILE_OK) {
		fclose(input_file);
		return FILE_ERROR;
	}

	calfLoadMatrix2Di(S, ly,lx, input_file);
	if (nS != NULL)
		calCopyBuffer2Di(nS, S, ly, lx);
	fclose(input_file);

	return FILE_OK;
}
//---------------------------------------------------------------------------------------------------
int loadAlreadyAllocatedMap(char *path, CALreal* S, CALreal* nS, CALint lx, CALint ly) {
	FILE *input_file;
	if ((input_file = fopen(path, "r")) == NULL)
		return FILE_ERROR;

	CALint gis_info_status = LeggiGISInfo(gis_info_generic, input_file);
	CALint gis_info_verify = VerificaGISInfo(gis_info_generic, gis_info_Sz);
	if (gis_info_status != GIS_FILE_OK || gis_info_verify != GIS_FILE_OK) {
		fclose(input_file);
		return FILE_ERROR;
	}

	calfLoadMatrix2Dr(S, ly,lx, input_file);
	if (nS != NULL)
		calCopyBuffer2Dr(nS, S, ly, lx);
	fclose(input_file);

	return FILE_OK;
}
//---------------------------------------------------------------------------------------------------
int loadConfiguration(char const *path, Sciara* sciara) {
	char configuration_path[1024];
//    int   gis_info_status;
//    int   gis_info_verify;

	//Apre il file di configurazione
	if (!loadParameters(path, sciara)) {
		strcat((char*)path, "_000000000000.cfg");
		if (!loadParameters(path, sciara))
			return FILE_ERROR;
	}

	//apre il file Morphology
	ConfigurationFilePath((char*)path, "Morphology", ".stt", configuration_path);
	if (!loadMorphology(configuration_path, sciara))
		return FILE_ERROR;

	//apre il file Vents
	ConfigurationFilePath((char*)path, "Vents", ".stt", configuration_path);
	if (!loadVents(configuration_path, sciara))
		return FILE_ERROR;

	//apre il file EmissionRate
	ConfigurationFilePath((char*)path, "EmissionRate", ".txt", configuration_path);
	if (!loadEmissionRate(configuration_path, sciara))
		return FILE_ERROR;

	//apre il file Thickness
	ConfigurationFilePath((char*)path, "Thickness", ".stt", configuration_path);
	loadAlreadyAllocatedMap(configuration_path, sciara->substates->Slt->current, sciara->substates->Slt->next, sciara->cols, sciara->rows);

	//apre il file Temperature
	ConfigurationFilePath((char*)path, "Temperature", ".stt", configuration_path);
	loadAlreadyAllocatedMap(configuration_path, sciara->substates->St->current, sciara->substates->St->next, sciara->cols, sciara->rows);

	//apre il file SolidifiedLavaThickness
	ConfigurationFilePath((char*)path, "SolidifiedLavaThickness", ".stt", configuration_path);
	loadAlreadyAllocatedMap(configuration_path, sciara->substates->Msl->current, NULL, sciara->cols, sciara->rows);

	//Imposta lo step in base al nome del file .cfg e aggiorna la barra di stato
	sciara->step = GetStepFromConfigurationFile((char*)path);


	return FILE_OK;
}

int saveConfiguration(char const *path, Sciara* sciara) {
//    int   gis_info_status;
//    int   gis_info_verify;

	//Apre il file di configurazione
	bool path_ok;
    char s[1024];

	//Salva il file di configurazione e i sottostati
    path_ok = ConfigurationFileSavingPath((char*)path, sciara->step, "", ".cfg", s);

    if (!path_ok || !saveParameters(s, sciara))
        return FILE_ERROR;



	//apre il file Morphology
	ConfigurationFileSavingPath((char*)path, sciara->step, "Morphology", ".stt", s);
	saveMatrixr(sciara->substates->Sz->current,s,sciara);

	//apre il file Vents
	ConfigurationFileSavingPath((char*)path, sciara->step, "Vents", ".stt", s);
	sciara->substates->Mv->current = calAllocBuffer2Di(sciara->rows,sciara->cols);
	rebuildVentsMatrix(sciara->substates->Mv->current,sciara->cols,sciara->rows,sciara->vent);
	saveMatrixi(sciara->substates->Mv->current,s,sciara);
	calDeleteBuffer2Di(sciara->substates->Mv->current);

	//apre il file EmissionRate
    if (!SaveConfigurationEmission(sciara, (char*)path, "EmissionRate"))
		return FILE_ERROR;

	//apre il file Thickness
	ConfigurationFileSavingPath((char*)path, sciara->step, "Thickness", ".stt", s);
	saveMatrixr(sciara->substates->Slt->current,s,sciara);

	//apre il file Temperature
	ConfigurationFileSavingPath((char*)path, sciara->step, "Temperature", ".stt", s);
	saveMatrixr(sciara->substates->St->current,s,sciara);

	//apre il file SolidifiedLavaThickness
	ConfigurationFileSavingPath((char*)path, sciara->step, "SolidifiedLavaThickness", ".stt", s);
	saveMatrixr(sciara->substates->Msl->current,s,sciara);

	return FILE_OK;
}
