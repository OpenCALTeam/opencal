//---------------------------------------------------------------------------
#ifndef io_h
#define io_h
//---------------------------------------------------------------------------
#include "GISInfo.h"
#include "Sciara.h"
#include "configurationPathLib.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

//---------------------------------------------------------------------------
// Autosave state variables
extern bool storing;		  //se � true avviene il salvataggio automatico
extern int storing_step;   //Ogni storing_step passi salva la configurazione
extern char storing_path[]; //percorso in cui viene salvata la configurazione
extern struct TGISInfo gis_info_Sz;
extern struct TGISInfo gis_info_generic;
extern struct TGISInfo gis_info_nodata0;
//---------------------------------------------------------------------------

int loadParameters(char* path, Sciara* sciara);
int saveParameters(char *path, Sciara* sciara);
void printParameters(Sciara* sciara);

int loadMorphology(char* path, Sciara* sciara);
int loadVents(char* path, Sciara* sciara);
int loadEmissionRate(char *path, Sciara* sciara);

int loadAlreadyAllocatedMap(char *path, int* S, int* nS, int lx, int ly);
int loadAlreadyAllocatedMap(char *path, double* S, double* nS, int lx, int ly);

int loadConfiguration(char *path, Sciara* sciara);
int saveConfiguration(char *path, Sciara* sciara);


//---------------------------------------------------------------------------
#endif
//---------------------------------------------------------------------------
