#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "configurationPathLib.h"


char * strrev (char * string)
{
	char *start = string;
	char *left = string;
	char ch;

	while (*string++) /* find end of string */
			;
	string -= 2;

	while (left < string)
	{
			ch = *left;
			*left++ = *string;
			*string-- = ch;
	}

	return(start);
}

void ConfigurationIdPath(char config_file_path[], char config_id_str[])
{
	int pos = 0;

	for (int i=strlen(config_file_path); i>=0; i--)
		//if ( config_file_path[i] == '/' || config_file_path[i] == '\\')
		if ( config_file_path[i] == '_' )
		{
			pos = i;
			break;
		}
	
	if (pos == 0)
		strcpy(config_id_str, "\0");
	else
	{
		strcpy(config_id_str, config_file_path);
		config_id_str[pos] = '\0';
	}
}

void ConfigurationFilePath(char config_file_path[], char const * name, char const *suffix, char file_path[])
{
    /*
      La funzione costruisce in file_path il percorso completo del filada aprire:
      config_file_path è il percorso completo del file di configurazione. Es: curti_000000000000.cfg
      name è il nome del sottostato da aprire. Es: Morphology
      suffix è l'estensione del file da aprire. Es: .stt oppure .txt

    */
    strcpy(file_path, "\0");            //inizializza file_path alla stringa vuota
    strcat(file_path, config_file_path);//file_path viene inizializzato al percorso completo del file di configurazione
    int lp = strlen(file_path)-4;       //lunghezza della stringa senza estensione .cfg
    file_path[lp] = '\0';               //file_path = file_path senza estensione .cfg
    strcat(file_path, "_");             //file_path = file_path + _ . Es: curti_000000000000_
    strcat(file_path, name);            //file_path = file_path + name . Es: curti_000000000000_Morphology
    strcat(file_path, suffix);          //file_path = file_path + .stt . Es: curti_000000000000_Morphology.stt
}
//---------------------------------------------------------------------------
int GetStepFromConfigurationFile(char config_file_path[])
{
    char step_str[150] = "\0";

    strcpy(step_str, config_file_path);
    step_str[strlen(step_str)-4] = '\0';
    strcpy(step_str, strrev(step_str));
    step_str[12] = '\0';
    strcpy(step_str, strrev(step_str));
    return atoi(step_str);
}
//---------------------------------------------------------------------------
bool ConfigurationFileSavingPath(char config_file_path[], int step, char const * name, char const * suffix, char file_path[])
{
    char p[32];                         //stringa contenete il passo di calcolo (step)
    char ps[] = "000000000000";         //stringa di 12 digts contenete il passo di calcolo (step)

    strcpy(file_path, "\0");            //inizializza file_path alla stringa vuota
    strcat(file_path, config_file_path);//file_path viene inizializzato al percorso del file di configurazione. Es: "c:\\simulazioni\\curti"
	if (step >= 0) {
		strcat(file_path, "_");         //file_path = file_path+ "_". Es: "c:\\simulazioni\\curti_"
		sprintf(p, "%d", step);			//conversione in stringa dello step dell'AC. Es: p = "345"
		int lp = strlen (p);            //calcolo lunghezza della stringha p
		int lps = strlen (ps);          //calcolo lunghezza della stringha ps
		if (lps < lp)                   //controllo
			return false;
		ps[lps-lp] = '\0';              //scarta gli ultimi lp caratteri della stringa ps = "000000000"
		strcat(ps, p);                  //ps = ps + p. Es. ps = "000000000345"
		strcat(file_path, ps);          //file_path = file_path + ps. Es: "c:\\simulazioni\\curti_000000000345"
		
	}
    if (strcmp(name, ""))               //se name non è la stringa vuota
	{
		strcat(file_path, "_");			//file_path = file_path + "_". Es: "c:\\simulazioni\\curti_000000000345_"
		strcat(file_path, name);        //file_path = file_path + name. Es: "c:\\simulazioni\\curti_000000000345_Morphology"
	}

    strcat(file_path, suffix);          //file_path = file_path + ."stt". Es: "c:\\simulazioni\\curti_000000000345_Morphology.stt"

    return true;
}
//---------------------------------------------------------------------------
