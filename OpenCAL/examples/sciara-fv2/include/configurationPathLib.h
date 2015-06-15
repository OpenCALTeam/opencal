//---------------------------------------------------------------------------

#ifndef configurationPathLib_h
#define configurationPathLib_h
//---------------------------------------------------------------------------
void ConfigurationIdPath(char config_file_path[], char config_dir[]);
void ConfigurationFilePath(char config_file_path[], char const *name, char const *suffix, char file_path[]);
int  GetStepFromConfigurationFile(char config_file_path[]);
bool ConfigurationFileSavingPath(char config_file_path[], int step, char const * name, char const *suffix, char file_path[]);

#endif
