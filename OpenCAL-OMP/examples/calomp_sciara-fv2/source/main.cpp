/*
 * main.cpp
 *
 *  Created on: 14/ott/2014
 *      Author: Maurizio
 */

#include "io.h"
#include "Sciara.h"
#include <time.h>

#define CONFIG_PATH "./data/2006/2006"
#define SAVE_PATH "./data/2006_SAVE/2006"
#define DEM_PATH CONFIG_PATH"_000000000000_Morphology.stt"

Sciara *sciara;
int active;
time_t start_time, end_time;

int main(int argc, char** argv) {
	start_time = time(NULL);

//	int steps = atoi(argv[1]);
//	char path[1024];
//	strcpy(path, argv[2]);
//	char * demPath = (char*)malloc(sizeof(char)*(strlen(path)+strlen("_000000000000_Morphology.stt")+1));
//	strcpy(demPath, path);
//	strcat(demPath, "_000000000000_Morphology.stt\0");
//	char * outputPath = argv[3];
//	active = atoi(argv[4]);

	int steps = 1000;
	char const* outputPath = SAVE_PATH;
	active = 0;

	char path[1024] = CONFIG_PATH;
	initSciara(DEM_PATH, steps);
	int err = loadConfiguration(path, sciara);
	if (err == 0) {
		perror("cannot load configuration\n");
		exit(EXIT_FAILURE);
	}

	runSciara();

	err = saveConfiguration(outputPath, sciara);

	if (err == 0) {
		perror("cannot save configuration\n");
		exit(EXIT_FAILURE);
	}
	end_time = time(NULL);
	printf("%lds", end_time - start_time);

	return 0;

}
