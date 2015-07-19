#ifndef iso_h
#define iso_h

#include <OpenCAL/cal2D.h>
#include <OpenCAL/cal2DIO.h>
#include <OpenCAL/cal2DRun.h>

//#define IDW		//if defined, the IDW algorithm is applied
//#define TEST		//if defined, the TEST configuration is loaded
//#define TEST_FIL	//if defined, the TEST_FIL configuration is loaded
//#define SANFILI	//if defined, the sanfili2 configuration is loaded
//#define DEBUG		//if defined, the algorithm is executed step by step
//#define SHOW_GRID	//if defined, the grid is shown
#define VERBOSE		//if defined, the verbose mode is enaled

//#define TWO_SRC		//if defined, only two sources are considered for steadify a new cell
#define EXP 1		//exponent for the iso and IDW algorithm

#define BOUND
#ifdef BOUND
#define istart 1
#define istop iso->rows - 1
#define jstart 1
#define jstop iso->columns -1
#else
#define istart 0
#define istop iso->rows
#define jstart 0
#define jstop iso->columns
#endif

#ifdef SANFILI
#define ROWS 127
#define COLS 64
#define STEPS 100
#define SOURCE_PATH "./data/sanfili/ferro.txt"
#define OUTPUT_PATH "./data/sanfili/ferro_values.out"
#define OUTPUT_PATH_STATE "./data/sanfili/ferro_states.out"
#else
	#ifdef TEST
		#define ROWS 10
		#define COLS 10
		#define STEPS 100
		#define SOURCE_PATH "./data/test.txt"
		#define OUTPUT_PATH "./data/test_output_value.txt"
		#define OUTPUT_PATH_STATE "./data/test_output_state.txt"
	#else
		#ifdef TEST_FIL
			#define ROWS 50
			#define COLS 50
			#define STEPS 1
			#define SOURCE_PATH "./data/new_sources.txt"
			#define OUTPUT_PATH "./data/new_output_value.txt"
			#define OUTPUT_PATH_STATE "./data/new_output_state.txt"
		#else 
			#define ROWS 100
			#define COLS 50
			#define STEPS 1
			#define SOURCE_PATH "./data/source.txt"
			#define OUTPUT_PATH "./data/output_value.txt"
			#define OUTPUT_PATH_STATE "./data/output_state.txt"
		#endif
	#endif
#endif

#define NODATA -9999
#define BLANK 0
#define UNSTEADY 1
#define TO_BE_STEADY 2
#define STEADY 3

//cadef and rundef
//extern struct CALModel2D* iso;
//extern struct CALRun2D* isoRun;

struct isoSubstates {
	struct CALSubstate2Dr *value;
	struct CALSubstate2Db *state;
	struct CALSubstate2Di *flux_count;
	struct CALSubstate2Di *source_row;
	struct CALSubstate2Di *source_col;
	struct CALSubstate2Di *next_source_row;
	struct CALSubstate2Di *next_source_col;
};

//extern struct isoSubstates Q;

struct CellularAutomata {
	struct CALModel2D* iso;			//the cellular automaton
	struct isoSubstates Q;			//the substates
	CALParameteri initial_sources;	//number of initial sources
	struct CALRun2D* isoRun;		//the simulartion run
};

#define numAC 5
extern CALint currentAC;
extern struct CellularAutomata ca[numAC];

void isoCADef(struct CellularAutomata* ca);
void isoLoadConfig(struct CellularAutomata* ca);
void isoSaveConfig(struct CellularAutomata* ca);
void isoExit(struct CellularAutomata* ca);

#endif