#ifndef iso_h
#define iso_h

#include <cal2D.h>
#include <cal2DIO.h>
#include <cal2DRun.h>

//#define IDW
//#define TEST
//#define DEBUG
//#define SHOW_GRID
#define VERBOSE

#define EXP 1

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

#ifdef TEST
#define ROWS 10
#define COLS 10
#define STEPS 100
#define SOURCE_PATH "./data/test.txt"
#define OUTPUT_PATH "./data/test_output_value.txt"
#define OUTPUT_PATH_STATE "./data/test_output_state.txt"
#else
#define ROWS 100
#define COLS 50
#define STEPS 1
#define SOURCE_PATH "./data/source.txt"
#define OUTPUT_PATH "./data/output_value.txt"
#define OUTPUT_PATH_STATE "./data/output_state.txt"
#endif

#define NODATA -9999
#define BLANK 0
#define UNSTEADY 1
#define TO_BE_STEADY 2
#define STEADY 3

//cadef and rundef
extern struct CALModel2D* iso;
extern struct CALRun2D* isoRun;

struct isoSubstates {
	struct CALSubstate2Dr *value;
	struct CALSubstate2Db *state;
	struct CALSubstate2Di *source_row;
	struct CALSubstate2Di *source_col;
	struct CALSubstate2Di *flux_count;
};

extern struct isoSubstates Q;

void isoCADef();
void isoLoadConfig();
void isoSaveConfig();
void isoExit();

#endif