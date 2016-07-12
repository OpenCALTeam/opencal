#ifndef life_h
#define life_h

#include <OpenCAL-CPU/opencal.h>

#define ROWS 1024
#define COLS 1024
#define STATE_DEAD 0
#define STATE_ALIVE 1

//cadef and rundef
struct CellularAutomaton {
    struct CALModel* model;		//the cellular automaton
    struct CALSubstate_i* Q;			//the set of call's states over the whole cellular space
};

extern struct CellularAutomaton life;

void CADef(struct CellularAutomaton* ca);
void isoExit(struct CellularAutomaton* ca);

#endif
