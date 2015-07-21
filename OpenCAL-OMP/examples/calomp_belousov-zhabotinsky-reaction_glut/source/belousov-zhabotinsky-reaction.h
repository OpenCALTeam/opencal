#ifndef zhaborinsky_h
#define zhaborinsky_h

/*! \brief Introduction.
 *         
 *
 *  Appeared in "Computer Recreations" section of the August 1988 issue of Scientific American (Professor A. K. Dewdney). Cells could be in any of n+1 states, where a cell in state 0 is "healthy", a cell in state n is "ill" and a cell in any intermediate state is "infected"

Cells in state 1 are always displayed in black (or, one might say, are not displayed). Cells in state q are always displayed in white. Colors may be used for cells in states 2 through q-1. The number of colors used for these cells can be set to 1, 2, 4 or 8. When 2 colors are used then all cells are either black (state <= q/2) or white (state > q/2). When 1 color is selected then cells in states 2 through q-1 (when q >= 3) are displayed in one color of differing intensity (higher states are shown in brighter color). When 4 or 8 colors are used then colors are distributed equally among the states 2 through q-1.

This cellular automaton provides an illustration of order emerging "spontaneously" from chaos. Starting from an array whose cells are in randomly assigned states, patterns eventually emerge. In other words, non-randomness emerges from randomness. This has significance for the question of whether complicated biological organisms can emerge from simpler ones as a result of chance occurrences (such as genetic mutations). Before drawing such a conclusion, however, we should note that in this cellular automaton pattern emerges from chaos only if the parameters governing the evolution of the system have values within a certain fairly narrow range. q = 200, k1 = 2, k2 = 3 and g = 70 produce spiral patterns, but spirals do not emerge for most other sets of values.
 */


/** RULES. 
(i) If the cell is healthy (i.e., in state 0) then its new state is [a/k1] + [b/k2], where a is the number of infected cells among its eight neighbors, b is the number of ill cells among its neighbors, and k1 and k2 are constants. Here "[]" means the integer part of the number enclosed, so that, for example, [7/3] = [2+1/3] = 2.

(ii) If the cell is ill (i.e., in state n) then it miraculously becomes healthy (i.e., its state becomes 0).

(iii) If the cell is infected (i.e., in a state other than 0 and n) then its new state is [s/(a+b+1)] + g, where a and b are as above, s is the sum of the states of the cell and of its neighbors and g is a constant. 

 Rule (iii) can be seen as stating that the new state of a cell is the average of its state and its neighbors states plus a constant which may be thought of as the tendency of the infection to spread.

(credits to: http://www.hermetic.ch/pca/bz.htm)
 */

#include <OpenCAL-OMP/cal2D.h>
#include <OpenCAL-OMP/cal2DRun.h>

#define ROWS 1024
#define COLS 1024
#define G (35)
#define k1 (2)
#define k2 (1)
#define QQ (180)

#define STATE_HEALTY 1
#define STATE_ILL (QQ)
//#define STATE_INFECTED 1

//cadef and rundef
struct CellularAutomaton {
	struct CALModel2D* model;		//the cellular automaton
	struct CALSubstate2Di* Q;			//the set of cell's states over the whole cellular space
	struct CALRun2D* run;		//the simulartion run
};

extern struct CellularAutomaton zhabotinsky;

void CADef(struct CellularAutomaton* ca);
void Init(struct CellularAutomaton* ca);
void isoExit(struct CellularAutomaton* ca);

#endif

