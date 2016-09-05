#ifndef cal2d_activecells_naive
#define cal2d_activecells_naive
#include <OpenCAL/calCommon.h>
#include <OpenCAL/cal2D.h>

/*! \brief Active cells structure.
*/
struct CALActiveCells2D {
    CALbyte* flags;				//!< Array of flags having the substates' dimension: flag is CAL_TRUE if the corresponding cell is active, CAL_FALSE otherwise.
    int size_next;	//!< Number of CAL_TRUE flags.
    struct CALCell2D* cells;	//!< Array of computational active cells.
        int size_current;					//!< Number of active cells in the current step.
};

/*! \brief Sets the cell (i,j) of the matrix flags to CAL_TRUE and increments the
    couter sizeof_active_flags.
*/
DllExport 
void calAddActiveCellNaive2D( struct CALActiveCells2D* activeCells, int i, int j, int columns );

/*! \brief \brief Sets the cell (i,j) of the matrix flags to CAL_FALSE and decrements the
    couter sizeof_active_flags.
*/
DllExport 
void calRemoveActiveCellNaive2D(struct CALActiveCells2D* activeCells, int i, int j , int columns);

/*! \brief \brief Release the memory
*/
DllExport 
void calApplyElementaryProcessActiveCellsNaive2D(CALModel2D *ca2D, CALCallbackFunc2D elementary_process);

DllExport 
void calFreeActiveCellsNaive2D( struct CALActiveCells2D* activeCells );

DllExport 
void calCopyBufferActiveCellsNaive2Db(CALbyte* M_src, CALbyte* M_dest,  struct CALModel2D* ca2D);

DllExport 
void calCopyBufferActiveCellsNaive2Di(CALint* M_src, CALint* M_dest,  struct CALModel2D* ca2D);

DllExport 
void calCopyBufferActiveCellsNaive2Dr(CALreal* M_src, CALreal* M_dest,  struct CALModel2D* ca2D);

DllExport 
void calSetActiveCellsNaiveBuffer2Db(CALbyte* M, CALbyte value, CALModel2D* ca2D);

DllExport 
void calSetActiveCellsNaiveBuffer2Di(CALint* M, CALint value, CALModel2D* ca2D);

DllExport 
void calSetActiveCellsNaiveBuffer2Dr(CALreal* M, CALreal value, CALModel2D* ca2D);

DllExport 
void calUpdateActiveCellsNaive2D(struct CALModel2D* ca2D);




#endif
