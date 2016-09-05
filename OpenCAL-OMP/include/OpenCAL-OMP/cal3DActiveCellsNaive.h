#ifndef cal3d_activecells_naive
#define cal3d_activecells_naive
#include <OpenCAL-OMP/calCommon.h>
#include <OpenCAL-OMP/cal3D.h>


typedef struct CALModel3D CALModel3D;

/*! \brief Active cells structure.
*/
struct CALActiveCells3D {
    CALbyte* flags;				//!< Array of flags having the substates' dimension: flag is CAL_TRUE if the corresponding cell is active, CAL_FALSE otherwise.
    int *size_next;	//!< Number of CAL_TRUE flags.
    struct CALCell3D* cells;	//!< Array of computational active cells.
    int size_current;					//!< Number of active cells in the current step.
    int num_threads;
};


/*! \brief Sets the cell (i,j,z) of the matrix flags to CAL_TRUE and increments the
    couter sizeof_active_flags.
*/
DllExport
void calAddActiveCellNaive3D(struct CALModel3D* ca3D, int i, int j, int k);

/*! \brief \brief Sets the cell (i,j,z) of the matrix flags to CAL_FALSE and decrements the
    couter sizeof_active_flags.
*/
DllExport
void calRemoveActiveCellNaive3D(struct CALModel3D* ca3D, int i, int j, int k);

/*! \brief \brief Release the memory
*/
DllExport
void calApplyElementaryProcessActiveCellsNaive3D(CALModel3D *ca2D, CALCallbackFunc3D elementary_process);

DllExport
void calFreeActiveCellsNaive3D( struct CALActiveCells3D* activeCells );

DllExport
void calCopyBufferActiveCellsNaive3Db(CALbyte* M_src, CALbyte* M_dest,  struct CALModel3D* ca3D);

DllExport
void calCopyBufferActiveCellsNaive3Di(CALint* M_src, CALint* M_dest,  struct CALModel3D* ca3D);

DllExport
void calCopyBufferActiveCellsNaive3Dr(CALreal* M_src, CALreal* M_dest,  struct CALModel3D* ca3D);

DllExport
void calSetActiveCellsNaiveBuffer3Db(CALbyte* M, CALbyte value, CALModel3D* ca3D);

DllExport
void calSetActiveCellsNaiveBuffer3Di(CALint* M, CALint value, CALModel3D* ca3D);

DllExport
void calSetActiveCellsNaiveBuffer3Dr(CALreal* M, CALreal value, CALModel3D* ca3D);

DllExport
void calUpdateActiveCellsNaive3D(struct CALModel3D* ca3D);

#endif
