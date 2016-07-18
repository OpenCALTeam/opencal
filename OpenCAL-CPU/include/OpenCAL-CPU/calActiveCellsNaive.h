#ifndef cal_active_cells_naive
#define cal_active_cells_naive

#include <OpenCAL-CPU/calModel.h>

struct CALActiveCellsNaive {
        struct CALActiveCells* inherited_pointer;

        CALbyte* flags;				//!< Array of flags having the substates' dimension: flag is CAL_TRUE if the corresponding cell is active, CAL_FALSE otherwise.
        int* size_next;	//!< size_next[i] stores how many active cells thread i adds/removes (can be negative)
        CALIndices* cells;	//!< Array of computational active cells.
        int size_current;					//!< Number of active cells in the current step.
        int num_threads; //!< number of threads using the data structure (used to iterate over size_next)
};

/*! \brief Sets the cell of the matrix flags to CAL_TRUE and increments the
    couter sizeof_active_flags.
*/
void calAddActiveCellNaive(struct CALModel* calModel, CALIndices cell);

/*! \brief \brief Sets the cell (i,j) of the matrix flags to CAL_FALSE and decrements the
    couter sizeof_active_flags.
*/
void calRemoveActiveCellNaive(struct CALModel* calModel, CALIndices cell);

/*! \brief \brief Release the memory
*/

void calApplyElementaryProcessActiveCellsNaive(struct CALModel *calModel, CALLocalProcess elementary_process);

void calFreeActiveCellsNaive(struct CALActiveCellsNaive* activeCells );

void calCopyBufferActiveCellsNaive_b(CALbyte* M_src, CALbyte* M_dest,  struct CALModel* calModel);
void calCopyBufferActiveCellsNaive_i(CALint* M_src, CALint* M_dest,  struct CALModel* calModel);
void calCopyBufferActiveCellsNaive_r(CALreal* M_src, CALreal* M_dest,  struct CALModel* calModel);


void calSetActiveCellsNaiveBuffer_b(CALbyte* M, CALbyte value, struct CALModel* calModel);
void calSetActiveCellsNaiveBuffer_i(CALint* M, CALint value, struct CALModel* calModel);
void calSetActiveCellsNaiveBuffer_r(CALreal* M, CALreal value, struct CALModel* calModel);


void calUpdateActiveCellsNaive(struct CALModel* calModel);

#endif

