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
static void calAddActiveCellNaive(struct CALActiveCellsNaive* A, CALIndices cell)
{
    struct CALModel* calModel = A->inherited_pointer->calModel;
    int linear_index = getLinearIndex(cell, calModel->coordinatesDimensions, calModel->numberOfCoordinates);
#if CAL_PARALLEL == 1
    CAL_SET_CELL_LOCK(linear_index, calModel->calRun->locks );
#endif

    if (!calGetMatrixElement(A->flags, linear_index))
    {
        calSetMatrixElement(A->flags, linear_index, CAL_TRUE);

        A->size_next[CAL_GET_THREAD_NUM()]++;
        return;
    }

#if CAL_PARALLEL == 1
    CAL_UNSET_CELL_LOCK(linear_index, calModel->calRun->locks );
#endif
}

/*! \brief \brief Sets the cell (i,j) of the matrix flags to CAL_FALSE and decrements the
    couter sizeof_active_flags.
*/

static void calRemoveActiveCellNaive(struct CALActiveCellsNaive* A, CALIndices cell)
{
    struct CALModel* calModel = A->inherited_pointer->calModel;
    int linear_index = getLinearIndex(cell, calModel->coordinatesDimensions, calModel->numberOfCoordinates);
#if CAL_PARALLEL == 1
    CAL_SET_CELL_LOCK(linear_index, calModel->calRun->locks );
#endif

    if (calGetMatrixElement(A->flags, linear_index))
    {
        calSetMatrixElement(A->flags, linear_index, CAL_FALSE);

        A->size_next[CAL_GET_THREAD_NUM()]--;
        return;
    }

#if CAL_PARALLEL == 1
    CAL_UNSET_CELL_LOCK(linear_index, calModel->calRun->locks );
#endif
}

void calRemoveInactiveCellsNaive(struct CALActiveCellsNaive* A, CALbyte (*active_cells_def)(struct CALModel*, CALIndices, int));

void calApplyElementaryProcessActiveCellsNaive(struct CALActiveCellsNaive *A, CALLocalProcess elementary_process);


void calCopyBufferActiveCellsNaive_b(CALbyte* M_src, CALbyte* M_dest,  struct CALActiveCellsNaive *A);
void calCopyBufferActiveCellsNaive_i(CALint* M_src, CALint* M_dest,  struct CALActiveCellsNaive *A);
void calCopyBufferActiveCellsNaive_r(CALreal* M_src, CALreal* M_dest,  struct CALActiveCellsNaive *A);


void calSetActiveCellsNaiveBuffer_b(CALbyte* M, CALbyte value, struct CALActiveCellsNaive *A);
void calSetActiveCellsNaiveBuffer_i(CALint* M, CALint value, struct CALActiveCellsNaive *A);
void calSetActiveCellsNaiveBuffer_r(CALreal* M, CALreal value, struct CALActiveCellsNaive *A);


void calUpdateActiveCellsNaive(struct CALActiveCellsNaive *A);

/*! \brief \brief Release the memory
*/
void calFreeActiveCellsNaive(struct CALActiveCellsNaive* activeCells );

#endif

