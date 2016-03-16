#ifndef calActiveCells_h
#define calActiveCells_h

#include <OpenCAL++/calCommon.h>
#include <OpenCAL++/calBuffer.h>

/*! \brief Active cells class.
*/
class CALActiveCells
{
private:
    CALBuffer <bool>* flags;			//!< Buffer of flags having the substates' dimension: flag is true if the corresponding cell is active, false otherwise.
    int* cells;	                        //!< Array of computational active cells.
    int size_current;					//!< Number of active cells in the current step.
    int size_next;                      //!< Number of true flags.
public:
    /*! \brief CALActiveCells' constructor with no parameter.
    */
    CALActiveCells ();
    /*! \brief CALActiveCells' constructor.
    */
    CALActiveCells (CALBuffer <bool>* flags, int size_next);
    CALActiveCells (CALBuffer <bool>* flags, int size_next, int* cells, int size_current);

    /*! \brief CALActiveCells' destructor.
    */
    ~ CALActiveCells ();
    CALActiveCells (const CALActiveCells & obj);

    /*! \brief CALActiveCells' setter and getter methods.
    */
    CALBuffer <bool>* getFlags ();
    void setFlags (CALBuffer <bool>* flags);
    int getSizeNext ();
    void setSizeNext (int size_next);
    int* getCells ();
    void setCells (int* cells);
    int getSizeCurrent ();
    void setSizeCurrent (int size_current);

    bool getElementFlags (int *indexes, int *coordinates, int dimension);

    /*! \brief Sets the cell of coordinates indexes of the matrix flags to parameter value. It
     * increments the couter size_current if value is true, it decreases otherwise.
    */
    void setElementFlags (int *indexes, int *coordinates, int dimension, bool value);

    /*! \brief Puts the cells marked as actives in flags into the array of active cells
        cells and sets its dimension, to size_of_actives, i.e. the actual
        number of active cells.
    */
    void update ();
    bool getFlag (int linearIndex);

    /*! \brief Sets the cell [linearIndex] of the matrix flags to parameter value. It
     * increments the couter size_current if value is true, it decreases otherwise.
    */
    void setFlag (int linearIndex, bool value);



};


#endif
