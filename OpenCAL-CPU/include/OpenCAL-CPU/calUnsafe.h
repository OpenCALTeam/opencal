#ifndef cal_unsafe
#define cal_unsafe

#include <OpenCAL-CPU/calModel.h>

/*! \brief Inits the given cell's n-th neighbour of a byte substate to value;
    it updates both the current and next matrix at that position.
    This operation is unsafe since it writes value in a neighbor, both on the current and next matrix.
*/
void calInitX_b(struct CALModel* calModel,	//!< Pointer to the cellular automaton structure.
                struct CALSubstate_b* Q,	//!< Pointer to a byte substate.
                CALIndices central_cell,    //!< The central cell's coordinates
                int n,						//!< Index of the n-th neighbor to be initialized.
                CALbyte value				//!< initializing value.
                );

/*! \brief Inits the given cell's n-th neighbour of a integer substate to value;
    it updates both the current and next matrix at that position.
    This operation is unsafe since it writes value in a neighbor, both on the current and next matrix.
*/
void calInitX_i(struct CALModel* calModel,	//!< Pointer to the cellular automaton structure.
                struct CALSubstate_i* Q,	//!< Pointer to a int substate.
                CALIndices central_cell,    //!< The central cell's coordinates
                int n,						//!< Index of the n-th neighbor to be initialized.
                CALint value				//!< initializing value.
                );

/*! \brief Inits the given cell's n-th neighbour of a real (floating point) substate to value;
    it updates both the current and next matrix at that position.
    This operation is unsafe since it writes value in a neighbor, both on the current and next matrix.
*/
void calInitX_r(struct CALModel* calModel,	//!< Pointer to the cellular automaton structure.
                struct CALSubstate_r* Q,	//!< Pointer to a real (floating point) substate.
                int n,						//!< Index of the n-th neighbor to be initialized.
                CALreal value				//!< initializing value.
                );



/*! \brief Returns the given cell's value of a byte substate from the next matrix.
    This operation is unsafe since it reads a value from the next matrix.
*/
CALbyte calGetNext_b(struct CALModel* calModel,		//!< Pointer to the cellular automaton structure.
                     struct CALSubstate_b* Q,       //!< Pointer to a 2D byte substate.
                     CALIndices central_cell        //!< The central cell's coordinates
                     );

/*! \brief Returns the given cell's value of an integer substate from the next matrix.
    This operation is unsafe since it reads a value from the next matrix.
*/
CALint calGetNext_i(struct CALModel* calModel,		//!< Pointer to the cellular automaton structure.
                    struct CALSubstate_i* Q,        //!< Pointer to a int substate.
                    CALIndices central_cell         //!< The central cell's coordinates
                    );

/*! \brief Returns the given cell's value of a real (floating point) substate from the next matrix.
    This operation is unsafe since it read a value from the next matrix.
*/
CALreal calGetNext_r(struct CALModel* calModel,		//!< Pointer to the cellular automaton structure.
                     struct CALSubstate_r* Q,       //!< Pointer to a real (floating point) substate.
                     CALIndices central_cell        //!< The central cell's coordinates
                     );



/*! \brief Returns the given cell's n-th neighbor value of a byte substate from the next matrix.
    This operation is unsafe since it reads a value from the next matrix.
*/
CALbyte calGetNextX_b(struct CALModel* calModel,		//!< Pointer to the cellular automaton structure.
                      struct CALSubstate_b* Q,          //!< Pointer to a real (floating point) substate.
                      CALIndices central_cell,          //!< The central cell's coordinates
                      int n                             //!< Index of the n-th neighbor
                      );

/*! \brief Returns the given cell's n-th neighbor value of an integer substate from the next matrix.
    This operation is unsafe since it reads a value from the next matrix.
*/
CALint calGetNextX_i(struct CALModel* calModel,		//!< Pointer to the cellular automaton structure.
                    struct CALSubstate_i* Q,		//!< Pointer to a real (floating point) substate.
                    CALIndices central_cell,        //!< The central cell's coordinates
                    int n							//!< Index of the n-th neighbor
                    );

/*! \brief Returns the given cell's n-th neighbor value of a real (floating point) substate from the next matrix.
    This operation is unsafe since it read a value from the next matrix.
*/
CALreal calGetNextX_r(struct CALModel* calModel,		//!< Pointer to the cellular automaton structure.
                      struct CALSubstate_r* Q,          //!< Pointer to a real (floating point) substate.
                      CALIndices central_cell,          //!< The central cell's coordinates
                      int n                             //!< Index of the n-th neighbor
                      );



/*! \brief Sets the value of the n-th neighbor of the given cell of a byte substate.
    This operation is unsafe since it writes a value in a neighbor of the next matrix.
*/
void calSetX_b(struct CALModel* calModel,	//!< Pointer to the cellular automaton structure.
               struct CALSubstate_b* Q,     //!< Pointer to a byte substate.
               CALIndices central_cell,     //!< The central cell's coordinates
               int n,						//!< Index of the n-th neighbor to be initialized.
               CALbyte value				//!< initializing value.
               );

/*! \brief Sets the value of the n-th neighbor of the given cell of an integer substate.
    This operation is unsafe since it writes a value in a neighbor of the next matrix.
*/
void calSetX_i(struct CALModel* calModel,	//!< Pointer to the cellular automaton structure.
               struct CALSubstate_i* Q,     //!< Pointer to a int substate.
               CALIndices central_cell,     //!< The central cell's coordinates
               int n,						//!< Index of the n-th neighbor to be initialized.
               CALint value                 //!< initializing value.
               );

/*! \brief Sets the value of the n-th neighbor of the given cell of a real (floating point) substate.
    This operation is unsafe since it writes a value in a neighbor of the next matrix.
*/
void calSetX_r(struct CALModel* calModel,	//!< Pointer to the cellular automaton structure.
               struct CALSubstate_r* Q,     //!< Pointer to a real (floating point) substate.
               CALIndices central_cell,     //!< The central cell's coordinates
               int n,						//!< Index of the n-th neighbor to be initialized.
               CALreal value				//!< initializing value.
               );



/*! \brief Sets the value of the n-th neighbor of the given cell of a byte substate of the CURRENT matrix.
    This operation is unsafe since it writes a value directly to the current matrix.
*/
void calSetCurrentX_b(struct CALModel* calModel,	//!< Pointer to the cellular automaton structure.
                      struct CALSubstate_b* Q,      //!< Pointer to a byte substate.
                      CALIndices central_cell,      //!< The central cell's coordinates
                      int n,                        //!< Index of the n-th neighbor to be initialized.
                      CALbyte value                 //!< initializing value.
                      );

/*! \brief Set the value of the n-th neighbor of the given cell of an int substate of the CURRENT matrix.
    This operation is unsafe since it writes a value directly to the current matrix.
*/
void calSetCurrentX_i(struct CALModel* calModel,	//!< Pointer to the cellular automaton structure.
                      struct CALSubstate_i* Q,      //!< Pointer to a int substate.
                      CALIndices central_cell,      //!< The central cell's coordinates
                      int n,                        //!< Index of the n-th neighbor to be initialized.
                      CALint value                  //!< initializing value.
                      );

/*! \brief Set the value of the n-th neighbor of the  given cell of a real (floating point) substate of the CURRENT matrix.
    This operation is unsafe since it writes a value directly to the current matrix.
*/
void calSetCurrentX_r(struct CALModel* calModel,	//!< Pointer to the cellular automaton structure.
                      struct CALSubstate_r* Q,      //!< Pointer to a int substate.
                      CALIndices central_cell,      //!< The central cell's coordinates
                      int n,                        //!< Index of the n-th neighbor to be initialized.
                      CALreal value                 //!< initializing value.
                      );



#endif
