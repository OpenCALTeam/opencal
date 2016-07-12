#ifndef cal_model
#define cal_model

#include <OpenCAL-CPU/calCommon.h>
#include <OpenCAL-CPU/calNeighborPool.h>

/*****************************************************************************
                        DEFINITIONS OF NEW DATA TYPES
 *****************************************************************************/

typedef int* CALNeighbourPattern;

#define CAL_HEXAGONAL_SHIFT 7			//<! Shif used for accessing to the correct neighbor in case hexagonal heighbourhood and odd column cell

struct CALModel;
/*! \brief Fake function pointer type.
*/

typedef void (* CALLocalProcess)(struct CALModel* calModel, CALIndices, int number_of_dimensions);
typedef void (* CALGlobalProcess)(struct CALModel* calModel);

struct CALProcess {
        CALLocalProcess  localProcess;
        CALGlobalProcess globalProcess;
        char type;
};

#include <OpenCAL-CPU/calRun.h>
/*! \brief Structure defining the cellular automaton.
*/
struct CALModel {
        int* coordinatesDimensions;
        int numberOfCoordinates;
        int cellularSpaceDimension;

        struct CALIndexesPool* calIndexesPool; //!<

        struct CALNeighborPool* calNeighborPool;

        enum CALOptimization OPTIMIZATION;	//!< Type of optimization used. It can be CAL_NO_OPT or CAL_OPT_ACTIVE_CELLS.
        //struct CALActiveCells2D A;			//!< Computational Active cells object. if A.actives==NULL no optimization is applied.

        CALIndices X;				//!< Array of cell coordinates defining the cellular automaton neighbourhood relation.
        int sizeof_X;						//!< Number of cells belonging to the neighbourhood. Note that predefined neighbourhoods include the central cell.

        struct CALSubstate_b** pQb_array;	//!< Array of pointers to 2D substates of type byte
        struct CALSubstate_i** pQi_array;	//!< Array of pointers to 2D substates of type int
        struct CALSubstate_r** pQr_array;	//!< Array of pointers to 2D substates of type real (floating point)
        int sizeof_pQb_array;				//!< Number of substates of type byte.
        int sizeof_pQi_array;				//!< Number of substates of type int.
        int sizeof_pQr_array;				//!< Number of substates of type real (floating point).

        struct CALProcess * model_processes; //!< Array of function pointers to the transition function's elementary processes or generic global functions.Note that a substates' update must be performed after each elementary process has been applied to each cell of the cellular space (see calGlobalTransitionFunction2D).
        int num_of_processes; //!< Number of function pointers to the transition functions's elementary processes callbacks.
        struct CALRun* calRun; //!< Pointer to a structure containing the appropriate functions which implementation should change according to the type of execution chosen (serial or parallel)

};


/******************************************************************************
                    DEFINITIONS OF FUNCTIONS PROTOTYPES
*******************************************************************************/

/*! \brief Creates an object of type CALModel2D, sets its records and returns it as a pointer; it defines the cellular automaton structure.
*/
struct CALModel* calCADef(int numberOfCoordinates, //!< Number of coordinates of the Cellular Space.
                          CALIndices coordinatesDimensions,
                          enum CALNeighborhood CAL_NEIGHBORHOOD,
                          enum CALSpaceBoundaryCondition CAL_TOROIDALITY, //!< Enumerator that specifies whether the execution flow must be serial or parallel.
                          enum CALOptimization CAL_OPTIMIZATION //!< Enumerator used for specifying the active cells optimization or no optimization.
                          , int initial_step, int final_step);


/*! \brief Adds a neighbour to CALModel::X and updates the value of CALModel::sizeof_X.
*/
void calAddNeighbor(struct CALModel* calModel, //!< Pointer to the cellular automaton structure.
                      CALIndices neighbourIndex  //!< Indexes of the n-th neighbour
                      );

int calGetSizeOfX(struct CALModel* calModel);


/*! \brief Creates and adds a new byte substate to CALModel2D::pQb_array and return a pointer to it.
*/
struct CALSubstate_b* calAddSubstate_b(struct CALModel* calModel,	//!< Pointer to the cellular automaton structure.
                                       enum CALInitMethod initMethod, //!< Tells if and which substate layer must be initialised.
                                       CALbyte value);

/*! \brief Creates and adds a new int substate to CALModel2D::pQi_array and return a pointer to it.
*/
struct CALSubstate_i* calAddSubstate_i(struct CALModel* calModel,	//!< Pointer to the cellular automaton structure.
                                       enum CALInitMethod initMethod, //!< Tells if and which substate layer must be initialised.
                                       CALint value);

/*! \brief Creates and adds a new real (floating point) substate to CALModel::pQr_array and return a pointer to it.
*/
struct CALSubstate_r* calAddSubstate_r(struct CALModel* calModel,	//!< Pointer to the cellular automaton structure.
                                       enum CALInitMethod initMethod, //!< Tells if and which substate layer must be initialised.
                                       CALreal value);



/*! \brief Creates a new single-layer byte substate and returns a pointer to it.
    Note that sinlgle-layer substates are not added to CALModel::pQ*_array because
    they do not nedd to be updated.
*/
struct CALSubstate_b* calAddSingleLayerSubstate_b(struct CALModel* calModel,	//!< Pointer to the cellular automaton structure.
                                                  CALbyte init_value );

/*! \brief Creates a new single-layer int substate and returns a pointer to it.
    Note that sinlgle-layer substates are not added to CALModel::pQ*_array because
    they do not nedd to be updated.
*/
struct CALSubstate_i* calAddSingleLayerSubstate_i(struct CALModel* calModel,	//!< Pointer to the cellular automaton structure.
                                                  CALint init_value);

/*! \brief Creates a new single-layer real (floating point) substate returns a pointer to it.
    Note that sinlgle-layer substates are not added to CALModel::pQ*_array because
    they do not nedd to be updated.
*/
struct CALSubstate_r* calAddSingleLayerSubstate_r(struct CALModel* calModel,	//!< Pointer to the cellular automaton structure.
                                                  CALreal init_value);

/*! \brief Inits the value of a byte substate in the given cell to value; it updates both the current and next matrices at that position.
*/
void calInitSubstate_b(struct CALModel* calModel,	//!< Pointer to the cellular automaton structure.
               struct CALSubstate_b* Q,
               CALbyte value				//!< initializing value for the substate at the cell (i, j).
               );

/*! \brief Inits the value of a byte substate in the given cell to value; it updates both the current and next matrices at that position.
*/
void calInitSubstate_i(struct CALModel* calModel,	//!< Pointer to the cellular automaton structure.
               struct CALSubstate_i* Q,
               CALint value				//!< initializing value for the substate at the cell (i, j).
               );

/*! \brief Inits the value of a byte substate in the given cell to value; it updates both the current and next matrices at that position.
*/
void calInitSubstate_r(struct CALModel* calModel,	//!< Pointer to the cellular automaton structure.
               struct CALSubstate_r* Q,
               CALreal value				//!< initializing value for the substate at the cell (i, j).
               );

void calInit_b(struct CALModel* calModel,	//!< Pointer to the cellular automaton structure.
               struct CALSubstate_b* Q,
               CALIndices central_cell,
               CALbyte value				//!< initializing value for the substate at the cell (i, j).
               );

void calInit_i(struct CALModel* calModel,	//!< Pointer to the cellular automaton structure.
               struct CALSubstate_i* Q,
               CALIndices central_cell,
               CALint value				//!< initializing value for the substate at the cell (i, j).
               );

void calInit_r(struct CALModel* calModel,	//!< Pointer to the cellular automaton structure.
               struct CALSubstate_r* Q,
               CALIndices central_cell,
               CALreal value				//!< initializing value for the substate at the cell (i, j).
               );
//extern CALbyte (* calGet_b)(struct CALModel* calModel, struct CALSubstate_b* Q, CALIndexes indexes);
//extern CALint (* calGet_i)(struct CALModel* calModel, struct CALSubstate_i* Q, CALIndexes indexes);
//extern CALreal (* calGet_r)(struct CALModel* calModel, struct CALSubstate_r* Q, CALIndexes indexes);

//extern CALbyte (* calGetX_b)(struct CALModel* calModel, struct CALSubstate_b* Q, CALIndexes central_cell, int n);
//extern CALint (* calGetX_i)(struct CALModel* calModel, struct CALSubstate_i* Q, CALIndexes central_cell, int n);
//extern CALreal (* calGetX_r)(struct CALModel* calModel, struct CALSubstate_r* Q, CALIndexes central_cell, int n);

//extern void (* calSet_b)(struct CALModel* calModel, struct CALSubstate_b* Q, CALIndexes central_cell,CALbyte value);
//extern void (* calSet_i)(struct CALModel* calModel, struct CALSubstate_i* Q, CALIndexes central_cell,CALbyte value);
//extern void (* calSet_r)(struct CALModel* calModel, struct CALSubstate_r* Q, CALIndexes central_cell,CALbyte value);

//extern void (* calSetCurrent_b)(struct CALModel* calModel, struct CALSubstate_b* Q, CALIndexes central_cell, CALbyte value);
//extern void (* calSetCurrent_i)(struct CALModel* calModel, struct CALSubstate_i* Q, CALIndexes central_cell, CALbyte value);
//extern void (* calSetCurrent_r)(struct CALModel* calModel, struct CALSubstate_r* Q, CALIndexes central_cell, CALbyte value);

/*! \brief Returns the given cell's value of a byte substate.
*/
CALbyte calGet_b(struct CALModel* calModel,	//!< Pointer to the cellular automaton structure.
                 struct CALSubstate_b* Q,	//!< Pointer to a byte substate.
                 CALIndices indexes
                 );

/*! \brief Returns the given cell's value of an integer substate.
*/
CALint calGet_i(struct CALModel* calModel,	//!< Pointer to the cellular automaton structure.
                struct CALSubstate_i* Q,	//!< Pointer to a int substate.
                CALIndices indexes
                );

/*! \brief Returns the given cell's value of the of a real (floating point) substate.
*/
CALreal calGet_r(struct CALModel* calModel,	//!< Pointer to the cellular automaton structure.
                 struct CALSubstate_r* Q,	//!< Pointer to a real (floating point) substate.
                 CALIndices indexes
                 );



/*! \brief Returns the n-th neighbor's value (in Q) of the given cell.
*/
CALbyte calGetX_b(struct CALModel* calModel,	//!< Pointer to the cellular automaton structure.
                  struct CALSubstate_b* Q,//!< Pointer to a byte substate.
                  CALIndices central_cell, //!< the central cell's coordinates
                  int n					//!< Index of the n-th neighbor.
                  );

/*! \brief Returns the n-th neighbor's value (in Q) of the given cell.
*/
CALint calGetX_i(struct CALModel* calModel,	//!< Pointer to the cellular automaton structure.
                 struct CALSubstate_i* Q,	//!< Pointer to a int substate.
                 CALIndices central_cell, //!< the central cell's coordinates
                 int n					//!< Index of the n-th neighbor.
                 );

/*! \brief Returns the n-th neighbor's value (in Q) of the given cell.
*/
CALreal calGetX_r(struct CALModel* calModel,	//!< Pointer to the cellular automaton structure.
                  struct CALSubstate_r* Q,//!< Pointer to a real (floating point) substate.
                  CALIndices central_cell, //!< the central cell's coordinates
                  int n					//!< Index of the n-th neighbor.
                  );



/*! \brief Sets the cell value of a byte substate.
*/
void calSet_b(struct CALModel* calModel,		//!< Pointer to the cellular automaton structure.
              struct CALSubstate_b* Q,	//!< Pointer to a byte substate.
              CALIndices central_cell, //!< the central cell's coordinates
              CALbyte value				//!< initializing value.
              );

/*! \brief Set the cell value of an integer substate.
*/
void calSet_i(struct CALModel* calModel,		//!< Pointer to the cellular automaton structure.
              struct CALSubstate_i* Q,	//!< Pointer to a int substate.
              CALIndices central_cell, //!< the central cell's coordinates
              CALint value					//!< initializing value.
              );

/*! \brief Set the cell value of a real (floating point) substate.
*/
void calSet_r(struct CALModel* calModel,		//!< Pointer to the cellular automaton structure.
              struct CALSubstate_r* Q,	//!< Pointer to a real (floating point) substate.
              CALIndices central_cell, //!< the central cell's coordinates
              CALreal value				//!< initializing value.
              );



/*! \brief Sets the value of the cell of a byte substate of the CURRENT matrix.
    This operation is unsafe since it writes a value directly to the current matrix.
*/
void calSetCurrent_b(struct CALModel* calModel,	//!< Pointer to the cellular automaton structure.
                     struct CALSubstate_b* Q,	//!< Pointer to a byte substate.
                     CALIndices central_cell, //!< the central cell's coordinates
                     CALbyte value				//!< initializing value.
                     );

/*! \brief Set the value the  cell of an int substate of the CURRENT matrix.
    This operation is unsafe since it writes a value directly to the current matrix.
*/
void calSetCurrent_i(struct CALModel* calModel,	//!< Pointer to the cellular automaton structure.
                     struct CALSubstate_i* Q,	//!< Pointer to a int substate.
                     CALIndices central_cell, //!< the central cell's coordinates
                     CALint value				//!< initializing value.
                     );

/*! \brief Set the value the  cell (i, j) of a real (floating point) substate of the CURRENT matrix.
    This operation is unsafe since it writes a value directly to the current matrix.
*/
void calSetCurrent_r(struct CALModel* calModel,	//!< Pointer to the cellular automaton structure.
                     struct CALSubstate_r* Q,	//!< Pointer to a int substate.
                     CALIndices central_cell, //!< the central cell's coordinates
                     CALreal value				//!< initializing value.
                     );

void calUpdateSubstate_b(struct CALModel* calModel,	//!< Pointer to the cellular automaton structure.
                         struct CALSubstate_b* Q	//!< Pointer to a byte substate.
                         );
/*! \brief Copies the next matrix of a integer substate to the current one: current = next.
    If the active cells optimization is considered, it only updates the active cells.
*/
void calUpdateSubstate_i(struct CALModel* calModel,	//!< Pointer to the cellular automaton structure.
                         struct CALSubstate_i* Q	//!< Pointer to a int substate.
                         );

/*! \brief Copies the next matrix of a real (floating point) substate to the current one: current = next.
    If the active cells optimization is considered, it only updates the active cells.
*/
void calUpdateSubstate_r(struct CALModel* calModel,	//!< Pointer to the cellular automaton structure.
                         struct CALSubstate_r* Q	//!< Pointer to a real (floating point) substate.
                         );

void calUpdate(struct CALModel* calModel);

void calFinalize(struct CALModel* calModel);
#endif

