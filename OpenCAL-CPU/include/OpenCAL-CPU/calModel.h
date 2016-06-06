﻿#ifndef cal_model
#define cal_model

#include <OpenCAL-CPU/calCommon.h>
#include <OpenCAL-CPU/calRun.h>
#include <OpenCAL-CPU/calRun.h>

/*****************************************************************************
                        DEFINITIONS OF NEW DATA TYPES
 *****************************************************************************/

typedef int* CALNeighbourPattern;

#define CAL_HEXAGONAL_SHIFT 7			//<! Shif used for accessing to the correct neighbor in case hexagonal heighbourhood and odd column cell

struct CALModel;
/*! \brief Fake function pointer type.
*/
typedef void (* CALLocalProcess)(struct CALModel* calModel, CALIndexes);
typedef void (* CALGlobalProcess)(struct CALModel* calModel);

struct CALProcess {
        CALLocalProcess * localFunction;
        CALGlobalProcess * globalFunction;
        char type;
};

/*! \brief Structure defining the cellular automaton.
*/
struct CALModel {
        int* coordinatesDimensions;
        int numberOfCoordinates;
        int cellularSpaceDimension;

        enum CALOptimization OPTIMIZATION;	//!< Type of optimization used. It can be CAL_NO_OPT or CAL_OPT_ACTIVE_CELLS.
        //struct CALActiveCells2D A;			//!< Computational Active cells object. if A.actives==NULL no optimization is applied.

        struct CALIndexes X;				//!< Array of cell coordinates defining the cellular automaton neighbourhood relation.
        int sizeof_X;						//!< Number of cells belonging to the neighbourhood. Note that predefined neighbourhoods include the central cell.

        struct CALSubstate_b** pQb_array;	//!< Array of pointers to 2D substates of type byte
        struct CALSubstate_i** pQi_array;	//!< Array of pointers to 2D substates of type int
        struct CALSubstate_r** pQr_array;	//!< Array of pointers to 2D substates of type real (floating point)
        int sizeof_pQb_array;				//!< Number of substates of type byte.
        int sizeof_pQi_array;				//!< Number of substates of type int.
        int sizeof_pQr_array;				//!< Number of substates of type real (floating point).

        struct CALProcess * model_functions; //!< Array of function pointers to the transition function's elementary processes or generic global functions.Note that a substates' update must be performed after each elementary process has been applied to each cell of the cellular space (see calGlobalTransitionFunction2D).
        int num_of_functions; //!< Number of function pointers to the transition functions's elementary processes callbacks.

        struct CALRun* calRun; //!< Pointer to a structure containing the appropriate functions which implementation should change according to the type of execution chosen (serial or parallel)
};


/******************************************************************************
                    DEFINITIONS OF FUNCTIONS PROTOTYPES
*******************************************************************************/



/*! \brief Creates an object of type CALModel2D, sets its records and returns it as a pointer; it defines the cellular automaton structure.
*/
struct CALModel* calCADef(int numberOfCoordinates, //!< Number of coordinates of the Cellular Space.
                          CALIndexes coordinatesDimensions,
                          enum CALSpaceBoundaryCondition CAL_TOROIDALITY, //!< Enumerator that identifies the type of cellular space: toroidal or non-toroidal.
                          enum CALExecutionType executionType, //!< Enumerator that specifies whether the execution flow must be serial or parallel.
                          enum CALOptimization CAL_OPTIMIZATION //!< Enumerator used for specifying the active cells optimization or no optimization.
                          );


/*! \brief Adds a neighbour to CALModel2D::X and updates the value of CALModel2D::sizeof_X.
*/
void calAddNeighbor2D(struct CALModel* calModel, //!< Pointer to the cellular automaton structure.
                      CALIndexes neighbourIndex  //!< Indexes of the n-th neighbour
                      );


enum CALInitMethod { CAL_NO_INIT = 0, CAL_INIT_CURRENT, CAL_INIT_NEXT, CAL_INIT_BOTH };

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

/*! \brief Adds a local function to the CALModel::modelFunctions array.
*/
void calAddLocalProcess(struct CALModel* calModel,	//!< Pointer to the cellular automaton structure.
                        CALLocalProcess elementary_process //!< Pointer to a transition function's elementary process.
                        );

/*! \brief Adds a global function to the CALModel::modelFunctions array.
*/
void calAddGlobalProcess(struct CALModel* calModel,	//!< Pointer to the cellular automaton structure.
                         CALGlobalProcess elementary_process //!< Pointer to a global function.
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



/*! \brief Apply a local process to all the cellular space.
*/
void calApplyLocalProcess(struct CALModel* calModel,	//!< Pointer to the cellular automaton structure.
                          CALLocalProcess local_process //!< Pointer to a local function.
                          );

/*! \brief Apply a global process to all the cellular space.
*/
void calApplyGlobalProcess(struct CALModel* calModel,	//!< Pointer to the cellular automaton structure.
                          CALGlobalProcess global_process //!< Pointer to a global function.
                          );




/*! \brief The cellular automaton global transition function.
    It applies the transition function to each cell of the cellular space.
    After each local process, a global substates update is performed.
*/
void calGlobalTransitionFunction(struct CALModel* calModel	//!< Pointer to the cellular automaton structure.
                                 );



/*! \brief Updates all the substates registered in CALModel::pQb_array,
    CALModel::pQi_array and CALModel::pQr_array.
    It is called by the global transition function.
*/
void calUpdate2D(struct CALModel* calModel	//!< Pointer to the cellular automaton structure.
                 );



/*! \brief Inits the value of a byte substate in the given cell to value; it updates both the current and next matrices at that position.
*/
void calInit_b(struct CALModel* calModel,	//!< Pointer to the cellular automaton structure.
               struct CALSubstate_b* Q,	//!< Pointer to a byte substate.
               CALIndexes indexes,
               CALbyte value				//!< initializing value for the substate at the cell (i, j).
               );

/*! \brief Inits the value of a byte substate in the given cell to value; it updates both the current and next matrices at that position.
*/
void calInit_i(struct CALModel* calModel,	//!< Pointer to the cellular automaton structure.
               struct CALSubstate_i* Q,	//!< Pointer to a int substate.
               CALIndexes indexes,
               CALint value				//!< initializing value for the substate at the cell (i, j).
               );

/*! \brief Inits the value of a byte substate in the given cell to value; it updates both the current and next matrices at that position.
*/
void calInit_r(struct CALModel* calModel,	//!< Pointer to the cellular automaton structure.
               struct CALSubstate_r* Q,	//!< Pointer to a real (floating point) substate.
               CALIndexes indexes,
               CALreal value				//!< initializing value for the substate at the cell (i, j).
               );



/*! \brief Returns the given cell's value of a byte substate.
*/
CALbyte calGet_b(struct CALModel* calModel,	//!< Pointer to the cellular automaton structure.
                 struct CALSubstate_b* Q,	//!< Pointer to a byte substate.
                 CALIndexes indexes
                 );

/*! \brief Returns the given cell's value of an integer substate.
*/
CALint calGet_i(struct CALModel* calModel,	//!< Pointer to the cellular automaton structure.
                struct CALSubstate_i* Q,	//!< Pointer to a int substate.
                CALIndexes indexes
                );

/*! \brief Returns the given cell's value of the of a real (floating point) substate.
*/
CALreal calGet_r(struct CALModel* calModel,	//!< Pointer to the cellular automaton structure.
                 struct CALSubstate_r* Q,	//!< Pointer to a real (floating point) substate.
                 CALIndexes indexes
                 );



/*! \brief Returns the n-th neighbor's value (in Q) of the given cell.
*/
CALbyte calGetX_b(struct CALModel* calModel,	//!< Pointer to the cellular automaton structure.
                  struct CALSubstate_b* Q,//!< Pointer to a byte substate.
                  CALIndexes central_cell, //!< the central cell's coordinates
                  int n					//!< Index of the n-th neighbor.
                  );

/*! \brief Returns the n-th neighbor's value (in Q) of the given cell.
*/
CALint calGetX_i(struct CALModel* calModel,	//!< Pointer to the cellular automaton structure.
                 struct CALSubstate_i* Q,	//!< Pointer to a int substate.
                 CALIndexes central_cell, //!< the central cell's coordinates
                 int n					//!< Index of the n-th neighbor.
                 );

/*! \brief Returns the n-th neighbor's value (in Q) of the given cell.
*/
CALreal calGetX_r(struct CALModel* calModel,	//!< Pointer to the cellular automaton structure.
                  struct CALSubstate_r* Q,//!< Pointer to a real (floating point) substate.
                  CALIndexes central_cell, //!< the central cell's coordinates
                  int n					//!< Index of the n-th neighbor.
                  );



/*! \brief Sets the cell value of a byte substate.
*/
void calSet_b(struct CALModel* calModel,		//!< Pointer to the cellular automaton structure.
              struct CALSubstate_b* Q,	//!< Pointer to a byte substate.
              CALIndexes central_cell, //!< the central cell's coordinates
              CALbyte value				//!< initializing value.
              );

/*! \brief Set the cell value of an integer substate.
*/
void calSet_i(struct CALModel* calModel,		//!< Pointer to the cellular automaton structure.
              struct CALSubstate_i* Q,	//!< Pointer to a int substate.
              CALIndexes central_cell, //!< the central cell's coordinates
              CALint value					//!< initializing value.
              );

/*! \brief Set the cell value of a real (floating point) substate.
*/
void calSet_r(struct CALModel* calModel,		//!< Pointer to the cellular automaton structure.
              struct CALSubstate_r* Q,	//!< Pointer to a real (floating point) substate.
              CALIndexes central_cell, //!< the central cell's coordinates
              CALreal value				//!< initializing value.
              );



/*! \brief Sets the value of the cell of a byte substate of the CURRENT matrix.
    This operation is unsafe since it writes a value directly to the current matrix.
*/
void calSetCurrent_b(struct CALModel* calModel,	//!< Pointer to the cellular automaton structure.
                     struct CALSubstate_b* Q,	//!< Pointer to a byte substate.
                     CALIndexes central_cell, //!< the central cell's coordinates
                     CALbyte value				//!< initializing value.
                     );

/*! \brief Set the value the  cell of an int substate of the CURRENT matrix.
    This operation is unsafe since it writes a value directly to the current matrix.
*/
void calSetCurrent_i(struct CALModel* calModel,	//!< Pointer to the cellular automaton structure.
                     struct CALSubstate_i* Q,	//!< Pointer to a int substate.
                     CALIndexes central_cell, //!< the central cell's coordinates
                     CALint value				//!< initializing value.
                     );

/*! \brief Set the value the  cell (i, j) of a real (floating point) substate of the CURRENT matrix.
    This operation is unsafe since it writes a value directly to the current matrix.
*/
void calSetCurrent_r(struct CALModel* calModel,	//!< Pointer to the cellular automaton structure.
                     struct CALSubstate_r* Q,	//!< Pointer to a int substate.
                     CALIndexes central_cell, //!< the central cell's coordinates
                     CALreal value				//!< initializing value.
                     );



/*! \brief Finalization function: it releases the memory allocated.
*/
void calFinalize2D(struct CALModel* calModel	//!< Pointer to the cellular automaton structure.
                   );








#endif
