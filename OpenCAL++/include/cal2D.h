// (C) Copyright University of Calabria and others.
// All rights reserved. This program and the accompanying materials
// are made available under the terms of the GNU Lesser General Public License
// (LGPL) version 2.1 which accompanies this distribution, and is available at
// http://www.gnu.org/licenses/lgpl-2.1.html
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
// Lesser General Public License for more details.

#ifndef cal2D_h
#define cal2D_h

#include <calCommon.h>
#include <ElementaryProcessFunctor2D.h>


/*****************************************************************************
						DEFINITIONS OF NEW DATA TYPES

 *****************************************************************************/

/*! \brief Enumeration of 2D neighbourhood.

	Enumeration that identifies the cellular automaton's 2D neighbourhood.
*/
enum CALNeighborhood2D {
	CAL_CUSTOM_NEIGHBORHOOD_2D,			//!< Enumerator used for the definition of a custom 2D neighbourhood; this is built by calling the function calAddNeighbor2D.
	CAL_VON_NEUMANN_NEIGHBORHOOD_2D,	//!< Enumerator used for specifying the 2D von Neumann neighbourhood; no calls to calAddNeighbor2D are needed.
	CAL_MOORE_NEIGHBORHOOD_2D,			//!< Enumerator used for specifying the 2D Moore neighbourhood; no calls to calAddNeighbor2D are needed.
	CAL_HEXAGONAL_NEIGHBORHOOD_2D,		//!< Enumerator used for specifying the 2D Moore Hexagonal neighbourhood; no calls to calAddNeighbor2D are needed.
	CAL_HEXAGONAL_NEIGHBORHOOD_ALT_2D	//!< Enumerator used for specifying the alternative 90� rotated 2D Moore Hexagonal neighbourhood; no calls to calAddNeighbor2D are needed.
};

#define CAL_HEXAGONAL_SHIFT 7			//<! Shif used for accessing to the correct neighbor in case hexagonal heighbourhood and odd column cell




/*! \brief Active cells structure.
*/
struct CALActiveCells2D {
	CALbyte* flags;				//!< Array of flags having the substates' dimension: flag is CAL_TRUE if the corresponding cell is active, CAL_FALSE otherwise.
	int size_next;	//!< Number of CAL_TRUE flags.
	struct CALCell2D* cells;	//!< Array of computational active cells.
        int size_current;					//!< Number of active cells in the current step.
};

struct CALModel2D;
/*! \brief Fake function pointer type.
*/
//typedef void (* CALCallbackFunc2D)(struct CALModel2D* ca2D, int i, int j);
typedef  ElementaryProcessFunctor2D* CALCallbackFunc2D;
/*! \brief Structure defining the 2D cellular automaton.
*/
struct CALModel2D {
	int rows;							//!< Number of rows of the 2D cellular space.
	int columns;						//!< Number of columns of the 2D cellular space.
	enum CALSpaceBoundaryCondition T;	//!< Type of cellular space: toroidal or non-toroidal.

	enum CALOptimization OPTIMIZATION;	//!< Type of optimization used. It can be CAL_NO_OPT or CAL_OPT_ACTIVE_CELLS.
	struct CALActiveCells2D A;			//!< Computational Active cells object. if A.actives==NULL no optimization is applied.
		
	struct CALCell2D* X;				//!< Array of cell coordinates defining the cellular automaton neighbourhood relation.
	int sizeof_X;						//!< Number of cells belonging to the neighbourhood. Note that predefined neighbourhoods include the central cell.
	enum CALNeighborhood2D X_id;		//!< Neighbourhood relation's id.

	struct CALSubstate2Db** pQb_array;	//!< Array of pointers to 2D substates of type byte
	struct CALSubstate2Di** pQi_array;	//!< Array of pointers to 2D substates of type int
	struct CALSubstate2Dr** pQr_array;	//!< Array of pointers to 2D substates of type real (floating point)
	int sizeof_pQb_array;				//!< Number of substates of type byte.
	int sizeof_pQi_array;				//!< Number of substates of type int.
	int sizeof_pQr_array;				//!< Number of substates of type real (floating point).

	CALCallbackFunc2D *elementary_processes; //!< Array of function pointers to the transition function's elementary processes callback functions. Note that a substates' update must be performed after each elementary process has been applied to each cell of the cellular space (see calGlobalTransitionFunction2D).
	int num_of_elementary_processes; //!< Number of function pointers to the transition functions's elementary processes callbacks.
};




/******************************************************************************
					DEFINITIONS OF FUNCTIONS PROTOTYPES

*******************************************************************************/



/*! \brief Creates an object of type CALModel2D, sets its records and returns it as a pointer; it defines the cellular automaton structure.
*/
struct CALModel2D* calCADef2D(int rows, //!< Number of rows of the 2D cellular space.
							  int columns, //!< Number of columns of the 2D cellular space.
							  enum CALNeighborhood2D CAL_NEIGHBORHOOD_2D, //!< Enumerator that identifies the type of neighbourhood relation to be used.
							  enum CALSpaceBoundaryCondition CAL_TOROIDALITY, //!< Enumerator that identifies the type of cellular space: toroidal or non-toroidal.
							  enum CALOptimization CAL_OPTIMIZATION //!< Enumerator used for specifying the active cells optimization or no optimization.
							  );



/*! \brief Sets the cell (i,j) of the matrix flags to CAL_TRUE and increments the 
	couter sizeof_active_flags.
*/
void calAddActiveCell2D(struct CALModel2D* ca2D,	//!< Pointer to the cellular automaton structure.
						int i,	//!< Row coordinate of the cell to be added.
						int j	//!< Column coordinate of the cell to be added.
						);


/*! \brief Sets the n-th neighbor of the cell (i,j) of the matrix flags to 
	CAL_TRUE and increments the couter sizeof_active_flags.
*/
void calAddActiveCellX2D(struct CALModel2D* ca2D,	//!< Pointer to the cellular automaton structure.
						 int i,	//!< Row coordinate of the central cell.
						 int j,	//!< Column coordinate of the central cell.
						 int n	//!< Index of the n-th neighbor to be added.
						 );

/*! \brief \brief Sets the cell (i,j) of the matrix flags to CAL_FALSE and decrements the 
	couter sizeof_active_flags.
*/
void calRemoveActiveCell2D(struct CALModel2D* ca2D,	//!< Pointer to the cellular automaton structure.
						   int i,	//!< Row coordinate of the cell to be removed.
						   int j	//!< Column coordinate of the cell to be removed.
						   );

/*! \brief Puts the cells marked as actives in A.flags into the array of active cells 
	A.cells and sets its dimension, A.size, to A.size_of_actives, i.e. the actual 
	number of active cells.
*/
void calUpdateActiveCells2D(struct CALModel2D* ca2D	//!< Pointer to the cellular automaton structure.
						   );



/*! \brief Adds a neighbour to CALModel2D::X and updates the value of CALModel2D::sizeof_X.
*/
struct CALCell2D*  calAddNeighbor2D(struct CALModel2D* ca2D, //!< Pointer to the cellular automaton structure.
									int i,	//!< Relative row coordinate with respect to the central cell (the north neighbour has i = -1, the south i = +1, etc.).
									int j	//!< Relative column coordinate with respect to the central cell (the east neighbour has j = -1, the west i = +1, etc.).
									);



/*! \brief Creates and adds a new byte substate to CALModel2D::pQb_array and return a pointer to it. 
*/
struct CALSubstate2Db* calAddSubstate2Db(struct CALModel2D* ca2D	//!< Pointer to the cellular automaton structure.
										 );

/*! \brief Creates and adds a new int substate to CALModel2D::pQi_array and return a pointer to it.
*/
struct CALSubstate2Di* calAddSubstate2Di(struct CALModel2D* ca2D	//!< Pointer to the cellular automaton structure.
										 );

/*! \brief Creates and adds a new real (floating point) substate to CALModel2D::pQr_array and return a pointer to it.
*/
struct CALSubstate2Dr* calAddSubstate2Dr(struct CALModel2D* ca2D	//!< Pointer to the cellular automaton structure.
										 );



/*! \brief Creates a new single-layer byte substate and returns a pointer to it.
	Note that sinlgle-layer substates are not added to CALModel2D::pQ*_array because
	they do not nedd to be updated.
*/
struct CALSubstate2Db* calAddSingleLayerSubstate2Db(struct CALModel2D* ca2D	//!< Pointer to the cellular automaton structure.
										            );

/*! \brief Creates a new single-layer int substate and returns a pointer to it.
	Note that sinlgle-layer substates are not added to CALModel2D::pQ*_array because
	they do not nedd to be updated.
*/
struct CALSubstate2Di* calAddSingleLayerSubstate2Di(struct CALModel2D* ca2D	//!< Pointer to the cellular automaton structure.
										            );

/*! \brief Creates a new single-layer real (floating point) substate returns a pointer to it.
	Note that sinlgle-layer substates are not added to CALModel2D::pQ*_array because
	they do not nedd to be updated.
*/
struct CALSubstate2Dr* calAddSingleLayerSubstate2Dr(struct CALModel2D* ca2D	//!< Pointer to the cellular automaton structure.
										            );



/*! \brief Adds a transition function's elementary process to the CALModel2D::elementary_processes array of callbacks pointers. 
	Note that the function calGlobalTransitionFunction2D calls a substates' update after each elementary process.
*/
CALCallbackFunc2D* calAddElementaryProcess2D(struct CALModel2D* ca2D,	//!< Pointer to the cellular automaton structure.
											CALCallbackFunc2D elementary_process //!< Pointer to a transition function's elementary process.
											 );



/*! \brief Initializes a byte substate to a constant value; both the current and next (if not single layer substate) matrices are initialized.
*/
void calInitSubstate2Db(struct CALModel2D* ca2D,	//!< Pointer to the cellular automaton structure.
						struct CALSubstate2Db* Q,	//!< Pointer to a 2D byte substate.
						CALbyte value				//!< Value to which each cell of the substate is set.
						);

/*! \brief Initializes a integer substate a constant value; both the current and next (if not single layer substate) matrices are initialized.
*/
void calInitSubstate2Di(struct CALModel2D* ca2D,	//!< Pointer to the cellular automaton structure.
						struct CALSubstate2Di* Q,	//!< Pointer to a 2D int substate.
						CALint value				//!< Value to which each cell of the substate is set.
						);

/*! \brief Initializes a real (floating point) substate a constant value; both the current and next (if not single layer substate) matrices are initialized.
*/
void calInitSubstate2Dr(struct CALModel2D* ca2D,	//!< Pointer to the cellular automaton structure.
						struct CALSubstate2Dr* Q,	//!< Pointer to a 2D real (floating point) substate.
						CALreal value				//!< Value to which each cell of the substate is set.
						);



/*! \brief Initializes a the next buffer of a byte substate to a constant value.
*/
void calInitSubstateNext2Db(struct CALModel2D* ca2D,	//!< Pointer to the cellular automaton structure.
							struct CALSubstate2Db* Q,	//!< Pointer to a 2D byte substate.
							CALbyte value				//!< Value to which each cell of the substate is set.
							);

/*! \brief Initializes a the next buffer of an integer substate to a constant value.
*/
void calInitSubstateNext2Di(struct CALModel2D* ca2D,	//!< Pointer to the cellular automaton structure.
							struct CALSubstate2Di* Q,	//!< Pointer to a 2D integer substate.
							CALint value				//!< Value to which each cell of the substate is set.
							);

/*! \brief Initializes a the next buffer of a real (floating point) substate to a constant value.
*/
void calInitSubstateNext2Dr(struct CALModel2D* ca2D,	//!< Pointer to the cellular automaton structure.
							struct CALSubstate2Dr* Q,	//!< Pointer to a 2D real (floating point) substate.
							CALreal value				//!< Value to which each cell of the substate is set.
							);


/*! \brief Copies the next matrix of a byte substate to the current one: current = next.
	If the active cells optimization is considered, it only updates the active cells.
*/
void calUpdateSubstate2Db(struct CALModel2D* ca2D,	//!< Pointer to the cellular automaton structure.
						  struct CALSubstate2Db* Q	//!< Pointer to a 2D byte substate.
						  );

/*! \brief Copies the next matrix of a integer substate to the current one: current = next.
	If the active cells optimization is considered, it only updates the active cells.
*/
void calUpdateSubstate2Di(struct CALModel2D* ca2D,	//!< Pointer to the cellular automaton structure.
						  struct CALSubstate2Di* Q	//!< Pointer to a 2D int substate.
						  );
/*! \brief Copies the next matrix of a real (floating point) substate to the current one: current = next.
	If the active cells optimization is considered, it only updates the active cells.
*/
void calUpdateSubstate2Dr(struct CALModel2D* ca2D,	//!< Pointer to the cellular automaton structure.
						  struct CALSubstate2Dr* Q	//!< Pointer to a 2D real (floating point) substate.
						  );



/*! \brief Apply an elementary process to all the cellular space.
*/
void calApplyElementaryProcess2D(struct CALModel2D* ca2D,	//!< Pointer to the cellular automaton structure.
											  CALCallbackFunc2D elementary_process //!< Pointer to a transition function's elementary process.
											   );



/*! \brief The cellular automaton global transition function.
	It applies the transition function to each cell of the cellular space.
	After each elementary process, a global substates update is performed.
*/
void calGlobalTransitionFunction2D(struct CALModel2D* ca2D	//!< Pointer to the cellular automaton structure.
								   );



/*! \brief Updates all the substates registered in CALModel2D::pQb_array, 
	CALModel2D::pQi_array and CALModel2D::pQr_array. 
	It is called by the global transition function.
*/
void calUpdate2D(struct CALModel2D* ca2D	//!< Pointer to the cellular automaton structure.
				 );



/*! \brief Inits the value of a byte substate in the cell (i, j) to value; it updates both the current and next matrices at the position (i, j).  
*/
void calInit2Db(struct CALModel2D* ca2D,	//!< Pointer to the cellular automaton structure.
				struct CALSubstate2Db* Q,	//!< Pointer to a 2D byte substate.
				int i,						//!< Row coordinate of the cell to be initialized.
				int j,						//!< Column coordinate of the cell to be initialized.
				CALbyte value				//!< initializing value for the substate at the cell (i, j).
				);

/*! \brief Inits the value of a integer substate in the cell (i, j) to value; it updates both the current and next matrices at the position (i, j).  
*/
void calInit2Di(struct CALModel2D* ca2D,	//!< Pointer to the cellular automaton structure.
				struct CALSubstate2Di* Q,	//!< Pointer to a 2D int substate.
				int i,						//!< Row coordinate of the cell to be initialized.
				int j,						//!< Column coordinate of the cell to be initialized.
				CALint value				//!< initializing value for the substate at the cell (i, j).
				);

/*! \brief Inits the value of a real (floating point) substate in the cell (i, j) to value; it updates both the current and next matrices at the position (i, j).  
*/
void calInit2Dr(struct CALModel2D* ca2D,	//!< Pointer to the cellular automaton structure.
				struct CALSubstate2Dr* Q,	//!< Pointer to a 2D real (floating point) substate.
				int i,						//!< Row coordinate of the cell to be initialized.
				int j,						//!< Column coordinate of the cell to be initialized.
				CALreal value				//!< initializing value for the substate at the cell (i, j).
				);



/*! \brief Returns the cell (i, j) value of a byte substate.  
*/
CALbyte calGet2Db(struct CALModel2D* ca2D,	//!< Pointer to the cellular automaton structure.
				  struct CALSubstate2Db* Q,	//!< Pointer to a 2D byte substate.	
				  int i,					//!< Row coordinate of the cell.
				  int j						//!< Column coordinate of the cell.
				  );

/*! \brief Returns the cell (i, j) value of an integer substate.  
*/
CALint calGet2Di(struct CALModel2D* ca2D,	//!< Pointer to the cellular automaton structure.
				 struct CALSubstate2Di* Q,	//!< Pointer to a 2D int substate.
				 int i,						//!< Row coordinate of the cell.
				 int j						//!< Column coordinate of the cell.
				 );

/*! \brief Returns the cell (i, j) value of the of a real (floating point) substate.  
*/
CALreal calGet2Dr(struct CALModel2D* ca2D,	//!< Pointer to the cellular automaton structure.
				  struct CALSubstate2Dr* Q,	//!< Pointer to a 2D real (floating point) substate.
				  int i,					//!< Row coordinate of the cell.
				  int j						//!< Column coordinate of the cell.
				  );



/*! \brief Returns the n-th neighbor of the cell (i, j) value of a byte substate.
*/
CALbyte calGetX2Db(struct CALModel2D* ca2D,	//!< Pointer to the cellular automaton structure.
				   struct CALSubstate2Db* Q,//!< Pointer to a 2D byte substate.
				   int i,					//!< Row coordinate of the central cell.
				   int j,					//!< Column coordinate of the central cell.
				   int n					//!< Index of the n-th neighbor.
				   );

/*! \brief Returns the n-th neighbor of the cell (i, j) value of an integer substate.
*/
CALint calGetX2Di(struct CALModel2D* ca2D,	//!< Pointer to the cellular automaton structure.
				  struct CALSubstate2Di* Q,	//!< Pointer to a 2D int substate.
				  int i,					//!< Row coordinate of the central cell.
				  int j,					//!< Column coordinate of the central cell.
				  int n						//!< Index of the n-th neighbor.
				  );

/*! \brief Returns the n-th neighbor of the cell (i, j) value of a real (floating point) substate.
*/
CALreal calGetX2Dr(struct CALModel2D* ca2D,	//!< Pointer to the cellular automaton structure.
				   struct CALSubstate2Dr* Q,//!< Pointer to a 2D real (floating point) substate.
				   int i,					//!< Row coordinate of the central cell.
				   int j,					//!< Column coordinate of the central cell.
				   int n					//!< Index of the n-th neighbor.
				   );



/*! \brief Sets the cell (i, j) value of a byte substate.  
*/
void calSet2Db(struct CALModel2D* ca2D,		//!< Pointer to the cellular automaton structure.
			   struct CALSubstate2Db* Q,	//!< Pointer to a 2D byte substate.
			   int i,						//!< Row coordinate of the cell.
			   int j,						//!< Column coordinate of the cell.
			   CALbyte value				//!< initializing value.
			   );

/*! \brief Set the cell (i, j) value of an integer substate.  
*/
void calSet2Di(struct CALModel2D* ca2D,		//!< Pointer to the cellular automaton structure.
			   struct CALSubstate2Di* Q,	//!< Pointer to a 2D int substate.
			   int i,						//!< Row coordinate of the cell.
			   int j,						//!< Column coordinate of the cell.
			   CALint value					//!< initializing value.
			   );

/*! \brief Set the cell (i, j) value of a real (floating point) substate.  
*/
void calSet2Dr(struct CALModel2D* ca2D,		//!< Pointer to the cellular automaton structure.
			   struct CALSubstate2Dr* Q,	//!< Pointer to a 2D real (floating point) substate.
			   int i,						//!< Row coordinate of the cell.
			   int j,						//!< Column coordinate of the cell.
			   CALreal value				//!< initializing value.
			   );



/*! \brief Sets the value of the cell (i, j) of a byte substate of the CURRENT matrix.
	This operation is unsafe since it writes a value directly to the current matrix.
*/
void calSetCurrent2Db(struct CALModel2D* ca2D,	//!< Pointer to the cellular automaton structure.
					  struct CALSubstate2Db* Q,	//!< Pointer to a 2D byte substate.
					  int i,					//!< Row coordinate of the central cell.
					  int j,					//!< Column coordinate of the central cell.
					  CALbyte value				//!< initializing value.
					  );

/*! \brief Set the value the  cell (i, j) of an int substate of the CURRENT matrix.
	This operation is unsafe since it writes a value directly to the current matrix.
*/
void calSetCurrent2Di(struct CALModel2D* ca2D,	//!< Pointer to the cellular automaton structure.
					  struct CALSubstate2Di* Q,	//!< Pointer to a 2D int substate.
					  int i,					//!< Row coordinate of the central cell.
					  int j,					//!< Column coordinate of the central cell.
					  CALint value				//!< initializing value.
					  );

/*! \brief Set the value the  cell (i, j) of a real (floating point) substate of the CURRENT matrix.
	This operation is unsafe since it writes a value directly to the current matrix.
*/
void calSetCurrent2Dr(struct CALModel2D* ca2D,	//!< Pointer to the cellular automaton structure.
					  struct CALSubstate2Dr* Q,	//!< Pointer to a 2D int substate.
					  int i,					//!< Row coordinate of the central cell.
					  int j,					//!< Column coordinate of the central cell.
					  CALreal value				//!< initializing value.
					  );



/*! \brief Finalization function: it releases the memory allocated.
*/
void calFinalize2D(struct CALModel2D* ca2D	//!< Pointer to the cellular automaton structure.
				  );



#endif
