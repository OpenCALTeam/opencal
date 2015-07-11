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

#ifndef cal3D_h
#define cal3D_h

#include <calCommon.h>
#include <ElementaryProcessFunctor3D.h>

/*****************************************************************************
						DEFINITIONS OF NEW DATA TYPES

 *****************************************************************************/

/*! \brief Enumeration of 3D neighbourhood.

	Enumeration that identifies the cellular automaton's 3D neighbourhood.
*/
enum CALNeighborhood3D {
	CAL_CUSTOM_NEIGHBORHOOD_3D,			//!< Enumerator used for the definition of a custom 3D neighbourhood; this is built by calling the function calAddNeighbor3D.
	CAL_VON_NEUMANN_NEIGHBORHOOD_3D,	//!< Enumerator used for specifying the 3D von Neumann neighbourhood; no calls to calAddNeighbor3D are needed.
	CAL_MOORE_NEIGHBORHOOD_3D			//!< Enumerator used for specifying the 3D Moore neighbourhood; no calls to calAddNeighbor3D are needed.
};



/*! \brief Active cells structure.
*/
struct CALActiveCells3D {
	CALbyte* flags;				//!< Array of flags having the substates' dimension: flag is CAL_TRUE if the corresponding cell is active, CAL_FALSE otherwise.
	int size_next;	//!< Number of CAL_TRUE flags.
	struct CALCell3D* cells;	//!< Array of computational active cells.
	int size_current;					//!< Number of active cells in the current step.
};


typedef  ElementaryProcessFunctor3D* CALCallbackFunc3D;

/*! \brief Structure defining the 3D cellular automaton.
*/
struct CALModel3D {
	int rows;							//!< Number of rows of the 3D cellular space.
	int columns;						//!< Number of columns of the 3D cellular space.
	int slices;							//!< Number of slices of the 3D cellular space.
	enum CALSpaceBoundaryCondition T;	//!< Type of cellular space: toroidal or non-toroidal.

	enum CALOptimization OPTIMIZATION;	//!< Type of optimization used. It can be CAL_NO_OPT or CAL_OPT_ACTIVE_CELLS.
	struct CALActiveCells3D A;			//!< Computational Active cells object. if A.actives==NULL no optimization is applied.

	struct CALCell3D* X;				//!< Array of cell coordinates defining the cellular automaton neighbourhood relation.
	int sizeof_X;						//!< Number of cells belonging to the neighbourhood. Note that predefined neighbourhoods include the central cell.
	enum CALNeighborhood3D X_id;		//!< Neighbourhood relation's id.

	struct CALSubstate3Db** pQb_array;	//!< Array of pointers to 3D substates of type byte
	struct CALSubstate3Di** pQi_array;	//!< Array of pointers to 3D substates of type int
	struct CALSubstate3Dr** pQr_array;	//!< Array of pointers to 3D substates of type real (floating point)
	int sizeof_pQb_array;				//!< Number of substates of type byte.
	int sizeof_pQi_array;				//!< Number of substates of type int.
	int sizeof_pQr_array;				//!< Number of substates of type real (floating point).

	CALCallbackFunc3D *elementary_processes;
	int num_of_elementary_processes; //!< Number of function pointers to the transition functions's elementary processes callbacks.
};

/*! \brief Fake function pointer type.
*/
//typedef void (* CALCallbackFunc3D)(struct CALModel3D* ca3D, int i, int j, int k);




/******************************************************************************
					DEFINITIONS OF FUNCTIONS PROTOTYPES

*******************************************************************************/



/*! \brief Creates an object of type CALModel3D, sets its records and returns it as a pointer; it defines the cellular automaton structure.
*/
struct CALModel3D* calCADef3D(int rows, //!< Number of rows of the 3D cellular space.
							  int columns, //!< Number of columns of the 3D cellular space.
							  int slices, //!< Number of slices of the 3D cellular space.
							  enum CALNeighborhood3D CAL_NEIGHBORHOOD_3D, //!< Enumerator that identifies the type of neighbourhood relation to be used.
							  enum CALSpaceBoundaryCondition CAL_TOROIDALITY, //!< Enumerator that identifies the type of cellular space: toroidal or non-toroidal.
							  enum CALOptimization CAL_OPTIMIZATION //!< Enumerator used for specifying the active cells optimization or no optimization.
							  );



/*! \brief Sets the cell (i,j) of the matrix flags to CAL_TRUE and increments the
	couter sizeof_active_flags.
*/
void calAddActiveCell3D(struct CALModel3D* ca3D,	//!< Pointer to the cellular automaton structure.
						int i,	//!< Row coordinate of the cell to be added.
						int j,	//!< Column coordinate of the cell to be added.
						int k	//!< Slice coordinate of the cell to be added.
						);


/*! \brief Sets the n-th neighbor of the cell (i,j) of the matrix flags to
	CAL_TRUE and increments the couter sizeof_active_flags.
*/
void calAddActiveCellX3D(struct CALModel3D* ca3D,	//!< Pointer to the cellular automaton structure.
						 int i,	//!< Row coordinate of the central cell.
						 int j,	//!< Column coordinate of the central cell.
						 int k,	//!< Slice coordinate of the central cell.
						 int n	//!< Index of the n-th neighbor to be added.
						 );

/*! \brief \brief Sets the cell (i,j) of the matrix flags to CAL_FALSE and decrements the
	couter sizeof_active_flags.
*/
void calRemoveActiveCell3D(struct CALModel3D* ca3D,	//!< Pointer to the cellular automaton structure.
						   int i,	//!< Row coordinate of the cell to be removed.
						   int j,	//!< Column coordinate of the cell to be removed.
						   int k	//!< Slice coordinate of the cell to be removed.
						   );

/*! \brief Puts the cells marked as actives in A.flags into the array of active cells
	A.cells and sets its dimension, A.size, to A.size_of_actives, i.e. the actual
	number of active cells.
*/
void calUpdateActiveCells3D(struct CALModel3D* ca3D	//!< Pointer to the cellular automaton structure.
						   );



/*! \brief Adds a neighbour to CALModel3D::X and updates the value of CALModel3D::sizeof_X.
*/
struct CALCell3D*  calAddNeighbor3D(struct CALModel3D* ca3D, //!< Pointer to the cellular automaton structure.
									int i,	//!< Relative row coordinate with respect to the central cell (the north neighbour has i = -1, the south i = +1, etc.).
									int j,	//!< Relative column coordinate with respect to the central cell (the east neighbour has j = -1, the west i = +1, etc.).
									int k	//!< Relative slice coordinate with respect to the central cell (that has relative slice 0).
									);



/*! \brief Creates and adds a new byte substate to CALModel3D::pQb_array and return a pointer to it.
*/
struct CALSubstate3Db* calAddSubstate3Db(struct CALModel3D* ca3D	//!< Pointer to the cellular automaton structure.
										 );

/*! \brief Creates and adds a new int substate to CALModel3D::pQi_array and return a pointer to it.
*/
struct CALSubstate3Di* calAddSubstate3Di(struct CALModel3D* ca3D	//!< Pointer to the cellular automaton structure.
										 );

/*! \brief Creates and adds a new real (floating point) substate to CALModel3D::pQr_array and return a pointer to it.
*/
struct CALSubstate3Dr* calAddSubstate3Dr(struct CALModel3D* ca3D	//!< Pointer to the cellular automaton structure.
										 );



/*! \brief Creates a new single-layer byte substate and returns a pointer to it.
	Note that sinlgle-layer substates are not added to CALModel3D::pQ*_array because
	they do not nedd to be updated.
*/
struct CALSubstate3Db* calAddSingleLayerSubstate3Db(struct CALModel3D* ca3D	//!< Pointer to the cellular automaton structure.
										            );

/*! \brief Creates a new single-layer int substate and returns a pointer to it.
	Note that sinlgle-layer substates are not added to CALModel3D::pQ*_array because
	they do not nedd to be updated.
*/
struct CALSubstate3Di* calAddSingleLayerSubstate3Di(struct CALModel3D* ca3D	//!< Pointer to the cellular automaton structure.
										            );

/*! \brief Creates a new single-layer real (floating point) substate returns a pointer to it.
	Note that sinlgle-layer substates are not added to CALModel3D::pQ*_array because
	they do not nedd to be updated.
*/
struct CALSubstate3Dr* calAddSingleLayerSubstate3Dr(struct CALModel3D* ca3D	//!< Pointer to the cellular automaton structure.
										            );


/*! \brief Adds a transition function's elementary process to the CALModel3D::elementary_processes array of callbacks pointers.
	Note that the function calGlobalTransitionFunction3D calls a substates' update after each elementary process.
*/
CALCallbackFunc3D* calAddElementaryProcess3D(struct CALModel3D* ca3D,	//!< Pointer to the cellular automaton structure.
											 CALCallbackFunc3D elementary_process //!< Pointer to a transition function's elementary process.
											 );



/*! \brief Initializes a byte substate to a constant value; both the current and next (if not single layer substate) matrices are initialized.
*/
void calInitSubstate3Db(struct CALModel3D* ca3D,	//!< Pointer to the cellular automaton structure.
						struct CALSubstate3Db* Q,	//!< Pointer to a 3D byte substate.
						CALbyte value				//!< Value to which each cell of the substate is set.
						);

/*! \brief Initializes a integer substate a constant value; both the current and next (if not single layer substate) matrices are initialized.
*/
void calInitSubstate3Di(struct CALModel3D* ca3D,	//!< Pointer to the cellular automaton structure.
						struct CALSubstate3Di* Q,	//!< Pointer to a 3D int substate.
						CALint value				//!< Value to which each cell of the substate is set.
						);

/*! \brief Initializes a real (floating point) substate a constant value; both the current and next (if not single layer substate) matrices are initialized.
*/
void calInitSubstate3Dr(struct CALModel3D* ca3D,	//!< Pointer to the cellular automaton structure.
						struct CALSubstate3Dr* Q,	//!< Pointer to a 3D real (floating point) substate.
						CALreal value				//!< Value to which each cell of the substate is set.
						);



/*! \brief Initializes a the next buffer of a byte substate to a constant value.
*/
void calInitSubstateNext3Db(struct CALModel3D* ca3D,	//!< Pointer to the cellular automaton structure.
							struct CALSubstate3Db* Q,	//!< Pointer to a 3D byte substate.
							CALbyte value				//!< Value to which each cell of the substate is set.
							);

/*! \brief Initializes a the next buffer of an integer substate to a constant value.
*/
void calInitSubstateNext3Di(struct CALModel3D* ca3D,	//!< Pointer to the cellular automaton structure.
							struct CALSubstate3Di* Q,	//!< Pointer to a 3D integer substate.
							CALint value				//!< Value to which each cell of the substate is set.
							);

/*! \brief Initializes a the next buffer of a real (floating point) substate to a constant value.
*/
void calInitSubstateNext3Dr(struct CALModel3D* ca3D,	//!< Pointer to the cellular automaton structure.
							struct CALSubstate3Dr* Q,	//!< Pointer to a 3D real (floating point) substate.
							CALreal value				//!< Value to which each cell of the substate is set.
							);



/*! \brief Copies the next 3D buffer of a byte substate to the current one: current = next.
	If the active cells optimization is considered, it only updates the active cells.
*/
void calUpdateSubstate3Db(struct CALModel3D* ca3D,	//!< Pointer to the cellular automaton structure.
						  struct CALSubstate3Db* Q	//!< Pointer to a 3D byte substate.
						  );

/*! \brief Copies the next 3D buffer of a integer substate to the current one: current = next.
	If the active cells optimization is considered, it only updates the active cells.
*/
void calUpdateSubstate3Di(struct CALModel3D* ca3D,	//!< Pointer to the cellular automaton structure.
						  struct CALSubstate3Di* Q	//!< Pointer to a 3D int substate.
						  );
/*! \brief Copies the next 3D buffer of a real (floating point) substate to the current one: current = next.
	If the active cells optimization is considered, it only updates the active cells.
*/
void calUpdateSubstate3Dr(struct CALModel3D* ca3D,	//!< Pointer to the cellular automaton structure.
						  struct CALSubstate3Dr* Q	//!< Pointer to a 3D real (floating point) substate.
						  );



/*! \brief Apply an elementary process to all the cellular space.
*/
void calApplyElementaryProcess3D(struct CALModel3D* ca3D,	//!< Pointer to the cellular automaton structure.
											   void (* elementary_process)(struct CALModel3D* ca2D, int i, int j, int k) //!< Pointer to a transition function's elementary process.
											   );



/*! \brief The cellular automaton global transition function.
	It applies the transition function to each cell of the cellular space.
	After each elementary process, a global substates update is performed.
*/
void calGlobalTransitionFunction3D(struct CALModel3D* ca3D	//!< Pointer to the cellular automaton structure.
								   );



/*! \brief Updates all the substates registered in CALModel3D::pQb_array,
	CALModel3D::pQi_array and CALModel3D::pQr_array.
	It is called by the global transition function.
*/
void calUpdate3D(struct CALModel3D* ca3D	//!< Pointer to the cellular automaton structure.
				 );



/*! \brief Inits the value of a byte substate in the cell (i, j, k) to value; it updates both the current and next buffers at the position (i, j, k).
*/
void calInit3Db(struct CALModel3D* ca3D,	//!< Pointer to the cellular automaton structure.
				struct CALSubstate3Db* Q,	//!< Pointer to a 3D byte substate.
				int i,						//!< Row coordinate of the cell to be initialized.
				int j,						//!< Column coordinate of the cell to be initialized.
				int k,						//!< Slice coordinate of the cell to be initialized.
				CALbyte value				//!< initializing value for the substate at the cell (i, j, k).
				);

/*! \brief Inits the value of a integer substate in the cell (i, j, k) to value; it updates both the current and next buffers at the position (i, j, k).
*/
void calInit3Di(struct CALModel3D* ca3D,	//!< Pointer to the cellular automaton structure.
				struct CALSubstate3Di* Q,	//!< Pointer to a 3D int substate.
				int i,						//!< Row coordinate of the cell to be initialized.
				int j,						//!< Column coordinate of the cell to be initialized.
				int k,						//!< Slice coordinate of the cell to be initialized.
				CALint value				//!< initializing value for the substate at the cell (i, j, k).
				);

/*! \brief Inits the value of a real (floating point) substate in the cell (i, j, k) to value; it updates both the current and next buffers at the position (i, j, k).
*/
void calInit3Dr(struct CALModel3D* ca3D,	//!< Pointer to the cellular automaton structure.
				struct CALSubstate3Dr* Q,	//!< Pointer to a 3D real (floating point) substate.
				int i,						//!< Row coordinate of the cell to be initialized.
				int j,						//!< Column coordinate of the cell to be initialized.
				int k,						//!< Slice coordinate of the cell to be initialized.
				CALreal value				//!< initializing value for the substate at the cell (i, j, k).
				);



/*! \brief Returns the cell (i, j, k) value of a byte substate.
*/
CALbyte calGet3Db(struct CALModel3D* ca3D,	//!< Pointer to the cellular automaton structure.
				  struct CALSubstate3Db* Q,	//!< Pointer to a 3D byte substate.
				  int i,					//!< Row coordinate of the cell.
				  int j,					//!< Column coordinate of the cell.
				  int k						//!< Slice coordinate of the cell.
				  );

/*! \brief Returns the cell (i, j, k) value of an integer substate.
*/
CALint calGet3Di(struct CALModel3D* ca3D,	//!< Pointer to the cellular automaton structure.
				 struct CALSubstate3Di* Q,	//!< Pointer to a 3D int substate.
				 int i,						//!< Row coordinate of the cell.
				 int j,						//!< Column coordinate of the cell.
				 int k						//!< Slice coordinate of the cell to be initialized.
				 );

/*! \brief Returns the cell (i, j, k) value of the of a real (floating point) substate.
*/
CALreal calGet3Dr(struct CALModel3D* ca3D,	//!< Pointer to the cellular automaton structure.
				  struct CALSubstate3Dr* Q,	//!< Pointer to a 3D real (floating point) substate.
				  int i,					//!< Row coordinate of the cell.
				  int j,					//!< Column coordinate of the cell.
				  int k						//!< Slice coordinate of the cell.
				  );



/*! \brief Returns the n-th neighbor of the cell (i, j, k) value of a byte substate.
*/
CALbyte calGetX3Db(struct CALModel3D* ca3D,	//!< Pointer to the cellular automaton structure.
				   struct CALSubstate3Db* Q,//!< Pointer to a 3D byte substate.
				   int i,					//!< Row coordinate of the central cell.
				   int j,					//!< Column coordinate of the central cell.
				   int k,					//!< Slice coordinate of the central cell.
				   int n					//!< Index of the n-th neighbor.
				   );

/*! \brief Returns the n-th neighbor of the cell (i, j, k) value of an integer substate.
*/
CALint calGetX3Di(struct CALModel3D* ca3D,	//!< Pointer to the cellular automaton structure.
				  struct CALSubstate3Di* Q,	//!< Pointer to a 3D int substate.
				  int i,					//!< Row coordinate of the central cell.
				  int j,					//!< Column coordinate of the central cell.
				  int k,					//!< Slice coordinate of the central cell.
				  int n						//!< Index of the n-th neighbor.
				  );

/*! \brief Returns the n-th neighbor of the cell (i, j, k) value of a real (floating point) substate.
*/
CALreal calGetX3Dr(struct CALModel3D* ca3D,	//!< Pointer to the cellular automaton structure.
				   struct CALSubstate3Dr* Q,//!< Pointer to a 3D real (floating point) substate.
				   int i,					//!< Row coordinate of the central cell.
				   int j,					//!< Column coordinate of the central cell.
				   int k,					//!< Slice coordinate of the central cell.
				   int n					//!< Index of the n-th neighbor.
				   );



/*! \brief Sets the cell (i, j, k) value of a byte substate.
*/
void calSet3Db(struct CALModel3D* ca3D,		//!< Pointer to the cellular automaton structure.
			   struct CALSubstate3Db* Q,	//!< Pointer to a 3D byte substate.
			   int i,						//!< Row coordinate of the cell.
			   int j,						//!< Column coordinate of the cell.
			   int k,						//!< Slice coordinate of the cell.
			   CALbyte value				//!< initializing value.
			   );

/*! \brief Set the cell (i, j, k) value of an integer substate.
*/
void calSet3Di(struct CALModel3D* ca3D,		//!< Pointer to the cellular automaton structure.
			   struct CALSubstate3Di* Q,	//!< Pointer to a 3D int substate.
			   int i,						//!< Row coordinate of the cell.
			   int j,						//!< Column coordinate of the cell.
			   int k,						//!< Slice coordinate of the cell.
			   CALint value					//!< initializing value.
			   );

/*! \brief Set the cell (i, j, k) value of a real (floating point) substate.
*/
void calSet3Dr(struct CALModel3D* ca3D,		//!< Pointer to the cellular automaton structure.
			   struct CALSubstate3Dr* Q,	//!< Pointer to a 3D real (floating point) substate.
			   int i,						//!< Row coordinate of the cell.
			   int j,						//!< Column coordinate of the cell.
			   int k,						//!< Slice coordinate of the cell.
			   CALreal value				//!< initializing value.
			   );



/*! \brief Sets the value of the cell (i, j) of a byte substate of the CURRENT matrix.
	This operation is unsafe since it writes a value directly to the current matrix.
*/
void calSetCurrent3Db(struct CALModel3D* ca3D,	//!< Pointer to the cellular automaton structure.
					  struct CALSubstate3Db* Q,	//!< Pointer to a 3D byte substate.
					  int i,					//!< Row coordinate of the central cell.
					  int j,					//!< Column coordinate of the central cell.
					  int k,					//!< Slice coordinate of the central cell.
					  CALbyte value				//!< initializing value.
					  );

/*! \brief Set the value the  cell (i, j) of an int substate of the CURRENT matrix.
	This operation is unsafe since it writes a value directly to the current matrix.
*/
void calSetCurrent3Di(struct CALModel3D* ca3D,	//!< Pointer to the cellular automaton structure.
					  struct CALSubstate3Di* Q,	//!< Pointer to a 3D int substate.
					  int i,					//!< Row coordinate of the central cell.
					  int j,					//!< Column coordinate of the central cell.
					  int k,					//!< Slice coordinate of the central cell.
					  CALint value				//!< initializing value.
					  );

/*! \brief Set the value the  cell (i, j) of a real (floating point) substate of the CURRENT matrix.
	This operation is unsafe since it writes a value directly to the current matrix.
*/
void calSetCurrent3Dr(struct CALModel3D* ca3D,	//!< Pointer to the cellular automaton structure.
					  struct CALSubstate3Dr* Q,	//!< Pointer to a 3D int substate.
					  int i,					//!< Row coordinate of the central cell.
					  int j,					//!< Column coordinate of the central cell.
					  int k,					//!< Slice coordinate of the central cell.
					  CALreal value				//!< initializing value.
					  );


/*! \brief Finalization function: it releases the memory allocated.
*/
void calFinalize3D(struct CALModel3D* ca3D	//!< Pointer to the cellular automaton structure.
				  );



#endif
