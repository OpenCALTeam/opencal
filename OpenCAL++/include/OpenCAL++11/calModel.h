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

#ifndef calModel_h
#define calModel_h

#include <OpenCAL++11/calCommon.h>
#include <OpenCAL++11/calNeighborhood.h>
#include <OpenCAL++11/calSubstate.h>
#include <OpenCAL++11/calActiveCells.h>
#include <stdlib.h>

#include <OpenCAL++11/calElementaryProcessFunctor.h>
#include <OpenCAL++11/calNeighborPool.h>




typedef  CALElementaryProcessFunctor* CALCallbackFunc;

/*! \brief Class defining the ND cellular automaton.
*/
class CALModel{
private:
    int * coordinates; //!
    size_t dimension;
    size_t size;
    enum calCommon:: CALSpaceBoundaryCondition CAL_TOROIDALITY;	//!< Type of cellular space: toroidal or non-toroidal.

    enum calCommon:: CALOptimization OPTIMIZATION;	//!< Type of optimization used. It can be CAL_NO_OPT or CAL_OPT_ACTIVE_CELLS.
    CALActiveCells* activeCells;			//!< Computational Active cells object. if activecells==NULL no optimization is applied.

               
    int sizeof_X;				//!< Number of cells belonging to the neighbourhood. Note that predefined neighbourhoods include the central cell.
    CALNeighborhood* X_id;		//!< Class that define the Neighbourhood relation.



    CALSubstateWrapper ** pQ_arrays; //!< Substates array.
    int sizeof_pQ_arrays;           //!< Number of substates.

    CALCallbackFunc* elementary_processes; //!< Array of transition function's elementary processes callback functions. Note that a substates' update must be performed after each elementary process has been applied to each cell of the cellular space.
    int num_of_elementary_processes; //!< Number of function pointers to the transition functions's elementary processes callbacks.


public:
/******************************************************************************
					DEFINITIONS OF FUNCTIONS PROTOTYPES

*******************************************************************************/

    /*! \brief Constructor of the object CALModel, sets and inizializes its records; it defines the cellular automaton object.
    */
    CALModel (int* coordinates, //!< Dimensions  of cellular space.
               size_t dimension, //!< Number of dimensions cellular space.
               CALNeighborhood* calNeighborhood, //!< Class that identifies the type of neighbourhood relation to be used.
               enum calCommon:: CALSpaceBoundaryCondition CAL_TOROIDALITY, //!< Enumerator that identifies the type of cellular space: toroidal or non-toroidal.
               enum calCommon:: CALOptimization CAL_OPTIMIZATION //!< Enumerator used for specifying the active cells optimization or no optimization.
               );

    ~ CALModel ();
    
    /*! \brief Sets a certain cell of the matrix flags to true and increments the
        couter sizeof_active_flags.
    */
    void addActiveCell(int * indexes);
    void addActiveCell(int linearIndex);

    /*! \brief Sets a specific cell of the matrix flags to false and decrements the
        couter sizeof_active_flags.
    */
    void removeActiveCell(int * indexes);
    void removeActiveCell(int linearIndex);

    /*! \brief Perform the update of CALActiveCells object.
    */
    void updateActiveCells();

    /*! \brief Adds a neighbour to CALNeighbourPool.
    */
    void  addNeighbor(int* indexes);

    /*! \brief Adds a neighbours to CALNeighbourPool.
    */
    void  addNeighbors(int** indexes,
                              size_t dimension
                              );

    /*! \brief Creates and adds a new substate to CALModel::pQ_arrays and return a pointer to it.
    */
    template <class T>
    CALSubstate<T>* addSubstate();

    /*! \brief Creates a new single-layer substate and returns a pointer to it.
        Note that sinlgle-layer substates are not added to CALModel::pQ_arrays because
        they do not need to be updated.
    */
    template <class T>
    CALSubstate<T>* addSingleLayerSubstate();


    /*! \brief Adds a transition function's elementary process to the CALModel::elementary_processes array of callbacks function.
        Note that the function globalTransitionFunction calls a substates' update after each elementary process.
    */
    void addElementaryProcess(CALCallbackFunc elementary_process 
                                                 );

    /*! \brief Initializes a substate to a constant value; both the current and next (if not single layer substate) matrices are initialized.
    */
    template <class T>
    void initSubstate(struct CALSubstate<T>*& Q,	
                            T value				
                            );

    /*! \brief Initializes a the next buffer of a substate to a constant value.
    */
    template <class T>
    void initSubstateNext(CALSubstate<T>*& Q,	
                                T value	
                                );

    /*! \brief Apply an elementary process to all the cellular space.
    */
    void applyElementaryProcess(CALCallbackFunc elementary_process 
                                                   );
    /*! \brief The cellular automaton global transition function.
        It applies the transition function to each cell of the cellular space.
        After each elementary process, a global substates update is performed.
    */
    void globalTransitionFunction();


    /*! \brief Updates all the substates registered in CALModel::pQ_array.
        It is called by the global transition function.
    */
    void update();

    /*! \brief Inits the value of a substate in a certain cell to value; it updates both the current and next matrices at the specified position.
    */
    template <class T>
    void init(struct CALSubstate<T>*& Q,	
                    int* indexes,				
                    T value		
                    );


    size_t getDimension ();

    int getSize ();

    int* getCoordinates ();

    int getNeighborhoodSize ();

    void setNeighborhoodSize (int sizeof_X);

    CALActiveCells* getActiveCells ();


/******************************************************************************
                            PRIVATE FUNCIONS

*******************************************************************************/
private:
   
    template <class T>
    calCommon :: CALbyte allocSubstate(CALSubstate<T>*& Q	
                                );

};

/******************************************************************************
                            PRIVATE FUNCIONS

*******************************************************************************/



template <class T>
calCommon :: CALbyte CALModel::allocSubstate(CALSubstate<T>*& Q	
                            )
{
    CALBuffer<T>* current = new CALBuffer<T> (this->coordinates, this->dimension);
    CALBuffer<T>* next = new CALBuffer<T> (this->coordinates, this->dimension);
    Q = new CALSubstate<T> (current, next);

    if (!Q->getCurrent() || !Q->getNext()){
        return CAL_FALSE;
    }

    return CAL_TRUE;
}


/******************************************************************************
                            PUBLIC FUNCIONS

*******************************************************************************/


template <class T>
CALSubstate<T> * CALModel::addSubstate(){
    CALSubstateWrapper** pQ_array_tmp = this->pQ_arrays;
    CALSubstateWrapper** pQ_array_new;
    CALSubstate<T>* Q;
    int i;


    pQ_array_new = new CALSubstateWrapper* [this->sizeof_pQ_arrays + 1];


    for (i = 0; i < this->sizeof_pQ_arrays; i++)
    {
        pQ_array_new[i] = pQ_array_tmp[i];
    }

    if (!allocSubstate<T>(Q))
    {
        return NULL;
    }

    pQ_array_new[this->sizeof_pQ_arrays] = (CALSubstateWrapper*) Q;

    this->pQ_arrays = pQ_array_new;


    delete [] pQ_array_tmp;

    this->sizeof_pQ_arrays++;
    return Q;
}

template <class T>
CALSubstate<T>* CALModel::addSingleLayerSubstate(){

    CALSubstate<T>* Q;
    CALBuffer<T> * current = new CALBuffer<T> (this->coordinates, this->dimension);
    Q = new CALSubstate <T> (current, NULL);

    return Q;
}

template <class T>
void CALModel::initSubstate(CALSubstate<T>*& Q, T value) {
    if (this->activeCells)
    {
        Q->setActiveCellsBufferCurrent(this->activeCells, value);
        if(Q->getNext())
            Q->setActiveCellsBufferNext(this->activeCells, value);

    }
    else
    {
        Q->setCurrentBuffer(value);
        if(Q->getNext())
            Q->setNextBuffer(value);

    }
}

template <class T>
void CALModel:: initSubstateNext(CALSubstate<T>*& Q, T value) {
    if (this->activeCells)
        Q->setActiveCellsBuffer(this->activeCells, value);
    else
        Q->setNextBuffer(value);
}



template <class T>
void CALModel::init(CALSubstate<T>*& Q, int* indexes, T value) {

    int linearIndex = calCommon::cellLinearIndex(indexes, this->coordinates, this->dimension);
    (*Q->getCurrent())[linearIndex] = value;
    (*Q->getNext())[linearIndex] = value;
}


#endif
