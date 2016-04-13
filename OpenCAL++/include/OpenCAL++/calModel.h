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

#ifndef CALMODEL_H_
#define CALMODEL_H_

#include<memory>
#include<array>

#include <OpenCAL++/calCommon.h>
#include <OpenCAL++/calBuffer.h>
#include<OpenCAL++/calActiveCells.h>
#include<OpenCAL++/calSubstate.h>



namespace opencal {


    template<uint DIMENSION, class NEIGHBORHOOD, typename COORDINATE_TYPE = int>
    class CALModel {


        typedef NEIGHBORHOOD* NEIGHBORHOOD_pointer;
        typedef NEIGHBORHOOD& NEIGHBORHOOD_reference;


        typedef SUBSTATE SUBSTATE_reference;
        typedef SUBSTATE SUBSTATE_pointer;

    public:

        //typedefs here

        /******************************************************************************
                        DEFINITIONS OF FUNCTIONS PROTOTYPES
        *******************************************************************************/
        /*! \brief Constructor of the object CALModel, sets and inizializes its records; it defines the cellular automaton object.
       */
        CALModel(std::array<COORDINATE_TYPE, DIMENSION>& _coordinates, //!< Dimensions  of cellular space.
                 NEIGHBORHOOD_pointer _calNeighborhood, //!< Class that identifies the type of neighbourhood relation to be used.
                 enum opencal::calCommon::CALSpaceBoundaryCondition _CAL_TOROIDALITY, //!< Enumerator that identifies the type of cellular space: toroidal or non-toroidal.
                 enum opencal::calCommon::CALOptimization _CAL_OPTIMIZATION //!< Enumerator used for specifying the active cells optimization or no optimization.
        ): coordinates(_coordinates) , CAL_TOROIDALITY(_CAL_TOROIDALITY) , CAL_OPTIMIZATION(_CAL_OPTIMIZATION)
        {

            this->size = opencal::calCommon::multiplier<DIMENSION,uint>(coordinates,0);

            //initialize indexpool and neighbor pool


            if (this->CAL_OPTIMIZATION == calCommon::CAL_OPT_ACTIVE_CELLS) {
                CALBuffer<bool , DIMENSION , COORDINATE_TYPE>* flags = new CALBuffer<bool , DIMENSION , COORDINATE_TYPE> (this->coordinates);
                flags->setBuffer(false);
                this->activeCells = new opencal::CALActiveCells< DIMENSION , COORDINATE_TYPE>(flags, 0);
            }
            else
                this->activeCells = NULL;
        }

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
        template <class PAYLOAD>
        CALSubstate<PAYLOAD, DIMENSION, COORDINATE_TYPE>* addSubstate();

        /*! \brief Creates a new single-layer substate and returns a pointer to it.
            Note that sinlgle-layer substates are not added to CALModel::pQ_arrays because
            they do not need to be updated.
        */
        template <class PAYLOAD>
        CALSubstate<PAYLOAD, DIMENSION, COORDINATE_TYPE>* addSingleLayerSubstate();



        uint getDimension() {
            return DIMENSION;
        }

    private:
        enum opencal::calCommon::CALSpaceBoundaryCondition CAL_TOROIDALITY;    //!< Type of cellular space: toroidal or non-toroidal.


        std::array <COORDINATE_TYPE, DIMENSION> coordinates;
        uint size;

        int sizeof_X;                //!< Number of cells belonging to the neighbourhood. Note that predefined neighbourhoods include the central cell.
        NEIGHBORHOOD_pointer X_id;    //!< Class that define the Neighbourhood relation.


        opencal::CALActiveCells<DIMENSION,COORDINATE_TYPE>* activeCells;			//!< Computational Active cells object. if activecells==NULL no optimization is applied.
        enum opencal::calCommon::CALOptimization CAL_OPTIMIZATION;    //!< Type of optimization used. It can be CAL_NO_OPT or CAL_OPT_ACTIVE_CELLS.


       // CALCallbackFunc* elementary_processes; //!< Array of transition function's elementary processes callback functions. Note that a substates' update must be performed after each elementary process has been applied to each cell of the cellular space.
        //int num_of_elementary_processes; //!< Number of function pointers to the transition functions's elementary processes callbacks.
    };


} //namespace opencal
#endif //CALMODEL_H_
