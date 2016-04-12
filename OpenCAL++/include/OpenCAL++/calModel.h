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




namespace opencal {


    template<uint DIMENSION, class NEIGHBORHOOD, typename COORDINATE_TYPE = int>
    class CALModel {


        typedef NEIGHBORHOOD *NEIGHBORHOOD_pointer;
        typedef NEIGHBORHOOD &NEIGHBORHOOD_reference;

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
        ): coordinates(_coordinates) , CAL_TOROIDALITY(_CAL_TOROIDALITY)
        {

        this->size = opencal::calCommon::multiplier<DIMENSION,uint>(coordinates,0);
        }

        uint getDimension() {
            return DIMENSION;
        }

    private:
        enum opencal::calCommon::CALSpaceBoundaryCondition CAL_TOROIDALITY;    //!< Type of cellular space: toroidal or non-toroidal.
        enum opencal::calCommon::CALOptimization OPTIMIZATION;    //!< Type of optimization used. It can be CAL_NO_OPT or CAL_OPT_ACTIVE_CELLS.

        std::array <COORDINATE_TYPE, DIMENSION> coordinates;
        uint size;

        int sizeof_X;                //!< Number of cells belonging to the neighbourhood. Note that predefined neighbourhoods include the central cell.
        NEIGHBORHOOD_pointer X_id;    //!< Class that define the Neighbourhood relation.

    };


} //namespace opencal
#endif //CALMODEL_H_
