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


#ifndef OPENCAL_ALL_CALCOMMON_H
#define OPENCAL_ALL_CALCOMMON_H

#include "functional_utilities.h"
#include <array>
#include <OpenCAL++/calIndexesPool.h>
namespace opencal {

    namespace calCommon {

        typedef unsigned int uint;

#define CAL_FALSE false        //!< Boolean alias for false
#define CAL_TRUE  true        //!< Boolean alias for true

        typedef bool CALbyte;	//!< Redefinition of the type char.


        /*!	\brief Enumeration used for cellular space toroidality setting.
        */
        enum CALSpaceBoundaryCondition {
            CAL_SPACE_FLAT = 0,            //!< Enumerator used for setting non-toroidal cellular space.
            CAL_SPACE_TOROIDAL            //!< Enumerator used for setting toroidal cellular space.
        };


        /*!	\brief Enumeration used for substate updating settings.
        */
        enum CALUpdateMode {
            CAL_UPDATE_EXPLICIT = 0,    //!< Enumerator used for specifying that explicit calls to calUpdateSubstate2D* and calUpdate2D are needed.
            CAL_UPDATE_IMPLICIT            //!< Enumerator used for specifying that explicit calls to calUpdateSubstate2D* and calUpdate2D are NOT needed.
        };


        /*!	\brief Enumeration used for optimization strategies. */
        enum CALOptimization {
            CAL_NO_OPT = 0,                //!< Enumerator used for specifying no optimizations.
            CAL_OPT_ACTIVE_CELLS        //!< Enumerator used for specifying the active cells optimization.
        };


        /*! \brief Macro recomputing the out of bound neighbourhood indexes in case of toroidal cellular space.
        */


        inline int getToroidalX (int index, uint size)
        {
          return (index)<0?((size)+(index)):( (index)>((size)-1)?((index)-(size)):(index) );
        }
//        #define calGetToroidalX(index, size) (   (index)<0?((size)+(index)):( (index)>((size)-1)?((index)-(size)):(index) )   )

        /*! Constant used to set the run final step to 0, correspondig to a loop condition.
        In this case, a stop condition should be defined.
        */
        #define CAL_RUN_LOOP 0


        /*! \brief Enumeration defining global reduction operations.
        Enumeration defining global reduction operations inside the
        steering function.
         */
        enum REDUCTION_OPERATION {
            REDUCTION_NONE = 0,
            REDUCTION_MAX,
            REDUCTION_MIN,
            REDUCTION_SUM,
            REDUCTION_PROD,
            REDUCTION_LOGICAL_AND,
            REDUCTION_BINARY_AND,
            REDUCTION_LOGICAL_OR,
            REDUCTION_BINARY_OR,
            REDUCTION_LOGICAL_XOR,
            REDUCTION_BINARY_XOR
        };


        /*! \brief Multiply cordinates array's element from startingIndex to dimension.
        */

        template<uint DIMENSION , class COORDINATE_TYPE>
        inline unsigned int multiplier ( std::array<COORDINATE_TYPE,DIMENSION>& coords, uint startIdx, uint endIdx = DIMENSION) {
                auto mult = [](const COORDINATE_TYPE& a, unsigned int acc){return acc*a;};
                return opencal::fold(coords.begin()+startIdx, coords.begin()+endIdx, 1 , mult );
        }


        /*! \brief Return the linearIndex of the cell with coordinates indexes.
     */
        template<uint DIMENSION , class COORDINATE_TYPE>
        inline unsigned int cellLinearIndex (const std::array<COORDINATE_TYPE,DIMENSION>& indices,
                                             const std::array<COORDINATE_TYPE,DIMENSION>& coords) {
            uint c = 0;
            uint multiplier = 1;
            uint n;
            for (uint i = 0 ; i < DIMENSION ; i++)
            {
                if (i == 1)
                    n = 0;
                else if (i == 0)
                    n = 1;
                else
                    n = i;
                c += indices[n] * multiplier;
                multiplier *= coords[n];
            }

            return c;
        }



//        /*! \brief Return multidimensional indexes of a certain cell.
//        */
//        template<uint DIMENSION, typename COORDINATE_TYPE>
//        inline std::array<COORDINATE_TYPE,DIMENSION>& cellMultidimensionalIndices (int index)
//        {
//            return CALIndexesPool<DIMENSION , COORDINATE_TYPE>:: getMultidimensionalIndexes(index);
//        }


        /*! \brief Return linear index of n^th neighbour of a certain cell.
     */
        template<uint DIMENSION, typename COORDINATE_TYPE>
        inline int getNeighborNLinear (int* indexes, int* neighbor,std::array<COORDINATE_TYPE,DIMENSION>& coordinates, enum CALSpaceBoundaryCondition CAL_TOROIDALITY)
        {
            int i;
            int c = 0;
            int t = multiplier (coordinates, 0, DIMENSION);
            if (CAL_TOROIDALITY == CAL_SPACE_FLAT)
                for (i = 0; i < DIMENSION; ++i)
                {
                    t= t/coordinates[i];
                    c+=(indexes[i] + neighbor[i])*t;
                }
            else
            {
                for (i=0; i< DIMENSION; i++)
                {
                    t= t/coordinates[i];
                    c+=(calGetToroidalX(indexes[i] + neighbor[i], coordinates[i]))*t;

                }

            }
            return c;
        }


    }//namespace calCommon

}//namespace opencal


#endif //OPENCAL_ALL_CALCOMMON_H
