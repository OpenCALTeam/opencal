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
#include <iostream>
#include <iterator>
#include <algorithm>
#include<functional>
#include <sstream>
#include <iomanip>

namespace opencal {

    template <class T>
    std::string tostring_fn_(const T& s, const int n = 6) {
      std::ostringstream out;
      out << std::setprecision(n + 1) << s;
      return out.str();
    }

    template <class T>
    const auto tostring_fn(const int n = 6) {
      using namespace std::placeholders;
      auto f = std::bind(tostring_fn_<T>, _1, n);
      return f;
    }

    // string -> signed integral
    inline int stoi(const std::string& s) { return std::stoi(s); }

    inline long int stol(const std::string& s) { return std::stol(s); }

    inline long long int stoll(const std::string& s) { return std::stoll(s); }

    // string -> unsigned integral
    inline unsigned long int stoul(const std::string& s) { return std::stol(s); }

    inline unsigned long long int stoull(const std::string& s) {
      return std::stoll(s);
    }

    // string -> floating
    inline float stof(const std::string& s) { return std::stof(s); }

    inline double stod(const std::string& s) { return std::stol(s); }

    inline long double stold(const std::string& s) { return std::stoll(s); }

    // string to boolean (aka int short)
    inline bool stob(const std::string& s) { return (bool)std::stoi(s); }



    namespace calCommon {
        typedef unsigned int uint;

        #define CAL_FALSE false // !< Boolean alias for false
        #define CAL_TRUE  true  // !< Boolean alias for true

        typedef bool CALbyte;   // !< Redefinition of the type char.


        /*!	\brief Enumeration used for cellular space toroidality setting.
         */
        enum CALSpaceBoundaryCondition {
          CAL_SPACE_FLAT = 0, // !< Enumerator used for setting non-toroidal cellular
                              // space.
          CAL_SPACE_TOROIDAL  // !< Enumerator used for setting toroidal cellular space.
        };


        /*!	\brief Enumeration used for substate updating settings.
         */
        enum CALUpdateMode {
          CAL_UPDATE_EXPLICIT = 0, // !< Enumerator used for specifying that explicit
                                   // calls to calUpdateSubstate2D* and calUpdate2D are
                                   // needed.
          CAL_UPDATE_IMPLICIT      // !< Enumerator used for specifying that explicit
                                   // calls to calUpdateSubstate2D* and calUpdate2D are
                                   // NOT needed.
        };


        /*!	\brief Enumeration used for optimization strategies. */
        enum CALOptimization {
          CAL_NO_OPT = 0,      // !< Enumerator used for specifying no optimizations.
          CAL_OPT_ACTIVE_CELLS // !< Enumerator used for specifying the active cells
                               // optimization.
        };


        /*! \brief Macro recomputing the out of bound neighbourhood indexes in case of
           toroidal cellular space.
         */
        inline int getToroidalX(const int index, const uint size)
        {
          return (index) <
                 0 ? ((size) + (index)) : ((index) >
                                           ((size) - 1) ? ((index) - (size)) : (index));
        }

        //        #define calGetToroidalX(index, size) (   (index)<0?((size)+(index)):(
        // (index)>((size)-1)?((index)-(size)):(index) )   )

        /*! Constant used to set the run final step to 0, correspondig to a loop
           condition.
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

        enum SUBSTATE_OPT { NO_OPT, OPT };


        /*! \brief Multiply cordinates array's element from startingIndex to dimension.
         */

        template<uint DIMENSION, class COORDINATE_TYPE>
        inline unsigned int multiplier(std::array<COORDINATE_TYPE, DIMENSION>& coords,
                                       uint startIdx,
                                       uint endIdx = DIMENSION) {
          auto mult = [](const COORDINATE_TYPE& a, unsigned int acc) { return acc * a; };

          return opencal::fold(coords.begin() + startIdx, coords.begin() + endIdx, 1,
                               mult);
        }

        using namespace std;
        template <class T, std::size_t N>
        std::ostream& operator<<(std::ostream& o, const std::array<T, N>& arr)
        {
            copy(arr.cbegin(), arr.cend(), ostream_iterator<T>(o, " "));
            return o;
        }
        /*! \brief Return the linearIndex of the cell with coordinates indexes.
         */
         template<uint DIMENSION, class COORDINATE_TYPE>
         inline unsigned int cellLinearIndex(const std::array<COORDINATE_TYPE,
                                                              DIMENSION>& indices,
                                             const std::array<COORDINATE_TYPE,
                                                              DIMENSION>& coords) {
           uint c          = 0;
           uint multiplier = 1;
           uint n;

           for (uint i = 0; i < DIMENSION; i++)
           {
             if (i == 1) n = 0;
             else if (i == 0) n = 1;
             else n = i;
             c          += indices[n] * multiplier;
             multiplier *= coords[n];
           }

           return c;
         }


        template<>
        inline unsigned int cellLinearIndex<2,uint>(const std::array<uint,
                                                             2>& indices,
                                            const std::array<uint,
                                                             2>& coords) {
           return indices[0]*coords[1]+indices[1];
        }

        template<>
        inline unsigned int cellLinearIndex<3,uint>(const std::array<uint,
                                                             3>& indices,
                                            const std::array<uint,
                                                             3>& coords) {
           return indices[0]*coords[1]+indices[1] + indices[2]*(coords[0]*coords[1]);
        }

        /*! \brief Return multidimensional indexes of a certain cell.
         */
        template<uint DIMENSION, typename COORDINATE_TYPE>
        inline std::array<COORDINATE_TYPE, DIMENSION>& cellMultidimensionalIndices(
          int index)
        {
          return CALIndexesPool<DIMENSION, COORDINATE_TYPE>::getMultidimensionalIndexes(
            index);
        }

        /*! \brief Return linear index of n^th neighbour of a certain cell.
         */
        template<uint DIMENSION, typename COORDINATE_TYPE>
        inline int getNeighborNLinear(int *indexes,
                                      int *neighbor,
                                      const std::array<COORDINATE_TYPE, DIMENSION>& coordinates,
                                      enum CALSpaceBoundaryCondition CAL_TOROIDALITY)
        {
          int i;
          int c = 0;
          int t = multiplier(coordinates, 0, DIMENSION);

          if (CAL_TOROIDALITY == CAL_SPACE_FLAT)
            for (i = 0; i < DIMENSION; ++i)
            {
              t  = t / coordinates[i];
              c += (indexes[i] + neighbor[i]) * t;
            }
          else
          {
            for (i = 0; i < DIMENSION; i++)
            {
              t  = t / coordinates[i];
              c += (calGetToroidalX(indexes[i] + neighbor[i], coordinates[i])) * t;
            }
          }
          return c;
        }

        // compile time pow operation
        template<class T>
        inline constexpr T pow_ct(const T base, unsigned const exponent)
        {
          // (parentheses not required in next line)
          return (exponent == 0)     ? 1 :
                 (exponent % 2 == 0) ? pow_ct(base, exponent / 2) * pow_ct(base,
                                                                           exponent / 2) :
                 base *pow_ct(base,
                              (exponent - 1) / 2) * pow_ct(base, (exponent - 1) / 2);
        }
    } // namespace calCommon
} // namespace opencal


#endif // OPENCAL_ALL_CALCOMMON_H
