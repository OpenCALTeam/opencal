/*
 * Copyright (c) 2016 OpenCALTeam (https://github.com/OpenCALTeam),
 * University of Calabria, Italy.
 *
 * This file is part of OpenCAL (Open Computing Abstraction Layer).
 *
 * OpenCAL is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3 of
 * the License, or (at your option) any later version.
 *
 * OpenCAL is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with OpenCAL. If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef calNeighborhoodPool_h
#define calNeighborhoodPool_h

#include <cassert>
#include <stdlib.h>
#include <iostream>
#include <OpenCAL++/calCommon.h>

class CALNeighborPool
{
protected:

    static CALNeighborPool* instance;
private:
    int ** neighborPool = NULL;
    int size;
    int * coordinates = NULL;
    size_t dimension;
    int size_of_X;
    enum calCommon::CALSpaceBoundaryCondition CAL_TOROIDALITY;

    CALNeighborPool (int size, int* coordinates, size_t dimension, enum calCommon::CALSpaceBoundaryCondition CAL_TOROIDALITY);

    ~CALNeighborPool();
public:
    static void destroy();
    static void init (int size, int* coordinates, size_t dimension, enum calCommon::CALSpaceBoundaryCondition CAL_TOROIDALITY);

    static CALNeighborPool* getInstance ();
    inline int getNeighborN (int linearIndex, int n)
    {
        //TODO manage hexagonalNeighborhood
        assert(linearIndex<instance->size);
        assert (n < instance->size_of_X);

        return instance->neighborPool[linearIndex][n];

    }

   void addNeighbor (int* cellPattern);

};





#endif
