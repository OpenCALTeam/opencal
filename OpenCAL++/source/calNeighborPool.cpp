/*
 * Copyright (c) 2016 OpenCALTeam (https://github.com/OpenCALTeam),
 * Telesio Research Group,
 * Department of Mathematics and Computer Science,
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

#include <OpenCAL++/calNeighborPool.h>



CALNeighborPool :: CALNeighborPool (int size, int* coordinates, size_t dimension, enum calCommon::CALSpaceBoundaryCondition CAL_TOROIDALITY)
{
    this->neighborPool = new int* [size];
    this->coordinates = coordinates;
    this->dimension = dimension;
    this->size = size;
    this->size_of_X = 0;
    this->CAL_TOROIDALITY = CAL_TOROIDALITY;

}

 void CALNeighborPool:: init (int size, int* coordinates, size_t dimension, enum calCommon::CALSpaceBoundaryCondition CAL_TOROIDALITY)
{
    if (instance == nullptr)
        instance = new CALNeighborPool (size,coordinates,dimension,CAL_TOROIDALITY);

}
CALNeighborPool* CALNeighborPool :: instance = nullptr;

 void CALNeighborPool:: destroy()
{
    delete instance;
}

CALNeighborPool* CALNeighborPool:: getInstance ()
{
    return instance;

}

CALNeighborPool:: ~CALNeighborPool ()
{
    for(int i = 0; i < this->size; i++)
    {
        delete[] this->neighborPool[i];
    }
    delete [] this->neighborPool;
}




void CALNeighborPool:: addNeighbor (int* cellPattern)
{
    for (int i = 0; i < instance->size; ++i)
    {
       int* neighborsTmp = instance->neighborPool[i];
       int* neighborsNew = new int [instance->size_of_X + 1];
       for (int k = 0; k < instance->size_of_X; ++k)
       {
            neighborsNew [k] = instance->neighborPool[i][k];
       }

       int* multidimensionalIndex = calCommon:: cellMultidimensionalIndexes(i);
       int toAdd = calCommon:: getNeighborNLinear(multidimensionalIndex,cellPattern,instance->coordinates, instance->dimension,instance->CAL_TOROIDALITY );
       neighborsNew[instance->size_of_X]= toAdd;
       instance->neighborPool[i]= neighborsNew;

       delete [] neighborsTmp;
    }
    instance->size_of_X++;

}
