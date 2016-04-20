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

#include <OpenCAL++/calIndexesPool.h>


CALIndexesPool :: CALIndexesPool (int size, int* coordinates, size_t dimension)
{
    this->indexesPool = new int* [size];
    this->coordinates = coordinates;
    this->dimension = dimension;
    this->size = size;
}

CALIndexesPool :: ~CALIndexesPool()
{
    for(int i = 0; i < this->size; i++)
    {
        delete[] this->indexesPool[i];
    }
    delete [] this->indexesPool;
}


CALIndexesPool* CALIndexesPool :: instance = nullptr;



void CALIndexesPool :: init (int size, int* coordinates, size_t dimension)
{
    if (instance == nullptr)
    {
        instance = new CALIndexesPool (size, coordinates, dimension);
        instance->initIndexes();
    }
}
int* CALIndexesPool:: getMultidimensionalIndexes (int linearIndex)
{
    assert (instance!= nullptr);
    assert(linearIndex<instance->size);
    return instance->indexesPool[linearIndex];

}

void CALIndexesPool::initIndexes ()
{
    for (int i = 0; i < this->size; ++i)
    {
        int n, k;
        int linearIndex = i;
        int* v = new int [this->dimension];
        int t =this-> size;
        for (n=this->dimension-1; n>=0; n--)
        {
            if (n ==1)
                k=0;
            else if (n==0)
                k=1;
            else
                k=n;

            t= (int)t/this->coordinates[k];
            v[k] = (int) linearIndex/t;
            linearIndex = linearIndex%t;
        }
        this->indexesPool[i] = v;

    }

}

void CALIndexesPool:: destroy ()
{
    delete instance;
}
