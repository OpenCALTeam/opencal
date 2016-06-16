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

#include <OpenCAL-CPU/calCommon.h>

struct CALIndexesPool* calDefIndexesPool(CALIndices coordinates_dimensions, int number_of_dimensions)
{
    struct CALIndexesPool* indexes_pool = (struct CALIndexesPool*) malloc(sizeof(struct CALIndexesPool));
    indexes_pool->coordinates_dimensions = coordinates_dimensions;
    indexes_pool->number_of_dimensions = number_of_dimensions;

    indexes_pool->cellular_space_dimension = 1;
    int n, i;

    for( n = 0; n < number_of_dimensions; n++ )
        indexes_pool->cellular_space_dimension *= coordinates_dimensions[n];

    indexes_pool->pool = (CALIndices*) malloc(sizeof(CALIndices) * indexes_pool->cellular_space_dimension);

    CALIndices current_cell = (CALIndices) malloc(sizeof(int) * number_of_dimensions);
    CALIndices first_cell = (CALIndices) malloc(sizeof(int) * number_of_dimensions);
    for( n = 0; n < number_of_dimensions; n++ )
    {
        current_cell[n] = 0;
        first_cell[n] = 0;
    }

    int current_dimension = number_of_dimensions - 1;
    n = 0;
    indexes_pool->pool[n] = first_cell;

    for(n = 1; n < indexes_pool->cellular_space_dimension; n++)
    {
        current_cell[current_dimension]++;

        while(current_cell[current_dimension] == coordinates_dimensions[current_dimension])
        {
            current_cell[current_dimension--] = 0;
            if( current_dimension < 0 )
                break;

            current_cell[current_dimension]++;
        }

        current_dimension = number_of_dimensions - 1;

        CALIndices cell_to_add = (CALIndices) malloc(sizeof(int) * number_of_dimensions);

        for( i = 0; i < number_of_dimensions; i++ )
            cell_to_add[i] = current_cell[i];

        indexes_pool->pool[n] = cell_to_add;
    }

    free(current_cell);
    return indexes_pool;

}

int getLinearIndex(CALIndices indices, CALIndices coordinates_dimensions, int cellular_space_dimension)
{

    int c = 0;
    uint multiplier = 1;
    uint n;

    for (uint i = 0; i < cellular_space_dimension; i++)
    {
        if (i == 1) n = 0;
        else if (i == 0) n = 1;
        else n = i;
        c          += indices[n] * multiplier;
        multiplier *= coordinates_dimensions[n];
    }
    return c;
}
