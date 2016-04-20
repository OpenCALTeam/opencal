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

#include <OpenCAL++/calVonNeumannNeighborhood.h>


void CALVonNeumannNeighborhood :: defineNeighborhood (CALModel* calModel)
{
    int n = calModel->getDimension();
    assert (n==2 || n==3);

    if (n==2)
    {
    /*
           | 1 |
        ---|---|---
         2 | 0 | 3
        ---|---|---
           | 4 |
   */
        int vonNeumannNeighborhoodIndexes [5][2] = {{0,0},
                                                    {-1,0},
                                                    {0,-1},
                                                    {0,1},
                                                    {1,0}};

        for (int i =0; i<5; i++)
        {
            calModel-> addNeighbor(vonNeumannNeighborhoodIndexes[i]);
        }


    }
    else
    {
    /*
         slice -1       slice 0       slice 1
         (sopra)					  (sotto)

           |   |         | 1 |         |   |
        ---|---|---   ---|---|---   ---|---|---
           | 5 |       2 | 0 | 3       | 6 |
        ---|---|---   ---|---|---   ---|---|---
           |   |         | 4 |         |   |
   */
        int vonNeumannNeighborhoodIndexes [7][3] = {{0,0,0},
                                                    {-1,0,0},
                                                    {0,-1,0},
                                                    {0,1,0},
                                                    {1,0,0},
                                                    {0,0,-1},
                                                    {0,0,1}};

        for (int i =0; i<7; i++)
        {
           calModel-> addNeighbor(vonNeumannNeighborhoodIndexes[i]);
        }
    }


}
