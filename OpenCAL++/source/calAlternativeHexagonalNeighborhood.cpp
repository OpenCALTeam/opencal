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

#include <OpenCAL++/calAlternativeHexagonalNeighborhood.h>


void CALAlternativeHexagonalNeighborhood :: defineNeighborhood (CALModel* calModel)
{
    int n = calModel->getDimension();
    assert (n==2);

    /*
        cell orientation

            /\
           /  \
           |  |
           \  /
            \/
    */
    /*
         2 | 1 |
        ---|---|---
         3 | 0 | 6		if (i%2 == 0), i.e. even rows
        ---|---|---
         4 | 5 |
    */

    int vonNeumannNeighborhoodIndexes [7][2] = {{0,0},
                                                {-1,0},
                                                {-1,-1},
                                                {0,-1},
                                                {1,-1},
                                                {1,0},
                                                {0,1}};

    for (int i =0; i<7; i++)
    {
       calModel-> addNeighbor(vonNeumannNeighborhoodIndexes[i]);
    }

// TODO controllare gli altri indici

//    calAddNeighbor2D(ca2D,   0,   0);
//    calAddNeighbor2D(ca2D, - 1,   0);
//    calAddNeighbor2D(ca2D, - 1, - 1);
//    calAddNeighbor2D(ca2D,   0, - 1);
//    calAddNeighbor2D(ca2D, + 1, - 1);
//    calAddNeighbor2D(ca2D, + 1,   0);
//    calAddNeighbor2D(ca2D,   0, + 1);

    /*
           | 2 | 1
        ---|---|---
         3 | 0 | 6		if (i%2 == 1), i.e. odd rows
        ---|---|---
           | 4 | 5
    */

//    calAddNeighbor2D(ca2D,   0,   0);
//    calAddNeighbor2D(ca2D, - 1, + 1);
//    calAddNeighbor2D(ca2D, - 1,   0);
//    calAddNeighbor2D(ca2D,   0, - 1);
//    calAddNeighbor2D(ca2D, + 1,   0);
//    calAddNeighbor2D(ca2D, + 1, + 1);
//    calAddNeighbor2D(ca2D,   0, + 1);

//   calModel->sizeof_X = 7;


}
