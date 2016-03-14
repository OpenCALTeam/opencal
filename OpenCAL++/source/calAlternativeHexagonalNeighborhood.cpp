
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
