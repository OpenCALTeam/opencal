<<<<<<< HEAD
<<<<<<< HEAD
#include <OpenCAL++/calHexagonalNeighborhood.h>
=======
#include <OpenCAL++11/calHexagonalNeighborhood.h>
>>>>>>> e44630b317eeb506eac14bb3076f71487fe5ed2d
=======
#include <OpenCAL++11/calHexagonalNeighborhood.h>
>>>>>>> e44630b317eeb506eac14bb3076f71487fe5ed2d

void CALHexagonalNeighborhood :: defineNeighborhood (CALModel* calModel)
{
    int n = calModel->getDimension();
    assert (n==2);
    /*
        cell orientation
             __
            /  \
            \__/
    */
    /*
         3 | 2 | 1
        ---|---|---
         4 | 0 | 6		if (j%2 == 0), i.e. even columns
        ---|---|---
           | 5 |
    */

    int vonNeumannNeighborhoodIndexes [7][2] = {{0,0},
                                                {-1,1},
                                                {-1,0},
                                                {-1,-1},
                                                {0,-1},
                                                {1,0},
                                                {0,1}};

    for (int i =0; i<7; i++)
    {
       calModel-> addNeighbor(vonNeumannNeighborhoodIndexes[i]);
    }

//    calModel->setNeighborhoodSize(7);
    // TODO controllare gli altri indici


//    calAddNeighbor2D(ca2D,   0,   0);
//    calAddNeighbor2D(ca2D, - 1, + 1);
//    calAddNeighbor2D(ca2D, - 1,   0);
//    calAddNeighbor2D(ca2D, - 1, - 1);
//    calAddNeighbor2D(ca2D,   0, - 1);
//    calAddNeighbor2D(ca2D, + 1,   0);
//    calAddNeighbor2D(ca2D,   0, + 1);

    /*
           | 2 |
        ---|---|---
         3 | 0 | 1		if (j%2 == 1), i.e. odd columns
        ---|---|---
         4 | 5 | 6
    */

//    calAddNeighbor2D(ca2D,   0,   0);
//    calAddNeighbor2D(ca2D,   0, + 1);
//    calAddNeighbor2D(ca2D, - 1,   0);
//    calAddNeighbor2D(ca2D,   0, - 1);
//    calAddNeighbor2D(ca2D, + 1, - 1);
//    calAddNeighbor2D(ca2D, + 1,   0);
//    calAddNeighbor2D(ca2D, + 1, + 1);

//    calModel->sizeof_X = 7;

}
