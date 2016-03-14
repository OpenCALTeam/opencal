#include <OpenCAL++11/calVonNeumannNeighborhood.h>

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
