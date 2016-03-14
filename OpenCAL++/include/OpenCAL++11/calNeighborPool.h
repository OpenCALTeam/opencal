#ifndef calNeighborhoodPool_h
#define calNeighborhoodPool_h

#include <cassert>
#include <stdlib.h>
#include <iostream>
#include <OpenCAL++11/calCommon.h>

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
