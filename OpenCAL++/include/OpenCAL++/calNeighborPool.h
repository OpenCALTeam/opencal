#ifndef calNeighborhoodPool_h
#define calNeighborhoodPool_h

#include <cassert>
#include <stdlib.h>
#include <iostream>
#include <OpenCAL++/calCommon.h>
#include<OpenCAL++/calIndexesPool.h>

namespace opencal{


    template<uint DIMENSION, typename COORDINATE_TYPE = int>
    class CALNeighborPool
    {
    protected:
        static int ** neighborPool;
        static int size;
        static std::array<COORDINATE_TYPE, DIMENSION> coordinates;
        static int size_of_X;
        static enum calCommon::CALSpaceBoundaryCondition CAL_TOROIDALITY;
    public:
        static inline void destroy();
        static inline void init (std::array<COORDINATE_TYPE, DIMENSION> _coordinates, enum calCommon::CALSpaceBoundaryCondition _CAL_TOROIDALITY);

        static inline int getNeighborN (int linearIndex, int n)
        {
            //TODO manage hexagonalNeighborhood
            assert(linearIndex<size);
            assert (n <size_of_X);

            return neighborPool[linearIndex][n];

        }

        static void addNeighbor (auto& cellPattern);


        static int getNeighborNLinear (const std::array<COORDINATE_TYPE,DIMENSION>& indexes, auto& neighbor);

        static void stampa ()
        {
            std::cout<<"STAMPA VICINATO "<<size_of_X<<std::endl;
            for (int i = 0; i < size; i++)
            {
                for (int j = 0; j < size_of_X; j++)
                {
                    std::cout<<neighborPool[i][j] <<" ";
                }
                std::cout<<std::endl;
            }
        }


    };

}//namespace opencal

template<uint DIMENSION, typename COORDINATE_TYPE>
int** opencal::CALNeighborPool<DIMENSION,COORDINATE_TYPE>:: neighborPool = nullptr;

template<uint DIMENSION, typename COORDINATE_TYPE>
int opencal::CALNeighborPool<DIMENSION,COORDINATE_TYPE>:: size = 0;

template<uint DIMENSION, typename COORDINATE_TYPE>
int opencal::CALNeighborPool<DIMENSION,COORDINATE_TYPE>:: size_of_X = 0;

template<uint DIMENSION, typename COORDINATE_TYPE>
std::array<COORDINATE_TYPE, DIMENSION> opencal::CALNeighborPool<DIMENSION,COORDINATE_TYPE>:: coordinates;

template<uint DIMENSION, typename COORDINATE_TYPE>
enum opencal::calCommon::CALSpaceBoundaryCondition opencal::CALNeighborPool<DIMENSION,COORDINATE_TYPE>:: CAL_TOROIDALITY;

template<uint DIMENSION, typename COORDINATE_TYPE>
void opencal::CALNeighborPool<DIMENSION, COORDINATE_TYPE>:: init (std::array<COORDINATE_TYPE, DIMENSION> _coordinates, enum calCommon::CALSpaceBoundaryCondition _CAL_TOROIDALITY)
{
    if (neighborPool == nullptr)
    {
        coordinates = _coordinates;
        size = opencal::calCommon::multiplier<DIMENSION,uint>(coordinates,0);
        neighborPool = new int* [size];
        CAL_TOROIDALITY = _CAL_TOROIDALITY;
        for (uint i = 0; i < size; ++i) {
            neighborPool[i] = nullptr;
    }
    }
}


template<uint DIMENSION, typename COORDINATE_TYPE>
void opencal::CALNeighborPool<DIMENSION, COORDINATE_TYPE>:: destroy()
{
    for(int i = 0; i < size; i++)
    {
        delete[] neighborPool[i];
    }
    delete [] neighborPool;

}


template<uint DIMENSION, typename COORDINATE_TYPE>
void opencal::CALNeighborPool<DIMENSION, COORDINATE_TYPE>:: addNeighbor (auto& cellPattern)
{
    for (int i = 0; i < size; ++i)
    {
        int* neighborsTmp = neighborPool[i];
        int* neighborsNew = new int [size_of_X + 1];
        for (int k = 0; k < size_of_X; ++k)
        {
            neighborsNew [k] = neighborPool[i][k];
        }

        std::array<COORDINATE_TYPE,DIMENSION> multidimensionalIndex = CALIndexesPool<DIMENSION, COORDINATE_TYPE>:: getMultidimensionalIndexes(i);
        int toAdd = getNeighborNLinear(multidimensionalIndex,cellPattern );
        neighborsNew[size_of_X]= toAdd;
        neighborPool[i]= neighborsNew;

        if (neighborsTmp)
            delete [] neighborsTmp;
    }

    size_of_X++;


}

template<uint DIMENSION, typename COORDINATE_TYPE>
int opencal::CALNeighborPool<DIMENSION, COORDINATE_TYPE> :: getNeighborNLinear (const std::array<COORDINATE_TYPE,DIMENSION>& indexes, auto& neighbor)
{
    int i;
    int c = 0;
    int t = size;
    if (CAL_TOROIDALITY == calCommon::CAL_SPACE_FLAT)
    {
        for (i = 0; i < DIMENSION; ++i)
        {
            t= t/coordinates[i];
            c+=(indexes[i] + neighbor[i])*t;
        }
    }
    else
    {
        for (i=0; i< DIMENSION; i++)
        {
            t= t/coordinates[i];
            c+=(calCommon::getToroidalX(indexes[i] + neighbor[i], coordinates[i]))*t;

        }
    }
    return c;
}





#endif
