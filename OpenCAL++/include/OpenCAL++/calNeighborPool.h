
//
// Created by knotman on 14/04/16.
//

#ifndef OPENCAL_ALL_CALNEIGHBORPOOL_H
#define OPENCAL_ALL_CALNEIGHBORPOOL_H

#include<array>
namespace opencal{
    template <uint DIMENSION , typename COORDINATE_TYPE>
    class CALNeighborPool {
    protected:

        static CALNeighborPool *instance;
    private:
        int **neighborPool = NULL;
        int size;
        std::array <COORDINATE_TYPE, DIMENSION> coordinates;


        int size_of_X;
        enum calCommon::CALSpaceBoundaryCondition CAL_TOROIDALITY;

        CALNeighborPool(int size, std::array <COORDINATE_TYPE, DIMENSION> &coordinates,
                        enum calCommon::CALSpaceBoundaryCondition CAL_TOROIDALITY){

            this->neighborPool = new int* [size];
            this->coordinates = coordinates;
            this->size = size;
            this->size_of_X = 0;
            this->CAL_TOROIDALITY = CAL_TOROIDALITY;
        }

        ~CALNeighborPool(){
            for(int i = 0; i < this->size; i++)
            {
                delete[] this->neighborPool[i];
            }
            delete [] this->neighborPool;
        }

    public:
        static void destroy(){
            delete instance;
        }

        static void init(int size, std::array <COORDINATE_TYPE, DIMENSION> &coordinates,
                         enum calCommon::CALSpaceBoundaryCondition CAL_TOROIDALITY){

            if (instance == nullptr)
                instance = new CALNeighborPool(size,coordinates,CAL_TOROIDALITY);
        }

        static CALNeighborPool *getInstance(){
            return instance;
        }

        inline int getNeighborN(int linearIndex, int n) {
            //TODO manage hexagonalNeighborhood
            assert(linearIndex < instance->size);
            assert(n < instance->size_of_X);

            return instance->neighborPool[linearIndex][n];

        }

        void addNeighbor(int *cellPattern){
            for (int i = 0; i < instance->size; ++i)
            {
                int* neighborsTmp = instance->neighborPool[i];
                int* neighborsNew = new int [instance->size_of_X + 1];
                for (int k = 0; k < instance->size_of_X; ++k)
                {
                    neighborsNew [k] = instance->neighborPool[i][k];
                }

                int* multidimensionalIndex = opencal::calCommon::cellMultidimensionalIndices<DIMENSION,COORDINATE_TYPE>(i);
                int toAdd = opencal::calCommon::getNeighborNLinear(multidimensionalIndex,cellPattern,instance->coordinates, instance->dimension,instance->CAL_TOROIDALITY );
                neighborsNew[instance->size_of_X]= toAdd;
                instance->neighborPool[i]= neighborsNew;

                delete [] neighborsTmp;
            }
            instance->size_of_X++;
        }
    };

    //initialize static member
    template <uint DIMENSION , typename COORDINATE_TYPE>
    CALNeighborPool<DIMENSION , COORDINATE_TYPE>* CALNeighborPool<DIMENSION , COORDINATE_TYPE>:: instance = nullptr;

} //namespace opencal
#endif //OPENCAL_ALL_CALNEIGHBORPOOL_H
=======
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
    static void destroy();
    static void init (std::array<COORDINATE_TYPE, DIMENSION> _coordinates, enum calCommon::CALSpaceBoundaryCondition _CAL_TOROIDALITY);

    static inline int getNeighborN (int linearIndex, int n)
    {
        //TODO manage hexagonalNeighborhood
        assert(linearIndex<size);
        assert (n <size_of_X);

        return neighborPool[linearIndex][n];

    }

    static void addNeighbor (std::array<COORDINATE_TYPE, DIMENSION>& cellPattern);
    static inline std::array<COORDINATE_TYPE,DIMENSION>& cellMultidimensionalIndexes(int index)
    {
        return CALIndexesPool<DIMENSION, COORDINATE_TYPE>:: getMultidimensionalIndexes(index);
    }

    static int getNeighborNLinear (const std::array<COORDINATE_TYPE,DIMENSION>& indexes, const std::array<COORDINATE_TYPE,DIMENSION>& neighbor);



};

}//namespace opencal

template<uint DIMENSION, typename COORDINATE_TYPE>
int** opencal::CALNeighborPool<DIMENSION,COORDINATE_TYPE>:: neighborPool = nullptr;

template<uint DIMENSION, typename COORDINATE_TYPE>
int opencal::CALNeighborPool<DIMENSION,COORDINATE_TYPE>:: size = 0;

template<uint DIMENSION, typename COORDINATE_TYPE>
int opencal::CALNeighborPool<DIMENSION,COORDINATE_TYPE>:: size_of_X = 0;

template<uint DIMENSION, typename COORDINATE_TYPE>
void opencal::CALNeighborPool<DIMENSION, COORDINATE_TYPE>:: init (std::array<COORDINATE_TYPE, DIMENSION> _coordinates, enum calCommon::CALSpaceBoundaryCondition _CAL_TOROIDALITY)
{
    if (neighborPool == nullptr)
    {
        size = opencal::calCommon::multiplier<DIMENSION,uint>(coordinates,0);
        neighborPool = new int* [size];
        coordinates = _coordinates;
        CAL_TOROIDALITY = _CAL_TOROIDALITY;
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
void opencal::CALNeighborPool<DIMENSION, COORDINATE_TYPE>:: addNeighbor (std::array<COORDINATE_TYPE, DIMENSION>& cellPattern)
{
    for (int i = 0; i < size; ++i)
    {
        int* neighborsTmp = neighborPool[i];
        int* neighborsNew = new int [size_of_X + 1];
        for (int k = 0; k < size_of_X; ++k)
        {
            neighborsNew [k] = neighborPool[i][k];
        }

        std::array<COORDINATE_TYPE,DIMENSION> multidimensionalIndex = cellMultidimensionalIndexes(i);
        int toAdd = getNeighborNLinear(multidimensionalIndex,cellPattern );
        neighborsNew[size_of_X]= toAdd;
        neighborPool[i]= neighborsNew;

        delete [] neighborsTmp;
    }
    size_of_X++;

}

template<uint DIMENSION, typename COORDINATE_TYPE>
int opencal::CALNeighborPool<DIMENSION, COORDINATE_TYPE> :: getNeighborNLinear (const std::array<COORDINATE_TYPE,DIMENSION>& indexes, const std::array<COORDINATE_TYPE,DIMENSION>& neighbor)
{
    int i;
    int c = 0;
    int t = size;
    if (CAL_TOROIDALITY == calCommon::CAL_SPACE_FLAT)
        for (i = 0; i < DIMENSION; ++i)
        {
            t= t/coordinates[i];
            c+=(indexes[i] + neighbor[i])*t;
        }
    else
    {
        for (i=0; i< DIMENSION; i++)
        {
            t= t/coordinates[i];
            c+=(calGetToroidalX(indexes[i] + neighbor[i], coordinates[i]))*t;

        }

    }
    return c;
}


#endif

