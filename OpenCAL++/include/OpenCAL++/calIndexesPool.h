#ifndef calIndexesPool_h
#define calIndexesPool_h

#include <cassert>
#include <stdlib.h>
#include <iostream>
#include <array>

#include<OpenCAL++/calCommon.h>

namespace opencal {

template<uint DIMENSION, typename COORDINATE_TYPE = int>
class CALIndexesPool
{
protected:
    static std::array <COORDINATE_TYPE, DIMENSION>* indexesPool;
    static std::array <COORDINATE_TYPE, DIMENSION> coordinates;
    static uint size;


    static void initIndexes ();
public:
    static void init (std::array<COORDINATE_TYPE, DIMENSION>& _coordinates);

    inline static std::array <COORDINATE_TYPE, DIMENSION>& getMultidimensionalIndexes (int linearIndex)
    {
        assert(linearIndex<size);
        return indexesPool[linearIndex];
    }

    static void destroy ();


};


} //namespace opencal




template<uint DIMENSION, typename COORDINATE_TYPE>
std::array <COORDINATE_TYPE, DIMENSION>* opencal::CALIndexesPool<DIMENSION,COORDINATE_TYPE>:: indexesPool = nullptr;
template<uint DIMENSION, typename COORDINATE_TYPE>
uint opencal::CALIndexesPool<DIMENSION,COORDINATE_TYPE>:: size = 0;

template<uint DIMENSION, typename COORDINATE_TYPE>
void opencal::CALIndexesPool<DIMENSION,COORDINATE_TYPE> :: init (std::array<COORDINATE_TYPE, DIMENSION>& _coordinates)
{

    size = opencal::calCommon::multiplier<DIMENSION,uint>(coordinates,0);
    coordinates = _coordinates;
    indexesPool = new std::array <COORDINATE_TYPE, DIMENSION> [size];
    initIndexes();
}

template<uint DIMENSION, typename COORDINATE_TYPE>
void opencal::CALIndexesPool<DIMENSION,COORDINATE_TYPE>::initIndexes ()
{
    for (int i = 0; i < size; ++i)
    {
        int n, k;
        int linearIndex = i;
        std::array <COORDINATE_TYPE, DIMENSION> v;
        int t =size;
        for (n=DIMENSION-1; n>=0; n--)
        {
            if (n ==1)
                k=0;
            else if (n==0)
                k=1;
            else
                k=n;

            t= (int)t/coordinates[k];
            v[k] = (int) linearIndex/t;
            linearIndex = linearIndex%t;
        }
        indexesPool[i] = &v;

    }

}

template<uint DIMENSION, typename COORDINATE_TYPE>
void opencal::CALIndexesPool<DIMENSION,COORDINATE_TYPE>:: destroy ()
{
    delete [] indexesPool;
}

#endif
