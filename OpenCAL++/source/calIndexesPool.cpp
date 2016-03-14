#include <OpenCAL++11/calIndexesPool.h>

CALIndexesPool :: CALIndexesPool (int size, int* coordinates, size_t dimension)
{
    this->indexesPool = new int* [size];
    this->coordinates = coordinates;
    this->dimension = dimension;
    this->size = size;
}

CALIndexesPool :: ~CALIndexesPool()
{
    for(int i = 0; i < this->size; i++)
    {
        delete[] this->indexesPool[i];
    }
    delete [] this->indexesPool;
}


CALIndexesPool* CALIndexesPool :: instance = nullptr;



void CALIndexesPool :: init (int size, int* coordinates, size_t dimension)
{
    if (instance == nullptr)
    {
        instance = new CALIndexesPool (size, coordinates, dimension);
        instance->initIndexes();
    }
}
int* CALIndexesPool:: getMultidimensionalIndexes (int linearIndex)
{
    assert (instance!= nullptr);
    assert(linearIndex<instance->size);
    return instance->indexesPool[linearIndex];

}

void CALIndexesPool::initIndexes ()
{
    for (int i = 0; i < this->size; ++i)
    {
        int n, k;
        int linearIndex = i;
        int* v = new int [this->dimension];
        int t =this-> size;
        for (n=this->dimension-1; n>=0; n--)
        {
            if (n ==1)
                k=0;
            else if (n==0)
                k=1;
            else
                k=n;

            t= (int)t/this->coordinates[k];
            v[k] = (int) linearIndex/t;
            linearIndex = linearIndex%t;
        }
        this->indexesPool[i] = v;

    }

}

void CALIndexesPool:: destroy ()
{
    delete instance;
}


