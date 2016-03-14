#ifndef calIndexesPool_h
#define calIndexesPool_h

#include <cassert>
#include <stdlib.h>
#include <iostream>

class CALIndexesPool
{
protected:

    static CALIndexesPool* instance;

private:
    int ** indexesPool = nullptr;
    int size;
    int * coordinates = nullptr;
    size_t dimension;



    CALIndexesPool (int size, int* coordinates, size_t dimension);
    ~CALIndexesPool ();

    void initIndexes ();
public:
    static void init (int size, int* coordinates, size_t dimension);

    static int* getMultidimensionalIndexes (int linearIndex);

    static void destroy ();


};





#endif
