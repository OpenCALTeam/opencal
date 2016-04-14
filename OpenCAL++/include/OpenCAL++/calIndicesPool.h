//
// Created by knotman on 14/04/16.
//

#ifndef OPENCAL_ALL_CALINDICESPOOL_H
#define OPENCAL_ALL_CALINDICESPOOL_H

#include <array>
namespace opencal {

    template<uint DIMENSION, typename COORDINATE_TYPE>
    class CALIndicesPool {
    protected:

        static CALIndicesPool<DIMENSION , COORDINATE_TYPE>* instance;

    private:
        int **indexesPool = nullptr;
        int size;
        std::array <COORDINATE_TYPE, DIMENSION> coordinates;



        CALIndicesPool(int size, std::array<COORDINATE_TYPE, DIMENSION>& coordinates){
            this->indexesPool = new int* [size];
            this->coordinates = coordinates;
            this->size = size;
        }

        ~CALIndicesPool(){
            for(int i = 0; i < this->size; i++)
            {
                delete[] this->indexesPool[i];
            }
            delete [] this->indexesPool;
        }

        void initIndexes(){
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

    public:
        static void init(int size, std::array<COORDINATE_TYPE, DIMENSION>& coordinates){
            if (instance == nullptr)
            {
                instance = new CALIndicesPool (size, coordinates);
                instance->initIndexes();
            }
        }

        static int *getMultidimensionalIndexes(int linearIndex){
            assert (instance!= nullptr);
            assert(linearIndex<instance->size);
            return instance->indexesPool[linearIndex];

        }

        static void destroy(){
            delete instance;
        }


    };

    template<uint DIMENSION, typename COORDINATE_TYPE>
    CALIndicesPool<DIMENSION,COORDINATE_TYPE>* CALIndicesPool<DIMENSION,COORDINATE_TYPE>::instance = nullptr;

} //namespace opencal

#endif //OPENCAL_ALL_CALINDICESPOOL_H
