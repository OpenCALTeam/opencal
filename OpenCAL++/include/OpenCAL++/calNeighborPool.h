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
