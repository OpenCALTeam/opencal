#include "OpenCAL-CPU/calNeighborPool.h"

static int pow_ct(const int base, unsigned const exponent)
{
  // (parentheses not required in next line)
  return (exponent == 0)     ? 1 :
         (exponent % 2 == 0) ? pow_ct(base, exponent / 2) * pow_ct(base,
                                                                   exponent / 2) :
         base *pow_ct(base,
                      (exponent - 1) / 2) * pow_ct(base, (exponent - 1) / 2);
}

int * generateAlphabethMooreNeighborhood(int radius,int alphabethsize)
{
    int* alphabet = (int *) malloc(sizeof(int)*alphabethsize);
    alphabet[0] = 0;

    for(int i = 1 , pos = 1 ; i <= radius ; i++ ){
        alphabet[pos++] = i;
        alphabet[pos++] = -i;

    }
    return alphabet;
}


static int getNeighborNLinear(struct CALNeighborPool * calNeighborPool,struct CALIndexesPool* calIndexesPool, int *indexes, int * neighbor){
    int i;
    int c = 0;
    int t = calNeighborPool->neighborPool_size;
    if (calNeighborPool->CAL_TOROIDALITY == CAL_SPACE_FLAT)
    {
        for (i = 0; i < calIndexesPool->cellular_space_dimension; ++i)
        {
            t= t/calIndexesPool->coordinates_dimensions[i];
            c+=(indexes[i] + neighbor[i])*t;
        }
    }
    else
    {
        for (i=0; i< calIndexesPool->cellular_space_dimension; i++)
        {
            t= t/calIndexesPool->coordinates_dimensions[i];
            c+=(calGetToroidalX(indexes[i] + neighbor[i], calIndexesPool->coordinates_dimensions[i]))*t;

        }
    }
    return c;

}

static void addNeighbor(struct CALNeighborPool * calNeighborPool,struct CALIndexesPool* calIndexesPool,int * cellPattern){
    for (int i = 0; i < calNeighborPool->neighborPool_size; ++i)
    {
        int* neighborsTmp = calNeighborPool->neighborPool[i];
        int* neighborsNew = (int*)malloc(sizeof(int)*(calNeighborPool->size_of_X + 1));
        for (int k = 0; k < calNeighborPool->size_of_X; ++k)
        {
            neighborsNew [k] = calNeighborPool->neighborPool[i][k];
        }

        int* multidimensionalIndex = calIndexesPool->pool[i];
        int toAdd = getNeighborNLinear(calNeighborPool, calIndexesPool, multidimensionalIndex, cellPattern);
        neighborsNew[calNeighborPool->size_of_X]= toAdd;
        calNeighborPool->neighborPool[i]= neighborsNew;

        if (neighborsTmp)
            free(neighborsTmp);
    }

    calNeighborPool->size_of_X++;

}

struct CALNeighborPool * calDefNeighborPool(struct CALIndexesPool* calIndexesPool, enum CALSpaceBoundaryCondition _CAL_TOROIDALITY,int ** cellPattern){

    struct CALNeighborPool *calNeighborPool = (struct CALNeighborPool *)malloc(sizeof(struct CALNeighborPool));

    for (int i = 0; i < calIndexesPool->cellular_space_dimension; ++i) {
        calNeighborPool->neighborPool_size*=calIndexesPool->coordinates_dimensions[i];
    }
    calNeighborPool->neighborPool =(int **)malloc (sizeof(int*)*calNeighborPool->neighborPool_size);
    calNeighborPool->CAL_TOROIDALITY = _CAL_TOROIDALITY;
    for (uint i = 0; i < calNeighborPool->neighborPool_size; ++i) {
        calNeighborPool->neighborPool[i] = NULL;
    }
    calNeighborPool->size_of_X = 0;
    addNeighbors(calNeighborPool, calIndexesPool, cellPattern);
    return calNeighborPool;
}

int ** defineMooreNeighborhood(int radius,int dimension) {
    int alphabethsize = 2*radius+1;
    int * alphabet = generateAlphabethMooreNeighborhood(radius,alphabethsize);

    int indices_size = pow_ct(radius, dimension);
    int ** indices = (int **) malloc(sizeof(int*) * indices_size);

    for (int i = 0; i < indices_size; ++i) {
        indices[i] = (int*) malloc(sizeof(int) * dimension);
    }

    for (int i = 0; i < indices_size; i++)
     for (int pos = 0 , v=i ; pos < dimension; pos++, v/=alphabethsize)
        indices[i][pos] = alphabet[v % alphabethsize];

     return indices;
  }

int **  defineVonNeumannNeighborhood(int radius,int dimension)
{
    int indices_size = pow_ct(radius, dimension);
    int ** indices = (int **) malloc(sizeof(int*) * indices_size);
    for (int i = 0; i < indices_size; ++i) {
        indices[i] = (int*) malloc(sizeof(int) * dimension);
    }


  //  indices[0] = NULL;//central cell
    //total number of insertions is 2*Dimension+1
    //TODO chiedere paola indices[i] = {0}
    for (int i = 1; i <= dimension; ++i)
    {
     for(int r = 1; r <= radius ; ++r){
        indices[i][i-1] = -r;
     }
    }
    int c = dimension -1;
    for (int i = dimension+1; i <= 2*dimension; ++i,--c)
    {
       for(int r = 1; r<=radius ; ++r){
        indices[i][c] = r;
       }
    }

    return indices;
}

void addNeighbors(struct CALNeighborPool * calNeighborPool,struct CALIndexesPool* calIndexesPool, int ** cellPattern){
    for (int i = 0; i < calIndexesPool->cellular_space_dimension; i++){
        addNeighbor(calNeighborPool, calIndexesPool, cellPattern[i]);
        //this->sizeof_X ++;
    }
}

void destroy(struct CALNeighborPool * calNeighborPool){

    for(int i = 0; i < calNeighborPool->neighborPool_size; i++)
    {
        free(calNeighborPool->neighborPool[i]);
    }
    free(calNeighborPool->neighborPool);

}


