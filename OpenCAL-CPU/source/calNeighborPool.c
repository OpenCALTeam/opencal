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


static int getNeighborNLinear(struct CALNeighborPool * calNeighborPool,struct CALIndexesPool* calIndexesPool, int *indexes, int * neighbor)
{
    int i;
    int c = 0;
    int t = calNeighborPool->neighborPool_size;
    if (calNeighborPool->CAL_TOROIDALITY == CAL_SPACE_FLAT)
    {
        for (i = 0; i < calIndexesPool->number_of_dimensions; ++i)
        {
            t = t/calIndexesPool->coordinates_dimensions[i];
            c += (indexes[i] + neighbor[i])*t;
        }
    }
    else
    {
        for (i = 0; i < calIndexesPool->number_of_dimensions; i++)
        {
            t = t / calIndexesPool->coordinates_dimensions[i];
            c += (calGetToroidalX((indexes[i] + neighbor[i]), calIndexesPool->coordinates_dimensions[i]))*t;

        }
    }
    return c;

}

static void addNeighbor(struct CALNeighborPool * calNeighborPool,struct CALIndexesPool* calIndexesPool,int * cellPattern)
{
    for (int i = 0; i < calNeighborPool->neighborPool_size; ++i)
    {
        int* neighborsTmp = calNeighborPool->neighborPool[i];
        int* neighborsNew = (int*)malloc(sizeof(int)*(calNeighborPool->numb_of_neighbours + 1));
        for (int k = 0; k < calNeighborPool->numb_of_neighbours; ++k)
        {
            neighborsNew [k] = calNeighborPool->neighborPool[i][k];
        }
        int* multidimensionalIndex = calIndexesPool->pool[i];
        int toAdd = getNeighborNLinear(calNeighborPool, calIndexesPool, multidimensionalIndex, cellPattern);
        neighborsNew[calNeighborPool->numb_of_neighbours]= toAdd;
        calNeighborPool->neighborPool[i]= neighborsNew;

        if (neighborsTmp)
            free(neighborsTmp);
    }

    calNeighborPool->numb_of_neighbours++;
}

struct CALNeighborPool * calDefNeighborPool(struct CALIndexesPool* calIndexesPool, enum CALSpaceBoundaryCondition _CAL_TOROIDALITY,struct CALNeighbourhoodPattern* cellPattern, int radius){

    struct CALNeighborPool *calNeighborPool = (struct CALNeighborPool *)malloc(sizeof(struct CALNeighborPool));

    calNeighborPool->neighborPool_size = calIndexesPool->cellular_space_dimension;
    calNeighborPool->neighborPool =(int **)malloc (sizeof(int*)*calNeighborPool->neighborPool_size);
    calNeighborPool->CAL_TOROIDALITY = _CAL_TOROIDALITY;
    for (uint i = 0; i < calNeighborPool->neighborPool_size; ++i) {
        calNeighborPool->neighborPool[i] = NULL;
    }
    calNeighborPool->numb_of_neighbours = 0;
    calNeighborPool->size_of_X = cellPattern->size_of_X;
    addNeighbors(calNeighborPool, calIndexesPool, cellPattern, radius);
    return calNeighborPool;
}

struct CALNeighbourhoodPattern* defineMooreNeighborhood(int radius,int dimension) {

    int alphabethsize = 2*radius+1;
    int * alphabet = generateAlphabethMooreNeighborhood(radius,alphabethsize);

    int indices_size = pow_ct(alphabethsize, dimension);
    int ** indices = (int **) malloc(sizeof(int*) * indices_size);

    for (int i = 0; i < indices_size; ++i) {
        indices[i] = (int*) malloc(sizeof(int) * dimension);
    }

    for (int i = 0; i < indices_size; i++)
        for (int pos = 0 , v=i ; pos < dimension; pos++, v/=alphabethsize)
            indices[i][pos] = alphabet[v % alphabethsize];

    struct CALNeighbourhoodPattern* pattern = (struct CALNeighbourhoodPattern*) malloc (sizeof(struct CALNeighbourhoodPattern));
    pattern->cell_pattern = indices;
    pattern->size_of_X = indices_size;
    return pattern;
}

struct CALNeighbourhoodPattern*  defineVonNeumannNeighborhood(int radius,int dimension)
{
    int indices_size = radius * 2 + dimension + 1;
    int ** indices = (int **) malloc(sizeof(int*) * indices_size);
    for (int i = 0; i < indices_size; ++i) {
        indices[i] = (int*) malloc(sizeof(int) * dimension);
        for(int n = 0; n < dimension; n++)
            indices[i][n] = 0;
    }

    for (int i = 1; i <= dimension; ++i)
        for(int r = 1; r <= radius ; ++r)
            indices[i][i-1] = -r;

    int c = dimension -1;
    for (int i = dimension+1; i <= 2*dimension; ++i,--c)
    {
        for(int r = 1; r <= radius ; ++r){
            indices[i][c] = r;
        }
    }

    struct CALNeighbourhoodPattern* pattern = (struct CALNeighbourhoodPattern*) malloc (sizeof(struct CALNeighbourhoodPattern));
    pattern->cell_pattern = indices;
    pattern->size_of_X = indices_size;
    return pattern;
}

void addNeighbors(struct CALNeighborPool * calNeighborPool,struct CALIndexesPool* calIndexesPool, struct CALNeighbourhoodPattern* cellPattern,int radius){

    for (int i = 0; i < cellPattern->size_of_X; i++)
        addNeighbor(calNeighborPool, calIndexesPool, cellPattern->cell_pattern[i]);

}

void destroy(struct CALNeighborPool * calNeighborPool){

    for(int i = 0; i < calNeighborPool->neighborPool_size; i++)
    {
        free(calNeighborPool->neighborPool[i]);
    }
    free(calNeighborPool->neighborPool);

}
