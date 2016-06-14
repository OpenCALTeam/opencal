#ifndef calNeighborPool_h
#define calNeighborPool_h
#include <OpenCAL-CPU/calCommon.h>


/*! \brief Enumeration of  neighbourhood.

    Enumeration that identifies the cellular automaton's  neighbourhood.
*/
enum CALNeighborhood {
    //TODO CAL_CUSTOM_NEIGHBORHOOD, CAL_HEXAGONAL_NEIGHBORHOOD, CAL_HEXAGONAL_NEIGHBORHOOD_ALT
    CAL_CUSTOM_NEIGHBORHOOD,		//!< Enumerator used for the definition of a custom  neighbourhood; this is built by calling the function calAddNeighbor.
    CAL_VON_NEUMANN_NEIGHBORHOOD,	//!< Enumerator used for specifying the  von Neumann neighbourhood; no calls to calAddNeighbor are needed.
    CAL_MOORE_NEIGHBORHOOD,			//!< Enumerator used for specifying the  Moore neighbourhood; no calls to calAddNeighbor are needed.
    CAL_HEXAGONAL_NEIGHBORHOOD,		//!< Enumerator used for specifying the  Moore Hexagonal neighbourhood; no calls to calAddNeighbor are needed.
    CAL_HEXAGONAL_NEIGHBORHOOD_ALT	//!< Enumerator used for specifying the alternative 90� rotated  Moore Hexagonal neighbourhood; no calls to calAddNeighbor are needed.
};


struct CALNeighborPool{
    int ** neighborPool;
    int neighborPool_size;
    int size_of_X;
    enum CALSpaceBoundaryCondition CAL_TOROIDALITY;
};

struct CALNeighborPool * calDefNeighborPool(struct CALIndexesPool* calIndexesPool, enum CALSpaceBoundaryCondition _CAL_TOROIDALITY, int **cellPattern);

int ** defineMooreNeighborhood(int radius,int dimension);

int ** defineVonNeumannNeighborhood(int radius,int dimension);

void addNeighbors(struct CALNeighborPool * calNeighborPool,struct CALIndexesPool* calIndexesPool, int ** cellPattern);

void destroy(struct CALNeighborPool * calNeighborPool);

#endif
