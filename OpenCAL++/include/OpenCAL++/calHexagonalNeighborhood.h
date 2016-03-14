#ifndef calHexagonalNeighborhood_h
#define calHexagonalNeighborhood_h

#include <OpenCAL++/calModel.h>


class CALHexagonalNeighborhood : public CALNeighborhood
{
public:
   virtual void defineNeighborhood (CALModel* calModel);
};

#endif
