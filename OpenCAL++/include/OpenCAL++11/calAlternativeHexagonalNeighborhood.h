#ifndef calAlternativeHexagonalNeighborhood_h
#define calAlternativeHexagonalNeighborhood_h

#include <OpenCAL++11/calModel.h>
#include <OpenCAL++11/calNeighborhood.h>

class CALAlternativeHexagonalNeighborhood : public CALNeighborhood
{
public:
   virtual void defineNeighborhood (CALModel* calModel);
};

#endif
