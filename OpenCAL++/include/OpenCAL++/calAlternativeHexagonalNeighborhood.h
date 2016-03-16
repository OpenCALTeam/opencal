#ifndef calAlternativeHexagonalNeighborhood_h
#define calAlternativeHexagonalNeighborhood_h

#include <OpenCAL++/calModel.h>
#include <OpenCAL++/calNeighborhood.h>

class CALAlternativeHexagonalNeighborhood : public CALNeighborhood
{
public:
   virtual void defineNeighborhood (CALModel* calModel);
};

#endif
