#ifndef calMooreNeighborhood_h
#define calMooreNeighborhood_h

#include <OpenCAL++/calNeighborhood.h>
#include <OpenCAL++/calModel.h>
#include <cmath>

class CALMooreNeighborhood : public CALNeighborhood
{
public:
    CALMooreNeighborhood(){}
   virtual void defineNeighborhood (CALModel* calModel);
};

#endif
