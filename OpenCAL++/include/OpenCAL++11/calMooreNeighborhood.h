#ifndef calMooreNeighborhood_h
#define calMooreNeighborhood_h

#include <OpenCAL++11/calNeighborhood.h>
#include <OpenCAL++11/calModel.h>
#include <cmath>

class CALMooreNeighborhood : public CALNeighborhood
{
public:
    CALMooreNeighborhood(){}
   virtual void defineNeighborhood (CALModel* calModel);
};

#endif
