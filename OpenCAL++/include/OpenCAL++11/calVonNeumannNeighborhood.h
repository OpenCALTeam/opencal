#ifndef calVonNeumannNeighborhood_h
#define calVonNeumannNeighborhood_h

#include <OpenCAL++11/calNeighborhood.h>
#include <OpenCAL++11/calModel.h>

class CALVonNeumannNeighborhood : public CALNeighborhood
{
public:
   virtual void defineNeighborhood (CALModel* calModel);
};

#endif
