#ifndef calVonNeumannNeighborhood_h
#define calVonNeumannNeighborhood_h

#include <OpenCAL++/calNeighborhood.h>
#include <OpenCAL++/calModel.h>

class CALVonNeumannNeighborhood : public CALNeighborhood
{
public:
   virtual void defineNeighborhood (CALModel* calModel);
};

#endif
