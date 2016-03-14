#ifndef calSubstateWrapper_h
#define calSubstateWrapper_h
#include <OpenCAL++11/calActiveCells.h>
#include <OpenCAL++11/calConverterIO.h>

class CALSubstateWrapper 
{
public:
    virtual ~ CALSubstateWrapper () {}
    virtual void update (CALActiveCells* activeCells) = 0;
    virtual void saveSubstate (int* coordinates, size_t dimension, CALConverterIO* calConverterInputOutput, char* path) = 0;
    virtual void loadSubstate (int* coordinates, size_t dimension, CALConverterIO* calConverterInputOutput, char* path) = 0;
};


#endif
