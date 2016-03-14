#ifndef calNeighborhood_h
#define calNeighborhood_h

#include<cassert>

class CALModel;

/*! \brief Class that define the neighbourhood relation.
*/
class CALNeighborhood
{
public:
    /*! \brief Add patterns to CALModel neighbours according to the concrete class instantiated
    */
   virtual void defineNeighborhood (CALModel* calModel //!<Pointer to the cellular automaton object
                                    ) = 0;
};

#endif
